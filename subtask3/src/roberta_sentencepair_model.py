
# INCOMPLETE

# Advice changing to Simple Transformers for RoBERTa
# https://medium.com/swlh/solving-sentence-pair-tasks-using-simple-transformers-2496fe79d616

import os
import sys
import io
import random
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components

import mxnet as mx
import gluonnlp as nlp
from gluonnlp.calibration import BertLayerCollector
from gluonnlp.data import get_tokenizer

from bert import data # # Download required scripts from - http://gluon-nlp.mxnet.io
from utils import *
from code_sample import read, evaluate

nlp.utils.check_version('0.8.1')

# CPU/GPU Setup
np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.cpu()

# change `ctx` to `mx.cpu()` if no GPU is available.
# change `ctx` to `mx.gpu(0)` if GPU is available.

# Loss Functions - Softmax Cross Entropy Loss for Classification
loss_function = mx.gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

class ROBERTASentencePair():

    def __init__(self, model_params_path=None, dataset_name='openwebtext_ccnews_stories_books_cased'):

        # 1. ROBERTA Pre-Trained Model
        roberta_base, vocabulary = nlp.model.get_model('roberta_12_768_12',
                                                       dataset_name=dataset_name, # 'wiki_multilingual_uncased'
                                                       pretrained=True, ctx=ctx, use_pooler=True,
                                                       use_decoder=False, use_classifier=False)
        # 2. ROBERTA Classifier
        roberta_classifier = nlp.model.RoBERTaClassifier(roberta_base, num_classes=2, dropout=0.1)

        # Only need to initialize the Classifier Layer
        roberta_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
        roberta_classifier.hybridize(static_alloc=True)

        # Load Model Parameters (if available)
        if isinstance(model_params_path, str):
            roberta_classifier.load_parameters(model_params_path, ctx=ctx)

        self.classifier = roberta_classifier
        self.classifier_to_evaluate = roberta_classifier
        self.vocabulary = vocabulary
        self.tokenizer = get_tokenizer(model_name='roberta_12_768_12', dataset_name=dataset_name)

        # To-fix:
        self.tokenizer = self.tokenizer(self.vocabulary, lower=True)

        # # Load Model
        # import warnings
        # with warnings.catch_warnings():
        #     sym = mx.sym.load_json(open('subtask3_models/en_full_train-symbol.json', 'r').read())
        #     deserialized_net = mx.gluon.SymbolBlock(outputs=sym, inputs=mx.sym.var('data'))
        #     deserialized_net.load_parameters('subtask3_models/en_full_train-0003.params', allow_missing=True, ctx=ctx, ignore_extra=True)

        # deserialized_net = mx.gluon.nn.SymbolBlock.imports('subtask3_models/en_full_train-symbol.json',
        #                                                      ['data'],
        #                                                      'subtask3_models/en_full_train-0003.params', ctx=ctx)
    def fit(self, data, save_model_as, col_id='id', shuffle=True, val_size=0.2,
            log_interval=4, num_epochs=3, optimize_metric='f1'):
        """
        Train and save best model, given document data (list of dict).
            • Given data ("data") is split into 2 groups: (i) Train Data, (ii) Validation Data.
            • The data is split such that all data with same id ("col_id") are in the same group,
              thus preserving unseen id in Validation Data.
            • Model is trained only on Train Data, and validated on Validation Data.
            • Model with best-performing metric ("optimize_metric") is saved.
        """
        # Convert to Pandas DataFrame (pairwise sentences)
        print('Extracting pairwise sentences and coreferential status from data...')
        if isinstance(data, list):
            df_pairwise = extract_pairwise_sentence_relationships(data)
        print('Extraction complete.\n')

        # Train-Test Split
        print('Splitting data...')
        train_data, val_data = train_test_split_by_segment(df_pairwise, test_size=val_size,
                                                           segment=col_id, shuffle=shuffle)
        print('Splitting data complete.\n')

        # Train model
        self.train(train_data, val_data, log_interval=log_interval, num_epochs=num_epochs,
                   save_model_as=save_model_as, optimize_metric=optimize_metric)

        pass

    def train(self, train_data, val_data, log_interval=4, num_epochs=3,
              save_model_as='roberta_sentencepair', optimize_metric='f1'):

        aesthetic_print_equals('Training Profile')

        model_identity = save_model_as.lower().replace(' ', '_')

        print(f'Train Data: {len(train_data)} entries')
        print(f'Validation Data: {len(val_data)} entries\n')
        print(f'Epochs: {num_epochs}')
        print(f'Save Model to: {model_identity}')
        print()

        aesthetic_print_equals('Model Training')

        # Hyperparameters
        batch_size = 32
        lr = 5e-6

        # Data Loader
        train_data_loader = self.make_data_loader(train_data, batch_size=batch_size)

        # Model
        model = self.classifier

        # Metrics
        metric = mx.metric.Accuracy()
        best_val_loss = 1000
        best_f1 = 0.0
        best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0, 'n': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

        # Loss Functions - Softmax Cross Entropy Loss for Classification
        loss_function = mx.gluon.loss.SoftmaxCELoss()
        loss_function.hybridize(static_alloc=True)

        # Trainer
        trainer = mx.gluon.Trainer(model.collect_params(), 'adam',
                                   {'learning_rate': lr, 'epsilon': 1e-9})

        # Collect all differentiable parameters
        # `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
        # The gradients for these params are clipped later
        params = [p for p in model.collect_params().values() if p.grad_req != 'null']
        grad_clip = 1

        # Epochs
        for epoch_id in range(num_epochs):

            metric.reset()
            step_loss = 0

            # (i) Training Loop
            for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(train_data_loader):

                with mx.autograd.record():

                    # Load the data to the GPU
                    token_ids = token_ids.as_in_context(ctx)
                    valid_length = valid_length.as_in_context(ctx)
                    segment_ids = segment_ids.as_in_context(ctx)
                    label = label.as_in_context(ctx)

                    # Forward Computation
                    out = model(token_ids, segment_ids, valid_length.astype('float32'))
                    ls = loss_function(out, label).mean()

                # Backwards Computation
                ls.backward()

                # Gradient Clipping
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(1)

                step_loss += ls.asscalar()
                metric.update([label], [out])

                # Model Performance
                if (batch_id + 1) % (log_interval) == 0:
                    print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                                 .format(epoch_id + 1, batch_id + 1, len(train_data_loader),
                                         step_loss / log_interval,
                                         trainer.learning_rate, metric.get()[1]))
                    step_loss = 0

            # (ii) Validation Loop
            self.classifier_to_evaluate = model
            val_fscore = self.evaluate_pairwise(val_data)
            print(f'[Epoch {epoch_id + 1} End of Batch], validation_f1={val_fscore["f1"]:.3f}, validation_recall={val_fscore["recall"]:.3f}, validation_accuracy={val_fscore["accuracy"]:.3f}')
            print(val_fscore)

            # Save Model Parameters only
            model_name = f'{model_identity}_model_epoch_{epoch_id + 1}'
            model_params_path = model_name + '.params'
            model.save_parameters(model_params_path)
            print(f'Saved model parameters to "{model_params_path}"')

            # Announce if achieve better metric
            if val_fscore[optimize_metric] > best_metrics[optimize_metric]:

                # Update Best Metrics
                best_metrics = val_fscore
                best_f1 = val_fscore['f1']
                # best_val_loss = val_loss_epoch

                # Update Best Model
                self.classifier = model
                self.best_metrics = val_fscore

                print(f'New best score {optimize_metric}: {val_fscore[optimize_metric]}')

            print()

        # Best Model
        self.classifier_to_evaluate = self.classifier

    def predict(self, data):
        """
        Return cluster predictions, given document data (list of dict).
        """
        preds = []

        for idx, instance in enumerate(data):

            # Get pairwise predictions
            df_instance_pairwise = extract_pairwise_sentence(instance)
            pairwise_predictions = self.predict_pairwise(test_data=df_instance_pairwise)
            df_instance_pairwise['pred_is_coreferential'] = pairwise_predictions

            # Sentence to Sentence ID
            sent_2_sent_id = get_sentence_2_sentence_id(instance['sentence_no'], instance['sentences'])
            df_instance_pairwise['sentence_1_id'] = df_instance_pairwise['sentence_1'].apply(lambda x: sent_2_sent_id[x])
            df_instance_pairwise['sentence_2_id'] = df_instance_pairwise['sentence_2'].apply(lambda x: sent_2_sent_id[x])

            # Get adjacency matrix
            df_adj_matrix = pd.DataFrame(0, index=instance['sentence_no'], columns=instance['sentence_no'])
            indexer = df_adj_matrix.index.get_indexer
            df_adj_matrix.values[indexer(df_instance_pairwise['sentence_1_id']), indexer(df_instance_pairwise['sentence_2_id'])] = df_instance_pairwise['pred_is_coreferential'].values
            df_adj_matrix.values[indexer(df_instance_pairwise['sentence_2_id']), indexer(df_instance_pairwise['sentence_1_id'])] = df_instance_pairwise['pred_is_coreferential'].values

            # Get clusters
            pred_total_clusters, pred_cluster_ids = connected_components(df_adj_matrix.values)

            result = dict()
            for i in range(pred_total_clusters):
                result[i] = []
            for cluster, sent_no in zip(pred_cluster_ids, instance['sentence_no']):
                result[cluster].append(sent_no)
            result = sorted(list(result.values()))

            preds.append({"id": instance['id'], "pred_clusters": result})

        return preds

    def predict_pairwise(self, test_data):
        """
        Return pairwise predictions, given sentence pairs in DataFrame (pandas.DataFrame) comprising columns:
            • 'sentence_1' - string of first sentence
            • 'sentence_2' - string of second sentence
        """
        # Data Loader
        test_data_loader = self.make_data_loader(test_data, is_test_dataset=True,
                                                 col_x1='sentence_1', col_x2= 'sentence_2', col_y='is_coreferential')

        # Model
        model = self.classifier

        # Predictions
        predictions = []

        for batch_id, (token_ids, segment_ids, valid_length) in enumerate(test_data_loader):

            # Load the data to the GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)

            # Forward Computation
            out = model(token_ids, segment_ids, valid_length.astype('float32'))
            out = [i.softmax() for i in out]
            predictions.append(out)

        predictions = list(itertools.chain.from_iterable(predictions))
        predictions = [pred.asnumpy().argmax() for pred in predictions]

        return predictions

    def evaluate_pairwise(self, val_data):
        """
        Given Validation Data, evaluate the model on sentence pair predictions, and return a
        dictionary of standard evaluation scores:
            • precision
            • recall
            • f1
            • accuracy
            • n - Total Number of
            • tp - True Positives
            • fp - False Positives
            • fn - False Negatives
            • tn - True Negatives

        Acceptable inputs for Validation Data (val_data):

            (1) Document data (list of dict) comprising keys:
                • 'event_clusters'
                • 'sentence_no'
                • 'sentences'
                • 'id'

            (2) DataFrame (pandas.DataFrame) comprising columns:
                • 'sentence_1' - string of first sentence
                • 'sentence_2' - string of second sentence
                • 'is_coreferential' - boolean of whether the two sentences are coreferential (ground truth)
        """
        # Convert to Pandas DataFrame (pairwise sentences)
        if isinstance(val_data, list):
            val_data = extract_pairwise_sentence_relationships(val_data)

        # Data Loader
        val_data_loader = self.make_data_loader(val_data,
                                                col_x1='sentence_1', col_x2= 'sentence_2', col_y='is_coreferential')

        # Model
        model = self.classifier_to_evaluate

        # Validation
        val_predictions = []
        val_truth = []
        val_loss = 0

        for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(val_data_loader):

            # Load the data to the GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)
            val_truth.append(label)

            # Forward Computation
            out = model(token_ids, segment_ids, valid_length.astype('float32'))

            ls = loss_function(out, label).mean()
            val_loss += ls.asscalar()

            val_pred = [i.softmax() for i in out]
            val_predictions.append(val_pred)

        val_predictions = list(itertools.chain.from_iterable(val_predictions))
        val_predictions = [pred.asnumpy().argmax() for pred in val_predictions]

        val_truth = list(itertools.chain.from_iterable(val_truth))
        val_truth = [pred.asnumpy()[0] for pred in val_truth]

        val_fscore = fscore(val_predictions, val_truth)

        return val_fscore

    def evaluate_scorch(self, val_data, only_min_clusters=None):
        """
        Given Validation Data, evaluate the model on the Validation Data, and
        print scorch evaluation scores:
            • MUC
            • B³
            • CEAF_m
            • CEAF_e
            • BLANC
            • CoNLL-2012 average score

        Acceptable inputs for Validation Data (val_data):

            (1) .json file path to Validation Data, where each document should be a dictionary with keys:
                • 'event_clusters'
                • 'sentence_no'
                • 'sentences'
                • 'id'

            (2) DataFrame (pandas.DataFrame) comprising columns:
                • 'event_clusters'
                • 'sentence_no'
                • 'sentences'
                • 'id'
        """
        if isinstance(val_data, str):
            data = read(val_data) # .json file to list of dict
        elif isinstance(val_data, pd.DataFrame):
            data = val_data.to_dict('records') # pd.DataFrame to list of dict
        else:
            raise Exception(f'Input "{val_data}" must be a string of a file path leading to a .json file, or a Pandas DataFrame.')

        # Limit Validation Data to those with indicated no of clusters
        if only_min_clusters and isinstance(only_min_clusters, int):
            data = [doc for doc in data if len(doc['event_clusters']) >= only_min_clusters]
            print(f'Restricted Validation Data to those with at least {only_min_clusters} event clusters.')

        if len(data):
            # Ground Truth - Save ground truth to .json file
            with open('temporary_ground_truth.json', "w", encoding="utf-8") as f:
                for doc in data:
                    f.write(json.dumps(doc, cls=NpEncoder) + "\n")

            # Generate cluster predictions
            predictions = self.predict(data)

            # Save cluster predictions to .json file
            with open('temporary_predictions.json', "w", encoding="utf-8") as f:
                for doc in predictions:
                    f.write(json.dumps(doc, cls=NpEncoder) + "\n")

            # Evaluate predictions against ground truth using scorch
            evaluate('temporary_ground_truth.json', 'temporary_predictions.json')

            # Delete temporary files
            attempt_delete_file('temporary_predictions.json')
            attempt_delete_file('temporary_ground_truth.json')
        else:
            print(f'No document with at least {only_min_clusters} event clusters.')

    def evaluate(self, ground_truth_json_file_path, only_min_clusters=None):
        """
        Given .json file path to Validation Data, evaluate the model on the Validation Data,
        and print the following:
            (i) standard evaluation scores - e.g F1, Recall, Precision, Accuracy etc.
            (ii) scorch evaluation scores - e.g MUC, B³, CoNLL-2012 average score etc.

        Each document in Validation Data (.json) should be a dictionary with keys:
            • 'event_clusters'
            • 'sentence_no'
            • 'sentences'
            • 'id'
        """
        if only_min_clusters:
            aesthetic_print_equals(f'Evaluation on {ground_truth_json_file_path} (with at least {only_min_clusters} event clusters)')
        else:
            aesthetic_print_equals(f'Evaluation on {ground_truth_json_file_path}')

        # Read from .json file. Get list of dict
        data = read(ground_truth_json_file_path)

        # Limit dataset to those with indicated no of clusters
        if only_min_clusters and isinstance(only_min_clusters, int):
            data = [doc for doc in data if len(doc['event_clusters']) >= only_min_clusters]

        if len(data):
            # Ground Truth - Save ground truth to .json file
            with open('temporary_ground_truth.json', "w", encoding="utf-8") as f:
                for doc in data:
                    f.write(json.dumps(doc, cls=NpEncoder) + "\n")

            # (i) Evaluation - fscore
            print('(i) fscore\n')
            doc_fscore = self.evaluate_pairwise(data)
            print(doc_fscore)

            # (ii) Evaluation - scorch
            print('\n(ii) scorch\n')
            self.evaluate_scorch('temporary_ground_truth.json')
            print()

            # Delete temporary file
            attempt_delete_file('temporary_ground_truth.json')
        else:
            print(f'No document with at least {only_min_clusters} event clusters.')

    def make_data_loader(self, dataset, is_test_dataset=False, batch_size=32,
                         col_x1='sentence_1', col_x2= 'sentence_2', col_y='is_coreferential'):

        dataset = dataset.copy()
        relevant_cols = [col_x1, col_x2]

        if is_test_dataset:
            dataset_type = 'testing'
            relevant_cols = [col_x1, col_x2]
            field_indices = [0, 1]
            has_label = False
            batch_size = 1
        else:
            dataset_type = 'training'
            relevant_cols = [col_x1, col_x2, col_y]
            field_indices = [0, 1, 2]
            has_label = True
            if batch_size > len(dataset):
                batch_size = 1
            dataset[col_y] = dataset[col_y].apply(lambda x: int(x))

        ## (i) Format Dataset - Pandas to .tsv
        dataset = dataset[relevant_cols]
        dataset_filename = f'roberta_pairwise_sentences_{dataset_type}.tsv'
        dataset.to_csv(dataset_filename, sep = '\t', index=False)

        ## (ii) Format Dataset - .tsv to TSVDataset

        # Skip the first line, which is the schema
        num_discard_samples = 1

        # Split fields by tabs
        field_separator = nlp.data.Splitter('\t')

        # Fields to select from the file
        dataset = nlp.data.TSVDataset(filename=dataset_filename,
                                      field_separator=field_separator,
                                      num_discard_samples=num_discard_samples,
                                      field_indices=field_indices)

        # Delete file (dataset_filename) # TODO:

        ## (iii) Tokenize Dataset

        # The maximum length of an input sequence
        max_len = 128

        # The labels for the two classes [(0 = not similar) or  (1 = similar)]
        all_labels = ["0", "1"]

        # whether to transform the data as sentence pairs.
        pair = True
        # for single sentence classification, set pair=False
        # for regression task, set class_labels=None
        # for inference without label available, set has_label=False
        transform = data.transform.BERTDatasetTransform(self.tokenizer, max_len,
                                                        class_labels=all_labels,
                                                        has_label=has_label,
                                                        pad=True,
                                                        pair=pair)
        dataset = dataset.transform(transform)

        # print('vocabulary used for tokenization = \n%s'%self.vocabulary)
        # print()
        #
        # print('%s token id = %s'%(vocabulary.padding_token, vocabulary[vocabulary.padding_token]))
        # print('%s token id = %s'%(vocabulary.cls_token, vocabulary[vocabulary.cls_token]))
        # print('%s token id = %s'%(vocabulary.sep_token, vocabulary[vocabulary.sep_token]))
        # print()
        #
        # print('token ids = \n%s'%dataset[sample_id][0])
        # print('segment ids = \n%s'%dataset[sample_id][1])
        # print('valid length = \n%s'%dataset[sample_id][2])
        #
        # if not is_test_dataset:
        #     print('label = \n%s'%dataset[sample_id][3])

        # sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in dataset],
        #                                       batch_size=batch_size, shuffle=shuffle)

        data_loader = mx.gluon.data.DataLoader(dataset, batch_size=batch_size)

        return data_loader

def fscore(predictions, truth, f_weight=1.0):
    predictions = [1 if x>0 else 0 for x in predictions]
    truth = [1 if x>0 else 0 for x in truth]
    # Calculate fscore
    tp,tn,fp,fn = 0,1e-8,1e-8,1e-8
    for p, t in zip(predictions, truth):
        if p==1 and t==1: tp+=1
        if p==0 and t==0: tn+=1
        if p==0 and t==1: fn+=1
        if p==1 and t==0: fp+=1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    # fscore = ((1 + f_weight**2) * precision * recall) / ((f_weight**2 * precision) + recall)
    fscore = ((1 + f_weight**2) * tp) / (((1 + f_weight**2) * tp) + (f_weight**2)*fn + fp)

    precision = float(format(precision,'.3f'))
    recall = float(format(recall,'.3f'))
    fscore = float(format(fscore,'.3f'))
    accuracy = float(format(accuracy,'.3f'))

    tp,tn,fp,fn = int(tp), int(tn), int(fp), int(fn)
    n = sum([tp,tn,fp,fn])

    metrics = {'precision': precision, 'recall': recall, 'f1': fscore, 'accuracy': accuracy, 'n':n, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn}

    return metrics

def get_sent_id_2_sentence(sentence_no, sentences):
    return {sent_no: sent for sent_no, sent in zip(sentence_no, sentences)}

def get_sentence_2_sentence_id(sentence_no, sentences):
    return {sent: sent_no for sent_no, sent in zip(sentence_no, sentences)}

def extract_pairwise_sentence(documents, col_sentence_no='sentence_no',
                              col_sentences='sentences', col_id='id'):
    """
    Given document data (list of dict), return DataFrame (pandas.DataFrame)
    comprising columns:
        • 'id' - int of document id
        • 'sentence_1' - string of first sentence
        • 'sentence_2' - string of second sentence
    """
    pairwise_sentences_1 = []
    pairwise_sentences_2 = []
    docs_id = []

    if isinstance(documents, dict):
        documents = [documents]

    for doc in documents:

        sentence_no = doc[col_sentence_no]
        sentences = doc[col_sentences]
        doc_id = doc[col_id]

        # id to sentence
        sent_id_2_sentence = get_sent_id_2_sentence(sentence_no, sentences)

        # Sentence pairs
        sentence_no_pairs = list(itertools.combinations(sentence_no, 2))

        for pair in sentence_no_pairs:

            sent_1 = sent_id_2_sentence[pair[0]]
            pairwise_sentences_1.append(sent_1)

            sent_2 = sent_id_2_sentence[pair[1]]
            pairwise_sentences_2.append(sent_2)

            docs_id.append(doc_id)

    df_pairwise = pd.DataFrame({'id': docs_id,
                                'sentence_1': pairwise_sentences_1,
                                'sentence_2': pairwise_sentences_2})

    return df_pairwise

def extract_pairwise_sentence_relationships(documents,
                                            col_event_clusters='event_clusters',
                                            col_sentence_no='sentence_no',
                                            col_sentences='sentences', col_id='id'):
    """
    Given document data (list of dict) with ground truth, return DataFrame (pandas.DataFrame)
    comprising columns:
        • 'id' - int of document id
        • 'sentence_1' - string of first sentence
        • 'sentence_2' - string of second sentence
        • 'is_coreferential' - boolean of whether the two sentences are coreferential (ground truth)
    """
    pairwise_sentences_1 = []
    pairwise_sentences_2 = []
    is_coreferential = []
    docs_id = []

    if isinstance(documents, dict):
        documents = [documents]

    for doc in documents:

        event_clusters = doc[col_event_clusters]
        sentence_no = doc[col_sentence_no]
        sentences = doc[col_sentences]
        doc_id = doc[col_id]

        # id to sentence
        sent_id_2_sentence = {sent_no: sent for sent_no, sent in zip(sentence_no, sentences)}

        # Related pairs
        related = []
        for inner in event_clusters:
            related.append(list(itertools.combinations(inner, 2)))
        related = list(itertools.chain.from_iterable(related))

        for pair in related:

            sent_1 = sent_id_2_sentence[pair[0]]
            pairwise_sentences_1.append(sent_1)

            sent_2 = sent_id_2_sentence[pair[1]]
            pairwise_sentences_2.append(sent_2)

            is_coreferential.append(True)

            docs_id.append(doc_id)

        # Unrelated pairs
        unrelated = []
        for first_idx, second_idx, in itertools.combinations(range(len(event_clusters)), 2):
            unrelated.append(list(itertools.product(event_clusters[first_idx], event_clusters[second_idx])))
        unrelated = list(itertools.chain.from_iterable(unrelated))

        for pair in unrelated:

            sent_1 = sent_id_2_sentence[pair[0]]
            pairwise_sentences_1.append(sent_1)

            sent_2 = sent_id_2_sentence[pair[1]]
            pairwise_sentences_2.append(sent_2)

            is_coreferential.append(False)

            docs_id.append(doc_id)

    df_pairwise = pd.DataFrame({'id': docs_id,
                                'sentence_1': pairwise_sentences_1,
                                'sentence_2': pairwise_sentences_2,
                                'is_coreferential': is_coreferential})

    return df_pairwise

def train_test_split_by_segment(df, test_size=0.1, segment='id', shuffle=True):

    total_entries = len(df)

    # Required quantity of test entries
    if test_size <= 1:
        required_test_size = int(test_size * total_entries)
        required_test_size = max(1, required_test_size) # at least one test entry

    # Collate test, train entries
    segmentation = df[segment].value_counts()

    if shuffle: # Shuffle for randomization
        segmentation = segmentation.sample(frac=1)

    test_index = []
    test_count = 0

    for index, count in zip(segmentation.index, segmentation.values):
        test_count += count
        test_index.append(index)
        if test_count >= required_test_size:
            break

    test = df[df[segment].apply(lambda x: x in test_index)]
    train = df[df[segment].apply(lambda x: x not in test_index)]

    return train, test
