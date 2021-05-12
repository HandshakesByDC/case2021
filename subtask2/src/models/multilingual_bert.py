import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class SentenceDataset(Dataset):

    def __init__(self, encodings, labels=None, length=None):
        self.encodings = encodings
        self.labels = labels
        self.length = length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx]).float()
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        if self.labels:
            return len(self.labels)
        else:
            return self.length

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
    # fscore = ((1 + f_weight**2) * precision * recall) / ((f_weight**2 * precision) + recall)
    fscore = ((1 + f_weight**2) * tp) / (((1 + f_weight**2) * tp) + (f_weight**2)*fn + fp)

    precision = float(format(precision,'.3f'))
    recall = float(format(recall,'.3f'))
    fscore = float(format(fscore,'.3f'))

    tp,tn,fp,fn = int(tp), int(tn), int(fp), int(fn)
    n = sum([tp,tn,fp,fn])
    metrics = {'precision': precision, 'recall': recall, 'f1':fscore, 'n':n, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn}
    return metrics

def prep_data(articles):
    texts = [a['sentence'] for a in articles]
    labels = [a['label'] for a in articles]
    return texts, labels


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def split_data(data:list, ratio:float=0.8):
    train = data[:int(len(data)*ratio)]
    val   = data[int(len(data)*ratio):]
    return train, val

def make_dataloader(articles, batch_size=8, shuffle=True):
    texts, labels = prep_data(articles)
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = SentenceDataset(encodings, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def make_submission_dataloader(articles, batch_size=1, shuffle=False):
    texts = [a['sentence'] for a in articles]
    encodings = tokenizer(texts, truncation=True, padding=True)
    dataset = SentenceDataset(encodings, labels=None, length=len(texts))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

from transformers import XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaTokenizer
model_name = "xlm-roberta-base"
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=2, local_files_only=False)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, local_files_only=False)

if __name__ == "__main__":

    ##### SUBTASK 1 ##########
    en_st1_train = pickle.load(open('../../data/processed/en_train_st1.pkl', 'rb'))
    es_st1_train = pickle.load(open('../../data/processed/es_train_st1.pkl', 'rb'))
    pr_st1_train = pickle.load(open('../../data/processed/pr_train_st1.pkl', 'rb'))

    en_st1_val = pickle.load(open('../../data/processed/en_test_st1.pkl', 'rb'))
    es_st1_val = pickle.load(open('../../data/processed/es_test_st1.pkl', 'rb'))
    pr_st1_val = pickle.load(open('../../data/processed/pr_test_st1.pkl', 'rb'))


    ##### SUBTASK 2 ##########
    en = pickle.load(open('../../data/processed/en_train.pkl', 'rb'))
    es = pickle.load(open('../../data/processed/es_train.pkl', 'rb'))
    pr = pickle.load(open('../../data/processed/pr_train.pkl', 'rb'))

    en_train, en_val = split_data(en, ratio=0.8)
    es_train, es_val = split_data(es, ratio=0.8)
    pr_train, pr_val = split_data(pr, ratio=0.8)

    train_articles = en_train + es_train + pr_train
    val_articles = en_val + es_val + pr_val

    train_articles += en_st1_train + es_st1_train + pr_st1_train
    val_articles += en_st1_val + es_st1_val + pr_st1_val

    train_texts, train_labels = prep_data(train_articles)
    val_texts, val_labels = prep_data(val_articles)

    print("tokenizing...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=450)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=450)

    train_dataset = SentenceDataset(train_encodings, train_labels)
    val_dataset = SentenceDataset(val_encodings, val_labels)


    training_args = TrainingArguments(
        output_dir='./results/XLMRoberta_st1trained_all',          # output directory
        num_train_epochs=100,              # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        learning_rate=5e-7,
        evaluation_strategy='epoch',
        logging_steps=500,
        save_total_limit=100,
        save_strategy='epoch',
        metric_for_best_model="eval_f1",
        seed=0,
        fp16=True,
    )

    print(f"Using: {model_name} model")

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics = compute_metrics
    )


    print("training...")
    trainer.train()
