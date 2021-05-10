import json
import itertools
import argparse
import subprocess
from embedding_model import EmbeddingModel
from utils import *

class OneClusterModel():

    def __init__(self):
        pass

    def fit(self,data):
        """
        Doesn't use the given data.
        Returns nothing.
        """
        return

    def predict(self,data):
        """
        Takes some data (.json) and makes predictions.
        Simply puts all sentences in a single cluster.
        """
        preds = []
        for idx,instance in enumerate(data):
            preds.append({"id":instance["id"], "pred_clusters": [instance['sentence_no']]})

        return preds

    def evaluate(self, val_data, only_min_clusters=None):
        """
        Given Validation Data, evaluate the model on the Validation Data, and print the following:
            (i) standard evaluation scores - e.g Fscore, Recall, Precision, Accuracy etc.
            (ii) scorch evaluation scores - e.g MUC, B³, CoNLL-2012 average score etc.

        Acceptable inputs for Validation Data (val_data):

            (1) .json file path to Validation Data, where each document should be a dictionary with keys:
                • 'event_clusters'
                • 'sentence_no'
                • 'sentences'
                • 'id'

            (2) Document data (list of dict) comprising keys:
                • 'event_clusters'
                • 'sentence_no'
                • 'sentences'
                • 'id'
        """
        # File types
        if isinstance(val_data, str):
            if only_min_clusters:
                aesthetic_print_equals(f'Evaluation on {val_data} (with at least {only_min_clusters} event clusters)')
            else:
                aesthetic_print_equals(f'Evaluation on {val_data}')
            # Read from .json file. Get list of dict
            data = read(val_data)
        elif isinstance(val_data, list):
            aesthetic_print_equals(f'Evaluation')
            data = val_data
        else:
            raise Exception(f'Input data must be (i) a string of a file path leading to a .json file, (ii) a list of dictionaries, or (iii) a Pandas DataFrame.')

        ## (i) Actual

        # Limit dataset to those with indicated no of clusters
        if only_min_clusters and isinstance(only_min_clusters, int):
            data = [doc for doc in data if len(doc['event_clusters']) >= only_min_clusters]

        if len(data):

            if only_min_clusters:
                print(f'\tTotal documents to evaluate (with at least {only_min_clusters} event clusters): {len(data)} \n')
            else:
                print(f'\tTotal documents to evaluate: {len(data)}\n')

            # Ground Truth - Save ground truth to .json file
            with open('temporary_ground_truth.json', "w", encoding="utf-8") as f:
                for doc in data:
                    f.write(json.dumps(doc, cls=NpEncoder) + "\n")

            ## (ii) Predictions

            # Generate cluster predictions
            predictions = self.predict(data)

            # Predictions - Save cluster predictions to .json file
            with open('temporary_predictions.json', "w", encoding="utf-8") as f:
                for doc in predictions:
                    f.write(json.dumps(doc, cls=NpEncoder) + "\n")

            ## (iii) Evaluation

            # Evaluation - Evaluate predictions against ground truth using scorch
            evaluate('temporary_ground_truth.json', 'temporary_predictions.json')

            # Delete temporary prediction files
            attempt_delete_file('temporary_ground_truth.json')
            attempt_delete_file('temporary_predictions.json')

        else:
            print(f'No document with at least {only_min_clusters} event clusters.')

def read(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            data.append(json.loads(instance))

    return data


def convert_to_scorch_format(docs, cluster_key="event_clusters"):
    # Merge all documents' clusters in a single list

    all_clusters = []
    for idx, doc in enumerate(docs):
        for cluster in doc[cluster_key]:
            all_clusters.append([str(idx) + "_" + str(sent_id) for sent_id in cluster])

    all_events = [event for cluster in all_clusters for event in cluster]
    all_links = sum([list(itertools.combinations(cluster,2)) for cluster in all_clusters],[])

    return all_links, all_events

def evaluate(goldfile, sysfile):
    """
    Uses scorch -a python implementaion of CoNLL-2012 average score- for evaluation. > https://github.com/LoicGrobol/scorch | pip install scorch
    Takes gold file path (.json), predicted file path (.json) and prints out the results.

    This function is the exact way the subtask3's submissions will be evaluated.
    """
    gold = read(goldfile)
    sys = read(sysfile)

    gold_links, gold_events = convert_to_scorch_format(gold)
    sys_links, sys_events = convert_to_scorch_format(sys, cluster_key="pred_clusters")

    with open("gold.json", "w") as f:
        json.dump({"type":"graph", "mentions":gold_events, "links":gold_links}, f)
    with open("sys.json", "w") as f:
        json.dump({"type":"graph", "mentions":sys_events, "links":sys_links}, f)

    subprocess.run(["scorch", "gold.json", "sys.json", "results.txt"])
    print(open("results.txt", "r").read())
    subprocess.run(["rm", "gold.json", "sys.json", "results.txt"])

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_file', '--train_file', required=True, help="The path to the training data json file")
    parser.add_argument('-prediction_output_file', '--prediction_output_file', required=True, help="The path to the prediction output json file")
    parser.add_argument('-test_file', '--test_file', required=False, help="The path to the test data json file")
    parser.add_argument('-embedding_key', '--embedding_key', required=False, help="The embedding key if Embedding Model is used")
    args = parser.parse_args()
    return args

def main(train_file, prediction_output_file, test_file=None, embedding_key='laser_embeddings'):

    #Create model.
    aesthetic_print_equals(f'Using Embedding Model ({embedding_key})')
    model = EmbeddingModel()
#     model = OneClusterModel()

    #Read training data.
    train_data = read(train_file)

    #Fit.
    model.fit(train_data)

    # Predict train data so that we can evaluate our system
    train_predictions = model.predict(train_data, embedding_key=embedding_key, metric='silhouette')
    with open(prediction_output_file, "w", encoding="utf-8") as f:
        for doc in train_predictions:
            f.write(json.dumps(doc) + "\n")

#     # Examine predictions
#     model.examine_predictions(train_data, train_predictions)

    # Evaluate sys outputs and print results.
    evaluate(train_file, prediction_output_file)

    # If there is a test file provided, make predictions and write them to file.
    if test_file:
        # Read test data.
        test_data = read(test_file)
        # Predict and save your predictions in the required format.
        test_predictions = model.predict(test_data)
        with open("sample_submission.json", "w", encoding="utf-8") as f:
            for doc in test_predictions:
                f.write(json.dumps(doc) + "\n")

if __name__ == "__main__":
    args = parse()
    main(train_file=args.train_file, prediction_output_file=args.prediction_output_file, test_file=args.test_file,
         embedding_key=args.embedding_key)

# python code_sample_subtask3.py --train_file embedding_test_samples.json --prediction_output_file embedding_test_samples_predictions.json --embedding_key tf_use_embeddings
