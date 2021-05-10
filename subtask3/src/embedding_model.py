import ast
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, k_means
from yellowbrick.cluster import KElbowVisualizer
from utils import *

class EmbeddingModel():

    def __init__(self):
        pass

    def fit(self, data):
        pass

    def predict(self, data, embedding_key='laser_embeddings', metric='distortion'):
        """
        Return cluster predictions, given document data (list of dict).
        """
        preds = []
        for idx, instance in enumerate(data):

            # Get Embeddings
            embeddings = pd.DataFrame(instance[embedding_key])

            try:

                # K-Means - Find optimal no. of cluster - Elbow method

                if metric=='distortion':
                    max_clusters = max(1, instance['total_sentences'])
                    elbow = KElbowVisualizer(KMeans(), k = (1, max_clusters+1))
                else:
                    max_clusters = max(2, instance['total_sentences'])
                    elbow = KElbowVisualizer(KMeans(), k = (2, max_clusters), metric = "silhouette")

                elbow.fit(embeddings)
                elbow.show()
                optimal_clusters = elbow.elbow_value_

                # K-Means - Find optimal no. of cluster - Elbow method
                predicted_clusters = KMeans(n_clusters = optimal_clusters, init = 'k-means++',
                                            random_state = 42).fit_predict(embeddings)

                # Format results - Convert cluster index to sentence index
                sentence_no = instance['sentence_no']
                result = dict()
                for i in range(max(predicted_clusters)+1):
                    result[i] = []
                for cluster, sent_no in zip(predicted_clusters, sentence_no):
                    result[cluster].append(sent_no)
                result = sorted(list(result.values()))

                preds.append({"id": instance['id'], "pred_clusters": result})

            except:
                print(f'Error with document \(id: {instance["id"]}\), returning all sentences as 1 cluster..')
                preds.append({"id": instance['id'], "pred_clusters": [instance['sentence_no']]})


        return preds

    def examine_predictions(self, data, preds, ground_truth_key='event_clusters',
                            prediction_key='pred_clusters', document_key='id'):

        def same_document(list_of_dictionaries, document_key='id'):
            if isinstance(list_of_dictionaries, list) and len(list_of_dictionaries)>=2:
                first_id = list_of_dictionaries[0][document_key]
                for dictionary in list_of_dictionaries[1:]:
                    if dictionary[document_key] != first_id:
                        return False
                return True
            else:
                return False

        for instance, pred in zip(data, preds):

            if not same_document([instance, pred], document_key):
                print(f'Data and Predictions must have sequentially matching Document ID Key "{document_key}"')
                break
            else:

                if ground_truth_key in instance.keys():

                    # Show actual and predicted - clusters of sentences number
                    aesthetic_print_stars(f'document id {instance["id"]}'.upper())
                    print(f'Actual: {instance[ground_truth_key]}')
                    print(f'Predictions: {pred[prediction_key]}')
                    print()

                # Show predicted - clusters of sentences
                sentences = instance['sentences']
                sentence_no = instance['sentence_no']
                aesthetic_print_stars(f'Predicted Clusters'.upper())

                for idx, cluster in enumerate(pred[prediction_key]):
                    aesthetic_print_equals(f'Predicted Cluster {idx}')
                    for c in cluster:
                        sentence_index = sentence_no.index(c)
                        print(f'Sentence {c}')
                        print(sentences[sentence_index])
                        print('\n')
