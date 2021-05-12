import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Needed for loading universal-sentence-encoder-cmlm/multilingual-preprocess
import numpy as np

preprocessor = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")


def labse_embed(article):
    sentence = tf.constant([article['sentence']])
    emb = encoder(preprocessor(sentence))["default"]
    emb = np.asarray(emb)
    return emb


article = {'sentence': "Testing labse embedding"}
print(labse_embed(article))
