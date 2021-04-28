from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import train_test_split

import torch

from transformers import BertTokenizerFast

def read_data(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        print(arr_offset.shape, len(doc_labels))
        print(sum((arr_offset[:,0]==0) & (arr_offset[:,1] != 0)))
        try:
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        except ValueError as e:
            print(e)
            import pdb; pdb.set_trace()
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

def align_labels(tags, encodings):
    labels = []
    for i, label in enumerate(tags):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(tag2id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(tag2id[label[word_idx]] if False else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    return labels

import torch

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

texts, tags = read_data('data/en-orig.txt')

train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)

unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
train_encodings = tokenizer(train_texts, is_split_into_words=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, padding=True, truncation=True)

train_labels = align_labels(train_tags, train_encodings)
val_labels = align_labels(val_tags, val_encodings)

train_dataset = NERDataset(train_encodings, train_labels)
val_dataset = NERDataset(val_encodings, val_labels)
