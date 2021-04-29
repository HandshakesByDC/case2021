from pathlib import Path
import re

from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from transformers import AdamW, BertTokenizerFast, BertForTokenClassification

from conlleval import evaluate as conll_evaluate

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

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_tags))
optim = AdamW(model.parameters(), lr=5e-5)

model.to(device)
model.train()

for epoch in range(5):
    model.train()
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

    model.eval()
    val_preds = []
    val_labels = []
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits, 2)
        val_pred = preds[attention_mask > 0]
        val_label = labels[attention_mask > 0]
        val_pred = val_pred[val_label > -1]
        val_label = val_label[val_label > -1]
        val_preds.append(val_pred)
        val_labels.append(val_label)

    val_preds = sum(map(lambda x: x.tolist(), val_preds), [])
    val_labels = sum(map(lambda x: x.tolist(), val_labels), [])
    val_preds_tag = [id2tag[i] for i in val_preds]
    val_labels_tag = [id2tag[i] for i in val_labels]
    conll_evaluate(val_labels_tag, val_preds_tag)

model.eval()
torch.save(model.state_dict(), 'baseline_bert.model')
