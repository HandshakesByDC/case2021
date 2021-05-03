from pathlib import Path
import re

from tqdm import tqdm

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from transformers import AdamW, BertTokenizerFast, BertForTokenClassification

from ..dataset import GloconDataset, conll_evaluate

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

train_dataset = GloconDataset.build('data/en-orig.txt', tokenizer)
es_dataset = GloconDataset.build('data/es-orig.txt', tokenizer)
pr_dataset = GloconDataset.build('data/pt-orig.txt', tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
es_loader = DataLoader(es_dataset, batch_size=2, shuffle=True)
pr_loader = DataLoader(pr_dataset, batch_size=2, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tag_map = train_dataset.tag_map
id2tag = tag_map.id2tag
num_tags = len(id2tag)

model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_tags)
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
    # val_preds = []
    # val_labels = []
    # for batch in tqdm(val_loader):
    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     labels = batch['labels'].to(device)
    #     outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #     loss = outputs[0]
    #     logits = outputs[1].to('cpu').detach()
    #     preds = torch.argmax(logits, 2)
    #     val_pred = preds[attention_mask > 0]
    #     val_label = batch['labels'][attention_mask > 0]
    #     val_pred = val_pred[val_label > -1]
    #     val_label = val_label[val_label > -1]
    #     val_preds.append(val_pred)
    #     val_labels.append(val_label)

    # val_preds = sum(map(lambda x: x.tolist(), val_preds), [])
    # val_labels = sum(map(lambda x: x.tolist(), val_labels), [])
    # val_preds_tag = [id2tag[i] for i in val_preds]
    # val_labels_tag = [id2tag[i] for i in val_labels]
    # _, _, f1 = conll_evaluate(val_labels_tag, val_preds_tag)

    es_preds = []
    es_labels = []
    for batch in tqdm(es_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs[1].to('cpu').detach()
        preds = torch.argmax(logits, 2)
        es_pred = preds[attention_mask > 0]
        es_label = batch['labels'][attention_mask > 0]
        es_pred = es_pred[es_label > -1]
        es_label = es_label[es_label > -1]
        es_preds.append(es_pred)
        es_labels.append(es_label)

    es_preds = sum(map(lambda x: x.tolist(), es_preds), [])
    es_labels = sum(map(lambda x: x.tolist(), es_labels), [])
    es_preds_tag = [id2tag[i] for i in es_preds]
    es_labels_tag = [id2tag[i] for i in es_labels]
    _, _, es_f1 = conll_evaluate(es_labels_tag, es_preds_tag)

    pr_preds = []
    pr_labels = []
    for batch in tqdm(pr_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs[1].to('cpu').detach()
        preds = torch.argmax(logits, 2)
        pr_pred = preds[attention_mask > 0]
        pr_label = batch['labels'][attention_mask > 0]
        pr_pred = pr_pred[pr_label > -1]
        pr_label = pr_label[pr_label > -1]
        pr_preds.append(pr_pred)
        pr_labels.append(pr_label)

    pr_preds = sum(map(lambda x: x.tolist(), pr_preds), [])
    pr_labels = sum(map(lambda x: x.tolist(), pr_labels), [])
    pr_preds_tag = [id2tag[i] for i in pr_preds]
    pr_labels_tag = [id2tag[i] for i in pr_labels]
    _, _, pr_f1 = conll_evaluate(pr_labels_tag, pr_preds_tag)

    print(f"es: {es_f1}, pr: {pr_f1}")

model.eval()
torch.save(model.state_dict(), 'baseline_bert.model')
