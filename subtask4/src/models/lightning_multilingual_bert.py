from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import AdamW, BertTokenizerFast, BertForTokenClassification

from ..dataset import GloconDataset, conll_evaluate

class MultilingualBertTokenClassifier(pl.LightningModule):
    def __init__(self, num_labels, tokenizer, batch_size):
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        return optimizer

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.bert(x["input_ids"], attention_mask=x["attention_mask"])
        return embedding

    def train_dataloader(self):
        en_dataset = GloconDataset.build('data/en-orig.txt', self.tokenizer)
        es_dataset = GloconDataset.build('src/models/UniTrans/data/ner/glocon/en2es/train.txt', self.tokenizer)
        pt_dataset = GloconDataset.build('src/models/UniTrans/data/ner/glocon/en2pt/train.txt', self.tokenizer)
        self.tag_map = en_dataset.tag_map
        train_dataset = torch.utils.data.ConcatDataset([en_dataset, es_dataset,
                                                       pt_dataset])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, labels=labels
        )
        return { "loss": outputs.loss}

    def val_dataloader(self):
        es_dataset = GloconDataset.build('data/es-orig.txt', self.tokenizer)
        pt_dataset = GloconDataset.build('data/pt-orig.txt', self.tokenizer)

        es_loader = DataLoader(es_dataset, batch_size=self.batch_size, shuffle=False)
        pt_loader = DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=False)

        return [es_loader, pt_loader]

    def validation_step(self, batch, batch_idx, dataloader_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, labels=labels
        )
        batch_preds = torch.argmax(outputs.logits, 2)[attention_mask > 0]
        batch_labels = labels[attention_mask > 0]
        final_preds = batch_preds[batch_labels > -1]
        final_labels = batch_labels[batch_labels > -1]

        return {
            f"val_loss_{dataloader_idx}": outputs.loss,
            f"val_preds_{dataloader_idx}": final_preds,
            f"val_labels_{dataloader_idx}": final_labels,
        }

    def validation_epoch_end(self, outs):
        tag_map = self.trainer.val_dataloaders[0].dataset.tag_map
        id2tag = tag_map.id2tag
        val_loss = 0
        for i, out in enumerate(outs):
            val_loss += sum(map(lambda x: x[f"val_loss_{i}"].tolist(), out), 0)

            preds = sum(map(lambda x: x[f"val_preds_{i}"].tolist(), out), [])
            labels = sum(map(lambda x: x[f"val_labels_{i}"].tolist(), out), [])
            preds_tag = [id2tag[i] for i in preds]
            labels_tag = [id2tag[i] for i in labels]
            _, _, f1 = conll_evaluate(labels_tag, preds_tag)

        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.transformer(
            input_ids, attention_mask=attention_mask
        )

def cli_main():
    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    en_dataset = GloconDataset.build('data/en-orig.txt', tokenizer)
    tag_map = en_dataset.tag_map
    id2tag = tag_map.id2tag
    num_tags = len(id2tag)

    model = MultilingualBertTokenClassifier(num_labels=num_tags, tokenizer=tokenizer, batch_size=args.batch_size)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=2)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode='min', verbose=True)

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping_callback, checkpoint_callback]
    )

    trainer.fit(model)

if __name__ == "__main__":
    cli_main()
