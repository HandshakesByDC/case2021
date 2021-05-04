from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import AdamW, BertTokenizerFast, BertForTokenClassification

from ..dataset import GloconDataset, conll_evaluate, TagMap
from ..models.viterbi_decoder import ViterbiDecoder

class MultilingualBertTokenClassifier(pl.LightningModule):
    def __init__(self, tokenizer, tag_map, batch_size, use_viterbi=True):
        super().__init__()
        self.tag_map = tag_map
        num_labels = len(self.tag_map.tag2id)
        self.bert = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.use_viterbi = use_viterbi
        self.viterbi_decoder = None

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        return optimizer

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.bert(x["input_ids"], attention_mask=x["attention_mask"])
        return embedding

    def train_dataloader(self):
        en_dataset, _ = GloconDataset.build('data/en-orig.txt', self.tokenizer, self.tag_map, test_split=0.05)
        es_dataset, _ = GloconDataset.build('src/models/UniTrans/data/ner/glocon/en2es/train.txt', self.tokenizer, self.tag_map, test_split=0.05)
        pt_dataset, _ = GloconDataset.build('src/models/UniTrans/data/ner/glocon/en2pt/train.txt', self.tokenizer, self.tag_map, test_split=0.05)
        # train_dataset = torch.utils.data.ConcatDataset([en_dataset, es_dataset,
                                                       # pt_dataset])
        train_dataset = en_dataset
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
        _, en_dataset = GloconDataset.build('data/en-orig.txt', self.tokenizer, self.tag_map, test_split=0.05)
        es_dataset = GloconDataset.build('data/es-orig.txt', self.tokenizer, self.tag_map)
        pt_dataset = GloconDataset.build('data/pt-orig.txt', self.tokenizer, self.tag_map)

        en_loader = DataLoader(en_dataset, batch_size=self.batch_size, shuffle=False)
        es_loader = DataLoader(es_dataset, batch_size=self.batch_size, shuffle=False)
        pt_loader = DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=False)

        return [en_loader, es_loader, pt_loader]

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.use_viterbi and self.viterbi_decoder is None:
            self.viterbi_decoder = ViterbiDecoder(self.tag_map.id2tag, -100, self.device)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, labels=labels
        )
        batch_labels = labels[attention_mask > 0]
        final_labels = batch_labels[batch_labels > -1]
        if self.use_viterbi:
            log_probs = torch.nn.functional.log_softmax(outputs.logits.detach(), dim=-1)
            batch_preds = self.viterbi_decoder.forward(log_probs, attention_mask, labels)
            final_preds = sum(map(lambda x: x, batch_preds), [])
        else:
            batch_preds = torch.argmax(outputs.logits, 2)[attention_mask > 0]
            final_preds = batch_preds[batch_labels > -1].tolist()

        return {
            f"val_loss_{dataloader_idx}": outputs.loss.item(),
            f"val_preds_{dataloader_idx}": final_preds,
            f"val_labels_{dataloader_idx}": final_labels.tolist(),
        }

    def validation_epoch_end(self, outs):
        id2tag = self.tag_map.id2tag
        val_loss = 0
        val_f1 = 0
        for i, out in enumerate(outs):
            val_loss += sum(map(lambda x: x[f"val_loss_{i}"], out), 0)

            preds = sum(map(lambda x: x[f"val_preds_{i}"], out), [])
            labels = sum(map(lambda x: x[f"val_labels_{i}"], out), [])
            preds_tag = [id2tag[i] for i in preds]
            labels_tag = [id2tag[i] for i in labels]
            _, _, f1 = conll_evaluate(labels_tag, preds_tag)
            val_f1 += f1

        self.log('val_loss', val_loss)
        self.log('val_f1', val_f1/(len(outs)))

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
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    tag_map = TagMap.build('src/models/UniTrans/data/ner/glocon/labels.txt')

    model = MultilingualBertTokenClassifier(tokenizer=tokenizer, tag_map=tag_map, batch_size=args.batch_size)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_f1", mode='max', patience=2)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_f1", mode='max', verbose=True)

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping_callback, checkpoint_callback],
        gpus=1,
        accumulate_grad_batches=8,
        precision=16,
    )

    trainer.fit(model)

if __name__ == "__main__":
    cli_main()
