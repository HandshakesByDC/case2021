from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    BertTokenizerFast,
    BertForTokenClassification,
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
)

from ..dataset import GloconDataset, conll_evaluate, TagMap
from ..models.viterbi_decoder import ViterbiDecoder

class MultilingualBertTokenClassifier(pl.LightningModule):
    def __init__(self, model_class, model_string, tokenizer, tag_map,
                 batch_size, use_viterbi=False, translate_data=False):
        super().__init__()
        self.tag_map = tag_map
        num_labels = len(self.tag_map.tag2id)
        self.bert = model_class.from_pretrained(model_string, num_labels=num_labels)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.use_viterbi = use_viterbi
        self.viterbi_decoder = None
        self.translate_data = translate_data

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        return optimizer

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.bert(x["input_ids"], attention_mask=x["attention_mask"])
        return embedding

    def train_dataloader(self):
        train_datasets = []
        en_dataset, _ = GloconDataset.build('data/en-orig.txt', self.tokenizer, self.tag_map, test_split=0.05)
        train_datasets.append(en_dataset)
        if self.translate_data:
            es_dataset, _ = GloconDataset.build('src/models/UniTrans/data/ner/glocon/en2es/train.txt', self.tokenizer, self.tag_map, test_split=0.05)
            train_datasets.append(es_dataset)
            pt_dataset, _ = GloconDataset.build('src/models/UniTrans/data/ner/glocon/en2pt/train.txt', self.tokenizer, self.tag_map, test_split=0.05)
            train_datasets.append(pt_dataset)
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
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
            batch_preds = self.viterbi_decoder.forward(log_probs, attention_mask)
            final_preds = torch.tensor(sum(map(lambda x: x, batch_preds), []))[batch_labels > -1]
        else:
            batch_preds = torch.argmax(outputs.logits, 2)[attention_mask > 0]
            final_preds = batch_preds[batch_labels > -1]

        return {
            f"val_loss_{dataloader_idx}": outputs.loss.item(),
            f"val_preds_{dataloader_idx}": final_preds.tolist(),
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

    def test_dataloader(self):
        _, en_dataset = GloconDataset.build('data/en-orig.txt', self.tokenizer, self.tag_map, test_split=0.05)
        es_dataset = GloconDataset.build('data/es-orig.txt', self.tokenizer, self.tag_map)
        pt_dataset = GloconDataset.build('data/pt-orig.txt', self.tokenizer, self.tag_map)

        en_loader = DataLoader(en_dataset, batch_size=self.batch_size, shuffle=False)
        es_loader = DataLoader(es_dataset, batch_size=self.batch_size, shuffle=False)
        pt_loader = DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=False)

        return [en_loader, es_loader, pt_loader]

    def test_step(self, batch, batch_idx, dataloader_idx):
        self.viterbi_decoder = ViterbiDecoder(self.tag_map.id2tag, -100, self.device)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, labels=labels
        )
        batch_labels = labels[attention_mask > 0]
        final_labels = batch_labels[batch_labels > -1]

        log_probs = torch.nn.functional.log_softmax(outputs.logits.detach(), dim=-1)
        batch_preds = self.viterbi_decoder.forward(log_probs, attention_mask)
        final_preds = torch.tensor(sum(map(lambda x: x, batch_preds), []))[batch_labels > -1]

        return {
            f"test_loss_{dataloader_idx}": outputs.loss.item(),
            f"test_preds_{dataloader_idx}": final_preds.tolist(),
            f"test_labels_{dataloader_idx}": final_labels.tolist(),
        }

    def test_epoch_end(self, outs):
        id2tag = self.tag_map.id2tag
        test_loss = 0
        test_f1 = 0
        for i, out in enumerate(outs):
            test_loss += sum(map(lambda x: x[f"test_loss_{i}"], out), 0)

            preds = sum(map(lambda x: x[f"test_preds_{i}"], out), [])
            labels = sum(map(lambda x: x[f"test_labels_{i}"], out), [])
            preds_tag = [id2tag[i] for i in preds]
            labels_tag = [id2tag[i] for i in labels]
            _, _, f1 = conll_evaluate(labels_tag, preds_tag)
            test_f1 += f1

        self.log('test_loss', test_loss)
        self.log('test_f1', test_f1/(len(outs)))

def cli_main():
    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--model_name", type=str, default='bert')
    args = parser.parse_args()

    if args.model_name == 'bert':
        tokenizer_class = BertTokenizerFast
        model_class = BertForTokenClassification
        model_string = "bert-base-multilingual-cased"
    elif args.model_name == 'roberta':
        tokenizer_class = XLMRobertaTokenizerFast
        model_class = XLMRobertaForTokenClassification
        model_string = "xlm-roberta-base"

    tokenizer = tokenizer_class.from_pretrained(model_string)

    tag_map = TagMap.build('src/models/UniTrans/data/ner/glocon/labels.txt')

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_f1", mode='max', patience=2)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_f1", mode='max', verbose=True)

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping_callback, checkpoint_callback],
        gpus=1,
        accumulate_grad_batches=8,
        precision=16,
    )

    if args.load is None:
        model = MultilingualBertTokenClassifier(
            model_class=model_class,
            model_string=model_string,
            tokenizer=tokenizer,
            tag_map=tag_map,
            batch_size=args.batch_size
        )
        trainer.fit(model)
        if not args.fast_dev_run:
            print(f"Load from checkpoint {trainer.checkpoint_callback.best_model_path}")
            model = MultilingualBertTokenClassifier.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,
                model_class=model_class,
                model_string=model_string,
                tokenizer=tokenizer,
                tag_map=tag_map,
                batch_size=args.batch_size
            )
    else:
        model = MultilingualBertTokenClassifier.load_from_checkpoint(
            args.load,
            model_class=model_class,
            model_string=model_string,
            tokenizer=tokenizer,
            tag_map=tag_map,
            batch_size=args.batch_size
        )

    trainer.test(model)

if __name__ == "__main__":
    cli_main()
