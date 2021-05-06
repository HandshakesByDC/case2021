import os
import json
from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    BertConfig,
    BertTokenizerFast,
    BertForTokenClassification,
    XLMRobertaConfig,
    XLMRobertaTokenizerFast,
    XLMRobertaForTokenClassification,
)

from dataset import GloconDataset
from conlleval import evaluate as conll_evaluate
from utils import TagMap
from viterbi_decoder import ViterbiDecoder

class MultilingualTokenClassifier(pl.LightningModule):
    def __init__(self, model_name, labels_path, batch_size, translate_data,
                 hidden_dropout_prob, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        """@nni.variable(nni.choice(0.1,0.2,0.3),name=hidden_dropout_prob)"""
        hidden_dropout_prob = hidden_dropout_prob
        if model_name == 'bert':
            config = BertConfig(hidden_dropout_prob=hidden_dropout_prob)
            tokenizer_class = BertTokenizerFast
            model_class = BertForTokenClassification
            model_string = "bert-base-multilingual-cased"
        elif model_name == 'roberta':
            config = XLMRobertaConfig(hidden_dropout_prob=hidden_dropout_prob)
            tokenizer_class = XLMRobertaTokenizerFast
            model_class = XLMRobertaForTokenClassification
            model_string = "xlm-roberta-base"

        self.tag_map = TagMap.build(labels_path)
        num_labels = len(self.tag_map.tag2id)
        self.bert = model_class.from_pretrained(model_string, num_labels=num_labels)
        self.tokenizer = tokenizer_class.from_pretrained(model_string)
        self.batch_size = batch_size
        self.viterbi_decoder = None
        """@nni.variable(nni.choice(True, False),name=self.translate_data)"""
        self.translate_data = translate_data
        # """@nni.variable(nni.choice(5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6, 5e-6, 1e-7),name=self.learning_rate)"""
        # """@nni.variable(nni.loguniform(1e-6, 5e-4),name=self.learning_rate)"""
        self.learning_rate = 5e-5

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MultilingualTokenClassifier")
        parser.add_argument('--model_name', type=str, default='bert')
        parser.add_argument('--labels_path', type=str, default='data/labels.txt')
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--translate_data', action='store_true')
        parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
        return parent_parser

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
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
            en2es_dataset, _ = GloconDataset.build('data/en2es.txt', self.tokenizer, self.tag_map, test_split=0.05)
            train_datasets.append(en2es_dataset)
            en2pt_dataset, _ = GloconDataset.build('data/en2pt.txt', self.tokenizer, self.tag_map, test_split=0.05)
            train_datasets.append(en2pt_dataset)
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
        return { "loss": outputs.loss }

    def val_dataloader(self):
        _, en_dataset = GloconDataset.build('data/en-orig.txt', self.tokenizer, self.tag_map, test_split=0.05)
        es_dataset = GloconDataset.build('data/es-orig.txt', self.tokenizer, self.tag_map)
        pt_dataset = GloconDataset.build('data/pt-orig.txt', self.tokenizer, self.tag_map)

        en_loader = DataLoader(en_dataset, batch_size=self.batch_size, shuffle=False)
        es_loader = DataLoader(es_dataset, batch_size=self.batch_size, shuffle=False)
        pt_loader = DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=False)

        return [en_loader, es_loader, pt_loader]

    def validation_step(self, batch, batch_idx, dataloader_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, labels=labels
        )
        batch_labels = labels[attention_mask > 0]
        final_labels = batch_labels[batch_labels > -1]

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

        avg_val_f1 = val_f1/len(outs)
        """@nni.report_intermediate_result(avg_val_f1)"""
        self.log('val_loss', val_loss)
        self.log('val_f1', avg_val_f1)

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

        avg_test_f1 = test_f1/len(outs)
        """@nni.report_final_result(avg_test_f1)"""
        self.log('test_loss', test_loss)
        self.log('test_f1', avg_test_f1)

        os.system(f"rm {self.trainer.checkpoint_callback.best_model_path}")

def cli_main():
    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser.add_argument("--load", type=str, default=None)
    parser = MultilingualTokenClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    print(json.dumps(dict(sorted(vars(args).items())), indent=2))

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_f1', mode='max', patience=2)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_f1', mode='max', verbose=True)

    """@nni.variable(nni.choice(16, 32),name=accumulate_grad_batches)"""
    accumulate_grad_batches=args.accumulate_grad_batches
    """@nni.variable(nni.choice(True, False),name=stochastic_weight_avg)"""
    stochastic_weight_avg=args.stochastic_weight_avg
    """@nni.variable(nni.choice(0, 0.5, 1),name=gradient_clip_val)"""
    gradient_clip_val=args.gradient_clip_val

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping_callback, checkpoint_callback],
        gpus=1,
        precision=16,
        accumulate_grad_batches=accumulate_grad_batches,
        stochastic_weight_avg=stochastic_weight_avg,
        gradient_clip_val=gradient_clip_val
    )

    if args.load is None:
        model = MultilingualTokenClassifier(**vars(args))
        trainer.fit(model)
        if not args.fast_dev_run:
            print(f"Load from checkpoint {trainer.checkpoint_callback.best_model_path}")
            model = MultilingualTokenClassifier.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,
            )
    else:
        model = MultilingualTokenClassifier.load_from_checkpoint(
            args.load,
        )

    trainer.test(model)

if __name__ == "__main__":
    """@nni.get_next_parameter()"""
    cli_main()
