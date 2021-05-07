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
    def __init__(self, model_name, labels_path, batch_size, source_data, translate_data,
                 hidden_dropout_prob, **kwargs):
        super().__init__()
        self.save_hyperparameters()
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
        self.source_data = source_data
        self.translate_data = translate_data
        self.learning_rate = 5e-5

    def on_post_move_to_device(self):
        self.viterbi_decoder = ViterbiDecoder(self.tag_map.id2tag, -100, self.device)

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MultilingualTokenClassifier")
        parser.add_argument('--model_name', type=str, default='bert')
        parser.add_argument('--labels_path', type=str, default='data/labels.txt')
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--source_data', action='store_true')
        parser.add_argument('--translate_data', action='store_true')
        parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
        return parent_parser

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]
        outputs = self.bert(
            input_ids, attention_mask=attention_mask
        )
        log_probs = torch.nn.functional.log_softmax(outputs.logits.detach(), dim=-1)
        batch_preds = self.viterbi_decoder.forward(log_probs, attention_mask)
        return batch_preds

    def train_dataloader(self):
        train_datasets = []
        if self.source_data:
            en_dataset, _ = GloconDataset.build('data/en-orig.txt', self.tokenizer, self.tag_map, test_split=0.05)
            train_datasets.append(en_dataset)
        if self.translate_data:
            en2es_dataset, _ = GloconDataset.build('data/en2es.txt', self.tokenizer, self.tag_map, test_split=0.05)
            train_datasets.append(en2es_dataset)
            en2pt_dataset, _ = GloconDataset.build('data/en2pt.txt', self.tokenizer, self.tag_map, test_split=0.05)
            train_datasets.append(en2pt_dataset)

        assert(len(train_datasets) > 0)
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
        val_loaders = []
        if self.source_data:
            _, en_dataset = GloconDataset.build('data/en-orig.txt', self.tokenizer, self.tag_map, test_split=0.05)
            en_loader = DataLoader(en_dataset, batch_size=self.batch_size, shuffle=False)
            val_loaders.append(en_loader)

        if self.translate_data:
            es_dataset = GloconDataset.build('data/es-orig.txt', self.tokenizer, self.tag_map)
            pt_dataset = GloconDataset.build('data/pt-orig.txt', self.tokenizer, self.tag_map)

            es_loader = DataLoader(es_dataset, batch_size=self.batch_size, shuffle=False)
            pt_loader = DataLoader(pt_dataset, batch_size=self.batch_size, shuffle=False)

            val_loaders.append(es_loader)
            val_loaders.append(pt_loader)

        return val_loaders

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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
        if isinstance(outs[0], list):
            is_multiple = True
        else:
            is_multiple = False
        if is_multiple:
            for i, out in enumerate(outs):
                val_loss += sum(map(lambda x: x[f"val_loss_{i}"], out), 0)

                preds = sum(map(lambda x: x[f"val_preds_{i}"], out), [])
                labels = sum(map(lambda x: x[f"val_labels_{i}"], out), [])
                preds_tag = [id2tag[i] for i in preds]
                labels_tag = [id2tag[i] for i in labels]
                _, _, f1 = conll_evaluate(labels_tag, preds_tag)
                val_f1 += f1
        else:
            val_loss += sum(map(lambda x: x[f"val_loss_0"], outs), 0)

            preds = sum(map(lambda x: x[f"val_preds_0"], outs), [])
            labels = sum(map(lambda x: x[f"val_labels_0"], outs), [])
            preds_tag = [id2tag[i] for i in preds]
            labels_tag = [id2tag[i] for i in labels]
            _, _, f1 = conll_evaluate(labels_tag, preds_tag)
            val_f1 += f1
        len_outs = len(outs) if is_multiple else 1
        avg_val_f1 = val_f1/len_outs
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

def cli_main():
    pl.seed_everything(1234)

    # args
    parser = ArgumentParser()
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--predict", action='store_true')
    parser.add_argument("--delete_checkpoint", action='store_true')
    parser = MultilingualTokenClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    print(json.dumps(dict(sorted(vars(args).items())), indent=2))

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_f1', mode='max', patience=2)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_f1', mode='max', verbose=True)

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[early_stopping_callback, checkpoint_callback],
        gpus=1,
        precision=16,
        accumulate_grad_batches=16,
        stochastic_weight_avg=True,
        gradient_clip_val=1.0
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
        if args.finetune:
            model_args = [x.dest for x in vars(parser._action_groups[2])['_group_actions']]
            for k in model_args:
                if hasattr(model, k) and getattr(model, k) != vars(args)[k]:
                        print(f"Change model.{k} from {getattr(model,k)} -> {vars(args)[k]}")
                        setattr(model, k, vars(args)[k])
            trainer.fit(model)

    trainer.test(model)

    if args.predict:
        en_predict_dataset = GloconDataset.build('data/en-test.txt',
                                                model.tokenizer, model.tag_map)
        es_predict_dataset = GloconDataset.build('data/es-test.txt',
                                                model.tokenizer, model.tag_map)
        pt_predict_dataset = GloconDataset.build('data/pt-test.txt',
                                                model.tokenizer, model.tag_map)

        en_predict_loader = DataLoader(en_predict_dataset, batch_size=model.batch_size, shuffle=False)
        es_predict_loader = DataLoader(es_predict_dataset, batch_size=model.batch_size, shuffle=False)
        pt_predict_loader = DataLoader(pt_predict_dataset, batch_size=model.batch_size, shuffle=False)

        preds = trainer.predict(model, dataloaders=[en_predict_loader, es_predict_loader, pt_predict_loader])
        en_labels = sum(map(lambda x: x, preds[0]), [])
        es_labels = sum(map(lambda x: x, preds[1]), [])
        pt_labels = sum(map(lambda x: x, preds[2]), [])
        en_predict_dataset.save(en_labels, model.tag_map, "predict-en.txt")
        es_predict_dataset.save(es_labels, model.tag_map, "predict-es.txt")
        pt_predict_dataset.save(pt_labels, model.tag_map, "predict-pt.txt")

    if args.delete_checkpoint:
        os.system(f"rm {trainer.checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    """@nni.get_next_parameter()"""
    cli_main()
