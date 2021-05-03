from pathlib import Path
import re

import torch

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class TagMap(metaclass=Singleton):
    def __init__(self, unique_tags):
        print("creating TagIdMap")
        self.tag2id = {tag: id for id, tag in enumerate(unique_tags)}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}

class GloconDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, tag_map):
        self.encodings = encodings
        self.labels = labels
        self.tag_map = tag_map

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    @classmethod
    def build(cls, file_path, tokenizer):
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
            while len(tokens) > 400:
                print(f"Split {len(tokens)} to -> :", end="")
                sep_idx = [i for i,t in enumerate(tokens) if t == '[SEP]']
                split_at = sep_idx[len(sep_idx)//2]
                token_docs.append(tokens[:split_at])
                tag_docs.append(tags[:split_at])
                tokens = tokens[split_at:]
                tokens[0] = 'SAMPLE_START'
                tags = tags[split_at:]
                print(f" {len(tokens)}")
            token_docs.append(tokens)
            tag_docs.append(tags)

        encodings = tokenizer(
            token_docs,
            is_split_into_words=True,
            padding=True,
            truncation=True
        )

        unique_tags = set(tag for doc in tag_docs for tag in doc)
        tag_map = TagMap(unique_tags)
        tag2id = tag_map.tag2id

        align_tags = []
        for i, tag in enumerate(tag_docs):
            word_ids = encodings.word_ids(batch_index=i)
            previous_word_idx = None
            tag_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label
                # to -100 so they are automatically ignored in the loss function.
                if word_idx is None:
                    tag_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    tag_ids.append(tag2id[tag[word_idx]])
                # For the other tokens in a word, we set the label to either the
                # current label or -100, depending on the label_all_tokens flag.
                else:
                    tag_ids.append(tag2id[tag[word_idx]] if False else -100)
                previous_word_idx = word_idx

            align_tags.append(tag_ids)

        return cls(encodings, align_tags, tag_map)
