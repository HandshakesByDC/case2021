import torch

from .fileio import GloconFile

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
    def build(cls, file_path, tokenizer, **kwargs):
        gf = GloconFile.build(file_path, **kwargs)

        encodings = tokenizer(
            gf.token_docs,
            is_split_into_words=True,
            padding=True,
            truncation=True
        )

        tag2id = gf.tag_map.tag2id

        align_tags = []
        for i, tag in enumerate(gf.tag_docs):
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

        return cls(encodings, align_tags, gf.tag_map)
