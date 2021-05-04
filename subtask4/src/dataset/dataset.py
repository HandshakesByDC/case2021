import torch

from sklearn.model_selection import train_test_split

from .fileio import GloconFile

class GloconDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    @classmethod
    def build(cls, file_path, tokenizer, tag_map, test_split=0, **kwargs):
        gf = GloconFile.build(file_path, **kwargs)
        tag2id = tag_map.tag2id

        if test_split > 0:
            train_token_docs, test_token_docs = train_test_split(gf.token_docs, random_state=0, test_size=test_split)
            train_tag_docs, test_tag_docs = train_test_split(gf.tag_docs, random_state=0, test_size=test_split)
            train_encodings = cls.create_encodings(train_token_docs, tokenizer)
            test_encodings = cls.create_encodings(test_token_docs, tokenizer)
            train_align_tags = cls.create_aligned_tags(train_encodings, train_tag_docs, tag2id)
            test_align_tags = cls.create_aligned_tags(test_encodings, test_tag_docs, tag2id)

            return cls(train_encodings, train_align_tags), cls(test_encodings, test_align_tags)
        else:
            encodings = cls.create_encodings(gf.token_docs, tokenizer)
            align_tags = cls.create_aligned_tags(encodings, gf.tag_docs, tag2id)

            return cls(encodings, align_tags)

    @staticmethod
    def create_encodings(token_docs, tokenizer):
        return tokenizer(
            token_docs,
            is_split_into_words=True,
            padding=True,
            truncation=True
        )

    @staticmethod
    def create_aligned_tags(encodings, tag_docs, tag2id):
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
        return align_tags
