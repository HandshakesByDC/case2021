from pathlib import Path
import re

class GloconFile:
    def __init__(self, token_docs, tag_docs):
        self.token_docs = token_docs
        self.tag_docs = tag_docs

    @classmethod
    def build(cls, file_path, max_tags=300):
        file_path = Path(file_path)

        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        token_docs = []
        tag_docs = []
        print(f"Reading {file_path}: ", end="")
        print(f"{len(raw_docs)} examples - ", end="")
        if '\t' in raw_docs[0].split('\n')[0]:
            labels_exists = True
            print("labels exist")
        else:
            labels_exists = False
            print("labels do not exist")
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                if labels_exists:
                    token, tag = line.split('\t')
                else:
                    token, tag = line, 'O'
                tokens.append(token)
                tags.append(tag)
            if max_tags == -1:
                sep_idx = [i for i,t in enumerate(tokens) if t == '[SEP]']
                tokens_split = [tokens[i:j] for i, j in zip([0]+sep_idx, sep_idx+[None])]
                tags_split = [tags[i:j] for i, j in zip([0]+sep_idx, sep_idx+[None])]
                for token_split, tag_split in zip(tokens_split, tags_split):
                    if token_split[0] == '[SEP]':
                        token_split[0] = 'SAMPLE_NOT_START'
                    token_docs.append(token_split)
                    tag_docs.append(tag_split)
            else:
                while len(tokens) > max_tags:
                    print(f"Split {len(tokens)} to -> :", end="")
                    sep_idx = [i for i,t in enumerate(tokens) if t == '[SEP]']
                    split_at = sep_idx[len(sep_idx)//2]
                    minus = 0
                    while split_at > max_tags:
                        minus += 1
                        split_at = sep_idx[len(sep_idx)//2 - minus]
                    print(f"{len(tokens[:split_at])} {len(tokens[split_at:])}")
                    token_docs.append(tokens[:split_at])
                    tag_docs.append(tags[:split_at])
                    tokens = tokens[split_at:]
                    tokens[0] = 'SAMPLE_NOT_START'
                    tags = tags[split_at:]
                token_docs.append(tokens)
                tag_docs.append(tags)

        return cls(token_docs, tag_docs)

    def save(self, file_path):
        print(f"Saving {file_path}: ", end="")
        lines = []
        minus = 0
        with open(file_path, 'w') as f:
            for tokens, tags in zip(self.token_docs, self.tag_docs):
                assert(len(tokens) == len(tags))
                for token, tag in zip(tokens, tags):
                    if token == 'SAMPLE_NOT_START':
                        lines.pop()
                        token = '[SEP]'
                        minus += 1
                    lines.append(f"{token}\t{tag}\n")
                lines.append("\n")
            f.writelines(lines)
            print(f"{len(self.token_docs) - minus} predictions")
