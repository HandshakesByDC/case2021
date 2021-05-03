from pathlib import Path
import re

class GloconFile:
    def __init__(self, token_docs, tag_docs):
        self.token_docs = token_docs
        self.tag_docs = tag_docs

    @classmethod
    def build(cls, file_path, max_tag=400):
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

        return cls(token_docs, tag_docs)

    def save(self, file_path):
        with open(file_path, 'w') as f:
            for tokens, tags in zip(self.token_docs, self.tag_docs):
                assert(len(tokens) == len(tags))
                for token, tag in zip(tokens, tags):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")
