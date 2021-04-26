import os, pickle
import numpy as np
from tqdm import tqdm
from generate_embeddings import read

def attach_emb(article, cache):
    fp_file=os.path.join(cache, f"{str(article['id'])}.npy")
    embedding_np = np.load(fp_file)
    article["laser_emb"] = embedding_np

def save_articles(path, cache, save_location, lang):
    articles = read(path)
    cache = cache+lang
    save_location = save_location+lang

    for a in tqdm( articles ):
        attach_emb(a, cache)
        a['lang'] = lang

    with open(save_location+".pkl", 'wb') as f:
        pickle.dump(articles, f)
        print(f"Saved to {save_location}")


if __name__ == "__main__":
    cache = "../../data/interim/laser/"
    save_location = "../../data/processed/"

    save_articles("../../data/raw/en-train.json", cache, save_location, "en")
    save_articles("../../data/raw/es-train.json", cache, save_location, "es")
    save_articles("../../data/raw/pr-train.json", cache, save_location, "pr")
