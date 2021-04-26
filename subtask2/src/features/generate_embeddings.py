import json
import os
import numpy as np
from tqdm import tqdm
from laser_embedding import laser_embed

def read(path):
    """
    Reads the file from the given path (json file).
    Returns list of instance dictionaries.
    """
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for instance in file:
            a = json.loads(instance)
            a['sentences'] = [a['sentence']] 
            data.append(a)

    return data

def save_embedding(embedding_np, article_id, cache):
    fp_file=os.path.join(cache, f"{str(article_id)}.npy")
    os.makedirs(os.path.dirname(fp_file), exist_ok=True)
    np.save(fp_file,  embedding_np)

def generate_embs(path, cache):
    articles = read(path)
    for a in tqdm( articles ):
        emb = laser_embed(a)[0]
        # print(emb)
        save_embedding(emb, a['id'], cache)



if __name__ == "__main__":

    cache = "../../data/interim/laser/"
    generate_embs("../../data/raw/en-train.json", cache+"en")
    generate_embs("../../data/raw/es-train.json", cache+"es")
    generate_embs("../../data/raw/pr-train.json", cache+"pr")
