import requests, json
import numpy as np

def laser_embed(article: dict):
    summary_url = f"http://192.168.2.72:3111/article/summary?sentences=1&embed=1&min_length=25"
    embedding_arr = requests.post(summary_url,
        headers={'Content-Type': 'application/json; charset=utf-8'},
        data=json.dumps(article),
    ).json()['embed']
    embedding_np = np.asarray(embedding_arr)

    return embedding_np

def joined_summary_ranked_embedding(article, max_sentences=100, summary_length=10, cache="./summary_ranked_cache", refresh_cache=False, save=True):
    article_id = article['article_id']
    # If sentences provided are less than summary_length, 'fp'(top sentences)=='fp_sentences'(all sentences)
    summary_length = min(len(article['sentences']), summary_length)
    if 'fp_sentences' and 'fp' in article:
        return
    fp_file=os.path.join(cache, str(article_id)[-2:], f"{str(article_id)}.npy")
    if refresh_cache or not os.path.isfile(fp_file):
        summary_url = f"http://{c_phraser.server.host}:{c_phraser.server.port}/article/summary?sentences={max_sentences}&embed=1&min_length=25"
        embedding_arr = requests.post(summary_url,
            headers={'Content-Type': 'application/json; charset=utf-8'},
            data=json.dumps(article),
        ).json()['embed']
        embedding_np = np.asarray(embedding_arr)
        article['fp'] = embedding_np[:summary_length,] # Top ranked embeddings
        article['fp_sentences'] = embedding_np
        # print(len(article['fp']), len(article['fp_sentences']))
        if save:
            print(f"  Storing {article_id} as '{fp_file}' - {str(embedding_np.shape)}")
            os.makedirs(os.path.dirname(fp_file), exist_ok=True)
            np.save(fp_file,  embedding_np)
    else:
        # print("loaded", fp_file)
        embedding_np = np.load(fp_file)
        article['fp'] = embedding_np[:summary_length,] # Top ranked embedding
        article['fp_sentences'] = embedding_np

