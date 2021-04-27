import pickle
import torch
import pandas as pd
from linear_layer_model import Net, make_dataloader
from train_model import split_data

en = pickle.load(open('../../data/processed/en.pkl', 'rb')) # 23000
es = pickle.load(open('../../data/processed/es.pkl', 'rb')) # 2700
pr = pickle.load(open('../../data/processed/pr.pkl', 'rb')) # 1200
print(len(en), len(es), len(pr))

en_train, en_val = split_data(en, ratio=0.8)
es_train, es_val = split_data(es, ratio=0.8)
pr_train, pr_val = split_data(pr, ratio=0.8)

testsets ={
        "en_100%": en,
        "es_100%": es,
        "pr_100%": pr,

        "en_20%": en_val,
        "es_20%": es_val,
        "pr_20%": pr_val,
        }

def test_all_languages(model, langs, trained_on):
    results = []
    for l in langs:
        dataloader = make_dataloader(testsets[l])
        r = model.TestCorpus(dataloader)
        r['trained on'] = trained_on
        r['tested on'] = l
        results.append(r)
    return results


all_results = []
save_location = "../../models"
en_net = torch.load(f'./{save_location}/en.pth')
es_net = torch.load(f'./{save_location}/es.pth')
pr_net = torch.load(f'./{save_location}/pr.pth')

en_test_langs = ["en_20%" , "es_100%", "pr_100%"]
es_test_langs = ["en_100%", "es_20%" , "pr_100%"]
pr_test_langs = ["en_100%", "es_100%", "pr_20%" ]

all_results += test_all_languages(en_net, en_test_langs, "en_80%")
all_results += test_all_languages(es_net, es_test_langs, "es_80%")
all_results += test_all_languages(pr_net, pr_test_langs, "pr_80%")

results_df = pd.DataFrame(all_results)
results_df.to_csv("./subtask_results.csv")
print(results_df)
