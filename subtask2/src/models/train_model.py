import pickle
import copy

import torch
from torch import optim

from linear_layer_model import Net, make_dataloader

def model_train(train_articles, val_articles, test_articles=None):
    torch.manual_seed(0)
    val_loss_best = 1000
    best_f1 = 0.0
    best_metrics = {}

    net = Net(nhid=[512, 128], idim=1024, sigmoid=False, activation='RELU', dropout=0.2)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1]))
    optimizer = optim.Adam(net.parameters(),
                           lr=0.00001,
                           weight_decay= 5e-5,
                           # betas=(0.9, 0.999),
                           # eps=1e-8,
                           # amsgrad=True)
                           )

    train_loader = make_dataloader(train_articles)
    val_loader = make_dataloader(val_articles)
    if test_articles:
        test_loader = make_dataloader(test_articles)
    else:
        test_loader = None

    for epoch in range(100):
        loss_epoch = 0.0
        for data in train_loader:
            inputs, labels = data
            net.train(mode=True)
            outputs = net(inputs).flatten()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        val_loss_epoch = 0.0
        for data in val_loader:
            inputs, labels = data
            net.train(mode=False)
            outputs = net(inputs).flatten()
            loss = criterion(outputs, labels)
            val_loss_epoch += loss.item()

        f = net.TestCorpus(val_loader)

        if f['f1'] > best_f1:
        # if val_loss_epoch < val_loss_best:
            print(f'Ep: {epoch}   loss: {val_loss_epoch}')
            print(f)
            if test_loader:
                net.train(mode=False)
                f_test = net.TestCorpus(test_loader)
                print(f'TEST: {f_test}')
            best_metrics = f
            best_f1 = f['f1']
            val_loss_best = val_loss_epoch
            net_best = copy.deepcopy(net)

    if best_f1==0:
        net_best = copy.deepcopy(net)
        best_metrics = {'recall':0, 'precision':0, 'f1':0}
    return net_best

def split_data(data:list, ratio:float=0.8):
    train = data[:int(len(data)*ratio)]
    val   = data[int(len(data)*ratio):]
    return train, val


if __name__ == "__main__":
    en = pickle.load(open('../../data/processed/en.pkl', 'rb')) # 23000
    es = pickle.load(open('../../data/processed/es.pkl', 'rb')) # 2700
    pr = pickle.load(open('../../data/processed/pr.pkl', 'rb')) # 1200
    print(len(en), len(es), len(pr))

    test_en_loader = make_dataloader(en)
    test_es_loader = make_dataloader(es)
    test_pr_loader = make_dataloader(pr)

    def test_all_languages(model):
        print("TEST English", model.TestCorpus(test_en_loader))
        print("TEST Spanish", model.TestCorpus(test_es_loader))
        print("TEST Portugese", model.TestCorpus(test_pr_loader))


    save_location = "../../models"
    print("ENGLISH TRAINED")
    en_train, en_val = split_data(en, ratio=0.8)
    en_net = model_train(en_train, en_val)
    print("ENGLISH TRAINED")
    test_all_languages(en_net)
    torch.save(en_net, f'./{save_location}/en.pth')


    print("SPANISH TRAINED")
    es_train, es_val = split_data(es, ratio=0.8)
    es_net = model_train(es_train, es_val)
    print("SPANISH TRAINED")
    test_all_languages(es_net)
    torch.save(es_net, f'./{save_location}/es.pth')

    print("PORTUGESE TRAINED")
    pr_train, pr_val = split_data(pr, ratio=0.8)
    pr_net = model_train(pr_train, pr_val)
    print("PORTUGESE TRAINED")
    test_all_languages(pr_net)
    torch.save(pr_net, f'./{save_location}/pr.pth')










