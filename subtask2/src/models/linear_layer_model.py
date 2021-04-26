import torch
import torch.nn as nn
import torch.utils.data as data_utils

import numpy as np

class Net(nn.Module):
    def __init__(self, idim=1024, odim=1, nhid=[0],
                 dropout=0.0, gpu=-1, activation='TANH', sigmoid=False):
        super(Net, self).__init__()
        modules = []
        print(' - mlp {:d}'.format(idim), end='')
        if len(nhid) > 0:
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
            nprev = idim
            for nh in nhid:
                if nh > 0:
                    modules.append(nn.Linear(nprev, nh))
                    nprev = nh
                    if activation == 'TANH':
                        modules.append(nn.Tanh())
                        print('-{:d}t'.format(nh), end='')
                    elif activation == 'RELU':
                        modules.append(nn.ReLU())
                        print('-{:d}r'.format(nh), end='')
                    else:
                       raise Exception('Unrecognized activation {activation}')
                    if dropout > 0:
                        modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(nprev, odim))
            print('-{:d}, dropout={:.1f}'.format(odim, dropout))
        else:
            modules.append(nn.Linear(idim, odim))
            print(' - mlp %d-%d'.format(idim, odim))
        if sigmoid:
            modules.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)

    def TestCorpus(self, dloader):
        predictions = []
        truth = []
        with torch.no_grad():
            for X, y in dloader:
                outputs = self.mlp(X)
                predicted = torch.where(outputs.data>0, torch.ones(1), torch.zeros(1)).flatten()
                label = torch.where(y>0.5, torch.ones(1), torch.zeros(1)).flatten()
                predictions += predicted.tolist()
                truth += label.tolist()

        #print(np.array(predictions))
        #print(np.array(truth))
        #print(truth[:30], predictions[:30])
        f = fscore(np.array(predictions), np.array(truth))
        return f

def make_dataloader(articles, batch_size=4, key='laser_emb'):
    embs = [a[key] for a in articles]
    lbl  = [a['label'] for a in articles]

    embs = np.array(embs)
    lbl = np.array(lbl)

    D = data_utils.TensorDataset(torch.from_numpy(embs).float(), torch.from_numpy(lbl).float())
    dataloader = data_utils.DataLoader(D, batch_size=batch_size, shuffle=True)
    return dataloader

def fscore(predictions, truth, f_weight=1.0):
    predictions = [1 if x>0 else 0 for x in predictions]
    truth = [1 if x>0 else 0 for x in truth]
    # Calculate fscore
    tp,tn,fp,fn = 0,1e-8,1e-8,1e-8
    for p, t in zip(predictions, truth):
        if p==1 and t==1: tp+=1
        if p==0 and t==0: tn+=1
        if p==0 and t==1: fn+=1
        if p==1 and t==0: fp+=1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    # fscore = ((1 + f_weight**2) * precision * recall) / ((f_weight**2 * precision) + recall)
    fscore = ((1 + f_weight**2) * tp) / (((1 + f_weight**2) * tp) + (f_weight**2)*fn + fp)

    precision = float(format(precision,'.3f'))
    recall = float(format(recall,'.3f'))
    fscore = float(format(fscore,'.3f'))

    tp,tn,fp,fn = int(tp), int(tn), int(fp), int(fn)
    n = sum([tp,tn,fp,fn])
    metrics = {'precision': precision, 'recall': recall, 'f1':fscore, 'n':n, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn}
    return metrics
