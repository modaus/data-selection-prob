## simple.py
## A simple example
import os
import sys

sys.path.append(os.path.abspath('../src/'))

import pickle
import torch
from pickle import load
from sklearn import preprocessing
from matplotlib import pyplot as plt

import valda
from valda.valuation import DataValuation
from valda.pyclassifier import PytorchClassifier
from valda.eval import data_removal, data_selection
from valda.metrics import weighted_acc_drop
from valda.params import Parameters
import numpy as np
np.random.seed(42)

# class LogisticRegression(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegression, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x):
#         outputs = self.softmax(self.linear(x))
#         return outputs


def main(dataset_name, sel):
    data = load(open(f'../data/{dataset_name}.pkl', 'rb'))
    trnX, trnY = data['trnX'], data['trnY']
    devX, devY = data['devX'], data['devY']
    tstX, tstY = data['tstX'], data['tstY']
    # print('trnX.shape = {}'.format(trnX.shape))

    labels = list(set(trnY))
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    trnY = le.transform(trnY)
    devY = le.transform(devY)
    tstY = le.transform(tstY)

    # model = LogisticRegression(input_dim=trnX.shape[1], output_dim=len(labels))
    # clf = PytorchClassifier(model, epochs=20, trn_batch_size=16,
    #                         dev_batch_size=16)
    vals_path = f'../data/{dataset_name}_sv.pkl'
    if os.path.exists(vals_path):
        vals = load(open(vals_path, 'rb'))
    else:
        dv = DataValuation(trnX, trnY, devX, devY)
        params = Parameters()

        vals = dv.estimate(method='tmc-shapley', params=params.get_values())
        with open(vals_path, 'wb') as f:
            pickle.dump(vals, f)
    
    # print(vals)
    # print("sum", np.sum(vals))
    # accs = data_removal(vals, trnX, trnY, tstX, tstY, clf)
    # print(vals)

    dict_acc_sel = dict()
    strategies = ['greedy', 's-greedy', 'random', 'prob', 'softmax',  'stratified', 'threshold']
    #  'prob', 'roulette',
    for strategy in strategies:
        acc_sel = data_selection(vals, 
                                 trnX, trnY, tstX, tstY, 
                                 sel=sel,
                                 strategy=strategy)
        dict_acc_sel[strategy] = acc_sel
    # print(dict_acc_sel)
    return dict_acc_sel
    # res = weighted_acc_drop(accs)
    # print("The weighted accuracy drop is {}".format(res))
    # plt.plot(res)
    # plt.show()
    # print(accs)

if __name__ == '__main__':
    for sel in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        for d in ['diabetes', 'click', '2dplanes',  'phoneme']: # 'diabetes', 'click', 'covertype',
            print(f'Processing {d} sel {sel}')
            res = []
            for i in range(50):
                res.append(list(main(d, sel).values()))
            print(np.mean(res, axis=0))
            print(np.std(res, axis=0))
        
