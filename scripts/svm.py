## simple.py
## A simple example
import os
import sys

sys.path.append(os.path.abspath('./src/'))

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


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        outputs = self.softmax(self.linear(x))
        return outputs


def main():
    data = load(open('./data/diabetes.pkl', 'rb'))
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

    model = LogisticRegression(input_dim=trnX.shape[1], output_dim=len(labels))
    clf = PytorchClassifier(model, epochs=20, trn_batch_size=16,
                            dev_batch_size=16)

    dv = DataValuation(trnX, trnY, devX, devY)

    params = Parameters()
    # params.update({'second_order_grad':True})
    # params.update({'for_high_value':False})

    vals = dv.estimate(clf=clf, method='inf-func', params=params.get_values())

    # print(vals)

    accs = data_removal(vals, trnX, trnY, tstX, tstY, clf)

    dict_acc_sel = dict()
    strategies = ['greedy', 'prob', 'softmax', 'roulette', 'stratified', 'threshold']
    for strategy in strategies:
        acc_sel = data_selection(vals, 
                                 trnX, trnY, tstX, tstY, 
                                 clf,
                                 sel=0.3,
                                 strategy=strategy)
        dict_acc_sel[strategy] = acc_sel
    print(dict_acc_sel)

    # res = weighted_acc_drop(accs)
    # print("The weighted accuracy drop is {}".format(res))
    # plt.plot(res)
    # plt.show()
    # print(accs)

if __name__ == '__main__':
    for _ in range(10):
        main()