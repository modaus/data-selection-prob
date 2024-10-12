# This script is used to process datasets to the required format

# Diabetes
# CPU
# Click
# Covertype
# CIFAR10 (binarized)
# FMNIST (binarized)
# MNIST (binarized and multi-class versions, denoted using -2 and -10, respectively)
# Phoneme

# 'diabetes'
all = {'cpu': 197, 
       'click' : 1216, 
    #    'covertype', 
    #    'cifar10', 
    #    'fmnist', 
    #    'mnist-2', 
    #    'minst-10', 
    #    'phoneme'
       }



import pickle
import openml
import numpy
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

trainsize = 500
devsize = trainsize
testsize = trainsize * 4
dataset_name = 'click'
dataset_id = all[dataset_name]
dataset = openml.datasets.get_dataset(dataset_id)
X, Y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

print(f"dataset name:\t\t {dataset.name}")
print(f"num of data points:\t {X.shape[0]}")
print(f"num of features:\t {X.shape[1]}")

idx = np.arange(len(Y))
slice_idx = numpy.random.choice(idx, trainsize+devsize+testsize, replace=False)
X = X.iloc[slice_idx].to_numpy()

le = LabelEncoder()
# encode non-number col
for i in range(X.shape[1]):
    if not np.issubdtype(X[:, i].dtype, np.number):
        X[:, i] = le.fit_transform(X[:, i])

Y = Y.iloc[slice_idx].to_numpy()

# force to binary class
Y = le.fit_transform(Y)
Y = np.array([1 if i == 1 else 0 for i in Y], dtype=int)

tmpX, tstX, tmpY, tstY = train_test_split(X, Y, test_size = testsize, stratify = Y)
trnX, devX, trnY, devY = train_test_split(tmpX, tmpY, test_size = devsize, stratify = tmpY)


result = {}
result["trnX"] = trnX
result["trnY"] = trnY
result["devX"] = devX
result["devY"] = devY
result["tstX"] = tstX
result["tstY"] = tstY

for key, value in result.items():
    print(f"{key}: {value.shape}")

spath = f'../data/{dataset_name}_{trainsize}.pkl'
print(f'store to {spath}')

with open(spath, "wb") as f:
    pickle.dump(result, f)

# # Train a logistic regression classifier and evaluate its accuracyc
# from sklearn.linear_model import LogisticRegression as LR
# from sklearn.metrics import accuracy_score
# clf = LR(solver="liblinear", max_iter=500, random_state=0)
# clf.fit(trnX, trnY)
# acc = accuracy_score(clf.predict(tstX), tstY)
# print(acc)

# import numpy as np
# from scipy.io import arff
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedShuffleSplit
# import pickle
# from sklearn.linear_model import LogisticRegression as LR
# from sklearn.metrics import accuracy_score

# trainsize = 500
# devsize = trainsize
# testsize = trainsize * 4
# dataset = "covertype"
# orgdata = arff.loadarff('../data/{}.arff'.format(dataset))
# data, _ = orgdata
# idx = np.arange(len(data))
# sliceidx = np.random.choice(idx, trainsize+devsize+testsize, replace=False)
# data = data[sliceidx]

# arr = np.array(data)
# data = np.array(arr.tolist())
# dataX = np.delete(data, -1, 1)
# le = LabelEncoder()

# # 对dataX中的每一列进行标签编码
# for i in range(dataX.shape[1]):
#     if not np.issubdtype(dataX[:, i].dtype, np.number):
#         dataX[:, i] = le.fit_transform(dataX[:, i])
# dataX = np.array(dataX, dtype=float)

# dataY = data[:, -1]
# dataY = le.fit_transform(dataY)
# # 多分类到二分类
# dataY = [1 if i == 1 else 0 for i in dataY]
# print(np.sum(dataY), dataY[:100])
# # 使用StratifiedShuffleSplit进行平衡划分
# sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.6666666, random_state=0)
# sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

# for train_index, test_index in sss1.split(dataX, dataY):
#     tmpX, tstX = dataX[train_index], dataX[test_index]
#     tmpY, tstY = np.array(dataY)[train_index], np.array(dataY)[test_index]

# for train_index, dev_index in sss2.split(tmpX, tmpY):
#     trnX, devX = tmpX[train_index], tmpX[dev_index]
#     trnY, devY = tmpY[train_index], tmpY[dev_index]

# result = {}
# result["trnX"] = trnX
# result["trnY"] = trnY
# result["devX"] = devX
# result["devY"] = devY
# result["tstX"] = tstX
# result["tstY"] = tstY
# with open("../data/{}.pkl".format(dataset, trainsize), "wb") as f:
#     pickle.dump(result, f)

# # Train a logistic regression classifier and evaluate its accuracy
# clf = LR(solver="liblinear", max_iter=500, random_state=0)
# clf.fit(trnX, trnY)
# acc = accuracy_score(clf.predict(tstX), tstY)
# print(acc)