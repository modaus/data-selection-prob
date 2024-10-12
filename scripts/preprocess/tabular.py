# This script is used to process datasets to the required format

import pickle
import openml
import numpy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

tabular_datasets = {
    '2dplanes': 727, 
    'click' : 1216, 
    'covertype': 1596,  
    'phoneme': 1489
}

trainsize = 500
devsize = trainsize
testsize = trainsize * 4

def process(dataset_name):
    dataset_id = tabular_datasets[dataset_name]
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

    spath = f'../../data/{dataset_name}.pkl' # _{trainsize}
    print(f'store to {spath}')

    with open(spath, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    for key, value in tabular_datasets.items():
        process(key)