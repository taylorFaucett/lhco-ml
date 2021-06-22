# standard library imports
from __future__ import absolute_import, division, print_function

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# standard numerical library imports
import numpy as np
import h5py
from tqdm import tqdm
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import ShuffleSplit
import scipy.stats

import matplotlib.pyplot as plt
import pathlib
import pandas as pd

path = pathlib.Path.cwd()

def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h    
    
def scale_data(x, mean=True):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1
    return (x - mean) / std    
    
def run_net(X, Y, bbx, network, JI, verbose=0, save_predictions=False):
    # do train/val/test split
    n = len(y)
    n_train = int(0.85 * n)
    n_test = int(0.15 * n)
    rs = ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.15)
    rs.get_n_splits(X)

    ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.15)
    straps = []
    aucs = []
    ados = []
    bs_count = 0
    t = tqdm(list(rs.split(X)))
    counter = 0
    params = {
        "objective": "binary:logistic",
        "base_score": np.mean(y),
        "scale_pos_weight": 1,
        "eta": 0.1,
        "learning_rate":0.01,  
        "colsample_bytree": 0.4,
        "subsample": 0.8,
        "n_estimators":1000, 
        "reg_alpha" : 0.3,
        "max_depth":4, 
        "gamma":10,
        "gpu_id":0,
        'tree_method':'gpu_hist', 
        'predictor':'gpu_predictor',
        "eval_metric": "auc"
    }  
    for train_index, test_index in t:        
        if JI:
            roc_file = path / "results" / "bootstrap" / f"{network}_JI"  / bbx / "roc" / f"roc_{bs_count}.csv"
        else:
            roc_file = path / "results" / "bootstrap" / network  / bbx / "roc" / f"roc_{bs_count}.csv"
        roc_file.parent.mkdir(parents=True, exist_ok=True)
        
#         full = xgb.DMatrix(data=X,label=y, weight=weights)
#         train = xgb.DMatrix(data=X[train_index],label=y[train_index], weight=weights[train_index])
#         test = xgb.DMatrix(data=X[test_index],label=y[test_index], weight=weights[test_index])
        full = xgb.DMatrix(data=X,label=y)
        train = xgb.DMatrix(data=X[train_index],label=y[train_index])
        test = xgb.DMatrix(data=X[test_index],label=y[test_index])

        model = xgb.train(params, 
                              train, 
                              evals=[(train, "train"), (test, "validation")], 
                              num_boost_round=1000, 
                              early_stopping_rounds=20,
                              verbose_eval=100
                             )
        val_predictions = np.hstack(model.predict(test))
        full_predictions = np.hstack(model.predict(full))
        fpr, tpr, _ = roc_curve(
            y[test_index], val_predictions,# sample_weight=weights[test_index]
        )
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr,})
        roc_df.to_csv(roc_file)
        bs_count += 1

if __name__ == "__main__": 
    N = 100000
    network = "xgb"
    bbx = "BlackBox1"
    JI = False
    n_splits = 200
    num_epoch = 200
    if JI:
        hf = h5py.File(path.parent / "data" / "FlowNetwork_from_pyjet" / f"{bbx}.h5", "r")
        X = hf["features"][:N]
        y = hf["targets"][:N]
    else:
        h5_file = path.parent / "data" / "raw" / "combined" / f"{bbx}.h5"
        X = pd.read_hdf(h5_file, "features", start=0, stop=N).to_numpy()
        y = np.concatenate(pd.read_hdf(h5_file, "targets", start=0, stop=N).to_numpy())
    X[X==0] = np.nan
    run_net(X, y, bbx, network, JI, verbose=0, save_predictions=True)