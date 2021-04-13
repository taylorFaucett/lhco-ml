# standard library imports
from __future__ import absolute_import, division, print_function

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# standard numerical library imports
import numpy as np
import h5py
from tqdm import tqdm
from rich import print
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# energyflow imports
import energyflow as ef
from energyflow.archs import EFN, PFN
from energyflow.utils import data_split, to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import ShuffleSplit
import scipy.stats
from get_best_sherpa_result import get_best_sherpa_result

import matplotlib.pyplot as plt
from average_decision_ordering import calc_ado
import pathlib
import pandas as pd

path = pathlib.Path.cwd()

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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
    
def run_net(X, Y, bbx, network, verbose=0, save_predictions=False):
    # Pre-process data
    for x in X:
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
        
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
    for train_index, test_index in t:
        if network == "EFN":
            z_train = X[:, :, 0][train_index]
            p_train = X[:, :, 1:][train_index]
            z_val = X[:, :, 0][test_index]
            p_val = X[:, :, 1:][test_index]
            X_train = [z_train, p_train]
            X_val = [z_val, p_val]
            X_complete = [X[:,:,0], X[:,:,1:]]
            input_dim = 3
        elif network == "PFN":
            X_train = X[train_index]
            X_val = X[test_index]
            X_complete = X
            input_dim = X_train.shape[-1]
        Y_train = y[train_index]
        Y_val = y[test_index]
        
        Y_train = to_categorical(Y_train, num_classes=2)
        Y_val = to_categorical(Y_val, num_classes=2)
        
        model_name = f"model_{bs_count}"
        model_file = path / "results" / "bootstrap" / network / bbx / "models" / f"{model_name}.h5"
        roc_file = path / "results" / "bootstrap" / network  / bbx / "roc" / f"roc_{bs_count}.csv"
        ll_prediction_file = path / "results" / "bootstrap" / network / bbx / "predictions" / f"ll_predictions_{bs_count}.npy"
        roc_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        ll_prediction_file.parent.mkdir(parents=True, exist_ok=True)

        tp = get_best_sherpa_result(network, bbx)
        if not model_file.exists():
            net = eval(network)(
                    input_dim=input_dim,
                    Phi_sizes = (300,300,300),
                    F_sizes = (300,300,300),
                    Phi_acts="relu", 
                    F_acts = "relu",
                    Phi_k_inits="glorot_normal",
                    F_k_inits="glorot_normal",
                    latent_dropout=0.2,
                    F_dropouts=0.0,
                    mask_val = 0,
                    loss="categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(lr=0.001),
                    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
                    output_act="softmax",
                    summary=False
                )           

            mc = tf.keras.callbacks.ModelCheckpoint(
                str(model_file),
                monitor="val_auc",
                verbose=verbose,
                save_best_only=True,
                mode="max",
            )
            es = tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", verbose=verbose, patience=5
            )
            try:
                # train model
                net.fit(
                    X_train,
                    Y_train,
                    epochs=num_epoch,
                    batch_size=16,#int(tp["batch_size"]),
                    validation_data=(X_val, Y_val),
                    callbacks=[es, mc],
                    verbose=verbose,
                    use_multiprocessing = True,
                    workers = 8
                )
            except KeyboardInterrupt:
                print(f"removing file: {str(model_file)}")
                os.remove(str(model_file))
                sys.exit()
        else:
            net = tf.keras.models.load_model(model_file)

        # get predictions on test data
        preds = net.predict(X_val)
        
        # Save the predictions
        if save_predictions:
            full_predictions = net.predict(X_complete)[:,1]
            np.save(ll_prediction_file, full_predictions)

        # get ROC curve
        fpr, tpr, threshs = roc_curve(Y_val[:, 1], preds[:, 1])

        roc_df = pd.DataFrame(
            {
                "fpr": fpr,
                "tpr": tpr,
            }
        )
        roc_df.to_csv(roc_file)  
        bs_count += 1

if __name__ == "__main__": 
    network = str(sys.argv[1])
    bbx = str(sys.argv[2])
    n_splits = 200
    num_epoch = 200
    h5_file = h5py.File(path.parent / "data" / "with_rotations" / f"{bbx}.h5", "r")
    X = h5_file["features"][:]
    y = h5_file["targets"][:]    
    X0 = X[y==0]
    X1 = X[y==1]
    max_size = min(len(X0), len(X1))
    X = np.vstack((X0[:max_size], X1[:max_size]))
    y = np.hstack((np.zeros(max_size), np.ones(max_size)))
    
    Y = to_categorical(y, num_classes=2)
    run_net(X, Y, bbx, network, verbose=1, save_predictions=True)