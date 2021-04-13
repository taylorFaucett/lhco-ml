import h5py
import pathlib
import numpy as np
import pandas as pd
from sklearn import preprocessing

path = pathlib.Path.cwd()


def scale_data(x, mean=True):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1
    return (x - mean) / std


def get_data(run_type, rinv, nb_chan, N=None):
    if run_type == "CNN_1" or "CNN_2":
        hf = h5py.File(path.parent / "data" / f"jet_images_chan_{nb_chan}.h5", "r")
        X = hf[rinv]["features"][:N]
        y = hf[rinv]["targets"][:N]
        X = np.log(1.0 + X) / 4.0
        X = scale_data(X)
        hf.close()
    elif run_type == "HL":
        hl_file = path.parent / "data" / run_type / f"HL-{rinv}.h5"
        X = pd.read_hdf(hl_file, "features").to_numpy()[:N]
        y = pd.read_hdf(hl_file, "targets").values[:N]
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
    return X, y
