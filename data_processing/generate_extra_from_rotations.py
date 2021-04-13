import os
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import h5py
import glob
from tqdm import tqdm
from tools import get_data

np.warnings.filterwarnings("ignore")

path = pathlib.Path.cwd()

def center_events(X):
    for x in tqdm(X):
        pt_max = np.argmax(x[:,0])
        x[:,1] -= x[:,1][pt_max]
        x[:,2] -= x[:,2][pt_max]
    return X

def randomly_rotate(X):
    rads = np.random.random(len(X)) * np.pi
    for (x, rad) in zip(X, rads):
        eta = x[:,1]
        phi = x[:,2]
        new_eta = np.cos(rad)*eta - np.sin(rad)*phi
        new_phi = np.sin(rad)*eta + np.cos(rad)*phi
        x[:,1] = new_eta
        x[:,2] = new_phi
    return X

def generate_extra_from_rotations(bbx, N=50000):
    h5_file = path.parent / "data" / "HL" / f"HL-{bbx}.h5"
    X, y = get_data(bbx, rotated=False)
    labels = list(set(y))
    Xs = []
    for label in labels:
        if label == 0:
            Xs.append(X[y==label][:N])
        else:
            Xs.append(X[y==label])
    data_shape = Xs[0].shape
    for ix, x in enumerate(Xs):
        center_events(x)
        if ix < 1:
            x_out = x
            y_out = np.zeros(len(x))
        else:
            x = np.resize(x, data_shape)
            x = randomly_rotate(x)
            x_out = np.concatenate((x_out, x))
            y_out = np.concatenate((y_out, np.ones(len(x))*ix)) 
    f = h5py.File(path.parent / "data" / "with_rotations" / f"{bbx}.h5", "w")
    f.create_dataset("features", data=x_out)
    f.create_dataset("targets", data=y_out)
    f.close()
    
if __name__ == "__main__":
    generate_extra_from_rotations(bbx)