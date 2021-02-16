from energyflow.datasets import qg_jets
import numpy as np
import pathlib

path = pathlib.Path.cwd()

def download_samples(N):
    X_file = path.parent / "data" / "X.npy"
    y_file = path.parent / "data" / "y.npy"
    if not X_file.exists():
        X, y = qg_jets.load(N)
        np.save(path.parent / "data" / "X.npy", X)
        np.save(path.parent / "data" / "y.npy", y)

if __name__ == "__main__":
    download_samples(N=10000)
