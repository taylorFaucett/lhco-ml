import numpy as np
import pandas as pd
import pathlib
path = pathlib.Path.cwd()

def generate_extra_with_rotations():
    h5_file = path.parent / "data" / f"HL.h5"
    X = np.load(path.parent / "data" / "X.npy")
    y = np.load(path.parent / "data" / "y.npy")
    y = pd.DataFrame({"targets":y})
    print(X, y)

if __name__ == "__main__":
    generate_extra_with_rotations()
