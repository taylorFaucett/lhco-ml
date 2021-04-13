import os
import pandas as pd
import numpy as np
import pathlib
import h5py
import glob
from tqdm import tqdm

np.warnings.filterwarnings("ignore")

path = pathlib.Path.cwd()

for entry in os.scandir("JSS"):
    if (
        entry.is_file()
        and "init" not in str(entry)
        and "jss_template" not in str(entry)
    ):
        string = f"from JSS import {entry.name}"[:-3]
        exec(string)


def load_modules():
    # Anything placed in JSS will be imported and used as a HL observable.
    # The input will be the prep_data file
    # The output should be a simple 1D numpy array
    jss_list = glob.glob("JSS/*.py")
    jss_list = [x.split("/")[-1].split(".py")[0] for x in jss_list]
    jss_list.remove("__init__")
    jss_list.remove("jss_template")
    return jss_list


def generate_hl_observables():
    JSS_list = load_modules()
    h5_file = path.parent / "data" / f"HL.h5"
    X = np.load(path.parent / "data" / "X.npy")
    y = np.load(path.parent / "data" / "y.npy")
    y = pd.DataFrame({"targets":y})
    HL_df = pd.DataFrame()
    
    # Calculate Jet PT
    jet_pt = []
    for x in tqdm(X):
        pt_val = x[:,0]
        pt_sum = np.sum(pt_val)
        jet_pt.append(pt_sum)
    jet_pt = pd.DataFrame({"pT": np.array(jet_pt)})
    HL_df = pd.concat([HL_df, jet_pt], axis=1)

    XX = []
    for x in tqdm(X):
        mask = x[:,0] > 0
        yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
        x[mask,1:3] -= yphi_avg
        x[mask,0] /= x[:,0].sum()
        XX.append(x[x[:,0]!=0])

    for JSS_calc in JSS_list:
        print(f"Calculating {JSS_calc}:")
        try:
            JSS_out = np.zeros(X.shape[0])
            exec("JSS_out[:] = %s.calc(XX)[:]" % JSS_calc)
            JSS_out = pd.DataFrame({JSS_calc: JSS_out})
            HL_df = pd.concat([HL_df, JSS_out], axis=1)
            print(HL_df)
        except Exception as e:
            print(
                f"JSS calculation for {JSS_calc} failed with error:"
            )
            print(e)

        # Re-organize columns alphabettically.
        # This guarantees the ordering is always the same
        HL_df = HL_df.reindex(sorted(HL_df.columns), axis=1)

        # Remove any NAN results
        HL_df = HL_df.fillna(0)
        HL_df.to_hdf(h5_file, key="features", mode="w")
        y.to_hdf(h5_file, key="targets", mode="a")


if __name__ == "__main__":
    generate_hl_observables()
