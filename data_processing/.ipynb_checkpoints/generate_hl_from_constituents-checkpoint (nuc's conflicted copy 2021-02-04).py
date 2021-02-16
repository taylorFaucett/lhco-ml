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


def generate_hl_observables(N=-1):
    JSS_list = load_modules()
    rinvs = ["1p0", "0p3", "0p0"]
    for rinv in rinvs:
        h5_file = path.parent / "data" / "CP" / f"HL-{rinv}.h5"
        if h5_file.exists():
            HL_df = pd.read_hdf(h5_file, "features")
            existing_jss = list(HL_df.columns)
        else:
            HL_df = pd.DataFrame()
            jet_mass = pd.DataFrame({"mass": np.load(path.parent / "data" / "trimmed_jets" / f"mass_from_trim-{rinv}.npy")[:N]})
            jet_pt = pd.DataFrame({"pT": np.load(path.parent / "data" / "trimmed_jets" / f"pT_from_trim-{rinv}.npy")[:N]})
            HL_df = pd.concat([jet_mass, jet_pt], axis=1)
            existing_jss = []

        X = np.load(path.parent / "data" / "trimmed_jets" / f"trim4HL-{rinv}.npy")[:N]
        y = pd.DataFrame({"targets":np.load(path.parent / "data" / "trimmed_jets" / f"y4HL-{rinv}.npy")[:N]})

        for x in tqdm(X):
            mask = x[:,0] > 0
            yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
            x[mask,1:3] -= yphi_avg
            x[mask,0] /= x[:,0].sum()

        for JSS_calc in JSS_list:
            if JSS_calc not in existing_jss:
                print(f"Calculating {JSS_calc} on data set: {rinv}")
                try:
                    JSS_out = np.zeros(X.shape[0])
                    exec("JSS_out[:] = %s.calc(X)[:]" % JSS_calc)
                    JSS_out = pd.DataFrame({JSS_calc: JSS_out})
                    HL_df = pd.concat([HL_df, JSS_out], axis=1)
                except Exception as e:
                    print(
                        f"JSS calculation for {JSS_calc} on data set {rinv} failed with error:"
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
