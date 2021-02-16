import pandas as pd
import numpy as np
import h5py
import sys
from tools import struc2arr, pad_array, mass_inv, calc_zg, calc_rg
from tqdm import tqdm
import os
from pyjet import cluster
from pyjet.testdata import get_event
import pathlib

path = pathlib.Path.cwd()


def jet_trimmer(tower, R0, R1, fcut, pt_min, pt_max, eta_cut):
    # R0 = Clustering radius for the main jets
    # R1 = Clustering radius for the subjets in the primary jet
    trim_pt, trim_eta, trim_phi, trim_mass = [], [], [], []

    # Convert the pandas dataframe to a structured array
    # (pT, eta, phi, mass)
    tower = tower.to_records(index=False)

    # Cluster the event
    sequence = cluster(tower, R=R0, p=-1)
    jets = sequence.inclusive_jets(ptmin=0)

    # check pt and eta cuts
    if pt_min < jets[0].pt < pt_max and -eta_cut < jets[0].eta < +eta_cut:
        # Grab the subjets by clustering with R1
        subjets = cluster(jets[0].constituents_array(), R=R1, p=1)
        subjet_array = subjets.inclusive_jets()

        # For each subjet, check (and trim) based on fcut
        trimmed_jet = []
        for subjet in subjet_array:
            if subjet.pt > jets[0].pt * fcut:
                # Get the subjets pt, eta, phi constituents
                subjet_data = subjet.constituents_array()
                subjet_data = struc2arr(subjet_data)
                trimmed_jet.append(subjet_data)
        m_inv = mass_inv(jets[0], jets[1])
        zg = calc_zg(jets[0], jets[1])
        rg = calc_rg(jets[0], jets[1])
        return np.concatenate(trimmed_jet), m_inv, zg, rg
    else:
        return None, None, None, None


def trim_towers(rinv):
    # Root files path
    root_exports_path = pathlib.Path.home().parent.parent / "media" / "Dropbox2" / "Projects" / "semi-visible-jets-ml" / "data" / "root_exports" / rinv
    h5_file = path.parent / "trimmed_jets" / f"trimmed_jet-{rinv}.h5"
    
    # Loop through root files and collect trimmed jets
    t0, m0, z0, r0 = [], [], [], []
    for idx, tower_file in enumerate(tqdm(list(root_exports_path.glob("*.h5")))):
        tower_events = pd.read_hdf(tower_file, "Tower")
        tower_events = tower_events.astype(np.float64)
        entries = len(tower_events.groupby("entry"))
        for entry, tower in tower_events.groupby("entry"):
            trimmed_jet, m_inv, zg, rg = jet_trimmer(
                tower=tower,
                R0=1.0,
                R1=0.2,
                fcut=0.05,
                pt_min=300,
                pt_max=400,
                eta_cut=2.0,
            )
            # Collect results
            if trimmed_jet is not None:
                t0.append(trimmed_jet)
                m0.append(m_inv)
                z0.append(zg)
                r0.append(rg)
    print(len(m0), len(z0), len(r0), len(t0))
    hf = h5py.File(h5_file, "w")
    hf.create_dataset("trimmed_jets", data=pad_array(t0))
    hf.create_dataset("mass", data=np.array(m0))
    hf.create_dataset("zg", data=np.array(z0))
    hf.create_dataset("rg", data=np.array(r0))
    hf.close()


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0", "bkg_qcd"]
    for rinv in rinvs:
        trim_towers(rinv)
