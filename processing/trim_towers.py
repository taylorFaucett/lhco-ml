import pandas as pd
import numpy as np
import h5py
import sys
from tools import struc2arr, pad_array
from tqdm import tqdm
import os
from pyjet import cluster
from pyjet.testdata import get_event
import pathlib

path = pathlib.Path.cwd()

# R0 = Clustering radius for the leading jet
# R1 = Clustering radius for the subjets
# fcut = percentage of the leading jet that all subjets must have to survive cuts
# pt_min = Minimum pT cut
# pt_max = Maximum pT cut
# eta_cut = cut on events by eta where -eta_cut < eta < +eta_cut


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
        return np.concatenate(trimmed_jet)
    else:
        return None


def trim_towers():
    # Grab exported tower data 
    # This format was created using UpRoot and saving in pandas format
    # This gives the "event" and "subevent" dataframe style
    tower_events = pd.read_hdf(path.parent / "data" / "root_sig.h5", "Tower")
    
    # pyjet requires float64 as the datatype
    tower_events = tower_events.astype(np.float64)

    # Loop through entries and run pyjet clustering
    X = []
    for entry, tower in tower_events.groupby("entry"):
        # Settings for jet trimming
        trimmed_jet = jet_trimmer(tower=tower, R0=1.0, R1=0.2, fcut=0.05, pt_min=300, pt_max=400, eta_cut=2.0)

        # Collect results from trimming
        if trimmed_jet is not None:
            X.append(trimmed_jet)
    X = pad_array(X) 
    np.save(path.parent / "data" / "X_from_towers.npy", X)


if __name__ == "__main__":
    trim_towers()
