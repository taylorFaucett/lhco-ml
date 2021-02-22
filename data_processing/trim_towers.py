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

def trim(event, R0=1, R1=0.2, fcut=0.05):            
    # Convert the pandas dataframe to a structured array
    # (pT, eta, phi, mass)
    event = pd.DataFrame(event)
    event = event.to_records(index=False)

    # Cluster the event
    sequence = cluster(event, R=R0, p=-1)
    jets = sequence.inclusive_jets(ptmin=20)

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
    
def process_file(data_file):
    N = 0
    step_size = 5000
    data_size = len(np.loadtxt(path.parent / "data" / "raw" / "bb1.masterkey"))
    trimmed_events = []
    pbar = tqdm(total=int(data_size/step_size))
    while N < data_size:
        events = pd.read_hdf(data_file, start=N, stop=N+step_size).to_numpy()
        events = np.reshape(events, (step_size, 700, 3))
        events = np.pad(events, [(0,0), (0,0), (0,1)], mode='constant')
        for event in events:
            trimmed_events.append(trim(event))
        N += step_size
        pbar.update(1)
    pbar.close()
    return pad_array(trimmed_events)
    


def trim_towers(bbx):
    # Grab exported tower data 
    # This format was created using UpRoot and saving in pandas format
    # This gives the "event" and "subevent" dataframe style
    data_file = path.parent / "data" / "raw" / f"{bbx}.h5"
    trim_file = path.parent / "data" / "trimmed" / f"{data_file.stem}.h5"
    if not trim_file.exists():
        hf = h5py.File(trim_file, "w")
        X = process_file(data_file)
        hf.create_dataset("features", data=X)
        y_file = path.parent / "data" / "raw" / f"{data_file.stem}.masterkey"
        if y_file.exists():
            y = np.loadtxt(y_file).astype("int8")[:len(X)]
            hf.create_dataset("targets", data=y)
        hf.close()

if __name__ == "__main__":
    trim_towers(bbx)
