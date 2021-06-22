import numpy as np
import pathlib
import h5py
import pandas as pd
import pyjet
from energyflow.utils import pixelate
from tqdm import tqdm
from tools import get_data

path = pathlib.Path.cwd()

def struc2arr(x):
    # pyjet outputs a structured array. This converts
    # the 4 component structured array into a simple
    # 4xN numpy array
    return x.view((float, len(x.dtype.names)))

def generate_jet_images(bbx, N):
    # Number of pixels in the image
    npix=32
    
    # Image width (0.8 corresponds to -0.4 to +0.4)
    img_width=0.8

    # h5 file to save results
    h5_file = path.parent / "data" / "jet_images" / f"jet_images_BlackBox{bbx}.h5"
    hf = h5py.File(h5_file, "w")
    for nb_chan in (1,2):
        images = np.asarray([pixelate(x, npix=npix, img_width=img_width, nb_chan=nb_chan, norm=False) for x in X])
        hf.create_dataset(f"{nb_chan}_chan", data=images)
    hf.create_dataset("targets", data=y)
    hf.close()

def preprocess(X):
    X = np.pad(X, [(0, 0), (0, 0), (0, 1)], "constant").astype(np.float64)
    jets_ = []
    for ix, x in tqdm(enumerate(X), total=N):
        x = x[x[:, 0] != 0]
        df = pd.DataFrame(x).to_records(index=False)
        sequence = pyjet.cluster(df, R=1, p=-1)
        jets = sequence.inclusive_jets(ptmin=0)
        trimmed_jets = []
        for jet in jets[0]:
            subjets = pyjet.cluster(jet.constituents_array(), R=0.2, p=1)

            # For each subjet, check (and trim) based on fcut  
            subjet_array = subjets.inclusive_jets(ptmin=jet.pt * 0.05)
            for subjet in subjet_array:
                # Get the subjets pt, eta, phi constituents
                subjet_data = subjet.constituents_array()
                subjet_data = struc2arr(subjet_data)
                trimmed_jets.append(subjet_data)
        jets_.append(np.vstack(trimmed_jets))
    return jets_
    
if __name__ == "__main__":
    N = 100000
    for bbx in ([1]):
        X, y = get_data(bbx, N)
        X = preprocess(X)
        generate_jet_images(bbx, N)
