import uproot
import numpy as np
from tqdm import tqdm, trange
import h5py
import pandas as pd
import pathlib

path = pathlib.Path.cwd()

labels = {
        "BlackBox1_qcd": 0,
        "BlackBox1_Z_XY_qq": 1,
        "BlackBox3_qcd":0,
        "BlackBox3_KKg2qq":1,
        "BlackBox3_KKg2gr":2,
        }

paddings = {
        "BlackBox1_qcd": 700,
        "BlackBox1_Z_XY_qq": 700,
        "BlackBox3_qcd":850,
        "BlackBox3_KKg2qq":850,
        "BlackBox3_KKg2gr":850,
        }    

def pad_array(events, padding_size):
    # Pad all events with zeros so that they all have
    # the same dimensions (to match event with largest number of constituents)
    event_array = np.zeros((len(events), padding_size, 3))
    for ix, event in tqdm(enumerate(events)):
        event_array[ix] = np.pad(
            event, [(0, padding_size - len(event)), (0, 0)], mode="constant"
        )
    return event_array

def process_data(input_file):
    output_file = path / "data" / "generated" / f"{input_file.stem}.h5"
    padding_size = paddings.get(input_file.stem)
    if not output_file.exists():
        print("Processing data from file:", input_file)
        # Set Jet PT cut, eta cut and number of constitutent padding
        jet_min = 1200
        eta_cut = 2.5

        # Open the root file
        file = uproot.open(input_file)["Delphes"]
        total_events = len(file["Event"].arrays("Event.Number", library="pd"))
        print("N =", total_events, "available")

        # Get the jet data and isolate events which pass the PT and |eta| cut from jet_min, eta_cut
        valid_events = []
        initial_cut = f"(Jet.PT > {jet_min}) & (Jet.Eta > -{eta_cut}) & (Jet.Eta < +{eta_cut})"
        jets = file["Jet"].arrays(["Jet.PT", "Jet.Eta"], initial_cut, library="pd")

        for jet_id, jet in tqdm(jets.groupby("entry"), desc=f"jet cuts (pT > {jet_min} GeV and |eta| < {eta_cut})"):
            for subjet_id, subjet in jet.groupby("subentry"):
                subjet_pt = subjet["Jet.PT"].values
                subjet_eta = subjet["Jet.Eta"].values
                if abs(subjet_eta) < eta_cut:
                    valid_events.append(jet_id)
                    break
        if len(valid_events) > 100:

            print("Loading EFlowTracks...")
            EFlowTrack = file["EFlowTrack"].arrays(["EFlowTrack.PT", "EFlowTrack.Eta", "EFlowTrack.Phi"], library="pd").loc[valid_events]
            EFlowTrack = EFlowTrack.rename(columns={"EFlowTrack.PT":"PT", "EFlowTrack.Eta":"Eta", "EFlowTrack.Phi":"Phi"})

            print("Loading EFlowNeutralHadron...")
            EFlowNeutralHadron = file["EFlowNeutralHadron"].arrays(["EFlowNeutralHadron.ET", "EFlowNeutralHadron.Eta", "EFlowNeutralHadron.Phi"], library="pd").loc[valid_events]
            EFlowNeutralHadron = EFlowNeutralHadron.rename(columns={"EFlowNeutralHadron.ET":"PT", "EFlowNeutralHadron.Eta":"Eta", "EFlowNeutralHadron.Phi":"Phi"})

            print("Loading EFlowPhoton...")
            EFlowPhoton = file["EFlowPhoton"].arrays(["EFlowPhoton.ET", "EFlowPhoton.Eta", "EFlowPhoton.Phi"], library="pd").loc[valid_events]
            EFlowPhoton = EFlowPhoton.rename(columns={"EFlowPhoton.ET":"PT", "EFlowPhoton.Eta":"Eta", "EFlowPhoton.Phi":"Phi"})

            print("Combining FlowParticles")
            FlowParticles = pd.concat([EFlowTrack, EFlowNeutralHadron, EFlowPhoton])
            X = []
            for FlowID, FlowParticle in FlowParticles.groupby("entry"):
                X.append(FlowParticle.to_numpy())

            print("Padding events array")
            X = pad_array(X, padding_size)
            X = np.reshape(X, ((len(X), 1, padding_size*3)))

            print("Converting np to dataframe")
            X = pd.DataFrame(data=X[:, 0])
            y = pd.DataFrame({"targets":np.full(len(X), labels.get(input_file.stem)).astype(np.int8)})

            print(f"Saving {len(X)} out of {total_events} initial events")
            X.to_hdf(output_file, "features", complevel=9)
            y.to_hdf(output_file, "targets", complevel=9)


if __name__ == "__main__":
    file_path = path / "data" / "raw"
    for file in file_path.glob("*.root"):
        process_data(file)
