import h5py
import numpy as np
import pandas as pd
import energyflow as ef
import glob
import os
import sys
from tqdm import tqdm, trange
import pathlib

path = pathlib.Path.cwd()


def efp(data, graph, kappa, beta):
    EFP_graph = ef.EFP(graph, measure="hadr", kappa=kappa, beta=beta, normed=True)
    X = EFP_graph.batch_compute(data)
    return X


def generate_efp():
    try:
        # Create an h5 file to store EFPs (or re-open to append)
        # if the file already exists
        hf = h5py.File(path.parent / "data" / "EFP.h5", "a")
        
        # Choose kappa, beta to produce
        kappas = [-1, 0, 1]
        betas = [1, 2]

        # Load sample data
        X = np.load(path.parent / "data" / "X.npy")
        y = np.load(path.parent / "data" / "y.npy")

        # Structure the h5 file. All efps go in the efp group
        if "efp" not in hf.keys():
            efp_grp = hf.create_group("efp")
        else:
            efp_grp = hf["efp"]
        if "targets" not in hf.keys():
            hf.create_dataset("targets", data=y)

        # Remove any zero padding added when storing
        # data as an array
        data = []
        for x in tqdm(X):
            data.append(x[x[:, 0] != 0])

        # Select graphs. "p==1" means prime (connected) graphs only
        # "d<=5" means all EFPs with <= 5 edges
        efpset = ef.EFPSet("d<=5", "p==1")
        
        # Load graphs from efpset
        graphs = efpset.graphs()
        print(f" Computing N={len(kappas)*len(betas)*len(graphs):,} graphs")
        t = tqdm(graphs)
        
        # Loop through all graphs, generate the efp and save to the h5 file
        for efp_ix, graph in enumerate(t):
            for kappa in kappas:
                for beta in betas:
                    # Get parameters for the graph
                    # n = num. of nodes
                    # d = num. o fedges
                    # k = unique id number for a given (n,d) pair
                    n, e, d, v, k, c, p, h = efpset.specs[efp_ix]
                    t.set_description(f"{n}_{d}_{k}_k_{kappa}_b_{beta}")
                    t.refresh()
                    data_label = f"{n}_{d}_{k}_k_{kappa}_b_{beta}"
                    if data_label not in efp_grp.keys():
                        efp_data = efp(
                            data=data, graph=graph, kappa=kappa, beta=beta
                        )
                        efp_grp.create_dataset(data_label, data=efp_data)
    except KeyboardInterrupt:
        print("Ending early and closing h5 file")
        hf.close()
        sys.exit()
    hf.close()


if __name__ == "__main__":
    generate_efp()
