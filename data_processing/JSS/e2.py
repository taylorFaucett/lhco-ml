import numpy as np
import tqdm
import energyflow as ef


def calc(X):
    efpset = ef.EFPSet("d<=5", "p==1")
    kappa, beta = 1, 1
    nn, dd, kk = 2, 1, 0
    graphs = efpset.graphs()
    for ix, graph in enumerate(graphs):
        n, e, d, v, k, c, p, h = efpset.specs[ix]
        if n==nn and d==dd and k==kk:
            EFP_graph = ef.EFP(graph, measure="hadr", kappa=kappa, beta=beta, normed=False)
            output = EFP_graph.batch_compute(X)
            return np.hstack(output)
