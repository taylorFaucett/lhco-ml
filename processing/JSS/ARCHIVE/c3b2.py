import numpy as np
import tqdm
import energyflow as ef


def calc(X):
    output = []
    hl_graph = ef.C3(
        measure="hadr",
        beta=2,
        reg=0.0,
        kappa=1,
        normed=False,
        coords=None,
        check_input=True,
    )
    output = []
    for x in tqdm(X):
        output.append(hl_graph.compute(x))
    output = np.array(output)

    return np.hstack(output)
