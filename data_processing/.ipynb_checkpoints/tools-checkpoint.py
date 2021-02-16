import numpy as np

def pad_array(X, dim=4):
    largest = 0
    for x in X:
        xSize = len(x)
        if xSize > largest:
            largest = xSize
    output = np.zeros((len(X), largest, dim))        
    for ix, x in enumerate(X):
        replacement = np.pad(x, [(0,largest-len(x)), (0,0)], mode="constant")
        output[ix] = replacement
    return output

def pad_pairs(X1, X2, dim=4):
    largest = max(X1.shape[1], X2.shape[1])
    output1 = np.zeros((len(X1), largest, dim))
    output2 = np.zeros((len(X2), largest, dim))
    for ix, x in enumerate(X1):
        replacement = np.pad(x, [(0,largest-len(x)), (0,0)], mode="constant")
        output1[ix] = replacement
    for ix, x in enumerate(X2):
        replacement = np.pad(x, [(0,largest-len(x)), (0,0)], mode="constant")
        output2[ix] = replacement
    return output1, output2

def strip_zeros(X):
    output = 0
    for x in X:
        output.append(x[x[:,0]!=0])
    return output

def mass_inv(j1, j2):
    return np.sqrt(
        2.0 * j1.pt * j2.pt * (np.cosh(j1.eta - j2.eta) - np.cos(j1.phi - j2.phi))
    )

def calc_zg(j1, j2):
    min_pt = min(j1.pt, j2.pt)
    total_pt = j1.pt + j2.pt
    return min_pt / total_pt

def calc_rg(j1, j2):
    t1 = (j2.eta - j1.eta)**2
    t2 = (j2.phi - j1.phi)**2
    return np.sqrt(t1 + t2)

def struc2arr(x):
    # pyjet outputs a structured array. This converts
    # the 4 component structured array into a simple
    # 4xN numpy array
    return x.view((float, len(x.dtype.names)))