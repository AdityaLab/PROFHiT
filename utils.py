import numpy as np

def lag_dataset(seqs, back):
    X, Y = [], []
    for i in range(back, seqs.shape[1]):
        X.append(seqs[:, :i-1])
        Y.append(seqs[:, i])
    return X, Y