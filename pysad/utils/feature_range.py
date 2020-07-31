import numpy as np


def get_minmax_array(X):
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    return min, max


def get_minmax_scalar(x):
    min = np.min(x)
    max = np.max(x)

    return min, max
