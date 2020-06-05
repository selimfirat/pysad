import os
import numpy as np

from streaming.array_iterator import ArrayIterator


class Data:

    def __init__(self, data_base_path="../data"):
        self.data_base_path = data_base_path

    def get_data_files(self):

        return [
            'arrhythmia.mat',
            'cardio.mat',
            'glass.mat',
            'ionosphere.mat',
            'letter.mat',
            'lympho.mat',
            'mnist.mat',
            'musk.mat',
            'optdigits.mat',
            'pendigits.mat',
            'pima.mat',
            'satellite.mat',
            'satimage-2.mat',
            'shuttle.mat',
            'vertebral.mat',
            'vowels.mat',
            'wbc.mat',
            "gisette_sampled.txt",
            "isolet_sampled.txt",
            "madelon_sampled.txt",
            "pima-indians_sampled.txt",
            "magic-telescope_sampled.txt",
        ]

    def load_via_txt(self, path):

        X = np.loadtxt(path, delimiter=",")

        return X

    def get_data(self, data_file):

        data_path = os.path.join(self.data_base_path, data_file)

        if ".mat" in data_file:
            from scipy.io import loadmat

            f = loadmat(data_path)

            X = f['X']
            y = f['y'].ravel()
        else:
            X = self.load_via_txt(data_path)

            y = X[:, -1].ravel()
            X = X[:, :-1]

        return X, y

    def get_iterator(self, data_file, shuffle=True, seed=None):

        if seed is not None:
            np.random.seed(seed)

        iterator = ArrayIterator(shuffle=shuffle)

        X, y = self.get_data(data_file)

        return iterator.iter(X, y)
