import os
import numpy as np
from pysad.utils.array_streamer import ArrayStreamer


class Data:
    """A helper class to load various data.

    Args:
        data_base_path (str): Base path that contains the data files.
    """

    def __init__(self, data_base_path="data"):
        self.data_base_path = data_base_path

    def _get_data_files(self):
        """ Helper method to return the names of the data files.

        Returns:
            file_names (list[str]): List of data file names.
        """
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

    def _load_via_txt(self, path):
        """Loads the data file from .txt file.

        Args:
            path (str): The path of data.

        Returns:
            X (np.float64 array of shape (num_instances, num_features)): Feature vectors.
        """
        X = np.loadtxt(path, delimiter=",")

        return X

    def get_data(self, data_file):
        """Loads the data given the path.

        Args:
            data_file: Path of the data.

        Returns:
            X (np.array of shape (num_instances, num_features)): Feature vectors.
            y (np.array of shape (num_instances,)): Labels.
        """
        data_path = os.path.join(self.data_base_path, data_file)

        if ".mat" in data_file:
            from scipy.io import loadmat

            f = loadmat(data_path)

            X = f['X']
            y = f['y'].ravel()
        else:
            X = self._load_via_txt(data_path)

            y = X[:, -1].ravel()
            X = X[:, :-1]

        return X, y

    def get_iterator(self, data_file, shuffle=True, seed=None):
        """The iterator function

        Args:
            data_file (str): Path of data.
            shuffle (bool): Whether to shuffle (Default=True).
            seed (int): Random seed (Default=None).

        Returns:
            iterator (The iterator): pysad.utils.array_streamer.ArrayStreamer.iter method applied with (X, y), where X is the variable containing feature vectors and y is the variable containing labels.

        """
        if seed is not None:
            np.random.seed(seed)

        iterator = ArrayStreamer(shuffle=shuffle)

        X, y = self.get_data(data_file)

        return iterator.iter(X, y)
