import csv
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Iterator
from typing import Any

import numpy as np

from pysad.utils.array_streamer import ArrayStreamer

# The ODDS files of PyOD's mirror at a fixed commit, so that the data cannot change without a
# change in PySAD.
_MIRROR = (
    "https://github.com/yzhao062/pyod/raw/bd110115c680d29561dd7e05d8f4b7af4f3512f7/notebooks/data/"
)
# The SHA-256 checksum of each file, which a download must match.
_CHECKSUMS = {
    "arrhythmia.mat": "e4d47e499591ae31e5d7f0cd983367092353eac240f48e14c9b36457edb3460d",
    "cardio.mat": "5e568b6a73711012c481fa6cc984ab9c68e9a4fa44d6aeb3b43efa381c3f695b",
    "glass.mat": "2c506e6e1e2ca1e82a905d10a90773151ff812c4bd969c71a12db5b2bb6e7921",
    "ionosphere.mat": "8ac5dfa6f47db4acdfe10615489b7f7dd015d79d5b381ce641e72baca3cfefb0",
    "letter.mat": "f4a229b0e2a6b0c9157ce27fa760bcbac0f948cd690b2f5d51c120e5450f2195",
    "lympho.mat": "031b907ecb50347720b08b38f8139d05635adeca9335a064d89d5569c27a5834",
    "mnist.mat": "9479e221f236684c09f3e11c7ad124725d34ecd301b9450425e8b113368413a6",
    "musk.mat": "50bbf414caceb66cfe6f6b1eb7c1ff7c6eb1e6d87173f7bb597ba55cd3dc5493",
    "optdigits.mat": "30c664cd6f3740a173277a27cd1028d471b06f0835a9902c9712749cec58c1c4",
    "pendigits.mat": "7bc32ac83b3d583443acea8b1342a54aaa59eaedb9996af405322f473f3c22bc",
    "pima.mat": "b6a5be8d548b9293d6958beceb1c77055591714d25d40d957eba87808456468f",
    "satellite.mat": "6feac3112b9c14e1c3e60afc437f2f3d29dc1000119c9f950c628078778d6aa0",
    "satimage-2.mat": "95c37a07cea2cb201f6900d14f521510e10b6b2f0074c06ac46d3157440ec67b",
    "shuttle.mat": "b6e396c92354d5dcf5036901e36dc58e2913f2d1a5e138c35a2bedf9c69db899",
    "vertebral.mat": "84b50b055f118f54b5c976650ccd284ab7c1adee822f8b3eb517c7e2c890d15e",
    "vowels.mat": "51d5397bad1ceeda22b7cd44fd58125c5d603c9c2edb53aa59933330e259ac7b",
    "wbc.mat": "afe6b26953cc196b9a480318c903e0edef22529c2f61bd02082438c6e70630a7",
}


# The Numenta Anomaly Benchmark (NAB) at the first commit under the MIT license. Its earlier
# commits are under the AGPL-3.0. Each univariate series is "nab/<category>/<name>.csv".
_NAB = "https://raw.githubusercontent.com/numenta/NAB/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/"
_NAB_WINDOWS = "1e1fbc4601321aad8d0f8b3784c8134299379f68f6c1f7777565f8ffd57ab6b1"
_NAB_CHECKSUMS = {
    "artificialNoAnomaly/art_daily_no_noise.csv": "ccb7743ac2de8ca110db4af796d11ba31a23ad448a28e353a935c15f637756bc",
    "artificialNoAnomaly/art_daily_perfect_square_wave.csv": "662ff0a295b81eb0ff8e81116a54a497cd6f6b07c19c7a5ea72cce494913ce62",
    "artificialNoAnomaly/art_daily_small_noise.csv": "c4b304229f69f108eaa024fb2406fcd4510c5c5d95cd0053e2da6989682e4e6d",
    "artificialNoAnomaly/art_flatline.csv": "1cc10761ddd6d55fb5f0d9cfc73610416abbb120ee14bb65eff288509a463397",
    "artificialNoAnomaly/art_noisy.csv": "ad455adeaf06852222500955f2aefe6bd389cc762ef68e7102ee543209c8607f",
    "artificialWithAnomaly/art_daily_flatmiddle.csv": "ea5df125947e9ee2b977ac3d341f3c2a029588862af97cae4a654fc56e838702",
    "artificialWithAnomaly/art_daily_jumpsdown.csv": "4af31802e87dff4b4ba3ceb21714ba9d905f7888e1f53d47cc6dd4f856f9debc",
    "artificialWithAnomaly/art_daily_jumpsup.csv": "f79f92b93ee4f9c10e2dc552505e8880b0c618639ed5e0f9816e044b75c26456",
    "artificialWithAnomaly/art_daily_nojump.csv": "65c85d1834376ed5e622f01721d478609577ea0df0a067c5a5ebe5edea03b077",
    "artificialWithAnomaly/art_increase_spike_density.csv": "0959d8871d5c1a3fc3120fb7421e4ef113fffb434af2889085b52aebdb69360c",
    "artificialWithAnomaly/art_load_balancer_spikes.csv": "99f7130e54c7c3e818df8e3bdc0974cd241914bb14a10a9c3fa7c1374125d224",
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv": "ab446fbd8b9f37507eb2fdb06315826d8daeef02e241133ce06e0ee571ba53d9",
    "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv": "8942e498de7b40f1b4a6d802755592c09b8fb73ce8f658763a09cb9ea1ea94ac",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv": "01613e6f632d067f11a5dfd40a188b0789752b388d9bc77a398bd06333878a76",
    "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv": "90ceabd570b449241ee24ff8a116a793979b7671ce4490707311bfea0e0aae1f",
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv": "d768419037c9db269343822957314f57ee21a7d9a4d41df2add0d1ba45ba84de",
    "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv": "749a15c2e1a4543c21fee9cbf3338cd8a7ed5f5f8a1308b9b099b06c2c66e66b",
    "realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv": "d936cea74682ed43ac96b778352d7de294c0b0c168f4a7e6817162346cdb28c1",
    "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv": "f3433f8171f4dcea86c0c7af9996d0f166f812fa0f4567f1d5cd85d2d2cd69b4",
    "realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv": "ce3d6c6a652ed7c31a6dd3727a36e128fcac6fb2381c63267ad0fbc02eccd9df",
    "realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv": "a0e734a66098bb81833835f0839dadafbc8ae829c51494670ee8849e46eb5b63",
    "realAWSCloudwatch/ec2_network_in_257a54.csv": "39104b08f2e0a673b5137eb7681897fcadf0955fedf565740a6a94edc63a81a4",
    "realAWSCloudwatch/ec2_network_in_5abac7.csv": "c27ef93f582af63bf42e80d804764cbe64ae987620f3cfab5477d6ebb3c6157e",
    "realAWSCloudwatch/elb_request_count_8c0756.csv": "74c26574a01ca9fb89dddb5021e2e13c3a93eb25dc640438a9acb1ceb00f1021",
    "realAWSCloudwatch/grok_asg_anomaly.csv": "86a0abe9d58e376858d2fd4f4a438f1997e24c64a09e5af03accd4e1b372ac47",
    "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv": "f115fbe6542159eed5612445bfa620708a28ab6b41349d1432bb15c3a3b8414d",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv": "d5df979ef85928769a016cd0f935919d8c2a113926e21e1eb3a5001d688760e2",
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv": "6b712b922ab3b3404c5a629d64a570c390c99471ee8b9b0ff7665f553fda92c1",
    "realAdExchange/exchange-2_cpc_results.csv": "d869ab165254f3ee77ee21fd9fab2be1d07c1f80b8133cf01bbaf2d768c2ac64",
    "realAdExchange/exchange-2_cpm_results.csv": "1f01b9438566d2fe1ac446faa202a395ac86db9240a1955bc1f118c8eada2a19",
    "realAdExchange/exchange-3_cpc_results.csv": "e70b24dba107a2589657cb3feb9ca5605b4f3f7de49ff445f32f5991bc0bdbe5",
    "realAdExchange/exchange-3_cpm_results.csv": "74d354a94fb1730e14b469c7c1c6e99c2ad69af53afdb85f65220bd3cf01ce4a",
    "realAdExchange/exchange-4_cpc_results.csv": "d4c13abd6f28594dc446f22106562ebe917fdbe76170f60a500c2a402ecc09e2",
    "realAdExchange/exchange-4_cpm_results.csv": "4e5c955b632145b77b93fba4a5e3bf615d47e00416e17091d64e05ae109c760e",
    "realKnownCause/ambient_temperature_system_failure.csv": "230b68ccca20f59d562afd5d24ad52939c9b784386bed0054018358bf9120581",
    "realKnownCause/cpu_utilization_asg_misconfiguration.csv": "58ba65dc0737cfbac11b51514476d50c438d44011232144bb8d93f392df58f9f",
    "realKnownCause/ec2_request_latency_system_failure.csv": "98378580aa80157e057c61d59d81daddccc6c65a2c0c800e3f01f603b8215c3f",
    "realKnownCause/machine_temperature_system_failure.csv": "92bf5b87fc7f9bba8ca0b7ec63ccaac8cb4a1371a258e8c29a10ae9c018d82a4",
    "realKnownCause/nyc_taxi.csv": "d8fa6f7f0734bf5c8be12c52a94e20a82664c397d9dec4449156bd453d32856d",
    "realKnownCause/rogue_agent_key_hold.csv": "47b110baa8e3636574c1033dda1cce780873647fab58223d159d4fee6f180e86",
    "realKnownCause/rogue_agent_key_updown.csv": "9797cfef477c13306f7a33874ab4d6cf700f8f2c7e5825a2a5bc02004370909a",
    "realTraffic/TravelTime_387.csv": "38e70e9ead8227fd023bff6e93247d5bc01ee008f781b921a156962419b399bb",
    "realTraffic/TravelTime_451.csv": "7c174adfc457b201cd111d60688f7ec37eb7201c8080790dc41eeff477bd7345",
    "realTraffic/occupancy_6005.csv": "cd357d7820d675074270fd976d4af1fc1e7854ecb764783028cbcb18d980c91d",
    "realTraffic/occupancy_t4013.csv": "5663a8122a300360eb51fbbd0f21706da05af1af55262926d6a226bb6d071704",
    "realTraffic/speed_6005.csv": "7976e7596cd1e579696c737576e96a26fe041484472ed7d0a463ce2b094d7aa7",
    "realTraffic/speed_7578.csv": "15ba994384730cb5a2e6818f17ded1430b40d92eb7a08e5b1a48a445c8c350f6",
    "realTraffic/speed_t4013.csv": "fa5532d6f7db36cadc73e657fd4dfef05cb1ec44d4010243b314d3f1bbd6a7b5",
    "realTweets/Twitter_volume_AAPL.csv": "826f5cf404c2890784a7824f7102fd00cb134a4948e12e44ec320d095cbbc217",
    "realTweets/Twitter_volume_AMZN.csv": "234f5e63f19a62720dfb7ccfd6ecc8349b94d93b33ef8aafbdb17edda0b654f3",
    "realTweets/Twitter_volume_CRM.csv": "e7133bf26479623d58fc7367107aea1cd7df6020420dece48b0eee4d26cdb28a",
    "realTweets/Twitter_volume_CVS.csv": "f0bdc89cba3dd2265b47069187c750eeae18ef48f76f86067ee26ea1af9c3114",
    "realTweets/Twitter_volume_FB.csv": "a7096bd646759486bda6796b5600cd33a4590f0842bf274ea7acba001b3ba591",
    "realTweets/Twitter_volume_GOOG.csv": "3a39cc23d1ff6a0f234b55298d0c46c21d2e94ddb81a0dad9e1e46e3aaa9ec90",
    "realTweets/Twitter_volume_IBM.csv": "4309bd56d28ddcbdc168e207da9717395fa2e9f74dba41fa768e0f7da8a116fa",
    "realTweets/Twitter_volume_KO.csv": "f3aa8157043c6569911031535bd1f7a73b428b5a2d95238d79b9d5b88ceebbd2",
    "realTweets/Twitter_volume_PFE.csv": "18d2804d6a5dae63fe711baaba9242dce38a32bfade3dcdee1d8a7c1a866e41b",
    "realTweets/Twitter_volume_UPS.csv": "be50f06bbb9d020f37653c8c5cf19af63f77aa0019d8cf10849d91f4af455af4",
}


def _source(name: str) -> tuple[str, str]:
    """The URL and the SHA-256 checksum of a remote file of a dataset."""
    if name == "nab/combined_windows.json":
        return _NAB + "labels/combined_windows.json", _NAB_WINDOWS
    if name.startswith("nab/"):
        return _NAB + "data/" + name[len("nab/") :], _NAB_CHECKSUMS[name[len("nab/") :]]

    return _MIRROR + name, _CHECKSUMS[name]


def _get_default_data_dir() -> str:
    """Returns the default directory for caching datasets.

    Defaults to the directory in the ``PYSAD_DATA_DIR`` environment variable if
    set, or ``~/.cache/pysad``.
    """
    return os.environ.get(
        "PYSAD_DATA_DIR", os.path.expanduser(os.path.join("~", ".cache", "pysad"))
    )


class Data:
    """A helper class to load and stream anomaly detection datasets.

    Datasets are loaded from the local filesystem or downloaded on demand from
    the Outlier Detection DataSets (ODDS) repository mirror
    (:cite:`rayana2016odds`, hosted in the PyOD repository at a fixed commit). Each
    download must match a SHA-256 checksum. :meth:`datasets` lists the datasets, and
    :meth:`get_stream` gives the stream of a run of the PySAD benchmark.

    Downloaded datasets are cached in a local directory. The cache directory
    defaults to ``~/.cache/pysad``, which can be overridden via the
    ``PYSAD_DATA_DIR`` environment variable or by passing an explicit
    ``data_base_path``.

    Args:
        data_base_path (str or None): Base directory that contains the data files and caches the downloaded datasets. If None, the class uses the directory in ``PYSAD_DATA_DIR`` if set, or ``~/.cache/pysad`` (Default=None).
    """

    def __init__(self, data_base_path: str | None = None) -> None:
        self.data_base_path = _get_default_data_dir() if data_base_path is None else data_base_path
        self._is_default_path = data_base_path is None

    @staticmethod
    def datasets(source: str = "odds") -> list[str]:
        """Returns the names of the datasets that this class can download.

        Args:
            source (str): ``"odds"`` for the 17 multivariate datasets of ODDS, for example ``"cardio.mat"``, or ``"nab"`` for the 58 univariate series of the Numenta Anomaly Benchmark, for example ``"nab/realKnownCause/nyc_taxi.csv"`` (Default="odds").

        Returns:
            list[str]: The names of the datasets, which :meth:`get_data` accepts.

        Raises:
            ValueError: If ``source`` is not ``"odds"`` or ``"nab"``.
        """
        if source == "odds":
            return list(_CHECKSUMS)
        if source == "nab":
            return ["nab/" + name for name in _NAB_CHECKSUMS]
        raise ValueError(f"source must be 'odds' or 'nab', got {source!r}.")

    def _download(self, file_name: str, target_path: str) -> None:
        """Downloads a dataset file from the remote mirror into target_path atomically.

        Args:
            file_name (str): Name of the remote file (e.g. 'cardio.mat').
            target_path (str): Destination file path.
        """
        url, expected = _source(file_name)
        target_dir = os.path.dirname(target_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)

        temp_file = tempfile.NamedTemporaryFile(
            dir=target_dir or None, prefix=f"tmp_{os.path.basename(file_name)}_", delete=False
        )
        temp_path = temp_file.name
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pysad"})
            with urllib.request.urlopen(req) as response, temp_file:
                shutil.copyfileobj(response, temp_file)
            with open(temp_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            if checksum != expected:
                raise ValueError(
                    f"The download of '{file_name}' has the SHA-256 checksum {checksum}, "
                    f"but PySAD expects {expected}."
                )
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    def _load_via_txt(self, path: str) -> np.ndarray:
        """Loads the data file from .txt file.

        Args:
            path (str): The path of data.

        Returns:
            np.float64 array of shape (num_instances, num_features): Feature vectors.
        """
        X = np.loadtxt(path, delimiter=",")

        return X

    def get_data(self, data_file: str, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Loads the data given the filename or path, downloading it if not present.

        Args:
            data_file (str): Path or filename of the data.
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            tuple[np.ndarray, np.ndarray]: The feature vectors of shape (num_instances, num_features) and the labels of shape (num_instances,).
        """
        if os.path.exists(data_file):
            data_path = data_file
        else:
            data_path = os.path.join(self.data_base_path, data_file)

        # An ODDS dataset is known by its file name, and a NAB series by its full name.
        file_name = data_file if data_file.startswith("nab/") else os.path.basename(data_file)

        if not os.path.exists(data_path):
            if self._is_default_path:
                local_fallback = os.path.join("data", data_file)
                if os.path.exists(local_fallback):
                    data_path = local_fallback
                else:
                    examples_fallback = os.path.join("examples", "data", data_file)
                    if os.path.exists(examples_fallback):
                        data_path = examples_fallback

        if not os.path.exists(data_path):
            if not download:
                raise FileNotFoundError(
                    f"Dataset file '{data_file}' was not found at '{data_path}' and download=False."
                )

            if file_name not in self.datasets() + self.datasets("nab"):
                raise FileNotFoundError(
                    f"Dataset file '{data_file}' was not found at '{data_path}' and is not a known remote dataset. "
                    f"Available datasets: {self.datasets()}, and the NAB series of datasets('nab')."
                )

            self._download(file_name, data_path)

        if file_name.startswith("nab/"):
            return self._load_nab(file_name, data_path, download)
        if data_path.endswith(".mat"):
            from scipy.io import loadmat

            f = loadmat(data_path)

            X = f["X"]
            y = f["y"].ravel()
        else:
            X = self._load_via_txt(data_path)

            y = X[:, -1].ravel()
            X = X[:, :-1]

        return X, y

    def _load_nab(self, name: str, data_path: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
        """Loads a NAB series and labels the points in its anomaly windows with 1."""
        windows_path = os.path.join(self.data_base_path, "nab", "combined_windows.json")
        if not os.path.exists(windows_path):
            if not download:
                raise FileNotFoundError(
                    f"The NAB labels were not found at '{windows_path}' and download=False."
                )
            self._download("nab/combined_windows.json", windows_path)
        with open(windows_path) as f:
            windows = json.load(f)[name[len("nab/") :]]
        with open(data_path, newline="") as f:
            rows = list(csv.reader(f))[1:]  # the header is "timestamp,value"

        times = np.array([row[0] for row in rows], dtype="datetime64[s]")
        y = np.zeros(len(rows), dtype=int)
        for start, end in windows:
            y[(times >= np.datetime64(start)) & (times <= np.datetime64(end))] = 1

        return np.array([[float(row[1])] for row in rows]), y

    def get_iterator(
        self,
        data_file: str,
        shuffle: bool = True,
        random_state: int | np.random.RandomState | None = None,
        download: bool = True,
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """Returns an iterator over the instances and labels of a dataset, in shuffled order by default.

        Args:
            data_file (str): Path or filename of data.
            shuffle (bool): Whether to shuffle the instances (Default=True).
            random_state (int, np.random.RandomState or None): Seed or random number generator for the shuffle, as in :class:`ArrayStreamer`. None draws from NumPy's global random state (Default=None).
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            Iterator[tuple[np.ndarray, Any]]: Iterator over ``(X, y)`` pairs, where ``X`` is the feature vector of shape (num_features,) and ``y`` is its label.
        """
        X, y = self.get_data(data_file, download=download)

        return ArrayStreamer(shuffle=shuffle, random_state=random_state).iter(X, y)

    def describe(self, data_file: str, download: bool = True) -> dict[str, Any]:
        """Returns the number of points, the number of features and the anomaly rate of a dataset.

        Args:
            data_file (str): Path or filename of data.
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            dict: ``points`` (int), ``features`` (int) and ``anomaly_rate`` (float, the fraction of points with label 1).
        """
        X, y = self.get_data(data_file, download=download)

        return {"points": len(X), "features": X.shape[1], "anomaly_rate": float(np.mean(y))}

    def get_stream(
        self,
        data_file: str,
        random_state: int | np.random.RandomState | None = None,
        max_points: int | None = None,
        scale: bool = False,
        download: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns a shuffled dataset as arrays, in the order of :meth:`get_iterator`.

        With ``random_state=seed``, ``max_points=5000`` and ``scale=True``, this is the stream of the PySAD benchmark for that seed.

        Args:
            data_file (str): Path or filename of data.
            random_state (int, np.random.RandomState or None): Seed or random number generator for the shuffle. None draws from NumPy's global random state (Default=None).
            max_points (int or None): The number of first points of the shuffled stream to keep. None keeps all of them (Default=None).
            scale (bool): Whether to scale each feature to the range from 0 to 1, with the minimum and the maximum of the points that it keeps. A constant feature becomes 0 (Default=False).
            download (bool): Whether to download the dataset if not present locally (Default=True).

        Returns:
            tuple[np.ndarray, np.ndarray]: The feature vectors of shape (num_points, num_features) and the labels of shape (num_points,), in stream order.
        """
        stream = list(self.get_iterator(data_file, random_state=random_state, download=download))
        stream = stream[:max_points]
        X, y = np.array([x for x, _ in stream]), np.array([label for _, label in stream])
        if scale:
            mins, maxes = X.min(axis=0), X.max(axis=0)
            X = (X - mins) / np.where(maxes > mins, maxes - mins, 1.0)

        return X, y
