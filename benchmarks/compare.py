"""Compare the multivariate detectors of PySAD with those of other libraries.

Every detector runs on the shuffled ODDS datasets of benchmark.py, with its protocol: the same
points, seed 0, each point scored before the detector learns from it, and AUROC and AUPR after the
warm-up. River, CapyMOA, StreamAD and Alibi Detect run their streaming detectors. PyOD runs its
batch detectors through pysad.integrations.ReferenceWindowModel.

StreamAD and Alibi Detect need NumPy 1, and CapyMOA needs NumPy 2, so no environment holds all the
libraries. The script runs the libraries that are installed. Run it from the repository root, once
with each dependency group:

    pip install -e . --group compare
    python benchmarks/compare.py
    pip install -e . --group compare-numpy1
    python benchmarks/compare.py

The results go to docs/benchmark/comparison_<library>.csv, which the pages in docs/comparisons/
show. Each table ranks the PySAD detectors with the detectors of one library. With the same library
versions, a rerun gives the same AUROC and AUPR. Only the times per point change.
"""

import argparse
import multiprocessing
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from importlib.util import find_spec

import benchmark
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from pysad.integrations import ReferenceWindowModel
from pysad.utils import fix_seed

SEED = 0
# Each PyOD detector fits on the warm-up points first, then refits on the last WINDOW points
# every SLIDE points.
WINDOW, SLIDE = 1000, 100


# Each detector gets its defaults, plus a seed if it takes one. The libraries are imported only
# if they are installed. X is scaled to [0, 1], the range that River's HalfSpaceTrees expects by
# default.
def river():
    from river import anomaly

    return {
        "HalfSpaceTrees": lambda X, seed: anomaly.HalfSpaceTrees(seed=seed),
        "LODA": lambda X, seed: anomaly.LODA(seed=seed),
        "LocalOutlierFactor": lambda X, seed: anomaly.LocalOutlierFactor(),
        "OneClassSVM": lambda X, seed: anomaly.OneClassSVM(),
    }


def pyod():
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF

    def window(cls, **kwargs):
        return lambda X, seed: ReferenceWindowModel(
            cls,
            window_size=WINDOW,
            sliding_size=SLIDE,
            initial_window_X=X[: benchmark.warmup(len(X))],
            **kwargs,
        )

    return {
        "hbos.HBOS": window(HBOS),
        "iforest.IForest": window(IForest, random_state=SEED),
        "knn.KNN": window(KNN),
        "lof.LOF": window(LOF),
    }


def capymoa():
    # Autoencoder is left out, because it needs PyTorch.
    from capymoa import anomaly
    from capymoa.stream import NumpyStream

    def schema(X):
        return NumpyStream(X, np.zeros(len(X), dtype=int), target_type="categorical").get_schema()

    return {
        "AdaptiveIsolationForest": lambda X, seed: anomaly.AdaptiveIsolationForest(
            schema(X), seed=seed
        ),
        "HalfSpaceTrees": lambda X, seed: anomaly.HalfSpaceTrees(schema(X), random_seed=seed),
        "IForestASD": lambda X, seed: anomaly.IForestASD(schema(X), random_state=seed),
        "Loda": lambda X, seed: anomaly.Loda(schema(X), random_state=seed),
        "OnlineIsolationForest": lambda X, seed: anomaly.OnlineIsolationForest(
            schema(X), random_seed=seed
        ),
        "RSHash": lambda X, seed: anomaly.RSHash(schema(X), seed=seed),
        "RobustRandomCutForest": lambda X, seed: anomaly.RobustRandomCutForest(
            schema(X), random_state=seed
        ),
        "StreamRHF": lambda X, seed: anomaly.StreamRHF(schema(X), random_seed=seed),
        "StreamingIsolationForest": lambda X, seed: anomaly.StreamingIsolationForest(
            schema(X), seed=seed
        ),
    }


def streamad():
    # The multivariate detectors, without the baseline RandomDetector. They take no seed and draw
    # from NumPy's global random state. score_first=True scores each point before the detector
    # learns from it. OCSVMDetector and RrcfDetector fail with it, so they keep the default, which
    # learns from each point first.
    from streamad import model

    return {
        "HSTreeDetector": lambda X, seed: model.HSTreeDetector(score_first=True),
        "LodaDetector": lambda X, seed: model.LodaDetector(score_first=True),
        "OCSVMDetector": lambda X, seed: model.OCSVMDetector(),
        "RShashDetector": lambda X, seed: model.RShashDetector(score_first=True),
        "RrcfDetector": lambda X, seed: model.RrcfDetector(),
        "xStreamDetector": lambda X, seed: model.xStreamDetector(score_first=True),
    }


def alibi_detect():
    # The outlier detector that learns from each batch that it scores. The batches have one point.
    from alibi_detect.od import Mahalanobis

    return {"Mahalanobis": lambda X, seed: Mahalanobis()}


# For each library: its distribution, the module of its detectors and the detectors.
LIBRARIES = {
    "PySAD": ("pysad", "pysad.models", lambda: benchmark.DETECTORS),
    "River": ("river", "river.anomaly", river),
    "PyOD": ("pyod", "pyod.models", pyod),
    "CapyMOA": ("capymoa", "capymoa.anomaly", capymoa),
    "StreamAD": ("streamad", "streamad.model", streamad),
    "Alibi Detect": ("alibi-detect", "alibi_detect.od", alibi_detect),
}


def installed():
    return [lib for lib, (dist, _, _) in LIBRARIES.items() if find_spec(dist.replace("-", "_"))]


def name(library, detector):
    if library == "PySAD":
        return f":class:`~pysad.models.{detector}`"
    return f"``{LIBRARIES[library][1]}.{detector}``"


def run(job):
    # One thread for BLAS and OpenMP in each worker, as in benchmark.py.
    with threadpool_limits(1), warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return _run(*job)


def _run(library, detector, dataset):
    X, y = benchmark.load(dataset, SEED)
    fix_seed(SEED)
    model = LIBRARIES[library][2]()[detector](X, SEED)
    # The points are converted to the input type of each library before the timer starts.
    if library == "River":
        points = [dict(enumerate(x.tolist())) for x in X]

        def step(x):
            score = model.score_one(x)
            model.learn_one(x)
            return score

    elif library == "CapyMOA":
        from capymoa.core import Instance

        points = [Instance.from_array(model.schema, x) for x in X]

        def step(x):
            score = model.score_instance(x)
            model.train(x)
            return score

    elif library == "StreamAD":
        points = X

        def step(x):
            score = model.fit_score(x)
            return 0.0 if score is None else score  # None until the first window is full

    elif library == "Alibi Detect":
        points = X[:, None]  # shape (num_points, 1, num_features), batches of one point

        def step(x):
            return model.score(x)[0]

    else:
        points, step = X, model.fit_score_partial
    start = time.perf_counter()
    scores = np.array([float(step(x)) for x in points])
    seconds = time.perf_counter() - start
    w = benchmark.warmup(len(X))
    valid = np.isfinite(scores).all()
    return {
        "key": (library, detector, dataset),
        "auroc": roc_auc_score(y[w:], scores[w:]) if valid else np.nan,
        "aupr": average_precision_score(y[w:], scores[w:]) if valid else np.nan,
        "ms_per_point": 1000 * seconds / len(X),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    for dataset in benchmark.DATASETS:  # download once, before the workers start
        benchmark.Data().get_data(f"{dataset}.mat")
    libraries = installed()
    print("Libraries:", *libraries)
    jobs = [
        (lib, d, ds) for lib in libraries for d in LIBRARIES[lib][2]() for ds in benchmark.DATASETS
    ]
    # Spawned workers, because CapyMOA starts a Java VM on import, and a forked copy of it hangs.
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = {row["key"]: row for row in pool.map(run, jobs)}
    versions = {lib: version(LIBRARIES[lib][0]) for lib in libraries}
    versions["PySAD"] += f"+{benchmark.commit()}"  # the run uses the code of the checkout
    for library in libraries[1:]:
        write_table(results, versions, library)


def write_table(results, versions, library):
    """Rank the PySAD detectors with the detectors of one other library."""
    detectors = [(lib, d) for lib in ("PySAD", library) for d in LIBRARIES[lib][2]()]

    def table(metric):  # shape (num_datasets, num_detectors)
        return np.array(
            [[results[(*d, ds)][metric] for d in detectors] for ds in benchmark.DATASETS]
        )

    auroc, aupr = table("auroc"), table("aupr")
    rank = np.mean([rankdata(-np.nan_to_num(row)) for row in auroc], axis=0)
    mean_auroc, mean_aupr = np.nanmean(auroc, axis=0), np.nanmean(aupr, axis=0)
    ms = np.median(table("ms_per_point"), axis=0)
    order = sorted(
        range(len(detectors)),
        key=lambda i: (round(rank[i], 1), -mean_auroc[i], -mean_aupr[i], ms[i]),
    )
    rows = [
        [
            name(*detectors[i]),
            f"{rank[i]:.1f}",
            f"{mean_auroc[i]:.3f}",
            f"{mean_aupr[i]:.3f}",
            f"{ms[i]:.2f}",
            versions[detectors[i][0]],
        ]
        for i in order
    ]
    header = ["Detector", "Mean rank", "Mean AUROC", "Mean AUPR", "Time per point (ms)", "Version"]
    file = library.lower().replace(" ", "_")
    benchmark.write_csv(f"comparison_{file}.csv", header, rows)
    for row in rows:
        print(*row, sep="  ")


if __name__ == "__main__":
    main()
