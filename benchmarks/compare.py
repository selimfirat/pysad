"""Compare the multivariate detectors of PySAD with those of River and PyOD.

Every detector runs on the shuffled ODDS datasets of benchmark.py, with its protocol: the same
points, seed 0, each point scored before the detector learns from it, and AUROC and AUPR after the
warm-up. River runs its streaming detectors. PyOD runs its batch detectors through
pysad.models.integrations.ReferenceWindowModel.

Install the libraries and run it from the repository root:

    pip install -e . --group compare
    python benchmarks/compare.py

The results go to docs/benchmark/comparison_river.csv and docs/benchmark/comparison_pyod.csv, which
docs/comparisons/river.rst and docs/comparisons/pyod.rst show. Each table ranks the PySAD detectors
with the detectors of one library. With the same library versions, a rerun gives the same AUROC
and AUPR. Only the times per point change.
"""

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version

import benchmark
import numpy as np
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from river import anomaly
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from pysad.models.integrations import ReferenceWindowModel
from pysad.utils import fix_seed

SEED = 0
# Each PyOD detector fits on the warm-up points first, then refits on the last WINDOW points
# every SLIDE points.
WINDOW, SLIDE = 1000, 100


def pyod(cls, **kwargs):
    return lambda X, seed: ReferenceWindowModel(
        cls,
        window_size=WINDOW,
        sliding_size=SLIDE,
        initial_window_X=X[: benchmark.warmup(len(X))],
        **kwargs,
    )


# Each detector gets its defaults, plus a seed if it draws random numbers. X is scaled to [0, 1],
# the range that River's HalfSpaceTrees expects by default.
DETECTORS = {
    "PySAD": benchmark.DETECTORS,
    "River": {
        "HalfSpaceTrees": lambda X, seed: anomaly.HalfSpaceTrees(seed=seed),
        "LODA": lambda X, seed: anomaly.LODA(seed=seed),
        "LocalOutlierFactor": lambda X, seed: anomaly.LocalOutlierFactor(),
        "OneClassSVM": lambda X, seed: anomaly.OneClassSVM(),
    },
    "PyOD": {
        "hbos.HBOS": pyod(HBOS),
        "iforest.IForest": pyod(IForest, random_state=SEED),
        "knn.KNN": pyod(KNN),
        "lof.LOF": pyod(LOF),
    },
}
NAMES = {
    "PySAD": ":class:`~pysad.models.{}`",
    "River": "``river.anomaly.{}``",
    "PyOD": "``pyod.models.{}``",
}


def run(job):
    # One thread for BLAS and OpenMP in each worker, as in benchmark.py.
    with threadpool_limits(1), warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return _run(*job)


def _run(library, detector, dataset):
    X, y = benchmark.load(dataset, SEED)
    fix_seed(SEED)
    model = DETECTORS[library][detector](X, SEED)
    if library == "River":
        points = [dict(enumerate(x.tolist())) for x in X]

        def step(x):
            score = model.score_one(x)
            model.learn_one(x)
            return score

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
    jobs = [
        (lib, d, ds)
        for lib, models in DETECTORS.items()
        for d in models
        for ds in benchmark.DATASETS
    ]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = {row["key"]: row for row in pool.map(run, jobs)}
    versions = {lib: version(lib.lower()) for lib in DETECTORS}
    versions["PySAD"] += f"+{benchmark.commit()}"  # the run uses the code of the checkout
    for library in ("River", "PyOD"):
        write_table(results, versions, library)


def write_table(results, versions, library):
    """Rank the PySAD detectors with the detectors of one other library."""
    detectors = [(lib, d) for lib in ("PySAD", library) for d in DETECTORS[lib]]

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
            f"{detectors[i][0]} {versions[detectors[i][0]]}",
            NAMES[detectors[i][0]].format(detectors[i][1]),
            f"{rank[i]:.1f}",
            f"{mean_auroc[i]:.3f}",
            f"{mean_aupr[i]:.3f}",
            f"{ms[i]:.2f}",
        ]
        for i in order
    ]
    header = ["Library", "Detector", "Mean rank", "Mean AUROC", "Mean AUPR", "Time per point (ms)"]
    benchmark.write_csv(f"comparison_{library.lower()}.csv", header, rows)
    for row in rows:
        print(*row, sep="  ")


if __name__ == "__main__":
    main()
