"""Compare PySAD detectors through the River adapter with River's own streaming detectors.

The harness wraps PySAD detectors in ``pysad.integrations.PySADToRiverAdapter``, which exposes
them as ``river.base.AnomalyDetector`` instances, and scores each point before the model learns
from it, exactly as PySAD does. It then compares the AUROC, AUPR and milliseconds per point of
the PySAD detectors with those of River's native streaming detectors on the shuffled ODDS
datasets of benchmarks/benchmark.py. The River detectors come from benchmarks/compare.py, so
both scripts compare the same detectors.

River is optional. The script imports river only when it is installed, and river is not a
runtime dependency of PySAD (it belongs to the test and compare dependency groups). When river
is not installed, running the script prints a message and writes nothing. No new dependency is
added.

Run it from the repository root:

    python benchmarks/compare_river_adapter.py
    python benchmarks/compare_river_adapter.py --jobs 2

The results go to docs/benchmark/comparison_river_adapter.csv.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.util import find_spec

import benchmark
import compare
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pysad.utils import fix_seed

SEED = 0


def installed() -> bool:
    """True when river is installed and the harness can run."""
    return find_spec("river") is not None


def pysad_adapter_detectors():
    """PySAD detectors exposed through the River adapter."""

    def factory(detector_name):
        def build(X, seed):
            from pysad.integrations import PySADToRiverAdapter

            return PySADToRiverAdapter(lambda: benchmark.DETECTORS[detector_name](X, seed))

        return build

    return {name: factory(name) for name in benchmark.DETECTORS}


def libraries():
    """The compared detectors, built only when river is installed."""
    return {
        "PySAD (adapter)": pysad_adapter_detectors(),
        "River": compare.river(),
    }


def _score_then_learn(model, x):
    """Score one instance, then learn from it, through the model's River interface."""
    score = model.score_one(x)
    model.learn_one(x)
    return score


def _run(library, detector, dataset):
    """Run one detector on one dataset in prequential order and return its metrics."""
    X, y = benchmark.load(dataset, SEED)
    fix_seed(SEED)
    model = libraries()[library][detector](X, SEED)
    points = [dict(enumerate(x.tolist())) for x in X]
    start = time.perf_counter()
    scores = np.array([float(_score_then_learn(model, x)) for x in points])
    seconds = time.perf_counter() - start
    w = benchmark.warmup(len(X))
    valid = np.isfinite(scores).all()
    both = valid and 0 < y.sum() < len(y)  # AUROC and AUPR need both kinds of points
    if both:
        auroc = round(float(roc_auc_score(y[w:], scores[w:])), 4)
        aupr = round(float(average_precision_score(y[w:], scores[w:])), 4)
    else:
        auroc, aupr = np.nan, np.nan
    return {
        "library": library,
        "detector": detector,
        "dataset": dataset,
        "auroc": auroc,
        "aupr": aupr,
        "ms_per_point": round(1000.0 * seconds / len(X), 3),
    }


def _run_job(job):
    """Run one detector on one dataset from a ``(library, detector, dataset)`` job tuple.

    ``ProcessPoolExecutor.map`` passes one argument per job, so this wrapper unpacks the tuple
    into the three arguments of :func:`_run`. A top-level function is required so the spawn
    start method can pickle it for the worker processes.
    """
    return _run(*job)


def _write_table(results):
    """Write the results to docs/benchmark/comparison_river_adapter.csv and print them."""
    header = ["Library", "Detector", "Dataset", "AUROC", "AUPR", "Time per point (ms)"]
    rows = []
    for job in sorted(results):
        row = results[job]
        rows.append(
            [
                row["library"],
                row["detector"],
                row["dataset"],
                row["auroc"],
                row["aupr"],
                row["ms_per_point"],
            ]
        )
    benchmark.write_csv("comparison_river_adapter.csv", header, rows)
    for row in rows:
        print(*row, sep="  ")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if not installed():
        print(
            "river is not installed; the River adapter benchmark does not run. "
            "Install it with `pip install river` (test or compare dependency group)."
        )
        return 0
    libs = libraries()
    for dataset in benchmark.DATASETS:  # download once, before the workers start
        benchmark.Data().get_data(f"{dataset}.mat")
    jobs = [(lib, d, ds) for lib in libs for d in libs[lib] for ds in benchmark.DATASETS]
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = {job: row for job, row in zip(jobs, pool.map(_run_job, jobs), strict=True)}
    _write_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
