"""Benchmark the multivariate detectors and the calibrators of PySAD on the datasets of ``pysad.utils.Data``.

Run it from the repository root:

    python benchmarks/benchmark.py                        # every detector
    python benchmarks/benchmark.py --detectors LODA,RSHash
    python benchmarks/benchmark.py --check origin/master  # rerun what changed, compare with the results

The results go to docs/benchmark/, which docs/benchmark.rst shows.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from pysad import models
from pysad.transform.probability_calibration import (
    ConformalProbabilityCalibrator,
    GaussianTailProbabilityCalibrator,
)
from pysad.utils import ArrayStreamer, Data, fix_seed, get_minmax_array

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmark"
DATASETS = [name.removesuffix(".mat") for name in Data()._get_data_files()]
SEEDS = range(5)
MAX_POINTS = 5000  # the first points of each shuffled stream
WARMUP_POINTS = 256  # the scores of the first points do not count, see warmup()
# --check accepts this difference in AUROC. Other platforms round differently, and on a small
# dataset that can swap one anomaly with one normal point. AUPR moves too much for such a swap.
TOLERANCE = 0.005
# --check accepts this difference in a false alert rate, because one alert more or less on a small
# dataset moves the rate by about 0.01.
ALERT_TOLERANCE = 0.02


def _ranges(X):
    return {"feature_mins": np.zeros(X.shape[1]), "feature_maxes": np.ones(X.shape[1])}


# Each detector gets its defaults, plus the arguments that it requires. X is scaled to [0, 1].
DETECTORS = {
    "ExactStorm": lambda X, seed: models.ExactStorm(),
    "HalfSpaceTrees": lambda X, seed: models.HalfSpaceTrees(**_ranges(X), random_state=seed),
    "IForestASD": lambda X, seed: models.IForestASD(random_state=seed),
    "Inqmad": lambda X, seed: models.Inqmad(X.shape[1], random_state=seed),
    "KitNet": lambda X, seed: models.KitNet(random_state=seed),
    "LODA": lambda X, seed: models.LODA(random_state=seed),
    "LocalOutlierProbability": lambda X, seed: models.LocalOutlierProbability(X[: warmup(len(X))]),
    "RobustRandomCutForest": lambda X, seed: models.RobustRandomCutForest(random_state=seed),
    "RSHash": lambda X, seed: models.RSHash(**_ranges(X), random_state=seed),
    "xStream": lambda X, seed: models.xStream(random_state=seed),
}
# The per-dataset tables use these names, which are common in the papers, to fit the page.
SHORT_NAMES = {
    "HalfSpaceTrees": "HST",
    "LocalOutlierProbability": "LoOP",
    "RobustRandomCutForest": "RRCF",
}
# Each calibrator, with its defaults, turns the scores of every run into alerts at each threshold,
# as the alerts guide does: an alert when the calibrated score is at least the threshold.
CALIBRATORS = {
    "conformal": ConformalProbabilityCalibrator,
    "gaussian": GaussianTailProbabilityCalibrator,
}
CALIBRATOR_NAMES = {"conformal": "Conformal", "gaussian": "Gaussian tail"}  # short, to fit the page
# "rate" is 1 minus the anomaly rate of the stream, so the alerts match the number of anomalies.
# It uses the labels, which a live stream does not have, so it shows the best case of a threshold.
THRESHOLDS = [0.95, 0.99, 0.995, "rate"]
ALERT_METRICS = ["false_alerts", "precision", "recall"]
ALERT_FIELDS = [f"{c}_{t}_{m}" for c in CALIBRATORS for t in THRESHOLDS for m in ALERT_METRICS]
FIELDS = ["dataset", "detector", "seed", "auroc", "aupr", "ms_per_point", *ALERT_FIELDS, "commit"]
COMMIT_URL = "https://github.com/selimfirat/pysad/commit/"


def warmup(n):
    """The number of first points of a stream whose scores do not count in the metrics.

    A detector learns from these points, but it has little or no model yet when it scores them.
    It is 256 points, or a quarter of a stream with fewer than 1,024 points.
    """
    return min(WARMUP_POINTS, n // 4)


def load(dataset, seed):
    X, y = Data().get_data(f"{dataset}.mat")
    stream = list(ArrayStreamer(shuffle=True, random_state=seed).iter(X, y))[:MAX_POINTS]
    X, y = np.array([x for x, _ in stream]), np.array([label for _, label in stream])
    mins, maxes = get_minmax_array(X)
    return (X - mins) / np.where(maxes > mins, maxes - mins, 1.0), y


def run(job):
    dataset, detector, seed = job
    X, y = load(dataset, seed)
    fix_seed(seed)
    model = DETECTORS[detector](X, seed)
    start = time.perf_counter()
    with np.errstate(all="ignore"):  # a score that is not finite makes the metrics NaN
        scores = np.array([float(model.fit_score_partial(x)) for x in X])
    seconds = time.perf_counter() - start
    valid = np.isfinite(scores).all()
    row = {"dataset": dataset, "detector": detector, "seed": seed}
    for name, calibrator in CALIBRATORS.items():
        transform = calibrator().fit_transform_partial
        with np.errstate(all="ignore"):
            calibrated = np.array([transform(s) for s in scores])
        row.update(alerts(name, calibrated[warmup(len(X)) :], y[warmup(len(X)) :], valid))
    scores, y = scores[warmup(len(X)) :], y[warmup(len(X)) :]
    return {
        **row,
        "auroc": round(roc_auc_score(y, scores), 4) if valid else np.nan,
        "aupr": round(average_precision_score(y, scores), 4) if valid else np.nan,
        "ms_per_point": round(1000 * seconds / len(X), 3),
    }


def alerts(calibrator, calibrated, y, valid):
    """The false alert rate, the precision and the recall of the alerts at each threshold."""
    row = {}
    for t in THRESHOLDS:
        alert = calibrated >= (1 - y.mean() if t == "rate" else t)
        values = [
            alert[y == 0].mean(),
            (alert & (y == 1)).sum() / alert.sum() if alert.any() else np.nan,
            alert[y == 1].mean(),
        ]
        for metric, value in zip(ALERT_METRICS, values, strict=True):
            row[f"{calibrator}_{t}_{metric}"] = round(value, 4) if valid else np.nan
    return row


def key(row):
    return row["dataset"], row["detector"], int(row["seed"])


def read_results():
    path = OUT / "results.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {key(row): row for row in csv.DictReader(f)}


def write_csv(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_tables(results):
    detectors = [d for d in DETECTORS if any(k[1] == d for k in results)]
    rows = sorted(results.values(), key=lambda r: (DATASETS.index(r["dataset"]), r["detector"]))
    write_csv("results.csv", FIELDS, [[r.get(f, "") for f in FIELDS] for r in rows])

    def mean(metric, dataset, detector):
        return np.mean(
            [float(r[metric]) for k, r in results.items() if k[:2] == (dataset, detector)]
        )

    tables = {
        metric: np.array([[mean(metric, ds, d) for d in detectors] for ds in DATASETS])
        for metric in ["auroc", "aupr"]
    }
    rank = np.mean([rankdata(-np.nan_to_num(row)) for row in tables["auroc"]], axis=0)
    ms = [
        np.median([float(r["ms_per_point"]) for k, r in results.items() if k[1] == d])
        for d in detectors
    ]
    auroc, aupr = (np.nanmean(tables[metric], axis=0) for metric in ["auroc", "aupr"])
    # Rank 1 is the highest AUROC of a dataset. Equal mean ranks, as the page rounds them, go by
    # mean AUROC, then mean AUPR, then time per point.
    order = sorted(
        range(len(detectors)), key=lambda i: (round(rank[i], 1), -auroc[i], -aupr[i], ms[i])
    )
    write_csv(
        "summary.csv",
        ["Detector", "Mean rank", "Mean AUROC", "Mean AUPR", "Time per point (ms)", "Commit"],
        [
            [
                f":class:`~pysad.models.{detectors[i]}`",
                f"{rank[i]:.1f}",
                f"{auroc[i]:.3f}",
                f"{aupr[i]:.3f}",
                f"{ms[i]:.2f}",
                ", ".join(
                    f"`{c} <{COMMIT_URL}{c.removesuffix('-dirty')}>`__"
                    for c in sorted(
                        {r.get("commit") or "" for k, r in results.items() if k[1] == detectors[i]}
                    )
                    if c
                ),
            ]
            for i in order
        ],
    )
    for metric, table in tables.items():  # the columns in the order of the summary
        rows = []
        for ds, row in zip(DATASETS, table[:, order], strict=True):
            cells = [f"{v:.3f}" for v in row]
            best = f"{np.nanmax(row):.3f}"
            rows.append([ds, *(f"**{c}**" if c == best else c for c in cells)])
        header = [SHORT_NAMES.get(detectors[i], detectors[i]) for i in order]
        write_csv(f"{metric}.csv", ["Dataset", *header], rows)

    write_csv(
        "alerts.csv",
        [
            "Calibrator",
            "Threshold",
            "Expected",
            "False alerts",
            "Range",
            "Precision",
            "Recall",
            "F1",
        ],
        alert_rows(results, detectors),
    )

    datasets = []
    for ds in DATASETS:
        X, y = Data().get_data(f"{ds}.mat")
        n = min(len(X), MAX_POINTS)
        datasets.append(
            [ds, f"{len(X):,}", f"{n:,}", f"{n - warmup(n):,}", X.shape[1], f"{100 * y.mean():.1f}"]
        )
    write_csv(
        "datasets.csv",
        ["Dataset", "Points", "Points used", "Points evaluated", "Features", "Anomalies (%)"],
        datasets,
    )


def number(value):
    """A value of a results row as a float, NaN if it is missing. 0.0 stays 0.0."""
    return np.nan if value is None or value == "" else float(value)


def alert_rows(results, detectors):
    """One row for each calibrator and threshold, with means over all the runs."""
    rows = []
    for name, calibrator in CALIBRATORS.items():
        for t in THRESHOLDS:
            values = {
                m: np.array([number(r.get(f"{name}_{t}_{m}")) for r in results.values()])
                for m in ALERT_METRICS
            }
            p, r = values["precision"], values["recall"]
            with np.errstate(all="ignore"):
                f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
            per_detector = [
                np.nanmean(
                    [
                        number(row.get(f"{name}_{t}_false_alerts"))
                        for k, row in results.items()
                        if k[1] == d
                    ]
                )
                for d in detectors
            ]
            rows.append(
                [
                    f":class:`{CALIBRATOR_NAMES[name]} <pysad.transform.probability_calibration.{calibrator.__name__}>`",
                    "1 - anomaly rate" if t == "rate" else t,
                    "" if t == "rate" else f"{1 - t:.3f}",
                    f"{np.nanmean(values['false_alerts']):.3f}",
                    f"{np.nanmin(per_detector):.3f} to {np.nanmax(per_detector):.3f}",
                    f"{np.nanmean(p):.3f}",
                    f"{np.nanmean(r):.3f}",
                    f"{np.nanmean(f1):.3f}",
                ]
            )
    return rows


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def commit():
    """The commit that the run uses, with -dirty when the code has changes that are not committed."""
    dirty = git("status", "--porcelain", "--", "pysad", "benchmarks", "pyproject.toml").strip()
    return git("rev-parse", "--short", "HEAD").strip() + ("-dirty" if dirty else "")


def changed_detectors(base):
    """The detectors whose results a change since ``base`` can move."""
    files = git("diff", "--name-only", base).split()
    common = pysad_files([ArrayStreamer, Data, fix_seed, get_minmax_array, *CALIBRATORS.values()])
    if common & set(files) or "benchmarks/benchmark.py" in files or "pyproject.toml" in files:
        return list(DETECTORS)
    edited = set()  # detectors whose committed results changed, so that every row comes from a run
    if "docs/benchmark/results.csv" in files:
        before = {
            key(row): row
            for row in csv.DictReader(
                git("show", f"{base}:docs/benchmark/results.csv").splitlines()
            )
        }
        edited = {k[1] for k, row in read_results().items() if before.get(k) != row}
    return [d for d in DETECTORS if pysad_files([getattr(models, d)]) & set(files) or d in edited]


def pysad_files(objects):
    """The pysad source files that define ``objects`` and the pysad names that those files import."""
    files, todo = set(), [obj.__module__ for obj in objects]
    while todo:
        module = sys.modules[todo.pop()]
        path = Path(module.__file__).relative_to(ROOT).as_posix()
        if path in files:
            continue
        files.add(path)
        for value in vars(module).values():
            name = (
                value.__name__
                if isinstance(value, ModuleType)
                else getattr(value, "__module__", "")
            )
            if isinstance(name, str) and name.split(".")[0] == "pysad":
                todo.append(name)
    return files


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detectors", help="comma-separated detector names (default: all)")
    parser.add_argument(
        "--check",
        nargs="?",
        const="",
        metavar="BASE",
        help="compare with docs/benchmark/ instead of writing it; with BASE, rerun only the detectors that changed since BASE",
    )
    parser.add_argument("--seeds", help="comma-separated seeds (default: 0,1,2,3,4)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    detectors = args.detectors.split(",") if args.detectors else list(DETECTORS)
    if args.check:
        detectors = changed_detectors(args.check)
    if args.check is not None:
        print("Detectors to check:", ", ".join(detectors) or "none")
    unknown = set(detectors) - set(DETECTORS)
    if unknown:
        parser.error(f"unknown detectors: {', '.join(sorted(unknown))}")

    for dataset in DATASETS:  # download each file once, before the workers start
        Data().get_data(f"{dataset}.mat")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    jobs = [(ds, d, s) for d in detectors for ds in DATASETS for s in seeds]
    with ProcessPoolExecutor(args.jobs) as pool:
        new = {key(row): row for row in pool.map(run, jobs)}

    results = read_results()
    if args.check is not None:
        changed = [
            (k, results.get(k), row)
            for k, row in new.items()
            if k not in results
            or not np.isclose(
                float(results[k]["auroc"]), row["auroc"], atol=TOLERANCE, equal_nan=True
            )
            or any(
                not np.isclose(
                    number(results[k].get(f)), row[f], atol=ALERT_TOLERANCE, equal_nan=True
                )
                for f in ALERT_FIELDS
                if f.endswith("false_alerts")
            )
        ]
        for k, old, row in changed:
            before = f"{old['auroc']} / {old['aupr']}" if old else "missing"
            print(
                f"{k[1]} on {k[0]}, seed {k[2]}: AUROC / AUPR {before} -> {row['auroc']} / {row['aupr']}"
            )
        if changed:
            print(
                f"{len(changed)} results changed. Run: python benchmarks/benchmark.py --detectors "
                f"{','.join(sorted({k[1] for k, _, _ in changed}))}, then commit docs/benchmark/."
            )
            sys.exit(1)
        print("The results in docs/benchmark/ are up to date.")
        return

    sha = commit()
    results.update({k: {**row, "commit": sha} for k, row in new.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    write_tables(results)


if __name__ == "__main__":
    main()
