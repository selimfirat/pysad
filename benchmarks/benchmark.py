"""Benchmark the detectors and the calibrators of PySAD on the datasets of ``pysad.utils.Data``.

The multivariate detectors and the calibrators run on the ODDS datasets, and the univariate
detectors on the series of the Numenta Anomaly Benchmark (NAB) and the UCR Time Series Anomaly
Archive.

Run it from the repository root:

    python benchmarks/benchmark.py                        # every detector
    python benchmarks/benchmark.py --detectors LODA,RSHash
    python benchmarks/benchmark.py --check origin/master  # rerun what changed, compare with the results

The results go to docs/benchmark/, which docs/benchmark.rst shows.
"""

import argparse
import csv
import functools
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
from pysad.utils import Data, fix_seed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmark"
DATASETS = [name.removesuffix(".mat") for name in Data.datasets()]
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


# The univariate detectors run once on each NAB series with anomalies, in the order of the file,
# because none of them draws random numbers. Each detector gets its defaults, plus the arguments
# that it requires. A series is scaled to [0, 1].
UNIVARIATE = {
    # KNNCAD learns during NAB's probationary period, as in NAB.
    "KNNCAD": lambda n, period: models.KNNCAD(probationary_period=nab_probation(n)),
    "MedianAbsoluteDeviation": lambda n, period: models.MedianAbsoluteDeviation(),
    "RelativeEntropy": lambda n, period: models.RelativeEntropy(min_val=0.0, max_val=1.0),
    "StandardAbsoluteDeviation": lambda n, period: models.StandardAbsoluteDeviation(),
    # A window of 3 periods and one point, because R's stl() needs more than 2 periods, and an odd
    # window makes the robust STL match R.
    "SeasonalESD": lambda n, period: models.SeasonalESD(period, 3 * period + 1),
    "SeasonalHybridESD": lambda n, period: models.SeasonalHybridESD(period, 3 * period + 1),
}
# The ESD detectors need the period of a series, so they run only on these series. Each period is a
# day, where the autocorrelation at a lag of one day is at least 0.5. Nine more NAB series have a
# daily period of 288 points, but a window of 3 such periods makes the STL of each point too slow.
PERIODS = {
    "nab/realKnownCause/ambient_temperature_system_failure.csv": 24,
    "nab/realKnownCause/nyc_taxi.csv": 48,
    "nab/realAdExchange/exchange-2_cpc_results.csv": 24,
    "nab/realAdExchange/exchange-2_cpm_results.csv": 24,
    "nab/realAdExchange/exchange-3_cpm_results.csv": 24,
}
SEASONAL = ["SeasonalESD", "SeasonalHybridESD"]


@functools.cache
def nab_series():
    """All the NAB series, as NAB scores them. 6 of them have no anomalies, so their AUROC is NaN."""
    return Data.datasets("nab")


def series_of(detector):
    return list(PERIODS) if detector in SEASONAL else nab_series()


# The UCR series with at most this many points, 84 of the 250, keep the run short. The ESD detectors
# do not run on them, because the archive does not give the periods.
UCR_MAX_POINTS = 20_000
# A run solves a UCR series when its highest score after the training part is at most this many
# points from the anomaly, the rule of the authors of the archive.
UCR_MARGIN = 100


@functools.cache
def ucr_series():
    return [
        name for name in Data.datasets("ucr") if len(Data().get_data(name)[0]) <= UCR_MAX_POINTS
    ]


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
FIELDS = [
    "dataset",
    "detector",
    "seed",
    "auroc",
    "aupr",
    "ms_per_point",
    "solved",  # UCR series only
    *ALERT_FIELDS,
    "commit",
]
COMMIT_URL = "https://github.com/selimfirat/pysad/commit/"


def warmup(n):
    """The number of first points of a stream whose scores do not count in the metrics.

    A detector learns from these points, but it has little or no model yet when it scores them.
    It is 256 points, or a quarter of a stream with fewer than 1,024 points.
    """
    return min(WARMUP_POINTS, n // 4)


def load(dataset, seed):
    return Data().get_stream(f"{dataset}.mat", random_state=seed, max_points=MAX_POINTS, scale=True)


def nab_probation(n):
    """NAB's probationary period of a series of n points: 15%, at most 750. NAB does not score it."""
    return min(n * 15 // 100, 750)


# These detectors give scores from 0 to 1, as NAB's detectors do. The scores of the others have
# another scale on each series, so they are calibrated before NAB uses one threshold for all. The
# calibrator compares each score with all the earlier scores of the series. With the default window
# of 300 scores, its values have steps of 1/300, and many points tie at the highest value.
NAB_UNIT_SCORES = ["KNNCAD", "RelativeEntropy"]

# NAB's scoring, ported from nab/sweeper.py and nab/optimizer.py of the Numenta Anomaly Benchmark
# (MIT license) at the commit that Data downloads the series from. Each profile gives the weights of
# a detected window, a false alert and a missed window.
NAB_PROFILES = {
    "standard": (1.0, 0.11, 1.0),
    "reward_low_fp": (1.0, 0.22, 1.0),
    "reward_low_fn": (1.0, 0.11, 2.0),
}
NAB_TOLERANCE = 1.0  # --check accepts this difference in a NAB score, which goes from 0 to 100
NAB_FIELDS = ["detector", "series", *NAB_PROFILES, "seasonal_standard"]


def _scaled_sigmoid(x):
    # NAB's weight of an alert at the relative position x of a window, 1 at the start of a window
    # and 0 at its end. A false alert at a distance x after a window gets the negative value.
    return np.where(x > 3.0, -1.0, 2.0 / (1.0 + np.exp(5.0 * np.minimum(x, 3.0))) - 1.0)


def nab_sweep(y, tp_weight, fp_weight):
    """NAB's score of an alert at each point, and the window of each point (-1 outside windows)."""
    sweep, window = np.empty(len(y)), np.full(len(y), -1)
    edges = np.flatnonzero(np.diff(np.r_[0, y, 0]))
    max_tp = float(_scaled_sigmoid(-1.0))
    position, previous = 0, None  # the point after the last window, and (its end, its width)
    for k, (start, end) in enumerate(zip(edges[::2], np.r_[edges[1::2] - 1], strict=True)):
        _sweep_outside(sweep, position, start, previous, fp_weight)
        i = np.arange(start, end + 1)
        width = end - start + 1
        sweep[i] = _scaled_sigmoid(-(end - i + 1) / width) * tp_weight / max_tp
        window[i] = k
        position, previous = end + 1, (end, width)
    _sweep_outside(sweep, position, len(y), previous, fp_weight)
    return sweep, window


def _sweep_outside(sweep, start, stop, previous, fp_weight):
    i = np.arange(start, stop)
    if previous is None:
        sweep[i] = -1.0 * fp_weight
    else:
        end, width = previous
        sweep[i] = _scaled_sigmoid(np.abs(end - i) / max(width - 1, 1)) * fp_weight


def nab_score(files, profile):
    """NAB's normalized score of a detector on the files, a list of (scores, labels) pairs.

    As NAB's optimizer does, it uses the one threshold for all the files that gives the highest
    score. It leaves out NAB's probationary period of each file: 15% of it, at most 750 points.
    """
    tp_weight, fp_weight, fn_weight = NAB_PROFILES[profile]
    scores, sweeps, windows, num_windows = [], [], [], 0
    for file_scores, y in files:
        if not np.isfinite(file_scores).all():
            return np.nan
        sweep, window = nab_sweep(y, tp_weight, fp_weight)
        keep = np.arange(len(y)) >= nab_probation(len(y))
        window = window[keep]
        ids = np.unique(window[window >= 0])  # the windows after the probationary period
        window = np.where(window >= 0, np.searchsorted(ids, window) + num_windows, -1)
        scores.append(file_scores[keep])
        sweeps.append(sweep[keep])
        windows.append(window)
        num_windows += len(ids)
    scores, sweeps, windows = (
        np.concatenate(scores),
        np.concatenate(sweeps),
        np.concatenate(windows),
    )
    best = total = -fn_weight * num_windows  # no alerts: every window is missed
    part = np.full(num_windows, -fn_weight)  # the best alert of each window so far
    order = np.argsort(-scores, kind="stable")
    for n, i in enumerate(order):  # lower the threshold one score at a time
        if windows[i] < 0:
            total += sweeps[i]
        elif sweeps[i] > part[windows[i]]:
            total += sweeps[i] - part[windows[i]]
            part[windows[i]] = sweeps[i]
        if n + 1 == len(order) or scores[order[n + 1]] != scores[i]:
            best = max(best, total)
    if num_windows == 0:  # nothing to detect
        return np.nan
    null, perfect = -fn_weight * num_windows, tp_weight * num_windows
    return 100.0 * (best - null) / (perfect - null)


def nab_rows(new):
    """The NAB scores of each univariate detector that ran, from the calibrated scores of its runs."""
    files = {}
    for (series, detector, _), row in new.items():
        if "_nab" in row:
            files.setdefault(detector, {})[series] = row.pop("_nab")
    rows = {}
    for detector, by_series in files.items():
        seasonal = [by_series[s] for s in PERIODS if s in by_series]
        rows[detector] = {
            "detector": detector,
            "series": len(by_series),
            **{p: round(nab_score(list(by_series.values()), p), 2) for p in NAB_PROFILES},
            "seasonal_standard": round(nab_score(seasonal, "standard"), 2) if seasonal else np.nan,
        }
    return rows


def read_nab():
    path = OUT / "nab.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {row["detector"]: row for row in csv.DictReader(f)}


def run_univariate(series, detector):
    X, y = Data().get_data(series)
    span = X.max() - X.min()
    X = (X - X.min()) / (span if span > 0 else 1.0)
    model = UNIVARIATE[detector](len(X), PERIODS.get(series))
    start = time.perf_counter()
    with np.errstate(all="ignore"):
        scores = np.array([float(model.fit_score_partial(x)) for x in X])
    seconds = time.perf_counter() - start
    valid = np.isfinite(scores).all()
    if detector in NAB_UNIT_SCORES:
        nab = (scores, y)
    else:
        transform = ConformalProbabilityCalibrator(window_size=len(X)).fit_transform_partial
        with np.errstate(all="ignore"):
            nab = (np.array([transform(s) for s in scores]), y)
    scores, y = scores[warmup(len(X)) :], y[warmup(len(X)) :]
    both = valid and 0 < y.sum() < len(y)  # AUROC and AUPR need both kinds of points
    return {
        "_nab": nab,
        "dataset": series,
        "detector": detector,
        "seed": 0,
        "auroc": round(roc_auc_score(y, scores), 4) if both else np.nan,
        "aupr": round(average_precision_score(y, scores), 4) if both else np.nan,
        "ms_per_point": round(1000 * seconds / len(X), 3),
    }


def run_ucr(series, detector):
    """Runs a detector on a UCR series, with the arguments that it gets on NAB, and the raw scores."""
    X, y = Data().get_data(series)
    span = X.max() - X.min()
    X = (X - X.min()) / (span if span > 0 else 1.0)
    model = UNIVARIATE[detector](len(X), None)
    start = time.perf_counter()
    with np.errstate(all="ignore"):
        scores = np.array([float(model.fit_score_partial(x)) for x in X])
    seconds = time.perf_counter() - start
    valid = np.isfinite(scores).all()
    train = int(series.split("_")[-3])  # the training part has no anomalies, so it does not count
    anomaly = np.flatnonzero(y)
    top = train + np.argmax(scores[train:])
    scores, y = scores[train:], y[train:]
    return {
        "dataset": series,
        "detector": detector,
        "seed": 0,
        "auroc": round(roc_auc_score(y, scores), 4) if valid else np.nan,
        "aupr": round(average_precision_score(y, scores), 4) if valid else np.nan,
        "ms_per_point": round(1000 * seconds / len(X), 3),
        "solved": int(anomaly[0] - UCR_MARGIN <= top <= anomaly[-1] + UCR_MARGIN)
        if valid
        else np.nan,
    }


def run(job):
    dataset, detector, seed = job
    if dataset.startswith("ucr/"):
        return run_ucr(dataset, detector)
    if detector in UNIVARIATE:
        return run_univariate(dataset, detector)
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


def write_tables(results, nab=None):
    rows = sorted(
        results.values(),
        key=lambda r: (
            DATASETS.index(r["dataset"]) if r["dataset"] in DATASETS else len(DATASETS),
            r["dataset"],
            r["detector"],
        ),
    )
    write_csv("results.csv", FIELDS, [[r.get(f, "") for f in FIELDS] for r in rows])
    write_univariate_tables(
        {k: r for k, r in results.items() if k[1] in UNIVARIATE}, nab or read_nab()
    )
    results = {k: r for k, r in results.items() if k[1] in DETECTORS}
    detectors = [d for d in DETECTORS if any(k[1] == d for k in results)]

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
                commit_links(results, detectors[i]),
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
        facts = Data().describe(f"{ds}.mat")
        n = min(facts["points"], MAX_POINTS)
        datasets.append(
            [
                ds,
                f"{facts['points']:,}",
                f"{n:,}",
                f"{n - warmup(n):,}",
                facts["features"],
                f"{100 * facts['anomaly_rate']:.1f}",
            ]
        )
    write_csv(
        "datasets.csv",
        ["Dataset", "Points", "Points used", "Points evaluated", "Features", "Anomalies (%)"],
        datasets,
    )


def commit_links(results, detector):
    """Links to the commits that gave the results of a detector."""
    commits = sorted({r.get("commit") or "" for k, r in results.items() if k[1] == detector} - {""})
    return ", ".join(f"`{c} <{COMMIT_URL}{c.removesuffix('-dirty')}>`__" for c in commits)


def write_univariate_tables(results, nab):
    """The tables of the univariate page: a summary, and all the detectors on the seasonal series."""
    detectors = [d for d in UNIVARIATE if d in nab and any(k[1] == d for k in results)]
    if not detectors:
        return

    def mean_auroc(detector, series):
        rows = [results.get((s, detector, 0)) for s in series]
        return np.nanmean([number(r["auroc"]) for r in rows if r])

    def score(detector, field):
        return number(nab[detector].get(field))

    write_csv(
        "univariate.csv",
        [
            "Detector",
            "Series",
            "Standard",
            "Low false alerts",
            "Low misses",
            "AUROC",
            "Time (ms)",
            "Commit",
        ],
        [
            [
                f":class:`~pysad.models.{d}`",
                len(series_of(d)),
                *(f"{score(d, p):.1f}" for p in NAB_PROFILES),
                f"{mean_auroc(d, series_of(d)):.3f}",
                f"{np.median([number(r['ms_per_point']) for k, r in results.items() if k[1] == d]):.2f}",
                commit_links(results, d),
            ]
            for d in sorted(detectors, key=lambda d: -score(d, "standard"))
        ],
    )
    ucr = [d for d in detectors if d not in SEASONAL]
    write_csv(
        "univariate_ucr.csv",
        ["Detector", "Series", "Solved (%)", "AUROC"],
        sorted(
            (
                [
                    f":class:`~pysad.models.{d}`",
                    len(ucr_series()),
                    f"{100 * np.nanmean([number(results[s, d, 0]['solved']) for s in ucr_series()]):.1f}",
                    f"{mean_auroc(d, ucr_series()):.3f}",
                ]
                for d in ucr
                if all((s, d, 0) in results for s in ucr_series())
            ),
            key=lambda row: -float(row[2]),
        ),
    )
    write_csv(
        "univariate_seasonal.csv",
        ["Detector", "Standard", "AUROC"],
        [
            [
                f":class:`~pysad.models.{d}`",
                f"{score(d, 'seasonal_standard'):.1f}",
                f"{mean_auroc(d, PERIODS):.3f}",
            ]
            for d in sorted(detectors, key=lambda d: -score(d, "seasonal_standard"))
        ],
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
    common = pysad_files([Data, fix_seed, *CALIBRATORS.values()])
    if common & set(files) or "benchmarks/benchmark.py" in files or "pyproject.toml" in files:
        return [*DETECTORS, *UNIVARIATE]
    edited = set()  # detectors whose committed results changed, so that every row comes from a run
    if "docs/benchmark/results.csv" in files:
        before = {
            key(row): row
            for row in csv.DictReader(
                git("show", f"{base}:docs/benchmark/results.csv").splitlines()
            )
        }
        edited = {k[1] for k, row in read_results().items() if before.get(k) != row}
    return [
        d
        for d in [*DETECTORS, *UNIVARIATE]
        if pysad_files([getattr(models, d)]) & set(files) or d in edited
    ]


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

    detectors = args.detectors.split(",") if args.detectors else [*DETECTORS, *UNIVARIATE]
    if args.check:
        detectors = changed_detectors(args.check)
    if args.check is not None:
        print("Detectors to check:", ", ".join(detectors) or "none")
    unknown = set(detectors) - set(DETECTORS) - set(UNIVARIATE)
    if unknown:
        parser.error(f"unknown detectors: {', '.join(sorted(unknown))}")

    for dataset in DATASETS:  # download each file once, before the workers start
        Data().get_data(f"{dataset}.mat")
    nab_series()
    ucr_series()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
    jobs = [(ds, d, s) for d in detectors if d in DETECTORS for ds in DATASETS for s in seeds]
    if 0 in seeds:  # the univariate detectors are not random, so they run once, with seed 0
        jobs += [(series, d, 0) for d in detectors if d in UNIVARIATE for series in series_of(d)]
        jobs += [
            (s, d, 0)
            for d in detectors
            if d in UNIVARIATE and d not in SEASONAL
            for s in ucr_series()
        ]
    with ProcessPoolExecutor(args.jobs) as pool:
        new = {key(row): row for row in pool.map(run, jobs)}
    nab = nab_rows(new)

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
                    number(results[k].get(f)),
                    row.get(f, np.nan),
                    atol=ALERT_TOLERANCE,
                    equal_nan=True,
                )
                for f in ALERT_FIELDS
                if f.endswith("false_alerts")
            )
            or not np.isclose(
                number(results[k].get("solved")), row.get("solved", np.nan), equal_nan=True
            )
        ]
        for k, old, row in changed:
            before = f"{old['auroc']} / {old['aupr']}" if old else "missing"
            print(
                f"{k[1]} on {k[0]}, seed {k[2]}: AUROC / AUPR {before} -> {row['auroc']} / {row['aupr']}"
            )
        old_nab = read_nab()
        changed_nab = [
            d
            for d, row in nab.items()
            if d not in old_nab
            or any(
                not np.isclose(
                    number(old_nab[d].get(f)), row[f], atol=NAB_TOLERANCE, equal_nan=True
                )
                for f in NAB_FIELDS[2:]
            )
        ]
        for d in changed_nab:
            before = [old_nab.get(d, {}).get(f, "missing") for f in NAB_FIELDS[2:]]
            print(f"{d}: NAB scores {before} -> {[nab[d][f] for f in NAB_FIELDS[2:]]}")
        if changed or changed_nab:
            detectors = sorted({k[1] for k, _, _ in changed} | set(changed_nab))
            print(
                f"{len(changed) + len(changed_nab)} results changed. Run: python benchmarks/benchmark.py "
                f"--detectors {','.join(detectors)}, then commit docs/benchmark/."
            )
            sys.exit(1)
        print("The results in docs/benchmark/ are up to date.")
        return

    sha = commit()
    results.update({k: {**row, "commit": sha} for k, row in new.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    nab = {**read_nab(), **nab}
    write_csv(
        "nab.csv",
        NAB_FIELDS,
        [[nab[d].get(f, "") for f in NAB_FIELDS] for d in UNIVARIATE if d in nab],
    )
    write_tables(results, nab)


if __name__ == "__main__":
    main()
