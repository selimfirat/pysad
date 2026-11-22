"""Benchmark the detectors and the calibrators of PySAD on the datasets of ``pysad.utils.Data``.

The multivariate detectors and the calibrators run on the shuffled ODDS datasets, and the
multivariate detectors also on streams with drift in their time order. The univariate
detectors on the series of the Numenta Anomaly Benchmark (NAB) and the UCR Time Series Anomaly
Archive.

Run it from the repository root:

    python benchmarks/benchmark.py                        # every detector
    python benchmarks/benchmark.py --detectors LODA,RSHash
    python benchmarks/benchmark.py --dimensions           # latency per point against dimension
    python benchmarks/benchmark.py --check origin/master  # rerun what changed, compare with the results
    python benchmarks/benchmark.py --parts drift          # only the streams with drift
    python benchmarks/benchmark.py --ci base.csv --since origin/master  # the short run of CI
    python benchmarks/benchmark.py --custom path.py:MyModel --latex  # custom model with LaTeX output

The results go to docs/benchmark/, which the pages in docs/benchmarks/ show.
The dimensionality sweep writes docs/benchmark/dimensions.csv and
docs/benchmark/dimensions.svg instead, and needs no downloads: it runs on
synthetic uniform streams.
"""

import argparse
import ast
import csv
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from threadpoolctl import threadpool_limits

import pysad
from pysad import models
from pysad.core.base_model import BaseModel
from pysad.transform import probability_calibration
from pysad.transform.preprocessing import ShingleTransformer
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import Data, fix_seed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmark"
DATASETS = [name.removesuffix(".mat") for name in Data.datasets()]
MAX_POINTS = 5000  # the first points of each shuffled stream
# With --ci FILE, every run keeps only its first CI_POINTS points, and the results go to FILE
# instead of docs/benchmark/. CI runs the code of the base branch and the code of the pull request
# in this way, on the same machine, and compares the two files with --compare.
CI_POINTS = 2000
CI_FIELDS = ["dataset", "detector", "seed", "auroc", "aupr", "solved"]
WARMUP_POINTS = 256  # the scores of the first points do not count, see warmup()
# --check accepts this difference in AUROC. Other platforms round differently, and on a small
# dataset that can swap one anomaly with one normal point. AUPR moves too much for such a swap.
TOLERANCE = 0.005
# --check accepts this difference in a false alert rate, because one alert more or less on a small
# dataset moves the rate by about 0.01.
ALERT_TOLERANCE = 0.02
SIGNIFICANT_DIGITS = 9  # the scores keep this many significant digits, see significant()


# The multivariate detectors also run on these streams in their time order, to show how they follow
# drift. Each stream keeps every k-th point, so that at most DRIFT_POINTS points remain and every
# drift of the stream stays in. The streams run with seed 0 only, to keep the run short. The short
# names fit the tables of the page.
DRIFT = {
    "kddcup99/http.csv": "KDD http",
    "insects/abrupt.csv": "INSECTS A",
    "insects/incremental.csv": "INSECTS I",
    "insects/incremental_gradual.csv": "INSECTS IG",
    "insects/incremental_reoccurring.csv": "INSECTS IR",
    "smd/machine-1-1.txt": "SMD 1-1",
    "smd/machine-2-1.txt": "SMD 2-1",
    "smd/machine-3-1.txt": "SMD 3-1",
}
DRIFT_POINTS = 10_000


# Fallback attributes keep this script compatible with base checkouts before _is_detector / _input_kind existed.
_FALLBACK_NON_DETECTORS = frozenset(
    {"ModelEnsemble", "NullModel", "PerfectModel", "RandomModel", "StreamPipeline"}
)
_FALLBACK_INPUT_KINDS = {"HTM": "timestamped", "MIDAS": "edge"}


def _is_detector_model(name):
    cls = getattr(models, name)
    flag = getattr(cls, "_is_detector", None)
    if flag is not None:
        return bool(flag)
    return name not in _FALLBACK_NON_DETECTORS


def _model_input_kind(name):
    cls = getattr(models, name)
    getter = getattr(cls, "_effective_input_kind", None)
    if callable(getter) and name not in _FALLBACK_INPUT_KINDS:
        return getter()
    kind = getattr(cls, "_input_kind", None)
    if kind is not None and kind != "multivariate":
        return kind
    if name in _FALLBACK_INPUT_KINDS:
        return _FALLBACK_INPUT_KINDS[name]
    if hasattr(sys.modules[cls.__module__], "check_univariate"):
        return "univariate"
    return kind or "multivariate"


def _accepts_random_state(sig):
    return "random_state" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _required_param_names(cls_or_sig):
    getter = getattr(cls_or_sig, "_required_init_params", None)
    if callable(getter):
        return list(getter())
    sig = (
        cls_or_sig
        if isinstance(cls_or_sig, inspect.Signature)
        else inspect.signature(cls_or_sig.__init__)
    )
    return [
        p.name
        for p in sig.parameters.values()
        if p.name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def _make_multivariate_factory(cls):
    sig = inspect.signature(cls.__init__)
    required = _required_param_names(cls)
    seeded = _accepts_random_state(sig)

    def factory(X, seed):
        kwargs = {}
        for param in required:
            if param == "feature_mins":
                kwargs["feature_mins"] = np.zeros(X.shape[1])
            elif param == "feature_maxes":
                kwargs["feature_maxes"] = np.ones(X.shape[1])
            elif param == "input_shape":
                kwargs["input_shape"] = X.shape[1]
            elif param == "initial_X":
                kwargs["initial_X"] = X[: warmup(len(X))]
            else:
                raise TypeError(
                    f"{cls.__name__} requires {param!r}; add an entry to DETECTOR_OVERRIDES."
                )
        if seeded:
            kwargs["random_state"] = seed
        return cls(**kwargs)

    return factory


def _make_univariate_factory(cls):
    sig = inspect.signature(cls.__init__)
    required = _required_param_names(cls)
    has_period = "period" in sig.parameters
    seeded = _accepts_random_state(sig)

    def factory(n, period):
        kwargs = {}
        for param in required:
            if param == "min_val":
                kwargs["min_val"] = 0.0
            elif param == "max_val":
                kwargs["max_val"] = 1.0
            elif param == "period":
                kwargs["period"] = period
            elif param in ("window_length", "subsequence_length"):
                kwargs[param] = SHINGLE_SIZE
            elif param == "window_size":
                # A window of 3 periods and one point, because R's stl() needs more than 2 periods,
                # and an odd window makes the robust STL match R.
                kwargs["window_size"] = 3 * period + 1 if has_period else WARMUP_POINTS
            elif param == "probationary_period":
                kwargs["probationary_period"] = nab_probation(n)
            else:
                raise TypeError(
                    f"{cls.__name__} requires {param!r}; add an entry to UNIVARIATE_OVERRIDES."
                )
        # Pass period for detectors that have it (seasonal detectors), even if it has a default.
        # The benchmark only runs these on series with a known period.
        if has_period and "period" not in kwargs:
            kwargs["period"] = period
        if seeded:
            kwargs["random_state"] = 0
        return cls(**kwargs)

    return factory


# Non-default benchmark arguments; every other detector in pysad.models.__all__ is auto-discovered.
DETECTOR_OVERRIDES = {
    "ZeroShotEmbeddingDetector": lambda X, seed: models.ZeroShotEmbeddingDetector(
        semantic_anchors=np.random.RandomState(seed).randn(10, X.shape[1])
    ),
}
UNIVARIATE_OVERRIDES = {
    # HTM learns during NAB's probationary period, as in NAB.
    "HTM": lambda n, period: models.HTM(
        min_val=0.0, max_val=1.0, probationary_period=nab_probation(n), random_state=0
    ),
    # STL needs more than two periods. An odd window matches R's robust STL.
    "SeasonalESD": lambda n, period: models.SeasonalESD(period=period, window_size=3 * period + 1),
    "SeasonalHybridESD": lambda n, period: models.SeasonalHybridESD(
        period=period, window_size=3 * period + 1
    ),
    # Fix the length without labels, as for shingles, and initialize in the standard warm-up.
    "SAND": lambda n, period: models.SAND(
        subsequence_length=SHINGLE_SIZE, batch_size=WARMUP_POINTS, random_state=0
    ),
}

# Each detector gets its defaults, plus the arguments that it requires. X is scaled to [0, 1].
DETECTORS = {
    name: DETECTOR_OVERRIDES.get(name) or _make_multivariate_factory(getattr(models, name))
    for name in models.__all__
    if _is_detector_model(name) and _model_input_kind(name) == "multivariate"
}
# The per-dataset tables use these names, which are common in the papers, to fit the page.
SHORT_NAMES = {
    "HalfSpaceTrees": "HST",
    "LocalOutlierProbability": "LoOP",
    "RobustRandomCutForest": "RRCF",
}


# The univariate detectors run once on each NAB and UCR series, in file order, with seed 0 for
# stochastic detectors. Each detector gets its defaults, plus the arguments that it requires.
# The values of a series are scaled to [0, 1].
UNIVARIATE = {
    name: UNIVARIATE_OVERRIDES.get(name) or _make_univariate_factory(getattr(models, name))
    for name in models.__all__
    if _is_detector_model(name) and _model_input_kind(name) in ("univariate", "timestamped")
}
# The seasonal detectors need the period of a series, so they run only on these series. Each period is a
# day, in points, where the autocorrelation at a lag of one day is at least 0.5.
PERIODS = {
    "nab/artificialWithAnomaly/art_daily_flatmiddle.csv": 288,
    "nab/artificialWithAnomaly/art_daily_jumpsdown.csv": 288,
    "nab/artificialWithAnomaly/art_daily_jumpsup.csv": 288,
    "nab/artificialWithAnomaly/art_daily_nojump.csv": 288,
    "nab/realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv": 288,
    "nab/realAWSCloudwatch/grok_asg_anomaly.csv": 288,
    "nab/realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv": 288,
    "nab/realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv": 288,
    "nab/realAdExchange/exchange-2_cpc_results.csv": 24,
    "nab/realAdExchange/exchange-2_cpm_results.csv": 24,
    "nab/realAdExchange/exchange-3_cpm_results.csv": 24,
    "nab/realKnownCause/ambient_temperature_system_failure.csv": 24,
    "nab/realKnownCause/cpu_utilization_asg_misconfiguration.csv": 288,
    "nab/realKnownCause/nyc_taxi.csv": 48,
}
SEASONAL = [
    name
    for name in UNIVARIATE
    if "period" in inspect.signature(getattr(models, name).__init__).parameters
]
# These detectors take each point as [timestamp, value]. The UCR series have no timestamps, so these
# detectors do not run on them.
TIMESTAMPED = [name for name in UNIVARIATE if _model_input_kind(name) == "timestamped"]


@functools.cache
def nab_series():
    """All the NAB series, as NAB scores them. 6 of them have no anomalies, so their AUROC is NaN."""
    return Data.datasets("nab")


def series_of(detector):
    return list(PERIODS) if detector in SEASONAL else nab_series()


# The multivariate detectors also run on the NAB series with seed 0, once on the single values and
# once on shingles of SHINGLE_SIZE points, which give them the recent values of the series. The
# results name a run on shingles "<detector>+shingles".
SHINGLE_SIZE = 10
SHINGLED = "+shingles"
# The parts of a run, which --parts selects: the shuffled ODDS datasets, the univariate detectors on
# NAB and UCR, the streams with drift, and the multivariate detectors on NAB without and with
# shingles. CI runs each part in its own job.
PARTS = ["odds", "univariate", "drift", "nab", "shingles"]
NAB_DETECTORS = [*UNIVARIATE, *DETECTORS, *(d + SHINGLED for d in DETECTORS)]


def label(detector):
    """The name of a detector in the tables of the univariate page."""
    if detector.endswith(SHINGLED):
        return f":class:`~pysad.models.{detector.removesuffix(SHINGLED)}`, shingles"
    return f":class:`~pysad.models.{detector}`"


# The UCR series with at most this many points, 84 of the 250, keep the run short. The seasonal detectors
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


# Each general zero-argument calibrator turns the scores of every run into alerts at each threshold,
# as the alerts guide does: an alert when the calibrated score is at least the threshold.
# AnomalyLikelihood calibrates HTM's raw scores; SPOT needs 1,000 initial scores (longer than 7 ODDS datasets).
_EXCLUDED_CALIBRATORS = frozenset(
    {"AnomalyLikelihoodProbabilityCalibrator", "SPOTProbabilityCalibrator"}
)
_CALIBRATOR_KEYS = {"GaussianTailProbabilityCalibrator": "gaussian"}
_CALIBRATOR_LABELS = {"gaussian": "Gaussian tail"}  # short, to fit the page
CALIBRATORS = {
    _CALIBRATOR_KEYS.get(name, name.removesuffix("ProbabilityCalibrator").lower()): getattr(
        probability_calibration, name
    )
    for name in probability_calibration.__all__
    if name not in _EXCLUDED_CALIBRATORS
    and not _required_param_names(
        inspect.signature(getattr(probability_calibration, name).__init__)
    )
}
CALIBRATOR_NAMES = {
    key: _CALIBRATOR_LABELS.get(key, key.replace("_", " ").capitalize()) for key in CALIBRATORS
}
# "rate" is 1 minus the anomaly rate of the stream, so the alerts match the number of anomalies.
# It uses the labels, which a live stream does not have, so it shows the best case of a threshold.
THRESHOLDS = [0.95, 0.99, 0.995, "rate"]
ALERT_METRICS = ["false_alerts", "precision", "recall", "mcc"]
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


CACHE_DIR = Path.home() / ".cache" / "pysad" / "benchmark"


def _result_cache_paths(jobs: list[tuple], cache_dir: Path) -> dict[tuple, Path]:
    """Build cache paths from source contents, job settings, and runtime versions."""
    shared = pysad_files([Data, fix_seed, ShingleTransformer, ConformalProbabilityCalibrator])
    shared |= pysad_files(
        [getattr(probability_calibration, n) for n in probability_calibration.__all__]
    )
    package_root = Path(pysad.__file__).resolve().parents[1]
    runtime = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    common = hashlib.sha256()
    common.update(json.dumps([sys.version, platform.platform(), runtime]).encode())
    for path in [Path(__file__), ROOT / "pyproject.toml"]:
        common.update(path.read_bytes())
    fingerprints = {}
    paths = {}
    for job in jobs:
        if len(job) > 5:  # Custom models can import files outside PySAD.
            continue
        detector = job[1].removesuffix(SHINGLED)
        if detector not in fingerprints:
            digest = common.copy()
            for name in sorted(shared | pysad_files([getattr(models, detector)])):
                digest.update(name.encode())
                digest.update((package_root / name).read_bytes())
            fingerprints[detector] = digest.digest()
        digest = hashlib.sha256(fingerprints[detector])
        digest.update(json.dumps(job).encode())
        paths[job] = cache_dir / (digest.hexdigest() + ".json")
    return paths


def _read_cached_result(path: Path) -> dict | None:
    """Read one cached row, or return None if the file is missing or damaged."""
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict) or not set(CI_FIELDS[:-1]) <= row.keys():
            return None
        if "_nab" in row:
            row["_nab"] = tuple(np.asarray(a) for a in row["_nab"])
        return row
    except (OSError, ValueError, TypeError):
        return None


def _write_cached_result(path: Path, row: dict) -> None:
    """Save a complete result before another run can read it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        temporary = Path(f.name)
        try:
            json.dump(row, f, default=lambda value: value.tolist())
            f.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _load_custom_model(spec):
    """Load a custom model class from a file path and class name.

    Args:
        spec (str): Specification in the format "path/to/file.py:ClassName".

    Returns:
        type: The model class.

    Raises:
        ValueError: If the spec format is invalid, the file does not exist,
            the class is not found, or the class does not inherit from BaseModel.
    """
    if ":" not in spec:
        raise ValueError(
            f"Custom model spec must be in format 'path/to/file.py:ClassName', got '{spec}'"
        )
    file_path, class_name = spec.rsplit(":", 1)
    path = Path(file_path).resolve()
    if not path.is_file():
        raise ValueError(f"Custom model file not found: {path}")
    module_name = path.stem
    spec_obj = importlib.util.spec_from_file_location(module_name, path)
    if spec_obj is None or spec_obj.loader is None:
        raise ValueError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise ValueError(f"Class '{class_name}' not found in {path}")
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseModel):
        raise ValueError(f"Class '{class_name}' must inherit from BaseModel")
    return cls


def _make_custom_factory(cls, input_kind):
    """Create a factory function for a custom model.

    Args:
        cls (type): The model class.
        input_kind (str): The input kind ("multivariate" or "univariate").

    Returns:
        callable: A factory function that creates model instances.
    """
    sig = inspect.signature(cls.__init__)
    required = _required_param_names(cls)
    seeded = _accepts_random_state(sig)

    if input_kind == "multivariate":

        def factory(X, seed):
            kwargs = {}
            for param in required:
                if param == "feature_mins":
                    kwargs["feature_mins"] = np.zeros(X.shape[1])
                elif param == "feature_maxes":
                    kwargs["feature_maxes"] = np.ones(X.shape[1])
                elif param == "input_shape":
                    kwargs["input_shape"] = X.shape[1]
                elif param == "initial_X":
                    kwargs["initial_X"] = X[: warmup(len(X))]
                else:
                    raise TypeError(
                        f"{cls.__name__} requires {param!r}; pass it via constructor defaults."
                    )
            if seeded:
                kwargs["random_state"] = seed
            return cls(**kwargs)

    else:  # univariate or timestamped

        def factory(n, period):
            kwargs = {}
            for param in required:
                if param == "min_val":
                    kwargs["min_val"] = 0.0
                elif param == "max_val":
                    kwargs["max_val"] = 1.0
                elif param == "period":
                    kwargs["period"] = period
                elif param in ("window_length", "subsequence_length"):
                    kwargs[param] = SHINGLE_SIZE
                elif param == "window_size":
                    has_period = "period" in sig.parameters
                    kwargs["window_size"] = 3 * period + 1 if has_period else WARMUP_POINTS
                elif param == "probationary_period":
                    kwargs["probationary_period"] = nab_probation(n)
                else:
                    raise TypeError(
                        f"{cls.__name__} requires {param!r}; pass it via constructor defaults."
                    )
            if hasattr(cls, "period") and "period" not in kwargs and period is not None:
                kwargs["period"] = period
            if seeded:
                kwargs["random_state"] = 0
            return cls(**kwargs)

    return factory


def _write_latex_summary(results, detectors, out_path):
    """Write a LaTeX summary table of AUROC results.

    Args:
        results (dict): Benchmark results keyed by (dataset, detector, seed).
        detectors (list): List of detector names in summary order (from write_ranking).
        out_path (Path): Output file path for the LaTeX table.
    """
    # Get all detectors from results, not just built-in ones
    all_detectors = sorted({k[1].removesuffix(SHINGLED) for k in results if k[0] in DATASETS})
    if not all_detectors:
        return

    datasets = [ds for ds in DATASETS if any(k[0] == ds for k in results)]
    if not datasets:
        return

    def mean_auroc(dataset, detector):
        vals = []
        for k, r in results.items():
            if k[:2] == (dataset, detector):
                auroc_val = r.get("auroc")
                if auroc_val is not None and auroc_val != "":
                    try:
                        val = float(auroc_val)
                        if not np.isnan(val):
                            vals.append(val)
                    except (ValueError, TypeError):
                        pass
        return np.mean(vals) if vals else np.nan

    table = np.array([[mean_auroc(ds, d) for d in all_detectors] for ds in datasets])

    mean_auroc_vals = np.nanmean(table, axis=0)
    order = np.argsort(-mean_auroc_vals)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Benchmark AUROC summary (higher is better).}")
    lines.append(r"\label{tab:benchmark-auroc}")
    cols = "l" + "c" * len(all_detectors)
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")
    header = ["Dataset"] + [SHORT_NAMES.get(all_detectors[i], all_detectors[i]) for i in order]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for j, ds in enumerate(datasets):
        row = [ds]
        for i in order:
            val = table[j, i]
            if np.isnan(val):
                row.append("--")
            else:
                cell = f"{val:.3f}"
                if val == np.nanmax(table[j]) and not np.isnan(np.nanmax(table[j])):
                    cell = rf"\textbf{{{cell}}}"
                if val <= 0.5:
                    cell = rf"\textit{{{cell}}}"
                row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def significant(scores):
    """The scores rounded to SIGNIFICANT_DIGITS significant digits.

    Platforms round matrix products and functions such as ``log`` differently in the last bits. Many
    series repeat a few values, so a detector gives many equal scores, and such a difference changes
    the order of these ties and so the metrics. The rounding keeps the ties equal.
    """
    return np.array([float(f"{s:.{SIGNIFICANT_DIGITS}g}") for s in scores])


def warmup(n):
    """The number of first points of a stream whose scores do not count in the metrics.

    A detector learns from these points, but it has little or no model yet when it scores them.
    It is 256 points, or a quarter of a stream with fewer than 1,024 points.
    """
    return min(WARMUP_POINTS, n // 4)


_WORKER_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] | None = None


def _init_worker():
    """Enable per-process dataset caching inside benchmark worker processes."""
    global _WORKER_CACHE
    _WORKER_CACHE = {}


def load(dataset, seed, points=None):
    cache_key = ("load", dataset, None if dataset in DRIFT else seed, points)
    if _WORKER_CACHE is not None and cache_key in _WORKER_CACHE:
        cached_X, cached_y = _WORKER_CACHE[cache_key]
        return cached_X.copy(), cached_y.copy()
    if dataset in DRIFT:
        X, y = Data().get_data(dataset)
        step = -(-len(X) // DRIFT_POINTS)  # the smallest step that keeps at most DRIFT_POINTS
        X, y = X[::step][:points], y[::step][:points]
        mins, maxes = X.min(axis=0), X.max(axis=0)
        result = ((X - mins) / np.where(maxes > mins, maxes - mins, 1.0), y)
    else:
        result = Data().get_stream(
            f"{dataset}.mat", random_state=seed, max_points=points or MAX_POINTS, scale=True
        )
    if _WORKER_CACHE is not None:
        _WORKER_CACHE[cache_key] = result
        return result[0].copy(), result[1].copy()
    return result


def anomaly_fraction(dataset):
    """The fraction of anomalies among the evaluated points of a dataset.

    The first points are the warm-up and do not count in the metrics. The value is the reference
    of a random detector for AUPR on this dataset.
    """
    _, y = load(dataset, 0)
    return float(y[warmup(len(y)) :].mean())


def nab_probation(n):
    """NAB's probationary period of a series of n points: 15%, at most 750. NAB does not score it."""
    return min(n * 15 // 100, 750)


# These detectors give scores from 0 to 1, as NAB's detectors do. The scores of the others have
# another scale on each series, so they are calibrated before NAB uses one threshold for all. The
# calibrator compares each score with all the earlier scores of the series. With the default window
# of 300 scores, its values have steps of 1/300, and many points tie at the highest value.
NAB_UNIT_SCORES = ["HTM", "KNNCAD", "RelativeEntropy"]

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


def score_stream(model: BaseModel, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Collect rounded scores and measure milliseconds per point.

    Args:
        model (BaseModel): Score each point before learning from it.
        X (np.ndarray): Stream with one point per row.

    Returns:
        tuple[np.ndarray, float]: Rounded scores and time per point. Timing excludes score rounding.
    """
    start = time.perf_counter()
    with np.errstate(all="ignore"):
        scores = np.array([float(model.fit_score_partial(x)) for x in X])
    seconds = time.perf_counter() - start
    return significant(scores), round(1000 * seconds / len(X), 3)


def _load_univariate_raw(series, timestamps):
    cache_key = ("univariate", series, timestamps)
    if _WORKER_CACHE is not None and cache_key in _WORKER_CACHE:
        cached_X, cached_y = _WORKER_CACHE[cache_key]
        return cached_X.copy(), cached_y.copy()
    # The base branch's Data.get_data has no timestamps argument, so pass it only when needed.
    X, y = Data().get_data(series, timestamps=True) if timestamps else Data().get_data(series)
    if _WORKER_CACHE is not None:
        _WORKER_CACHE[cache_key] = (X, y)
        return X.copy(), y.copy()
    return X, y


def run_univariate(
    series, detector, points=None, ci=False, model_type=None, input_kind=None, custom_spec=None
):
    X, y = _load_univariate_raw(series, detector in TIMESTAMPED)
    X, y = X[:points].copy(), y[:points]
    values = X[:, -1]  # the timestamps, if any, come first
    span = values.max() - values.min()
    X[:, -1] = (values - values.min()) / (span if span > 0 else 1.0)

    if model_type == "custom":
        cls = _load_custom_model(custom_spec)
        factory = _make_custom_factory(cls, input_kind)
        model = factory(len(X), PERIODS.get(series))
    elif detector in UNIVARIATE:
        model = UNIVARIATE[detector](len(X), PERIODS.get(series))
    else:
        if detector.endswith(SHINGLED):
            X = ShingleTransformer(SHINGLE_SIZE).fit_transform(X)
        fix_seed(0)
        model = DETECTORS[detector.removesuffix(SHINGLED)](X, 0)

    scores, ms_per_point = score_stream(model, X)
    valid = np.isfinite(scores).all()
    if not ci:
        if detector in NAB_UNIT_SCORES:
            nab = (scores, y)
        else:
            with np.errstate(all="ignore"):
                nab = (ConformalProbabilityCalibrator(window_size=len(X)).fit_transform(scores), y)
    scores, y = scores[warmup(len(X)) :], y[warmup(len(X)) :]
    both = valid and 0 < y.sum() < len(y)  # AUROC and AUPR need both kinds of points
    row = {
        "dataset": series,
        "detector": detector,
        "seed": 0,
        "auroc": round(roc_auc_score(y, scores), 4) if both else np.nan,
        "aupr": round(average_precision_score(y, scores), 4) if both else np.nan,
        "ms_per_point": ms_per_point,
    }
    if not ci:
        row["_nab"] = nab
    return row


def run_ucr(series, detector, model_type=None, input_kind=None, custom_spec=None):
    """Runs a detector on a UCR series, with the arguments that it gets on NAB, and the raw scores."""
    X, y = Data().get_data(series)
    span = X.max() - X.min()
    X = (X - X.min()) / (span if span > 0 else 1.0)

    if model_type == "custom":
        cls = _load_custom_model(custom_spec)
        factory = _make_custom_factory(cls, input_kind)
        model = factory(len(X), None)
    else:
        model = UNIVARIATE[detector](len(X), None)

    scores, ms_per_point = score_stream(model, X)
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
        "ms_per_point": ms_per_point,
        "solved": int(anomaly[0] - UCR_MARGIN <= top <= anomaly[-1] + UCR_MARGIN)
        if valid
        else np.nan,
    }


def run(job):
    dataset, detector, seed, points, *rest = job
    ci = bool(rest[0]) if rest else False
    model_type = rest[1] if len(rest) > 1 else None
    input_kind = rest[2] if len(rest) > 2 else None
    custom_spec = rest[3] if len(rest) > 3 else None
    # One thread for BLAS and OpenMP in each worker, so that the workers do not compete for the cores
    # and the times per point stay comparable. The scores do not change.
    with threadpool_limits(1):
        return _run(
            dataset,
            detector,
            seed,
            points,
            ci=ci,
            model_type=model_type,
            input_kind=input_kind,
            custom_spec=custom_spec,
        )


def _run(
    dataset, detector, seed, points, ci=False, model_type=None, input_kind=None, custom_spec=None
):
    if dataset.startswith("ucr/"):
        return run_ucr(
            dataset, detector, model_type=model_type, input_kind=input_kind, custom_spec=custom_spec
        )
    if dataset.startswith("nab/"):
        return run_univariate(
            dataset,
            detector,
            points,
            ci=ci,
            model_type=model_type,
            input_kind=input_kind,
            custom_spec=custom_spec,
        )

    X, y = load(dataset, seed, points)
    fix_seed(seed)

    if model_type == "custom":
        # Custom model: load from spec in worker
        cls = _load_custom_model(custom_spec)
        factory = _make_custom_factory(cls, input_kind)
        model = factory(X, seed)
    else:
        model = DETECTORS[detector](X, seed)

    scores, ms_per_point = score_stream(model, X)
    valid = np.isfinite(scores).all()
    row = {"dataset": dataset, "detector": detector, "seed": seed}
    calibrators = [] if (ci or dataset in DRIFT) else CALIBRATORS.items()
    for name, calibrator in calibrators:
        with np.errstate(all="ignore"):
            calibrated = calibrator().fit_transform(scores)
        row.update(alerts(name, calibrated[warmup(len(X)) :], y[warmup(len(X)) :], valid))
    scores, y = scores[warmup(len(X)) :], y[warmup(len(X)) :]
    both = valid and 0 < y.sum() < len(y)  # AUROC and AUPR need both kinds of points
    return {
        **row,
        "auroc": round(roc_auc_score(y, scores), 4) if both else np.nan,
        "aupr": round(average_precision_score(y, scores), 4) if both else np.nan,
        "ms_per_point": ms_per_point,
    }


def alerts(calibrator, calibrated, y, valid):
    """Return alert rates, precision, recall, and MCC at each threshold."""
    row = {}
    for t in THRESHOLDS:
        alert = calibrated >= (1 - y.mean() if t == "rate" else t)
        values = [
            alert[y == 0].mean(),
            (alert & (y == 1)).sum() / alert.sum() if alert.any() else np.nan,
            alert[y == 1].mean(),
            matthews_corrcoef(y, alert) if valid else np.nan,
        ]
        for metric, value in zip(ALERT_METRICS, values, strict=True):
            row[f"{calibrator}_{t}_{metric}"] = round(value, 4) if valid else np.nan
    return row


def key(row):
    return row["dataset"], row["detector"], int(row["seed"])


def read_results(path=OUT / "results.csv"):
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
        {k: r for k, r in results.items() if k[0].startswith(("nab/", "ucr/"))}, nab or read_nab()
    )
    write_ranking({k: r for k, r in results.items() if k[0] in DRIFT}, DRIFT, "drift_")
    results = {k: r for k, r in results.items() if k[0] in DATASETS}
    detectors = write_ranking(results, {ds: ds for ds in DATASETS}, "")

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
            "MCC",
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
    streams = []
    for stream in DRIFT:
        X, y = load(stream, 0)
        streams.append(
            [
                stream,
                f"{Data().describe(stream)['points']:,}",
                f"{len(X):,}",
                X.shape[1],
                f"{100 * y.mean():.1f}",
            ]
        )
    write_csv(
        "drift_streams.csv",
        ["Stream", "Points", "Points used", "Features", "Anomalies (%)"],
        streams,
    )


def write_ranking(results, datasets, prefix):
    """The summary, the AUROC and the AUPR tables of the multivariate detectors.

    ``datasets`` maps each dataset to its name in the tables.
    Italics mark AUROC values at or below 0.5 and AUPR values at or below the anomaly fraction.
    The chance comparison rounds the anomaly fraction to four decimal places, like stored metrics.
    Bold marks the best value in each row.
    """
    detectors = [d for d in DETECTORS if any(k[1] == d for k in results)]
    if not detectors:
        return detectors

    def mean(metric, dataset, detector):
        return np.mean(
            [float(r[metric]) for k, r in results.items() if k[:2] == (dataset, detector)]
        )

    fractions = np.array([anomaly_fraction(ds) for ds in datasets])
    tables = {
        metric: np.array([[mean(metric, ds, d) for d in detectors] for ds in datasets])
        for metric in ["auroc", "aupr"]
    }
    rank = np.mean([rankdata(-np.nan_to_num(row)) for row in tables["auroc"]], axis=0)
    ms = [
        np.median([float(r["ms_per_point"]) for k, r in results.items() if k[1] == d])
        for d in detectors
    ]
    auroc = np.nanmean(tables["auroc"], axis=0)
    # Rank 1 is the highest AUROC of a dataset. Equal mean ranks, as the page rounds them, go by
    # mean AUROC, then time per point. AUPR is not averaged over datasets, because its value for a
    # random detector is the anomaly rate of the dataset, which is different on each dataset.
    order = sorted(range(len(detectors)), key=lambda i: (round(rank[i], 1), -auroc[i], ms[i]))
    write_csv(
        f"{prefix}summary.csv",
        ["Detector", "Mean rank", "Mean AUROC", "Time per point (ms)", "Commit"],
        [
            [
                f":class:`~pysad.models.{detectors[i]}`",
                f"{rank[i]:.1f}",
                f"{auroc[i]:.3f}",
                f"{ms[i]:.2f}",
                commit_links(results, detectors[i]),
            ]
            for i in order
        ],
    )
    for metric, table in tables.items():  # the columns in the order of the summary
        rows = []
        for j, (ds, row) in enumerate(zip(datasets, table[:, order], strict=True)):
            cells = [f"{v:.3f}" for v in row]
            best = f"{np.nanmax(row):.3f}"
            marked = []
            for c, v in zip(cells, row, strict=True):
                # Metrics in results.csv have four decimal places.
                below = v <= (0.5 if metric == "auroc" else round(fractions[j], 4))
                if c == best and below:
                    marked.append(f":strong-emphasis:`{c}`")
                elif c == best:
                    marked.append(f"**{c}**")
                elif below:
                    marked.append(f"*{c}*")
                else:
                    marked.append(c)
            if metric == "aupr":
                rows.append([datasets[ds], f"{fractions[j]:.3f}", *marked])
            else:
                rows.append([datasets[ds], *marked])
        header = [SHORT_NAMES.get(detectors[i], detectors[i]) for i in order]
        if metric == "aupr":
            header = ["Dataset", "Fraction of anomalies", *header]
        else:
            header = ["Dataset", *header]
        write_csv(f"{prefix}{metric}.csv", header, rows)
    return detectors


def commit_links(results, detector):
    """Links to the commits that gave the results of a detector."""
    commits = sorted({r.get("commit") or "" for k, r in results.items() if k[1] == detector} - {""})
    return ", ".join(f"`{c} <{COMMIT_URL}{c.removesuffix('-dirty')}>`__" for c in commits)


def write_univariate_tables(results, nab):
    """The tables of the univariate page: a summary, and all the detectors on the seasonal series."""
    detectors = [d for d in NAB_DETECTORS if d in nab and any(k[1] == d for k in results)]
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
                label(d),
                len(series_of(d)),
                *(f"{score(d, p):.1f}" for p in NAB_PROFILES),
                f"{mean_auroc(d, series_of(d)):.3f}",
                f"{np.median([number(r['ms_per_point']) for k, r in results.items() if k[1] == d]):.2f}",
                commit_links(results, d),
            ]
            for d in sorted(detectors, key=lambda d: -score(d, "standard"))
        ],
    )
    ucr = [d for d in detectors if d in UNIVARIATE and d not in SEASONAL + TIMESTAMPED]
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
                label(d),
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
                    f"{np.nanmean(p):.3f}" if np.isfinite(p).any() else "",  # no run had alerts
                    f"{np.nanmean(r):.3f}",
                    f"{np.nanmean(f1):.3f}",
                    f"{np.nanmean(values['mcc']):.3f}" if np.isfinite(values["mcc"]).any() else "",
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


_DETECTOR_TABLES = {
    "DETECTORS",
    "DETECTOR_OVERRIDES",
    "UNIVARIATE",
    "UNIVARIATE_OVERRIDES",
    "SHORT_NAMES",
    "SEASONAL",
    "TIMESTAMPED",
    "NAB_UNIT_SCORES",
}


def _parse_detector_tables(source):
    """The detector registration tables in a benchmark.py source string, and the rest of its AST."""
    tree = ast.parse(source)
    tables, body = {}, []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in _DETECTOR_TABLES
            and isinstance(node.value, ast.Dict | ast.List)
        ):
            name = node.targets[0].id
            if isinstance(node.value, ast.Dict):
                tables[name] = {
                    ast.literal_eval(k): ast.dump(v)
                    for k, v in zip(node.value.keys, node.value.values, strict=True)
                }
            else:
                tables[name] = {ast.literal_eval(elt) for elt in node.value.elts}
        else:
            body.append(node)
    tree.body = body
    return tables, ast.dump(tree)


def _changed_benchmark_detectors(base_source, head_source):
    """Detectors changed between two benchmark.py sources, or None if shared benchmark code changed."""
    base_tables, base_rest = _parse_detector_tables(base_source)
    head_tables, head_rest = _parse_detector_tables(head_source)
    if base_rest != head_rest:
        return None
    changed = set()
    for name in _DETECTOR_TABLES:
        before, after = base_tables.get(name, {}), head_tables.get(name, {})
        if isinstance(before, dict) and isinstance(after, dict):
            changed |= {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
        else:
            changed |= set(before) ^ set(after)
    return changed


def _runtime_toml(source):
    """The build, project, and runtime dependency sections of pyproject.toml."""
    return source.split("\ntest = [", 1)[0]


def changed_detectors(base, path=None, ci=False):
    """The detectors whose results a change since ``base`` can move.

    With ``path``, also the detectors whose committed results in ``path`` changed since ``base``.
    """
    files = git("diff", "--name-only", base).split()
    shared = [Data, fix_seed] if ci else [Data, fix_seed, ShingleTransformer, *CALIBRATORS.values()]
    common = pysad_files(shared)
    if common & set(files):
        return [*DETECTORS, *UNIVARIATE]
    if "pyproject.toml" in files and _runtime_toml(
        git("show", f"{base}:pyproject.toml")
    ) != _runtime_toml((ROOT / "pyproject.toml").read_text(encoding="utf-8")):
        return [*DETECTORS, *UNIVARIATE]
    edited = set()  # detectors whose committed results or registration changed
    if "benchmarks/benchmark.py" in files:
        bench_changed = _changed_benchmark_detectors(
            git("show", f"{base}:benchmarks/benchmark.py"),
            Path(__file__).read_text(encoding="utf-8"),
        )
        if bench_changed is None:
            return [*DETECTORS, *UNIVARIATE]
        edited |= bench_changed
    name = path and path.relative_to(ROOT).as_posix()
    if name in files:
        before = {
            key(row): row for row in csv.DictReader(git("show", f"{base}:{name}").splitlines())
        }
        edited |= {
            k[1].removesuffix(SHINGLED)
            for k, row in read_results(path).items()
            if before.get(k) != row
        }
    return [
        d
        for d in [*DETECTORS, *UNIVARIATE]
        if pysad_files([getattr(models, d)]) & set(files) or d in edited
    ]


def progress(done, total, seconds, remaining):
    """Prints how many runs are done, and which runs remain, so that a slow run shows in the log."""
    line = f"{done}/{total} runs done ({100 * done // total}%) after {seconds:.0f} s."
    if remaining and len(remaining) <= 5:
        line += " Remaining: " + ", ".join(f"{d} on {ds}" for ds, d, *_ in remaining) + "."
    elif remaining:
        counts = Counter(d for _, d, *_ in remaining).most_common()
        line += " Remaining: " + ", ".join(f"{d} {n}" for d, n in counts) + "."
    print(line, flush=True)


def compare(before, after):
    """Prints the runs whose AUROC or solved series differ between two --ci runs.

    In GitHub Actions, it also adds them to the summary of the job as a table.
    """
    moved = []
    for k, row in sorted(after.items()):
        old = before.get(k)
        if old is None:
            moved.append((k, "new", row["auroc"]))
        elif not np.isclose(
            number(old["auroc"]), number(row["auroc"]), atol=TOLERANCE, equal_nan=True
        ) or not np.isclose(
            number(old.get("solved")), number(row.get("solved", np.nan)), equal_nan=True
        ):
            moved.append((k, old["auroc"], row["auroc"]))
    lines = [f"| {k[1]} | {k[0]} | {a} | {b} |" for k, a, b in moved]
    report = (
        [
            f"{len(moved)} of {len(after)} runs moved.",
            "",
            "| Detector | Dataset | AUROC before | AUROC after |",
            "| --- | --- | --- | --- |",
            *lines,
        ]
        if moved
        else [f"No run moved, out of {len(after)}."]
    )
    print("\n".join(report))
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("\n".join(report) + "\n")


def pysad_files(objects):
    """The pysad source files that define ``objects`` and the pysad names that those files import."""
    files, todo = set(), [obj.__module__ for obj in objects]
    while todo:
        module = sys.modules[todo.pop()]
        # relative to the folder of the installed pysad, which can be another checkout in CI
        path = Path(module.__file__).relative_to(Path(pysad.__file__).parents[1]).as_posix()
        if path in files:
            continue
        files.add(path)
        if path.endswith("/__init__.py"):
            continue
        for value in vars(module).values():
            name = (
                value.__name__
                if isinstance(value, ModuleType)
                else getattr(value, "__module__", "")
            )
            if isinstance(name, str) and name.split(".")[0] == "pysad":
                todo.append(name)
        for node in ast.walk(ast.parse(Path(module.__file__).read_text(encoding="utf-8"))):
            target = sys.modules.get(node.module) if isinstance(node, ast.ImportFrom) else None
            if (
                target is not None
                and target.__name__.split(".")[0] == "pysad"
                and target.__file__ is not None
                and not target.__file__.endswith("__init__.py")
            ):
                todo.append(target.__name__)
    return files


# The dimensionality sweep (--dimensions) runs these detectors on synthetic
# uniform streams of increasing feature dimension, to show how latency per point
# scales with d. Every detector sees the same stream at a given dimension, so the
# comparison does not depend on the order of the runs.
DIMENSION_DETECTORS = ["LODA", "xStream", "RSHash", "HalfSpaceTrees", "Inqmad", "ExactStorm"]
DIMENSIONS = [2, 10, 50, 100, 500, 1000]
DIMENSION_POINTS = 2000
DIMENSION_FIELDS = ["detector", "dimension", "points", "ms_per_point"]
DIMENSION_COLORS = ["#00757a", "#5c6bc0", "#c26d1f", "#2a9d9f", "#64748b", "#e63946"]
DIMENSION_COLORS_DARK = ["#2dd4bf", "#818cf8", "#f59e0b", "#99f6e4", "#94a3b8", "#f45b69"]


def dimension_model(detector, d, seed):
    """A detector for a synthetic stream of d features in [0, 1].

    The array carries only the shape and the [0, 1] ranges that
    HalfSpaceTrees, RSHash, and Inqmad need; the stream itself is
    generated one point at a time by run_dimension.
    """
    return DETECTORS[detector](np.zeros((1, d)), seed)


def run_dimension(detector, d, points, seed):
    """Latency per point of a detector on a synthetic stream of d features.

    A fixed warmup run first, so even short timed runs exclude setup costs,
    then the timed points, each scored before the model fits it, as in
    the rest of the benchmark.
    """
    model = dimension_model(detector, d, seed)
    rng = np.random.RandomState(seed)
    for _ in range(WARMUP_POINTS):
        model.fit_score_partial(rng.rand(d))
    start = time.perf_counter()
    with np.errstate(all="ignore"):
        for _ in range(points):
            model.fit_score_partial(rng.rand(d))
    seconds = time.perf_counter() - start
    return {
        "detector": detector,
        "dimension": d,
        "points": points,
        "ms_per_point": round(1000 * seconds / points, 3),
    }


def dimension_rows(detectors, dimensions, points, seed):
    """One latency row per detector and dimension, in detector order."""
    return [run_dimension(detector, d, points, seed) for detector in detectors for d in dimensions]


def write_dimensions_svg(rows, detectors):
    """Scalability curves: ms per point against dimension, log scale on d.

    Written with the standard library only, so the sweep adds no
    dependency: one line per detector, with a tick at each dimension.
    """
    width, height = 640, 400
    left, right, top, bottom = 70, 150, 30, 50
    plot_w, plot_h = width - left - right, height - top - bottom
    dims = sorted({row["dimension"] for row in rows})
    max_ms = max(row["ms_per_point"] for row in rows)

    def x(d):
        span = np.log10(max(dims)) - np.log10(min(dims))
        return left + plot_w * (np.log10(d) - np.log10(min(dims))) / span

    def y(ms):
        return top + plot_h - plot_h * ms / (max_ms * 1.1)

    sans = 'Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif'
    mono = '"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace'
    light_rules = "".join(f".s{i}{{stroke:{color}}}" for i, color in enumerate(DIMENSION_COLORS))
    dark_rules = "".join(
        f".s{i}{{stroke:{color}}}" for i, color in enumerate(DIMENSION_COLORS_DARK)
    )
    style = (
        f"text{{font-family:{sans}}}"
        f".title{{fill:#0d1b22;font-size:14px;font-weight:600}}"
        f".axis{{fill:#5b6f78;font-size:12px;font-weight:500}}"
        f".tick{{fill:#5b6f78;font-family:{mono};font-size:11px}}"
        f".legend{{fill:#0d1b22;font-size:12px;font-weight:500}}"
        f".grid{{stroke:#dce3e2;stroke-width:1}}"
        f"{light_rules}"
        f"@media (prefers-color-scheme:dark){{"
        f".title,.legend{{fill:#edf5f6}}"
        f".axis,.tick{{fill:#99b0b6}}"
        f".grid{{stroke:#1e343d}}"
        f"{dark_rules}"
        f"}}"
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="Latency per point by dimension">',
        f"<style>{style}</style>",
        f'<text x="{width // 2}" y="20" text-anchor="middle" class="title">'
        "Latency per point by dimension</text>",
    ]
    for ms in np.linspace(0, max_ms, 6):
        lines.append(
            f'<line x1="{left}" y1="{y(ms):.1f}" x2="{left + plot_w}" y2="{y(ms):.1f}" class="grid"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y(ms) + 4:.1f}" text-anchor="end" class="tick">{ms:.3g}</text>'
        )
    for d in dims:
        lines.append(
            f'<line x1="{x(d):.1f}" y1="{top}" x2="{x(d):.1f}" y2="{top + plot_h}" class="grid"/>'
        )
        lines.append(
            f'<text x="{x(d):.1f}" y="{top + plot_h + 20}" text-anchor="middle" class="tick">{d}</text>'
        )
    lines.append(
        f'<text x="{left + plot_w // 2}" y="{height - 10}" text-anchor="middle" class="axis">'
        "dimension d (log scale)</text>"
    )
    lines.append(
        f'<text x="20" y="{top + plot_h // 2}" text-anchor="middle" '
        f'transform="rotate(-90 20 {top + plot_h // 2})" class="axis">ms per point</text>'
    )
    for i, detector in enumerate(detectors):
        cls = f"s{i % len(DIMENSION_COLORS)}"
        cells = [row for row in rows if row["detector"] == detector]
        points_attr = " ".join(f"{x(r['dimension']):.1f},{y(r['ms_per_point']):.1f}" for r in cells)
        lines.append(
            f'<polyline fill="none" class="{cls}" stroke-width="2" points="{points_attr}"/>'
        )
        legend_y = top + 15 * i
        lines.append(
            f'<line x1="{left + plot_w + 10}" y1="{legend_y}" x2="{left + plot_w + 30}" '
            f'y2="{legend_y}" class="{cls}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{left + plot_w + 35}" y="{legend_y + 4}" class="legend">{detector}</text>'
        )
    lines.append("</svg>")
    with open(OUT / "dimensions.svg", "w") as f:
        f.write("\n".join(lines) + "\n")


_FALLBACK_PART_WEIGHTS: dict[str, dict[str, float]] = {
    "odds": {
        "PCBIsolationForest": 180.0,
        "MiLOF": 50.0,
        "IDKS": 35.0,
        "GNGA": 25.0,
        "FRAHST": 20.0,
        "FeedbackHST": 18.0,
        "CostSensitiveStreamModel": 15.0,
        "SDOstream": 15.0,
    },
    "drift": {
        "PCBIsolationForest": 140.0,
        "MiLOF": 25.0,
        "IDKS": 20.0,
        "GNGA": 15.0,
        "FRAHST": 12.0,
        "FeedbackHST": 10.0,
        "CostSensitiveStreamModel": 8.0,
        "SDOstream": 8.0,
    },
    "univariate": {
        "OeSNNUAD": 25.0,
        "FluxEV": 12.0,
        "PEWMA": 8.0,
        "StreamingAutoregressiveDetector": 8.0,
    },
}


def _write_ci_csv(out_path: Path, rows_by_key: dict[tuple[str, str, int], dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=out_path.parent, delete=False
    ) as f:
        tmp_path = Path(f.name)
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CI_FIELDS)
        writer.writerows(
            [r.get(field, "") for field in CI_FIELDS] for _, r in sorted(rows_by_key.items())
        )
    tmp_path.replace(out_path)


def _detector_weights(results_path=OUT / "results.csv"):
    """Load per-(dataset, detector) and per-part detector timings from results.csv."""
    cell_weights: dict[tuple[str, str], float] = {}
    part_weights: dict[str, dict[str, float]] = {"odds": {}, "drift": {}, "univariate": {}}
    if results_path.exists():
        drift_keys = set(DRIFT)
        multivariate_dets = set(DETECTORS)
        univariate_dets = set(UNIVARIATE)
        with open(results_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ds = row.get("dataset", "")
                det = row.get("detector", "").removesuffix(SHINGLED)
                try:
                    ms = float(row.get("ms_per_point", ""))
                except (TypeError, ValueError):
                    continue
                cell_weights[(ds, det)] = ms
                if ds in drift_keys and det in multivariate_dets:
                    part = "drift"
                elif ds.startswith("nab/") and det in univariate_dets:
                    part = "univariate"
                elif not ds.startswith(("nab/", "ucr/")) and det in multivariate_dets:
                    part = "odds"
                else:
                    continue
                part_weights[part][det] = part_weights[part].get(det, 0.0) + ms
    for part, defaults in _FALLBACK_PART_WEIGHTS.items():
        for det, weight in defaults.items():
            part_weights[part].setdefault(det, weight)
    return cell_weights, part_weights


def _ci_shard_count(part: str, count: int) -> int:
    if count <= 1:
        return max(1, count)
    if part == "odds":
        target = (
            1 if count <= 4 else 2 if count <= 9 else 3 if count <= 15 else 4 if count <= 22 else 5
        )
    elif part == "drift":
        target = 1 if count <= 5 else 2 if count <= 14 else 3
    else:
        target = 1 if count <= 4 else 2 if count <= 10 else 3
    return min(target, count)


def _pack_shards(
    detectors: list[str], num_shards: int, weights: dict[str, float]
) -> list[list[str]]:
    """Greedy LPT bin-packing of detectors into balanced shards."""
    if num_shards <= 1 or len(detectors) <= 1:
        return [list(detectors)] if detectors else []
    known = [weights[d] for d in detectors if d in weights]
    default_weight = float(np.median(known)) if known else 1.0
    ordered = sorted(detectors, key=lambda d: (-weights.get(d, default_weight), d))
    bins: list[tuple[float, list[str]]] = [(0.0, []) for _ in range(num_shards)]
    for det in ordered:
        bins.sort(key=lambda b: b[0])
        load_so_far, bucket = bins[0]
        bucket.append(det)
        bins[0] = (load_so_far + weights.get(det, default_weight), bucket)
    return [bucket for _, bucket in bins if bucket]


def _cache_file(cache_dir: Path, detector: str, det_hash: str) -> Path:
    return cache_dir / f"{detector}-{det_hash}.csv"


def _det_base_hash(
    part: str, bench_sha: str, detector: str, deps: set[str], ls_map: dict[str, str]
) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"{part}\n{bench_sha}\n{detector}\n".encode())
    for dep in sorted(deps):
        if dep in ls_map:
            hasher.update((ls_map[dep] + "\n").encode())
    return hasher.hexdigest()[:16]


def compare_ci_dirs(base_dir: Path, head_dir: Path) -> None:
    """Compare all ``--ci`` CSV artifacts collected under ``base_dir`` and ``head_dir``."""
    before: dict[tuple[str, str, int], dict] = {}
    for csv_path in sorted(base_dir.rglob("*.csv")):
        before.update(read_results(csv_path))
    after: dict[tuple[str, str, int], dict] = {}
    for csv_path in sorted(head_dir.rglob("*.csv")):
        after.update(read_results(csv_path))
    if not after:
        print("No head benchmark results to compare.")
        return
    if not before:
        print(f"No base benchmark results to compare ({len(after)} head runs).")
        return
    compare(before, after)


def ci_plan(base: str, *, all_detectors: bool = False) -> dict:
    """Compute sharded GitHub Actions matrix and content-addressed base cache keys for --ci."""
    tree_ref = (base or "HEAD") if all_detectors else base
    detectors = (
        [*DETECTORS, *UNIVARIATE]
        if (all_detectors or not base)
        else changed_detectors(base, ci=True)
    )
    multivariate = [d for d in detectors if d in DETECTORS]
    univariate = [d for d in detectors if d in UNIVARIATE]
    _, part_weights = _detector_weights()
    shared_deps = pysad_files([Data, fix_seed]) | {"pyproject.toml"}
    bench_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

    ls_map: dict[str, str] = {}
    if tree_ref:
        for line in git("ls-tree", "-r", tree_ref, "--", "pysad", "pyproject.toml").splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                ls_map[parts[1]] = line

    include = []
    for part, part_detectors in [
        ("odds", multivariate),
        ("drift", multivariate),
        ("univariate", univariate),
    ]:
        if not part_detectors:
            continue
        shards = _pack_shards(
            part_detectors,
            _ci_shard_count(part, len(part_detectors)),
            part_weights[part],
        )
        for shard_id, shard_dets in enumerate(shards):
            has_base = bool(tree_ref) and any(
                f"{getattr(models, d).__module__.replace('.', '/')}.py" in ls_map
                for d in shard_dets
            )
            det_hashes = {
                d: _det_base_hash(
                    part,
                    bench_sha,
                    d,
                    shared_deps | pysad_files([getattr(models, d)]),
                    ls_map,
                )
                for d in sorted(shard_dets)
            }
            shard_hash = hashlib.sha256(
                (
                    f"{part}\n{bench_sha}\n" + "\n".join(f"{k}={v}" for k, v in det_hashes.items())
                ).encode()
            ).hexdigest()[:16]
            include.append(
                {
                    "part": part,
                    "shard": ",".join(shard_dets),
                    "shard_id": str(shard_id),
                    "has_base": has_base,
                    "base_hash": shard_hash,
                    "det_hashes": json.dumps(det_hashes, separators=(",", ":")),
                }
            )
    return {"has_jobs": bool(include), "include": include}


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
    parser.add_argument(
        "--parts",
        help=f"comma-separated parts of the benchmark (default: all): {', '.join(PARTS)}",
    )
    parser.add_argument(
        "--ci",
        metavar="FILE",
        help=f"run each dataset on its first {CI_POINTS} points, without UCR, and write the "
        "results to FILE instead of docs/benchmark/",
    )
    parser.add_argument(
        "--since", metavar="BASE", help="run only the detectors that changed since BASE"
    )
    parser.add_argument(
        "--ignore-unknown",
        action="store_true",
        help="ignore detector names not present in this checkout",
    )
    parser.add_argument(
        "--ci-plan",
        nargs="?",
        const="",
        metavar="BASE",
        help="print JSON sharded CI plan and content-addressed base cache keys for changes since BASE",
    )
    parser.add_argument(
        "--all-detectors",
        action="store_true",
        help="with --ci-plan, include every detector instead of only detectors changed since BASE",
    )
    parser.add_argument(
        "--base-cache-dir",
        metavar="DIR",
        help="with --ci, read and populate per-detector <detector>-<hash>.csv cache files in DIR",
    )
    parser.add_argument(
        "--det-hashes",
        metavar="JSON",
        help="with --base-cache-dir, JSON mapping detector name to its base hash",
    )
    parser.add_argument(
        "--compare",
        metavar="FILE",
        help="with --ci, print the AUROC of each run that differs from FILE, a file of another --ci run",
    )
    parser.add_argument(
        "--compare-dirs",
        nargs=2,
        metavar=("BASE_DIR", "HEAD_DIR"),
        help="compare all --ci CSV files under BASE_DIR against HEAD_DIR",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument(
        "--dimensions",
        action="store_true",
        help="run the dimensionality sweep and write docs/benchmark/dimensions.csv and dimensions.svg",
    )
    parser.add_argument(
        "--dimension-points",
        type=int,
        default=DIMENSION_POINTS,
        help="points per detector and dimension in the dimensionality sweep",
    )
    parser.add_argument(
        "--custom",
        metavar="SPEC",
        help="custom model to benchmark, as 'path/to/file.py:ClassName'",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="write a LaTeX summary table to docs/benchmark/summary.tex",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="directory for cached pair results (default: %(default)s)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="run pairs without reading or writing the result cache",
    )
    args = parser.parse_args()
    if args.compare_dirs:
        compare_ci_dirs(Path(args.compare_dirs[0]), Path(args.compare_dirs[1]))
        return
    if args.ci_plan is not None:
        print(json.dumps(ci_plan(args.ci_plan, all_detectors=args.all_detectors)))
        return
    if args.dimensions:
        if args.dimension_points <= 0:
            parser.error("--dimension-points must be positive.")
        rows = dimension_rows(DIMENSION_DETECTORS, DIMENSIONS, args.dimension_points, 0)
        OUT.mkdir(parents=True, exist_ok=True)
        write_csv(
            "dimensions.csv",
            DIMENSION_FIELDS,
            [[row[field] for field in DIMENSION_FIELDS] for row in rows],
        )
        write_dimensions_svg(rows, DIMENSION_DETECTORS)
        for row in rows:
            print(
                f"{row['detector']:<16} d={row['dimension']:>4} {row['ms_per_point']:>8.3f} ms/point"
            )
        return
    path = OUT / "results.csv"
    ci_mode = bool(args.ci)
    points = CI_POINTS if ci_mode else None

    # Load custom model if specified
    custom_model = None
    custom_name = None
    custom_input_kind = None
    if args.custom:
        custom_model = _load_custom_model(args.custom)
        custom_name = custom_model.__name__
        custom_input_kind = custom_model._input_kind
        if custom_input_kind not in ("multivariate", "univariate", "timestamped"):
            custom_input_kind = "multivariate"

    detectors = args.detectors.split(",") if args.detectors else [*DETECTORS, *UNIVARIATE]
    if args.check or args.since:
        detectors = changed_detectors(
            args.check or args.since, None if ci_mode else path, ci=ci_mode
        )
    if args.ignore_unknown:
        known = set(DETECTORS) | set(UNIVARIATE)
        detectors = [d for d in detectors if d in known]
    if args.check is not None or args.since:
        print("Detectors to run:", ", ".join(detectors) or "none")
    unknown = set(detectors) - set(DETECTORS) - set(UNIVARIATE)
    if unknown:
        parser.error(f"unknown detectors: {', '.join(sorted(unknown))}")
    parts = args.parts.split(",") if args.parts else PARTS
    if set(parts) - set(PARTS):
        parser.error(f"unknown parts: {', '.join(sorted(set(parts) - set(PARTS)))}")

    # Create factory for custom model
    custom_is_multivariate = False
    custom_is_univariate = False
    if custom_model:
        _make_custom_factory(custom_model, custom_input_kind)  # validate factory creation
        custom_is_multivariate = custom_input_kind == "multivariate"
        custom_is_univariate = custom_input_kind in ("univariate", "timestamped")

    # Each part downloads its data once, before the workers start. Every part runs once, with seed 0.
    cache_dir = Path(args.base_cache_dir) if (ci_mode and args.base_cache_dir) else None
    det_hashes: dict[str, str] = json.loads(args.det_hashes) if args.det_hashes else {}
    cached_rows: dict[tuple[str, str, int], dict] = {}
    run_detectors = list(detectors)
    if cache_dir is not None and det_hashes:
        run_detectors = []
        for d in detectors:
            det_hash = det_hashes.get(d)
            cache_path = _cache_file(cache_dir, d, det_hash) if det_hash else None
            if cache_path is not None and cache_path.is_file():
                cached_rows.update(read_results(cache_path))
            else:
                run_detectors.append(d)

    jobs = []
    multivariate = [d for d in run_detectors if d in DETECTORS]
    univariate = [d for d in run_detectors if d in UNIVARIATE]

    if "odds" in parts and (multivariate or custom_is_multivariate):
        for dataset in DATASETS:
            Data().get_data(f"{dataset}.mat")
        jobs += [(ds, d, 0, points, ci_mode) for d in multivariate for ds in DATASETS]
        if custom_is_multivariate:
            jobs += [
                (ds, custom_name, 0, points, ci_mode, "custom", custom_input_kind, args.custom)
                for ds in DATASETS
            ]
    if {"univariate", "nab", "shingles"} & set(parts) and (
        univariate
        or ({"nab", "shingles"} & set(parts) and multivariate)
        or custom_is_univariate
        or ({"nab", "shingles"} & set(parts) and custom_is_multivariate)
    ):
        for series in nab_series():
            Data().get_data(series)
    if "univariate" in parts and (univariate or custom_is_univariate):
        jobs += [(series, d, 0, points, ci_mode) for d in univariate for series in series_of(d)]
        if custom_is_univariate:
            jobs += [
                (series, custom_name, 0, points, ci_mode, "custom", custom_input_kind, args.custom)
                for series in nab_series()
            ]
        if not ci_mode:
            jobs += [
                (s, d, 0, None, ci_mode)
                for d in univariate
                if d not in SEASONAL + TIMESTAMPED
                for s in ucr_series()
            ]
    if "drift" in parts and (multivariate or custom_is_multivariate):
        for stream in DRIFT:
            Data().get_data(stream)
        jobs += [(stream, d, 0, points, ci_mode) for d in multivariate for stream in DRIFT]
        if custom_is_multivariate:
            jobs += [
                (stream, custom_name, 0, points, ci_mode, "custom", custom_input_kind, args.custom)
                for stream in DRIFT
            ]
    if "nab" in parts and (multivariate or custom_is_multivariate):
        jobs += [(series, d, 0, points, ci_mode) for d in multivariate for series in nab_series()]
        if custom_is_multivariate:
            jobs += [
                (series, custom_name, 0, points, ci_mode, "custom", custom_input_kind, args.custom)
                for series in nab_series()
            ]
    if "shingles" in parts and (multivariate or custom_is_multivariate):
        jobs += [
            (series, d + SHINGLED, 0, points, ci_mode)
            for d in multivariate
            for series in nab_series()
        ]
        if custom_is_multivariate:
            jobs += [
                (
                    series,
                    custom_name + SHINGLED,
                    0,
                    points,
                    ci_mode,
                    "custom",
                    custom_input_kind,
                    args.custom,
                )
                for series in nab_series()
            ]
    cache_paths = (
        _result_cache_paths(jobs, args.cache_dir)
        if not args.no_cache and args.check is None
        else {}
    )
    pending = []
    for job in jobs:
        row = _read_cached_result(cache_paths[job]) if job in cache_paths else None
        if row is None:
            pending.append(job)
        else:
            cached_rows[key(row)] = row
    print(f"Pair cache: {len(jobs) - len(pending)} hits, {len(pending)} runs.")
    jobs = pending
    cell_weights, part_weights = _detector_weights()
    drift_keys = set(DRIFT)

    def _job_weight(job: tuple) -> float:
        ds, det_raw = job[0], job[1].removesuffix(SHINGLED)
        if (ds, det_raw) in cell_weights:
            return cell_weights[(ds, det_raw)]
        part = "drift" if ds in drift_keys else "univariate" if ds.startswith("nab/") else "odds"
        return part_weights[part].get(det_raw, float("inf"))

    jobs.sort(key=_job_weight, reverse=True)
    new = {}
    run_commit = commit() if not ci_mode else None
    if jobs:
        with ProcessPoolExecutor(args.jobs, initializer=_init_worker) as pool:
            remaining = {pool.submit(run, job): job for job in jobs}
            start, step = time.perf_counter(), max(1, len(jobs) // 20)
            for future in as_completed(list(remaining)):
                row = future.result()
                if run_commit is not None:
                    row["commit"] = run_commit
                job = remaining.pop(future)
                if job in cache_paths:
                    _write_cached_result(cache_paths[job], row)
                new[key(row)] = row
                if len(new) % step == 0 or not remaining:
                    progress(len(new), len(jobs), time.perf_counter() - start, remaining.values())
    new = {**cached_rows, **new}
    nab = {} if ci_mode else nab_rows(new)  # the NAB score needs the whole series
    for row in new.values():
        row.pop("_nab", None)

    if ci_mode:
        if cache_dir is not None and det_hashes:
            for d in run_detectors:
                det_hash = det_hashes.get(d)
                if not det_hash:
                    continue
                det_rows = {k: r for k, r in new.items() if k[1].removesuffix(SHINGLED) == d}
                if det_rows:
                    _write_ci_csv(_cache_file(cache_dir, d, det_hash), det_rows)
        combined = new
        _write_ci_csv(Path(args.ci), combined)
        if args.compare:
            compare(read_results(Path(args.compare)), combined)
        return

    results = read_results(path)
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
                if f.endswith("false_alerts") and f in results[k]
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
            detectors = sorted(
                {d.removesuffix(SHINGLED) for d in [*(k[1] for k, _, _ in changed), *changed_nab]}
            )
            print(
                f"{len(changed) + len(changed_nab)} results changed. Run: python benchmarks/benchmark.py "
                f"--detectors {','.join(detectors)}, then commit docs/benchmark/."
            )
            sys.exit(1)
        print("The results in docs/benchmark/ are up to date.")
        return

    results.update(new)
    OUT.mkdir(parents=True, exist_ok=True)
    nab = {**read_nab(), **nab}
    write_csv(
        "nab.csv",
        NAB_FIELDS,
        [[nab[d].get(f, "") for f in NAB_FIELDS] for d in NAB_DETECTORS if d in nab],
    )
    detectors_order = write_tables(results, nab)
    if args.latex:
        _write_latex_summary(results, detectors_order, OUT / "summary.tex")
        print(f"LaTeX summary written to {OUT / 'summary.tex'}")


if __name__ == "__main__":
    main()
