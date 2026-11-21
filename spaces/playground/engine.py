"""Core streaming evaluation, calibration, injection, and explanation engine for the playground."""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.core.base_postprocessor import BasePostprocessor
from pysad.evaluation import AUPRMetric, AUROCMetric, PrecisionMetric, RecallMetric
from pysad.models import (
    KNNCAD,
    LODA,
    HalfSpaceTrees,
    LocalOutlierProbability,
    MedianAbsoluteDeviation,
    RobustRandomCutForest,
    RSHash,
    SeasonalHybridESD,
    SmartSifter,
    xStream,
)
from pysad.transform.probability_calibration import (
    ConformalProbabilityCalibrator,
    SPOTProbabilityCalibrator,
)
from pysad.utils import Data, get_minmax_array

SPACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPACE_DIR.parents[1] if len(SPACE_DIR.parents) > 1 else SPACE_DIR

STREAM_SYNTHETIC = "Synthetic Drift + Spikes (4 sensors)"
STREAM_NYC_TAXI = "NYC Taxi Demand (NAB, univariate)"
STREAM_ARRHYTHMIA = "Arrhythmia (ODDS, 274 features)"
STREAM_CREDITCARD = "ULB Credit Card Fraud (30 features)"
STREAM_CSV = "Upload CSV"
STREAM_CHOICES = [
    STREAM_SYNTHETIC,
    STREAM_NYC_TAXI,
    STREAM_ARRHYTHMIA,
    STREAM_CREDITCARD,
    STREAM_CSV,
]
BROWSER_STREAM_CHOICES = [
    STREAM_SYNTHETIC,
    STREAM_NYC_TAXI,
    STREAM_ARRHYTHMIA,
    STREAM_CSV,
]

CALIBRATOR_CONFORMAL = "Conformal (ConformalProbabilityCalibrator)"
CALIBRATOR_SPOT = "SPOT / EVT (SPOTProbabilityCalibrator)"
CALIBRATOR_CHOICES = [CALIBRATOR_CONFORMAL, CALIBRATOR_SPOT]

DETECTOR_CHOICES = [
    "LocalOutlierProbability",
    "RSHash",
    "LODA",
    "SmartSifter",
    "RobustRandomCutForest",
    "HalfSpaceTrees",
    "KNNCAD",
    "SeasonalHybridESD",
]
_SUPPORTED_DETECTORS = set(DETECTOR_CHOICES) | {"MedianAbsoluteDeviation", "xStream"}
DEFAULT_DETECTORS = ["LocalOutlierProbability", "RSHash", "LODA"]

_LABEL_COLUMN_NAMES = {"label", "labels", "anomaly", "is_anomaly", "class", "target", "y"}


@dataclass(frozen=True)
class StreamData:
    """Loaded streaming dataset ready for detector evaluation and plotting."""

    name: str
    description: str
    X: np.ndarray
    y: np.ndarray | None
    feature_names: list[str]
    timestamps: list[str] | None
    seasonal_period: int

    @property
    def num_points(self) -> int:
        return int(self.X.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def is_univariate(self) -> bool:
        return self.num_features == 1


@dataclass(frozen=True)
class DetectorTrace:
    """Outputs and streaming evaluation metrics for one detector run."""

    name: str
    display_name: str
    raw_scores: np.ndarray
    calibrated_scores: np.ndarray
    alerts: np.ndarray
    us_per_point: float
    points_per_second: float
    auroc: float | None
    aupr: float | None
    precision: float | None
    recall: float | None
    injected_caught: int
    injected_total: int


@dataclass(frozen=True)
class ExplanationResult:
    """Feature attribution result at a selected stream position."""

    step: int
    feature_names: list[str]
    contributions: np.ndarray
    loda_raw_score: float
    note: str


@dataclass(frozen=True)
class PlaygroundResult:
    """Complete execution output rendered by the interactive UI."""

    stream: StreamData
    injected_mask: np.ndarray
    warmup: int
    threshold: float
    calibrator_name: str
    traces: list[DetectorTrace]
    explanation: ExplanationResult
    pipeline_code: str
    status_note: str


def make_synthetic_drift_stream(num_points: int = 1200, seed: int = 42) -> StreamData:
    """Generate a deterministic 4-sensor stream with concept drift and labeled anomalies."""
    n = int(max(300, num_points))
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)

    base_cycle = np.sin(2.0 * np.pi * t / 48.0)
    drift_start = int(0.52 * n)
    drift_ramp = np.clip((t - drift_start) / max(30.0, 0.08 * n), 0.0, 1.0)

    s1 = 1.0 + 0.65 * base_cycle + 1.1 * drift_ramp + rng.normal(0.0, 0.22, n)
    s2 = 0.0 + 0.50 * np.cos(2.0 * np.pi * t / 48.0) - 0.8 * drift_ramp + rng.normal(0.0, 0.22, n)
    s3 = (
        -0.5
        + 0.45 * np.sin(2.0 * np.pi * t / 24.0 + 0.7)
        + 0.9 * drift_ramp
        + rng.normal(0.0, 0.20, n)
    )
    s4 = 0.4 + 0.35 * np.sin(2.0 * np.pi * t / 72.0) + rng.normal(0.0, 0.22, n)

    X = np.column_stack([s1, s2, s3, s4])
    y = np.zeros(n, dtype=int)

    # Plant distinct anomaly shapes across different sensors before and after drift.
    spike_idx = int(0.24 * n)
    X[spike_idx, 0] += 4.8
    y[spike_idx] = 1

    plateau_start = int(0.39 * n)
    plateau_end = min(n, plateau_start + 5)
    X[plateau_start:plateau_end, 1] += 3.6
    y[plateau_start:plateau_end] = 1

    dip_idx = int(0.69 * n)
    X[dip_idx, 2] -= 4.6
    y[dip_idx] = 1

    burst_idx = min(n - 2, int(0.86 * n))
    X[burst_idx : burst_idx + 2, 0] += 4.0
    X[burst_idx : burst_idx + 2, 3] -= 4.2
    y[burst_idx : burst_idx + 2] = 1

    return StreamData(
        name=STREAM_SYNTHETIC,
        description=(
            f"4 synthetic sensors, {n} steps, regime shift near step {drift_start}, "
            f"{int(y.sum())} labeled anomalies."
        ),
        X=X,
        y=y,
        feature_names=["sensor_1", "sensor_2", "sensor_3", "sensor_4"],
        timestamps=None,
        seasonal_period=48,
    )


def _find_local_data_dir(rel_path: str) -> str | None:
    """Return a local directory containing bundled dataset `rel_path` if available."""
    for candidate in (SPACE_DIR / "data", REPO_ROOT / "examples" / "data"):
        if (candidate / rel_path).is_file():
            return str(candidate)
    return None


@lru_cache(maxsize=1)
def _load_full_nyc_taxi() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and cache the full NYC Taxi series with ISO timestamps."""
    local_dir = _find_local_data_dir("nab/realKnownCause/nyc_taxi.csv")
    loader = Data(local_dir) if local_dir is not None else Data()
    raw, y = loader.get_data(
        "nab/realKnownCause/nyc_taxi.csv",
        download=(local_dir is None),
        timestamps=True,
    )
    timestamps = [
        datetime.fromtimestamp(float(ts), tz=UTC).strftime("%Y-%m-%d %H:%M") for ts in raw[:, 0]
    ]
    return raw[:, 1:2].astype(np.float64), y.astype(int), timestamps


def load_nyc_taxi_stream(max_points: int = 1500) -> StreamData:
    """Load a contiguous slice of NAB's NYC Taxi passenger stream covering real anomaly windows."""
    X_full, y_full, ts_full = _load_full_nyc_taxi()
    n = min(int(max(300, max_points)), len(X_full))

    anom_indices = np.flatnonzero(y_full == 1)
    start = 0
    if len(anom_indices) > 0 and n < len(X_full):
        first_anom = int(anom_indices[0])
        lead = min(450, n // 3)
        start = max(0, min(first_anom - lead, len(X_full) - n))

    end = start + n
    X_slice = X_full[start:end].copy()
    y_slice = y_full[start:end].copy()
    ts_slice = ts_full[start:end]

    return StreamData(
        name=STREAM_NYC_TAXI,
        description=(
            f"NYC Taxi 30-min passenger counts: {n} steps ({ts_slice[0]} to {ts_slice[-1]}), "
            f"{int(y_slice.sum())} points in NAB anomaly windows."
        ),
        X=X_slice,
        y=y_slice,
        feature_names=["passengers"],
        timestamps=ts_slice,
        seasonal_period=48,
    )


@lru_cache(maxsize=1)
def _load_full_arrhythmia() -> tuple[np.ndarray, np.ndarray]:
    """Load and cache the ODDS Arrhythmia benchmark dataset."""
    local_dir = _find_local_data_dir("arrhythmia.mat")
    loader = Data(local_dir) if local_dir is not None else Data()
    X, y = loader.get_data("arrhythmia.mat", download=(local_dir is None))
    return X.astype(np.float64), y.astype(int)


def load_arrhythmia_stream(max_points: int = 1500) -> StreamData:
    """Load the bundled ODDS Arrhythmia stream (up to 452 instances x 274 features)."""
    X_full, y_full = _load_full_arrhythmia()
    n = min(int(max(300, max_points)), len(X_full))
    X_slice = X_full[:n].copy()
    y_slice = y_full[:n].copy()
    feature_names = [f"attr_{i + 1}" for i in range(X_slice.shape[1])]
    return StreamData(
        name=STREAM_ARRHYTHMIA,
        description=(
            f"ODDS Arrhythmia records: {n} steps, {X_slice.shape[1]} features, "
            f"{int(y_slice.sum())} labeled anomalies."
        ),
        X=X_slice,
        y=y_slice,
        feature_names=feature_names,
        timestamps=None,
        seasonal_period=24,
    )


@lru_cache(maxsize=1)
def _load_full_creditcard() -> tuple[np.ndarray, np.ndarray]:
    """Load and cache the ULB Credit Card Fraud stream."""
    X, y = Data().get_data("creditcard/creditcard.arff", download=True)
    return X.astype(np.float64), y.astype(int)


def load_creditcard_stream(max_points: int = 1500) -> StreamData:
    """Load a chronological slice of ULB Credit Card transactions containing real frauds."""
    X_full, y_full = _load_full_creditcard()
    n = min(int(max(300, max_points)), len(X_full))

    fraud_indices = np.flatnonzero(y_full == 1)
    start = 0
    if len(fraud_indices) > 0 and n < len(X_full):
        # Position near an early fraud cluster while keeping an anomaly-free warm-up prefix.
        target_fraud = int(fraud_indices[min(2, len(fraud_indices) - 1)])
        start = max(0, min(target_fraud - min(450, n // 3), len(X_full) - n))

    end = start + n
    X_slice = X_full[start:end].copy()
    y_slice = y_full[start:end].copy()
    feature_names = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

    return StreamData(
        name=STREAM_CREDITCARD,
        description=(
            f"Chronological slice of {n} European card transactions (30 features, "
            f"{int(y_slice.sum())} fraudulent transactions)."
        ),
        X=X_slice,
        y=y_slice,
        feature_names=feature_names,
        timestamps=None,
        seasonal_period=24,
    )


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_csv_stream(csv_input: str | bytes, max_points: int = 2500) -> StreamData:
    """Parse an uploaded CSV file into a clean numeric StreamData object."""
    if isinstance(csv_input, bytes):
        raw_text = csv_input.decode("utf-8", errors="replace")
    else:
        raw_text = csv_input

    rows = [row for row in csv.reader(io.StringIO(raw_text)) if any(cell.strip() for cell in row)]
    if len(rows) < 10:
        raise ValueError("Uploaded CSV must contain at least 10 non-empty data rows.")

    first_row = [cell.strip() for cell in rows[0]]
    has_header = any(cell != "" and not _is_float(cell) for cell in first_row)
    if has_header:
        headers = [cell or f"feature_{idx}" for idx, cell in enumerate(first_row)]
        data_rows = rows[1:]
    else:
        headers = [f"feature_{idx}" for idx in range(len(first_row))]
        data_rows = rows

    if len(data_rows) < 10:
        raise ValueError("Uploaded CSV must contain at least 10 data rows after the header.")

    data_rows = data_rows[: int(max(10, max_points))]

    # Identify columns that parse cleanly as numeric across at least half the rows.
    numeric_cols: list[int] = []
    label_col: int | None = None
    for col_idx, col_name in enumerate(headers):
        values = [r[col_idx].strip() if col_idx < len(r) else "" for r in data_rows]
        numeric_count = sum(1 for v in values if v != "" and _is_float(v))
        if numeric_count < max(1, len(data_rows) // 2):
            continue
        if col_name.lower() in _LABEL_COLUMN_NAMES and label_col is None:
            parsed = [float(v) for v in values if v != "" and _is_float(v)]
            if set(parsed) <= {0.0, 1.0}:
                label_col = col_idx
                continue
        numeric_cols.append(col_idx)

    if not numeric_cols:
        raise ValueError("Uploaded CSV has no numeric feature columns.")

    X = np.empty((len(data_rows), len(numeric_cols)), dtype=np.float64)
    for out_j, col_idx in enumerate(numeric_cols):
        last_val = 0.0
        for row_i, row in enumerate(data_rows):
            raw_cell = row[col_idx].strip() if col_idx < len(row) else ""
            val = float(raw_cell) if raw_cell != "" and _is_float(raw_cell) else np.nan
            if not np.isfinite(val):
                val = last_val
            else:
                last_val = val
            X[row_i, out_j] = val

    y: np.ndarray | None = None
    if label_col is not None:
        y = np.array(
            [
                int(float(row[label_col].strip()))
                if label_col < len(row) and _is_float(row[label_col].strip())
                else 0
                for row in data_rows
            ],
            dtype=int,
        )

    feature_names = [headers[idx] for idx in numeric_cols]
    label_note = f"{int(y.sum())} labeled anomalies" if y is not None else "no label column"
    return StreamData(
        name="Uploaded CSV",
        description=f"Uploaded CSV stream: {len(X)} rows, {len(feature_names)} numeric features ({label_note}).",
        X=X,
        y=y,
        feature_names=feature_names,
        timestamps=None,
        seasonal_period=24,
    )


def load_stream(
    stream_name: str,
    max_points: int = 1200,
    csv_data: bytes | str | None = None,
) -> StreamData:
    """Load the requested stream by name, falling back cleanly if CSV is omitted."""
    if stream_name == STREAM_SYNTHETIC:
        return make_synthetic_drift_stream(num_points=max_points)
    if stream_name == STREAM_NYC_TAXI:
        return load_nyc_taxi_stream(max_points=max_points)
    if stream_name == STREAM_ARRHYTHMIA:
        return load_arrhythmia_stream(max_points=max_points)
    if stream_name == STREAM_CREDITCARD:
        return load_creditcard_stream(max_points=max_points)
    if stream_name == STREAM_CSV:
        if csv_data is None:
            raise ValueError("Upload a CSV file to run in 'Upload CSV' mode.")
        return parse_csv_stream(csv_data, max_points=max_points)
    raise ValueError(f"Unknown stream choice: {stream_name!r}")


def apply_injected_anomalies(
    stream: StreamData,
    injected: list[dict[str, Any]],
) -> tuple[StreamData, np.ndarray]:
    """Return a copy of `stream` with user-injected anomalies applied."""
    X = stream.X.copy()
    y = stream.y.copy() if stream.y is not None else np.zeros(len(X), dtype=int)
    has_any_labels = stream.y is not None
    injected_mask = np.zeros(len(X), dtype=bool)
    stds = np.std(X, axis=0)
    scales = np.where(stds > 1e-8, stds, 1.0)

    for item in injected:
        idx = int(item["index"])
        feat_idx = int(item["feature_idx"])
        sigma = float(item["magnitude_sigma"])
        if 0 <= idx < len(X) and 0 <= feat_idx < X.shape[1]:
            X[idx, feat_idx] += sigma * float(scales[feat_idx])
            y[idx] = 1
            injected_mask[idx] = True
            has_any_labels = True

    updated = StreamData(
        name=stream.name,
        description=stream.description,
        X=X,
        y=y if has_any_labels else None,
        feature_names=stream.feature_names,
        timestamps=stream.timestamps,
        seasonal_period=stream.seasonal_period,
    )
    return updated, injected_mask


def _safe_feature_bounds(X_prefix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite, strictly ordered bounds from an initial prefix."""
    mins, maxes = get_minmax_array(X_prefix)
    span = np.where(maxes > mins, maxes - mins, 1.0)
    # Give HalfSpaceTrees modest headroom around the warm-up prefix.
    return mins - 0.1 * span, maxes + 0.1 * span


def _make_detector(
    name: str,
    X: np.ndarray,
    warmup: int,
    seasonal_period: int,
    primary_feature_idx: int,
) -> tuple[BaseModel, str, int | None]:
    """Instantiate a detector and return `(model, display_name, univariate_feature_idx)`."""
    is_uni = X.shape[1] == 1
    if name == "LocalOutlierProbability":
        init_len = min(len(X), max(15, min(60, warmup)))
        num_neighbors = min(10, max(1, init_len - 1))
        return (
            LocalOutlierProbability(initial_X=X[:init_len], num_neighbors=num_neighbors),
            "LocalOutlierProbability",
            None,
        )
    if name == "RSHash":
        mins, maxes = _safe_feature_bounds(X[: max(10, warmup)])
        return (
            RSHash(feature_mins=mins, feature_maxes=maxes, random_state=0),
            "RSHash",
            None,
        )
    if name == "SmartSifter":
        return SmartSifter(random_state=0), "SmartSifter", None
    if name == "KNNCAD":
        feat_idx = 0 if is_uni else primary_feature_idx
        return (
            KNNCAD(probationary_period=max(20, min(warmup, len(X) // 3))),
            "KNNCAD",
            feat_idx,
        )
    if name == "LODA":
        return LODA(random_state=0), "LODA", None
    if name == "HalfSpaceTrees":
        mins, maxes = _safe_feature_bounds(X[: max(10, warmup)])
        return (
            HalfSpaceTrees(
                feature_mins=mins,
                feature_maxes=maxes,
                window_size=min(200, max(50, warmup)),
                num_trees=25,
                random_state=0,
            ),
            "HalfSpaceTrees",
            None,
        )
    if name == "RobustRandomCutForest":
        return (
            RobustRandomCutForest(
                num_trees=20,
                tree_size=128,
                shingle_size=4 if is_uni else 1,
                time_decay=0.01,
                random_state=0,
            ),
            "RobustRandomCutForest",
            None,
        )
    if name == "xStream":
        return (
            xStream(
                num_components=min(60, max(16, X.shape[1] * 4)),
                n_chains=40,
                depth=12,
                window_size=50,
                random_state=0,
            ),
            "xStream",
            None,
        )
    if name == "SeasonalHybridESD":
        period = max(4, min(seasonal_period, max(4, len(X) // 6)))
        win = min(len(X) - 1, max(2 * period + 2, 4 * period))
        feat_idx = 0 if is_uni else primary_feature_idx
        return (
            SeasonalHybridESD(period=period, window_size=win, direction="both", alpha=0.01),
            "SeasonalHybridESD",
            feat_idx,
        )
    if name == "MedianAbsoluteDeviation":
        feat_idx = 0 if is_uni else primary_feature_idx
        return (
            MedianAbsoluteDeviation(window_size=min(300, max(50, warmup))),
            "MedianAbsoluteDeviation",
            feat_idx,
        )
    raise ValueError(f"Unsupported detector: {name!r}")


def _make_calibrator(
    calibrator_name: str,
    calibrator_window: int,
    threshold: float,
    warmup: int,
    num_points: int,
) -> BasePostprocessor:
    """Instantiate the selected probability calibrator with valid parameters."""
    if calibrator_name == CALIBRATOR_CONFORMAL:
        return ConformalProbabilityCalibrator(window_size=int(max(20, calibrator_window)))
    if calibrator_name == CALIBRATOR_SPOT:
        # Enforce SPOT constraints: 0 < q < 1 - level and num_init < stream length.
        level = min(0.95, max(0.80, threshold - 0.02))
        q = max(1e-4, min(1.0 - level - 1e-4, 1.0 - threshold))
        num_init = min(max(60, warmup), max(60, num_points // 3))
        return SPOTProbabilityCalibrator(
            q=float(q),
            num_init=int(num_init),
            level=float(level),
            depth=0,
            max_peaks=500,
        )
    raise ValueError(f"Unsupported calibrator: {calibrator_name!r}")


def _compute_metrics(
    y: np.ndarray | None,
    raw_scores: np.ndarray,
    alerts: np.ndarray,
    warmup: int,
    injected_mask: np.ndarray,
) -> tuple[float | None, float | None, float | None, float | None, int, int]:
    """Compute post-warmup AUROC, AUPR, precision, recall, and injected catch count."""
    injected_indices = np.flatnonzero(injected_mask)
    injected_total = int(len(injected_indices))
    injected_caught = int(alerts[injected_indices].sum()) if injected_total > 0 else 0

    if y is None or warmup >= len(y):
        return None, None, None, None, injected_caught, injected_total

    y_eval = y[warmup:]
    scores_eval = raw_scores[warmup:]
    alerts_eval = alerts[warmup:]
    if len(np.unique(y_eval)) < 2:
        return None, None, None, None, injected_caught, injected_total

    auroc_m = AUROCMetric()
    aupr_m = AUPRMetric()
    prec_m = PrecisionMetric()
    rec_m = RecallMetric()
    auroc_m.update_batch(y_eval, scores_eval)
    aupr_m.update_batch(y_eval, scores_eval)
    prec_m.update_batch(y_eval, alerts_eval)
    rec_m.update_batch(y_eval, alerts_eval)
    return (
        auroc_m.get(),
        aupr_m.get(),
        prec_m.get(),
        rec_m.get(),
        injected_caught,
        injected_total,
    )


def _explain_step(
    stream: StreamData,
    inspect_step: int,
    loda_pre_fit: tuple[np.ndarray, float] | None,
) -> ExplanationResult:
    """Compute LODA.explain_partial() attribution at `inspect_step`."""
    x_point = stream.X[inspect_step]
    if loda_pre_fit is None:
        loda = LODA(random_state=0)
        for idx in range(inspect_step):
            loda.fit_partial(stream.X[idx])
        raw_score = float(loda.score_partial(x_point))
        raw_contrib = loda.explain_partial(x_point)
    else:
        raw_contrib, raw_score = loda_pre_fit

    if stream.is_univariate:
        hist = stream.X[: max(1, inspect_step), 0]
        med = float(np.median(hist))
        mad = float(np.median(np.abs(hist - med))) * 1.4826
        scale = mad if mad > 1e-8 else (float(np.std(hist)) or 1.0)
        z = abs(float(x_point[0]) - med) / scale
        return ExplanationResult(
            step=inspect_step,
            feature_names=stream.feature_names,
            contributions=np.array([z], dtype=np.float64),
            loda_raw_score=float(raw_score),
            note=(
                "Univariate stream (1 feature): showing robust |x - median| / MAD deviation, "
                "since LODA.explain_partial() requires at least 2 features."
            ),
        )

    contributions = np.nan_to_num(raw_contrib, nan=0.0)
    top_idx = int(np.argmax(contributions))
    return ExplanationResult(
        step=inspect_step,
        feature_names=stream.feature_names,
        contributions=contributions,
        loda_raw_score=float(raw_score),
        note=(
            f"LODA.explain_partial() one-tailed t-statistics at step {inspect_step}: "
            f"top feature is '{stream.feature_names[top_idx]}' ({contributions[top_idx]:.2f})."
        ),
    )


def generate_pipeline_code(
    stream_name: str,
    detectors: list[str],
    calibrator_name: str,
    threshold: float,
    calibrator_window: int,
    warmup: int,
) -> str:
    """Build a reproducible PySAD pipeline snippet matching the active configuration."""
    primary = detectors[0]
    imports: list[str] = []
    if stream_name not in (STREAM_NYC_TAXI, STREAM_ARRHYTHMIA, STREAM_CREDITCARD):
        imports.append("import numpy as np")
    imports.append(f"from pysad.models import {primary}")
    if calibrator_name == CALIBRATOR_CONFORMAL:
        cal_cls = "ConformalProbabilityCalibrator"
        cal_init = f"ConformalProbabilityCalibrator(window_size={int(calibrator_window)})"
    else:
        cal_cls = "SPOTProbabilityCalibrator"
        level = min(0.95, max(0.80, threshold - 0.02))
        q = max(1e-4, min(1.0 - level - 1e-4, 1.0 - threshold))
        cal_init = (
            f"SPOTProbabilityCalibrator(q={q:.4g}, num_init={int(warmup)}, level={level:.2f})"
        )
    imports.append(f"from pysad.transform.probability_calibration import {cal_cls}")
    if stream_name in (STREAM_NYC_TAXI, STREAM_ARRHYTHMIA, STREAM_CREDITCARD):
        imports.append("from pysad.utils import ArrayStreamer, Data")
    else:
        imports.append("from pysad.utils import ArrayStreamer")

    if stream_name == STREAM_NYC_TAXI:
        data_line = 'X, y = Data().get_data("nab/realKnownCause/nyc_taxi.csv")'
    elif stream_name == STREAM_ARRHYTHMIA:
        data_line = 'X, y = Data().get_data("arrhythmia.mat")'
    elif stream_name == STREAM_CREDITCARD:
        data_line = 'X, y = Data().get_data("creditcard/creditcard.arff")'
    else:
        data_line = "# Stream rows of shape (num_features,) in arrival order"
        if stream_name == STREAM_SYNTHETIC:
            data_line += "\nX = np.random.default_rng(42).normal(size=(1200, 4))\ny = None"
        else:
            data_line += "\nX = np.loadtxt('stream.csv', delimiter=',', skiprows=1)\ny = None"

    primary = detectors[0]
    if primary == "LocalOutlierProbability":
        model_expr = f"LocalOutlierProbability(initial_X=X[:{min(60, warmup)}])"
    elif primary == "RSHash":
        model_expr = (
            f"RSHash(feature_mins=X[:{warmup}].min(axis=0), "
            f"feature_maxes=X[:{warmup}].max(axis=0), random_state=0)"
        )
    elif primary == "HalfSpaceTrees":
        model_expr = (
            f"HalfSpaceTrees(feature_mins=X[:{warmup}].min(axis=0), "
            f"feature_maxes=X[:{warmup}].max(axis=0), random_state=0)"
        )
    elif primary == "SeasonalHybridESD":
        model_expr = "SeasonalHybridESD(period=48, window_size=192, direction='both')"
    elif primary == "MedianAbsoluteDeviation":
        model_expr = f"MedianAbsoluteDeviation(window_size={min(300, max(50, warmup))})"
    elif primary == "KNNCAD":
        model_expr = f"KNNCAD(probationary_period={warmup})"
    else:
        model_expr = f"{primary}(random_state=0)"

    explain_block = ""
    if primary == "LODA":
        explain_block = (
            "\n        # Rank features contributing to this alert\n"
            "        top = pipeline.models[0].explain_partial(x).argsort()[::-1][:3]"
        )

    comparison_note = (
        [f"# Also compared in playground: {', '.join(detectors[1:])}"] if len(detectors) > 1 else []
    )
    lines = [
        *imports,
        "",
        data_line,
        *comparison_note,
        f"pipeline = {model_expr} | {cal_init}",
        "alerts = []",
        "",
        "for t, x in enumerate(ArrayStreamer().iter(X)):",
        "    score = pipeline.fit_score_partial(x)",
        f"    if t >= {warmup} and score >= {threshold:.3f}:",
        f"        alerts.append(t){explain_block}",
    ]
    return "\n".join(lines)


def run_playground(
    stream_name: str,
    detectors: list[str],
    calibrator_name: str,
    threshold: float = 0.985,
    calibrator_window: int = 350,
    max_points: int = 1200,
    csv_data: bytes | str | None = None,
    injected: list[dict[str, Any]] | None = None,
    inspect_step: int | None = None,
    primary_feature_name: str | None = None,
) -> PlaygroundResult:
    """Execute selected streaming detectors side by side on the chosen stream."""
    selected = [d for d in detectors if d in _SUPPORTED_DETECTORS][:3]
    if not selected:
        selected = ["LocalOutlierProbability"]

    base_stream = load_stream(stream_name, max_points=max_points, csv_data=csv_data)
    stream, injected_mask = apply_injected_anomalies(base_stream, injected or [])
    n = stream.num_points
    warmup = min(max(60, n // 6), max(60, n - 40))

    primary_feat_idx = 0
    if primary_feature_name and primary_feature_name in stream.feature_names:
        primary_feat_idx = stream.feature_names.index(primary_feature_name)

    # Pre-determine inspection step if not explicitly supplied, preferring injected anomalies.
    target_inspect_step: int | None = None
    if inspect_step is not None:
        target_inspect_step = int(np.clip(inspect_step, min(warmup, n - 1), n - 1))
    elif injected_mask.any():
        target_inspect_step = int(np.flatnonzero(injected_mask)[-1])

    traces: list[DetectorTrace] = []
    loda_snapshots: dict[int, tuple[np.ndarray, float]] = {}

    for det_name in selected:
        model, display_name, uni_feat_idx = _make_detector(
            det_name,
            stream.X,
            warmup=warmup,
            seasonal_period=stream.seasonal_period,
            primary_feature_idx=primary_feat_idx,
        )
        calibrator = _make_calibrator(
            calibrator_name,
            calibrator_window=calibrator_window,
            threshold=threshold,
            warmup=warmup,
            num_points=n,
        )
        if uni_feat_idx is not None and not stream.is_univariate:
            display_name = f"{display_name} ({stream.feature_names[uni_feat_idx]})"

        raw_scores = np.empty(n, dtype=np.float64)
        cal_scores = np.empty(n, dtype=np.float64)

        total_model_sec = 0.0
        for t in range(n):
            x_in = (
                stream.X[t, uni_feat_idx : uni_feat_idx + 1]
                if uni_feat_idx is not None
                else stream.X[t]
            )
            if (
                isinstance(model, LODA)
                and target_inspect_step is not None
                and t == target_inspect_step
            ):
                # Capture the exact pre-fit explanation and score at step t.
                loda_snapshots[t] = (model.explain_partial(x_in), float(model.score_partial(x_in)))

            t0 = time.perf_counter()
            raw = float(model.fit_score_partial(x_in))
            total_model_sec += time.perf_counter() - t0

            cal = float(calibrator.fit_transform_partial(raw))
            raw_scores[t] = raw
            cal_scores[t] = cal

        alerts = (np.arange(n) >= warmup) & (cal_scores >= threshold)
        us_per_point = (total_model_sec / max(1, n)) * 1e6
        pps = n / max(total_model_sec, 1e-9)
        auroc, aupr, precision, recall, inj_caught, inj_total = _compute_metrics(
            stream.y,
            raw_scores,
            alerts,
            warmup=warmup,
            injected_mask=injected_mask,
        )
        traces.append(
            DetectorTrace(
                name=det_name,
                display_name=display_name,
                raw_scores=raw_scores,
                calibrated_scores=cal_scores,
                alerts=alerts,
                us_per_point=us_per_point,
                points_per_second=pps,
                auroc=auroc,
                aupr=aupr,
                precision=precision,
                recall=recall,
                injected_caught=inj_caught,
                injected_total=inj_total,
            )
        )

    if target_inspect_step is None:
        # Pick the highest-scoring post-warmup point from the primary trace.
        post_scores = traces[0].calibrated_scores[warmup:]
        target_inspect_step = warmup + int(np.argmax(post_scores)) if len(post_scores) else n - 1

    explanation = _explain_step(
        stream,
        inspect_step=target_inspect_step,
        loda_pre_fit=loda_snapshots.get(target_inspect_step),
    )

    code = generate_pipeline_code(
        stream_name=stream_name,
        detectors=selected,
        calibrator_name=calibrator_name,
        threshold=threshold,
        calibrator_window=calibrator_window,
        warmup=warmup,
    )
    status_note = f"{stream.description} Warm-up: first {warmup} steps."
    if injected_mask.any():
        status_note += f" Injected anomalies: {int(injected_mask.sum())}."

    return PlaygroundResult(
        stream=stream,
        injected_mask=injected_mask,
        warmup=warmup,
        threshold=threshold,
        calibrator_name=calibrator_name,
        traces=traces,
        explanation=explanation,
        pipeline_code=code,
        status_note=status_note,
    )


def run_playground_json(
    stream_name: str,
    detectors: list[str],
    calibrator_name: str,
    threshold: float = 0.985,
    calibrator_window: int = 350,
    max_points: int = 1200,
    csv_data: bytes | str | None = None,
    injected: list[dict[str, Any]] | None = None,
    inspect_step: int | None = None,
    primary_feature_name: str | None = None,
) -> str:
    """Serialize `run_playground(...)` output cleanly as JSON for Pyodide / JS UI callers."""
    result = run_playground(
        stream_name=stream_name,
        detectors=detectors,
        calibrator_name=calibrator_name,
        threshold=threshold,
        calibrator_window=calibrator_window,
        max_points=max_points,
        csv_data=csv_data,
        injected=injected,
        inspect_step=inspect_step,
        primary_feature_name=primary_feature_name,
    )
    stream = result.stream
    exp = result.explanation

    if stream.num_features > 4:
        ranked = np.argsort(exp.contributions)[::-1][:4].tolist()
        feature_indices = sorted(set(ranked))
    else:
        feature_indices = list(range(stream.num_features))

    plotted_features = [
        {
            "name": stream.feature_names[idx],
            "values": [round(float(v), 5) for v in stream.X[:, idx]],
        }
        for idx in feature_indices
    ]
    labeled_indices = (
        np.flatnonzero(stream.y == 1).astype(int).tolist()
        if stream.y is not None and stream.y.any()
        else []
    )
    primary_plot_idx = feature_indices[0]
    labeled_primary_values = [
        round(float(stream.X[i, primary_plot_idx]), 5) for i in labeled_indices
    ]

    order = np.argsort(exp.contributions)
    if len(order) > 12:
        order = order[-12:]

    payload = {
        "stream": {
            "name": stream.name,
            "description": stream.description,
            "num_points": stream.num_points,
            "num_features": stream.num_features,
            "is_univariate": stream.is_univariate,
            "feature_names": stream.feature_names,
            "timestamps": stream.timestamps is not None,
            "x_axis": stream.timestamps
            if stream.timestamps is not None
            else list(range(stream.num_points)),
            "plotted_features": plotted_features,
            "labeled_indices": labeled_indices,
            "labeled_primary_values": labeled_primary_values,
        },
        "warmup": result.warmup,
        "threshold": result.threshold,
        "calibrator_name": result.calibrator_name,
        "status_note": result.status_note,
        "pipeline_code": result.pipeline_code,
        "injected_indices": np.flatnonzero(result.injected_mask).astype(int).tolist(),
        "traces": [
            {
                "name": tr.name,
                "display_name": tr.display_name,
                "calibrated_scores": [round(float(v), 5) for v in tr.calibrated_scores],
                "alert_indices": np.flatnonzero(tr.alerts).astype(int).tolist(),
                "alert_scores": [
                    round(float(tr.calibrated_scores[i]), 5) for i in np.flatnonzero(tr.alerts)
                ],
                "alerts_count": int(tr.alerts.sum()),
                "auroc": None if tr.auroc is None else round(float(tr.auroc), 4),
                "aupr": None if tr.aupr is None else round(float(tr.aupr), 4),
                "precision": None if tr.precision is None else round(float(tr.precision), 4),
                "recall": None if tr.recall is None else round(float(tr.recall), 4),
                "injected_caught": tr.injected_caught,
                "injected_total": tr.injected_total,
                "us_per_point": round(float(tr.us_per_point), 2),
                "points_per_second": round(float(tr.points_per_second), 1),
            }
            for tr in result.traces
        ],
        "explanation": {
            "step": exp.step,
            "loda_raw_score": round(float(exp.loda_raw_score), 4),
            "note": exp.note,
            "feature_names": [exp.feature_names[int(i)] for i in order],
            "contributions": [round(float(exp.contributions[int(i)]), 4) for i in order],
        },
    }
    return json.dumps(payload)
