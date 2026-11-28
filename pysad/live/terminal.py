"""Terminal dashboard for live streaming anomaly detection using ANSI escape codes."""

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysad.utils import ArrayStreamer, Data, generate_stream

# ANSI escape codes
CSI = "\x1b["
RESET = f"{CSI}0m"
CLEAR_LINE = f"{CSI}2K"
CLEAR_SCREEN = f"{CSI}2J"
CURSOR_HOME = f"{CSI}H"
CURSOR_UP = f"{CSI}1A"
CURSOR_DOWN = f"{CSI}1B"
CURSOR_SAVE = f"{CSI}s"
CURSOR_RESTORE = f"{CSI}u"
HIDE_CURSOR = f"{CSI}?25l"
SHOW_CURSOR = f"{CSI}?25h"

# Colors
RED = f"{CSI}31m"
GREEN = f"{CSI}32m"
YELLOW = f"{CSI}33m"
BLUE = f"{CSI}34m"
MAGENTA = f"{CSI}35m"
CYAN = f"{CSI}36m"
WHITE = f"{CSI}37m"
BRIGHT_RED = f"{CSI}91m"
BRIGHT_GREEN = f"{CSI}92m"
BRIGHT_YELLOW = f"{CSI}93m"
BRIGHT_BLUE = f"{CSI}94m"
BRIGHT_MAGENTA = f"{CSI}95m"
BRIGHT_CYAN = f"{CSI}96m"
BRIGHT_WHITE = f"{CSI}97m"

# Background colors
BG_RED = f"{CSI}41m"
BG_GREEN = f"{CSI}42m"
BG_YELLOW = f"{CSI}43m"
BG_BLUE = f"{CSI}44m"

FEATURE_COLORS = (CYAN, BLUE, MAGENTA, YELLOW, GREEN, BRIGHT_CYAN)
SCORE_COLOR = GREEN
ALERT_COLOR = RED
THRESHOLD_COLOR = YELLOW
RIBBON_COLOR = BLUE
LABEL_COLOR = WHITE
DIM_COLOR = f"{CSI}90m"


@dataclass
class TerminalConfig:
    """Configuration for the terminal dashboard."""

    width: int = 80
    height: int = 24
    max_features: int = 4
    history_length: int = 100
    refresh_rate: float = 0.1  # seconds
    show_labels: bool = True
    show_ribbon: bool = True


@dataclass
class StreamState:
    """Current state of the streaming dashboard."""

    step: int = 0
    feature_values: list[list[float]] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    calibrated_scores: list[float] = field(default_factory=list)
    lower_bounds: list[float] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)
    alerts: list[bool] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    threshold: float = 0.95
    detector_name: str = "Detector"
    calibrator_name: str = "Calibrator"


class TerminalDashboard:
    """Live terminal dashboard for streaming anomaly detection."""

    def __init__(self, config: TerminalConfig | None = None, output: Any = None) -> None:
        self.config = config or TerminalConfig()
        self.output = output or sys.stdout
        self.state = StreamState()
        self._first_render = True
        self._lines_rendered = 0

    def _write(self, text: str) -> None:
        self.output.write(text)
        self.output.flush()

    def _move_cursor_up(self, lines: int) -> None:
        if lines > 0:
            self._write(f"{CSI}{lines}A")

    def _clear_line(self) -> None:
        self._write(CLEAR_LINE)

    def _clear_screen(self) -> None:
        self._write(CLEAR_SCREEN + CURSOR_HOME)

    def _hide_cursor(self) -> None:
        self._write(HIDE_CURSOR)

    def _show_cursor(self) -> None:
        self._write(SHOW_CURSOR)

    def _render_sparkline(
        self,
        values: list[float],
        width: int,
        color: str = WHITE,
        reference: float | None = None,
    ) -> str:
        """Render a sparkline (mini line chart) for the given values."""
        if not values:
            return " " * width

        # Take the last `width` values
        recent = values[-width:]
        if len(recent) < 2:
            return " " * (width - len(recent)) + "●" * len(recent)

        min_val = min(recent)
        max_val = max(recent)
        if max_val == min_val:
            max_val = min_val + 1e-6

        # Sparkline characters (8 levels)
        sparks = " ▁▂▃▄▅▆▇█"
        num_levels = len(sparks) - 1

        result = []
        for v in recent:
            normalized = (v - min_val) / (max_val - min_val)
            level = int(normalized * num_levels)
            level = max(0, min(level, num_levels))
            result.append(sparks[level])

        spark_str = "".join(result)

        # Add reference line marker if provided
        if reference is not None and min_val <= reference <= max_val:
            ref_level = int(((reference - min_val) / (max_val - min_val)) * num_levels)
            ref_level = max(0, min(ref_level, num_levels))
            # We can't easily mark the reference in a single-line sparkline,
            # so we'll just note it in the label

        return f"{color}{spark_str}{RESET}"

    def _render_score_bar(self, score: float, threshold: float, width: int = 30) -> str:
        """Render a horizontal bar for the current score."""
        filled = int(score * width)
        filled = max(0, min(filled, width))
        thresh_pos = int(threshold * width)
        thresh_pos = max(0, min(thresh_pos, width))

        bar = []
        for i in range(width):
            if i < filled:
                if i == thresh_pos:
                    bar.append(f"{ALERT_COLOR}▓{RESET}")
                elif score > threshold:
                    bar.append(f"{ALERT_COLOR}█{RESET}")
                else:
                    bar.append(f"{SCORE_COLOR}█{RESET}")
            elif i == thresh_pos:
                bar.append(f"{THRESHOLD_COLOR}▁{RESET}")
            else:
                bar.append(f"{DIM_COLOR}░{RESET}")

        return "".join(bar)

    @staticmethod
    def _norm_pos(value: float, min_val: float, span: float, width: int) -> int:
        """Normalize a metric into `[0, width - 1]` for ASCII ribbon placement."""
        pos = int(((value - min_val) / span) * (width - 1))
        return max(0, min(pos, width - 1))

    @staticmethod
    def _ribbon_chars(l_pos: int, u_pos: int, s_pos: int, width: int) -> tuple[str, str]:
        """Return `(upper_char, lower_char)` for one ribbon column."""
        if width <= 0:
            return " ", " "
        upper_char = f"{RIBBON_COLOR}═{RESET}"
        if l_pos > u_pos:
            return upper_char, " "
        if l_pos == s_pos:
            return upper_char, f"{SCORE_COLOR}●{RESET}"
        return upper_char, f"{RIBBON_COLOR}░{RESET}"

    def _render_ribbon(self, width: int) -> str:
        """Render the conformal ribbon (confidence band)."""
        if not self.state.lower_bounds or not self.state.upper_bounds:
            return ""

        recent_lower = self.state.lower_bounds[-width:]
        recent_upper = self.state.upper_bounds[-width:]
        recent_scores = self.state.calibrated_scores[-width:]
        if len(recent_lower) < 2:
            return ""

        min_val = min(min(recent_lower), min(recent_scores))
        max_val = max(max(recent_upper), max(recent_scores))
        span = max_val - min_val if max_val != min_val else 1e-6

        upper_line: list[str] = []
        lower_line: list[str] = []
        for i in range(len(recent_lower)):
            s_val = recent_scores[i] if i < len(recent_scores) else 0
            u_pos = self._norm_pos(recent_upper[i], min_val, span, width)
            l_pos = self._norm_pos(recent_lower[i], min_val, span, width)
            s_pos = self._norm_pos(s_val, min_val, span, width)
            u_char, l_char = self._ribbon_chars(l_pos, u_pos, s_pos, width)
            upper_line.append(u_char)
            lower_line.append(l_char)

        return f"{''.join(upper_line)}\n{''.join(lower_line)}"

    def _build_header_box(self) -> str:
        """Build the bordered ANSI header box for the terminal dashboard."""
        cfg = self.config
        state = self.state
        plain_header = (
            f"PySAD Live Anomaly Detection │ "
            f"{state.detector_name} + {state.calibrator_name} │ "
            f"Step: {state.step} │ "
            f"Threshold: {state.threshold:.3f}"
        )
        colored_header = (
            f"{BRIGHT_CYAN}PySAD Live Anomaly Detection{RESET} │ "
            f"{CYAN}{state.detector_name}{RESET} + {CYAN}{state.calibrator_name}{RESET} │ "
            f"{YELLOW}Step: {state.step}{RESET} │ "
            f"{RED}Threshold: {state.threshold:.3f}{RESET}"
        )
        box_width = max(cfg.width, len(plain_header) + 4)
        pad = " " * (box_width - 4 - len(plain_header))
        border = "═" * (box_width - 2)
        return (
            f"{BRIGHT_WHITE}╔{border}╗{RESET}\n"
            f"{BRIGHT_WHITE}║{RESET} {colored_header}{pad} {BRIGHT_WHITE}║{RESET}\n"
            f"{BRIGHT_WHITE}╚{border}╝{RESET}"
        )

    def _build_feature_lines(self) -> list[str]:
        """Render recent feature sparklines capped at `config.max_features`."""
        state = self.state
        if not state.feature_values:
            return []
        cfg = self.config
        num_features = min(len(state.feature_values[0]), cfg.max_features)
        lines: list[str] = []
        for feat_idx in range(num_features):
            feat_name = (
                state.feature_names[feat_idx]
                if feat_idx < len(state.feature_names)
                else f"feature_{feat_idx}"
            )
            feat_vals = [v[feat_idx] for v in state.feature_values]
            color = FEATURE_COLORS[feat_idx % len(FEATURE_COLORS)]
            spark = self._render_sparkline(feat_vals, cfg.width - 20, color)
            lines.append(f"  {color}{feat_name:<16}{RESET} {spark} {feat_vals[-1]:>8.3f}")
        return lines

    def _build_score_lines(self) -> list[str]:
        """Render current score bar and optional conformal prediction ribbon."""
        state = self.state
        if not state.calibrated_scores:
            return []
        cfg = self.config
        current_score = state.calibrated_scores[-1]
        is_alert = current_score > state.threshold
        alert_badge = f" {BRIGHT_RED}🔔 ALERT{RESET}" if is_alert else ""
        lines = [
            (
                f"  {SCORE_COLOR}Score:{RESET}      "
                f"{self._render_score_bar(current_score, state.threshold, cfg.width - 25)} "
                f"{ALERT_COLOR if is_alert else SCORE_COLOR}{current_score:.4f}{RESET}"
                f"{alert_badge}"
            )
        ]
        if cfg.show_ribbon and state.lower_bounds and state.upper_bounds:
            ribbon = self._render_ribbon(cfg.width - 10)
            if ribbon:
                lines.append(f"  {RIBBON_COLOR}Ribbon:{RESET}")
                lines.extend(f"         {r_line}" for r_line in ribbon.split("\n"))
        return lines

    @staticmethod
    def _build_history_line(label: str, color: str, items: Sequence[int] | Sequence[bool], on_symbol: str) -> str:
        """Render a compact 20-step status strip for alerts or ground-truth labels."""
        recent = list(items[-20:])
        strip = "".join(on_symbol if item else "⚪" for item in recent)
        total = sum(int(bool(item)) for item in recent)
        return f"  {color}{label:<12}{RESET}{strip} ({total}/{len(recent)})"

    def render(self) -> None:
        """Render the current dashboard state."""
        if self._first_render:
            self._clear_screen()
            self._hide_cursor()
            self._first_render = False

        if self._lines_rendered > 0:
            self._move_cursor_up(self._lines_rendered)

        cfg = self.config
        state = self.state
        lines = [self._build_header_box()]
        lines.extend(self._build_feature_lines())
        lines.append(f"  {DIM_COLOR}{'─' * (cfg.width - 4)}{RESET}")
        lines.extend(self._build_score_lines())

        if state.alerts:
            lines.append(self._build_history_line("Alerts:", YELLOW, state.alerts, "🔴"))
        if cfg.show_labels and state.labels:
            lines.append(
                self._build_history_line(
                    "Labels:", GREEN, [lab == 1 for lab in state.labels], "🟢"
                )
            )

        elapsed = time.time() - getattr(self, "_start_time", time.time())
        rate = state.step / elapsed if elapsed > 0 else 0
        lines.append(
            f"  {DIM_COLOR}Points: {state.step} │ Rate: {rate:.1f} pts/s │ "
            f"Alerts: {sum(state.alerts)} │ "
            f"Press Ctrl+C to stop{RESET}"
        )

        self._write("\n".join(lines) + "\n")
        self._lines_rendered = len(lines)

    def update(
        self,
        feature_vector: NDArray[Any],
        score: float,
        calibrated_score: float,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        alert: bool = False,
        label: int | None = None,
    ) -> None:
        """Update the dashboard with a new data point."""
        state = self.state

        # Initialize feature names if first update
        if not state.feature_names and len(feature_vector) > 0:
            state.feature_names = [f"sensor_{i + 1}" for i in range(len(feature_vector))]

        state.step += 1
        state.feature_values.append(feature_vector.tolist())
        state.scores.append(score)
        state.calibrated_scores.append(calibrated_score)

        if lower_bound is not None:
            state.lower_bounds.append(lower_bound)
        if upper_bound is not None:
            state.upper_bounds.append(upper_bound)

        state.alerts.append(alert)
        if label is not None:
            state.labels.append(label)

        # Trim history
        max_hist = self.config.history_length
        for attr in [
            "feature_values",
            "scores",
            "calibrated_scores",
            "lower_bounds",
            "upper_bounds",
            "alerts",
            "labels",
        ]:
            lst = getattr(state, attr)
            if len(lst) > max_hist:
                setattr(state, attr, lst[-max_hist:])

        self.render()

    def set_metadata(
        self,
        detector_name: str,
        calibrator_name: str,
        threshold: float,
        feature_names: list[str] | None = None,
    ) -> None:
        """Set metadata for the dashboard."""
        self.state.detector_name = detector_name
        self.state.calibrator_name = calibrator_name
        self.state.threshold = threshold
        if feature_names:
            self.state.feature_names = feature_names

    def start_timer(self) -> None:
        """Start the timing for rate calculation."""
        self._start_time = time.time()

    def cleanup(self) -> None:
        """Clean up terminal state."""
        self._show_cursor()
        self._write("\n")


_DATASET_STREAMS = {
    "nyc_taxi": ("nab/realKnownCause/nyc_taxi.csv", True),
    "arrhythmia": ("arrhythmia.mat", False),
    "creditcard": ("creditcard/creditcard.arff", False),
}


def _stream_synthetic(
    max_points: int | None, **kwargs: Any
) -> Iterator[tuple[NDArray[Any], int | None]]:
    """Yield `(x, label)` pairs from the synthetic benchmark generator."""
    n_instances = max(max_points or 1000, 200)
    X, y = generate_stream(num_instances=n_instances, **kwargs)
    streamer = ArrayStreamer()
    for x, label in zip(streamer.iter(X), y, strict=False):
        yield x, int(label)


def _stream_remote_dataset(
    dataset_file: str,
    drop_timestamp: bool,
    max_points: int | None,
) -> Iterator[tuple[NDArray[Any], int | None]]:
    """Yield `(x, label)` pairs from a named cached/remote dataset."""
    X, y = Data().get_data(dataset_file, download=True)
    if drop_timestamp:
        X = X[:, 1:]
    if max_points:
        X = X[:max_points]
        y = y[:max_points] if y is not None else None
    streamer = ArrayStreamer()
    labels: Any = y if y is not None else [None] * len(X)
    for x, label in zip(streamer.iter(X), labels, strict=False):
        yield x, int(label) if label is not None else None


def _stream_csv(
    source: str,
    max_points: int | None,
    delimiter: str,
    has_header: bool,
    kwargs: dict[str, Any],
) -> Iterator[tuple[NDArray[Any], int | None]]:
    """Yield `(x, label)` pairs from a local CSV file."""
    from pysad.utils import CSVStreamer

    csv_streamer = CSVStreamer(
        feature_columns=kwargs.pop("feature_columns", None),
        label_column=kwargs.pop("label_column", None),
        delimiter=delimiter,
        has_header=has_header,
    )
    count = 0
    for item in csv_streamer.iter(source):
        if max_points and count >= max_points:
            break
        if isinstance(item, tuple):
            x_arr, y_val = item
            label = int(y_val) if y_val is not None else None
        else:
            x_arr = item
            label = None
        yield x_arr, label
        count += 1


def create_data_stream(
    source: str,
    max_points: int | None = None,
    **kwargs: Any,
) -> Iterator[tuple[NDArray[Any], int | None]]:
    """Create a data stream from various sources.

    Yields:
        Tuples of (feature_vector, label_or_none).
    """
    delimiter = kwargs.pop("delimiter", ",")
    has_header = kwargs.pop("has_header", True)

    if source == "synthetic":
        return _stream_synthetic(max_points, **kwargs)
    if source in _DATASET_STREAMS:
        dataset_file, drop_ts = _DATASET_STREAMS[source]
        return _stream_remote_dataset(dataset_file, drop_ts, max_points)
    if source.endswith(".csv"):
        return _stream_csv(source, max_points, delimiter, has_header, kwargs)
    raise ValueError(f"Unknown data source: {source}")


def compute_feature_bounds(
    source: str,
    max_points: int | None = None,
    sample_size: int = 1000,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute feature min/max bounds by sampling from the data source."""
    stream = create_data_stream(source, max_points=max_points, **kwargs)
    samples = []
    for i, (x, _) in enumerate(stream):
        samples.append(x)
        if i >= sample_size - 1:
            break
    if not samples:
        # Fallback bounds
        return np.array([-10.0]), np.array([10.0])
    arr = np.array(samples)
    return np.min(arr, axis=0), np.max(arr, axis=0)


def get_detector(
    name: str,
    feature_mins: NDArray[Any] | None = None,
    feature_maxes: NDArray[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Get a detector instance by name.

    Args:
        name: Detector name.
        feature_mins: Minimum feature values (required for some detectors).
        feature_maxes: Maximum feature values (required for some detectors).
        **kwargs: Additional detector arguments.

    Returns:
        Detector instance.
    """
    detectors = {
        "loda": ("pysad.models", "LODA"),
        "half_space_trees": ("pysad.models", "HalfSpaceTrees"),
        "rs_hash": ("pysad.models", "RSHash"),
        "local_outlier_probability": ("pysad.models", "LocalOutlierProbability"),
        "robust_random_cut_forest": ("pysad.models", "RobustRandomCutForest"),
        "knncad": ("pysad.models", "KNNCAD"),
        "smart_sifter": ("pysad.models", "SmartSifter"),
        "seasonal_hybrid_esd": ("pysad.models", "SeasonalHybridESD"),
        "xstream": ("pysad.models", "xStream"),
        "median_absolute_deviation": ("pysad.models", "MedianAbsoluteDeviation"),
    }

    name_lower = name.lower().replace("-", "_")
    if name_lower not in detectors:
        raise ValueError(f"Unknown detector: {name}. Available: {list(detectors.keys())}")

    module_name, class_name = detectors[name_lower]
    module = __import__(module_name, fromlist=[class_name])
    detector_class = getattr(module, class_name)

    # Inject feature bounds for detectors that require them
    if name_lower in ("half_space_trees", "rs_hash"):
        if feature_mins is not None and feature_maxes is not None:
            kwargs.setdefault("feature_mins", feature_mins)
            kwargs.setdefault("feature_maxes", feature_maxes)

    return detector_class(**kwargs)


_CALIBRATOR_CLASSES = {
    "conformal": "ConformalProbabilityCalibrator",
    "spot": "SPOTProbabilityCalibrator",
    "conformal_tsfm": "ConformalTSFMCalibrator",
    "gaussian_tail": "GaussianTailProbabilityCalibrator",
    "anomaly_likelihood": "AnomalyLikelihoodProbabilityCalibrator",
    "group_conditional_conformal": "GroupConditionalConformalCalibrator",
}


def get_calibrator(name: str, **kwargs: Any) -> Any:
    """Get a calibrator instance by name."""
    name_lower = name.lower().replace("-", "_")
    class_name = _CALIBRATOR_CLASSES.get(name_lower)
    if class_name is None:
        raise ValueError(
            f"Unknown calibrator: {name}. Available: {list(_CALIBRATOR_CLASSES.keys())}"
        )

    module = importlib.import_module("pysad.transform.probability_calibration")
    return getattr(module, class_name)(**kwargs)


def _quantile_interval(sorted_scores: Sequence[float], confidence: float) -> tuple[float, float]:
    """Compute clipped lower/upper quantile bounds from pre-sorted calibration scores."""
    n = len(sorted_scores)
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    lower_idx = max(0, min(int(np.floor(alpha / 2 * n)), n - 1))
    upper_idx = max(0, min(int(np.ceil((1 - alpha / 2) * n)) - 1, n - 1))
    return float(sorted_scores[lower_idx]), float(sorted_scores[upper_idx])


def compute_conformal_bounds(
    calibrator: Any,
    score: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute conformal prediction bounds (lower, upper) for a score.

    For the ConformalProbabilityCalibrator, the bounds are based on the
    quantiles of the calibration window.
    """
    if hasattr(calibrator, "window") and hasattr(calibrator, "_sorted"):
        return _quantile_interval(calibrator._sorted, confidence)
    if hasattr(calibrator, "window_") and hasattr(calibrator.window_, "__iter__"):
        return _quantile_interval(sorted(calibrator.window_), confidence)
    if hasattr(calibrator, "threshold_"):
        return 0.0, float(calibrator.threshold_)
    return 0.0, 1.0
