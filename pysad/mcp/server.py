"""MCP server for PySAD streaming anomaly detection.

This module provides a Model Context Protocol (MCP) server that exposes PySAD's
streaming anomaly detectors, drift meters, and explainability tools to AI assistants.

It provides two interfaces:
1. Session-based API (backward compatible): initialize_detector, score_point, etc.
2. Auto-discovered tools: One tool per detector/statistic with explicit parameters.

Usage:
    python -m pysad.mcp.server

Or via the console script:
    pysad-mcp
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mcp.server.mcpserver import Context, MCPServer
from numpy.typing import NDArray

from pysad.mcp.tools import register_all_tools
from pysad.models import __all__ as MODEL_NAMES
from pysad.utils import ArrayStreamer, CSVStreamer, Data, generate_stream

_CALIBRATION_MODULE = "pysad.transform.probability_calibration"
_BUNDLED_STREAM_SOURCES: dict[str, tuple[str, bool]] = {
    "nyc_taxi": ("nab/realKnownCause/nyc_taxi.csv", True),
    "arrhythmia": ("arrhythmia.mat", False),
    "creditcard": ("creditcard/creditcard.arff", False),
}


def _class_name_to_key(name: str) -> str:
    """Convert a class name to a user-facing snake_case key.

    Handles common acronyms (LODA, RS, KNN, etc.) as single words.
    """
    # Known acronyms that should stay together
    acronyms = {
        "LODA": "loda",
        "RS": "rs",
        "KNN": "knn",
        "CAD": "cad",
        "ESD": "esd",
        "HTM": "htm",
        "IDKS": "idks",
        "KELOS": "kelos",
        "MCOD": "mcod",
        "MIDAS": "midas",
        "NETS": "nets",
        "OESNN": "oesnn",
        "UAD": "uad",
        "PCB": "pcb",
        "PEWMA": "pewma",
        "TEDA": "teda",
        "SAND": "sand",
        "SDO": "sdo",
        "DILOF": "dilof",
        "DAMP": "damp",
        "CPOD": "cpod",
        "D3": "d3",
        "ABSAD": "absad",
        "ADeMS": "adems",
        "AnyOut": "anyout",
        "EXPoSE": "expose",
        "FluxEV": "fluxev",
        "FRAHST": "frahst",
        "GNGA": "gnga",
        "IForestASD": "iforestasd",
        "IncrementalLOF": "incrementallof",
        "Inqmad": "inqmad",
        "KitNet": "kitnet",
        "MiLOF": "milof",
        "MStream": "mstream",
        "MemStream": "memstream",
        "NullModel": "nullmodel",
        "PerfectModel": "perfectmodel",
        "RandomModel": "randommodel",
        "ZeroShotEmbeddingDetector": "zeroshotembeddingdetector",
        "StreamingAutoregressiveDetector": "streamingautoregressivedetector",
        "StreamKMPlusPlus": "streamkmplusplus",
        "StandardAbsoluteDeviation": "standardabsolutedeviation",
        "MedianAbsoluteDeviation": "medianabsolutedeviation",
        "LocalOutlierProbability": "localoutlierprobability",
        "RobustRandomCutForest": "robustrandomcutforest",
        "OnlineIsolationForest": "onlineisolationforest",
        "OnlineOneClassSVM": "onlineoneclasssvm",
        "HalfSpaceTrees": "halfspacetrees",
        "SmartSifter": "smartsifter",
        "SeasonalHybridESD": "seasonalhybrid_esd",
        "xStream": "xstream",
        "RSForest": "rsforest",
        "RSHash": "rshash",
        "RelativeEntropy": "relativeentropy",
        "LocalPolynomialDetector": "localpolynomialdetector",
        "MatrixProfile": "matrixprofile",
        "ChangeFinder": "changefinder",
        "CostSensitiveStreamModel": "costsensitivestreammodel",
        "ApproxStorm": "approxstorm",
        "ExactStorm": "exactstorm",
        "DriftAdaptiveDetector": "driftadaptivedetector",
        "FeedbackHST": "feedbackhst",
        "HoltWinters": "holtwinters",
        "SpectralResidual": "spectralresidual",
        "ThreshLEAP": "threshleap",
        "DenStream": "denstream",
    }
    if name in acronyms:
        return acronyms[name]

    # Default conversion: insert underscore before uppercase letters
    # but handle consecutive uppercase as a single word
    result = []
    for i, c in enumerate(name):
        if c.isupper():
            if i > 0 and (name[i - 1].islower() or (i + 1 < len(name) and name[i + 1].islower())):
                result.append("_")
            result.append(c.lower())
        else:
            result.append(c)
    return "".join(result)


@dataclass
class SessionState:
    """Per-session state for the MCP server."""

    detector: Any = None
    calibrator: Any = None
    stream_data: list[NDArray[Any]] = field(default_factory=list)
    stream_labels: list[int] | None = field(default_factory=list)
    step: int = 0
    threshold: float = 0.95
    detector_name: str = "LODA"
    calibrator_name: str = "ConformalProbabilityCalibrator"
    initialized: bool = False


async def _log(ctx: Context[Any, Any] | None, message: str, level: str = "info") -> None:
    """Log *message* through the MCP context at *level* when a context is available."""
    if ctx is None:
        return
    with contextlib.suppress(Exception):
        await getattr(ctx, level)(message)


def _get_state(ctx: Context[Any, Any]) -> SessionState:
    """Get or create session state from MCP context."""
    context: Any = ctx
    if not hasattr(context, "_pysad_state"):
        context._pysad_state = SessionState()
    state: SessionState = context._pysad_state
    return state


def _build_detector_map() -> dict[str, tuple[str, str]]:
    """Map snake_case detector keys to ``(module, class_name)`` entries in pysad.models."""
    skip_names = {"BaseModel", "StreamPipeline", "ModelEnsemble"}
    detectors: dict[str, tuple[str, str]] = {}
    for name in MODEL_NAMES:
        if name in skip_names or name.startswith("_"):
            continue
        detectors[_class_name_to_key(name)] = ("pysad.models", name)
    return detectors


def _set_stream_data(state: SessionState, X: NDArray[Any], y: NDArray[Any] | None) -> None:
    """Populate session stream state from a feature matrix and an optional label vector."""
    state.stream_data = [x for x in ArrayStreamer().iter(X)]
    state.stream_labels = y.tolist() if y is not None else None
    state.step = 0


def _load_stream_data(
    path: str, max_points: int | None, drop_first_column: bool = False
) -> tuple[NDArray[Any], NDArray[Any] | None]:
    """Load a bundled data source and trim it to *max_points* when given."""
    data = Data()
    X, y = data.get_data(path, download=True)
    if drop_first_column:
        X = X[:, 1:]
    if max_points:
        X = X[:max_points]
        y = y[:max_points] if y is not None else None
    return X, y


def _calibrate(state: SessionState, score: float, thresh: float) -> dict[str, Any]:
    """Calibrate *score* with the session calibrator and return result fields."""
    calibrated = state.calibrator.fit_transform_partial(score)
    return {
        "calibrated_score": float(calibrated),
        "alert": bool(calibrated > thresh),
    }


def create_mcp_server() -> MCPServer:
    """Create and configure the PySAD MCP server.

    Returns:
        Configured MCPServer instance with both session-based and
        auto-discovered tools.
    """
    mcp = MCPServer(
        name="PySAD Streaming Anomaly Detection",
        instructions=(
            "PySAD MCP server for streaming anomaly detection. "
            "Provides session-based tools (initialize_detector, score_point, etc.) "
            "and auto-discovered tools for every detector, drift meter, and statistic."
        ),
    )

    # Register session-based tools (backward compatible)
    mcp.add_tool(initialize_detector)
    mcp.add_tool(initialize_calibrator)
    mcp.add_tool(initialize_stream)
    mcp.add_tool(score_point)
    mcp.add_tool(score_batch)
    mcp.add_tool(get_status)
    mcp.add_tool(reset_session)
    mcp.add_tool(list_detectors)
    mcp.add_tool(list_calibrators)
    mcp.add_tool(list_data_sources)

    # Register auto-discovered tools for all detectors, drift meters, etc.
    register_all_tools(mcp)

    return mcp


# Session-based tool definitions (backward compatible)
async def initialize_detector(
    detector_name: str = "loda",
    detector_kwargs: str = "{}",
    random_state: int | None = None,
    ctx: Context[Any, Any] | None = None,
) -> dict[str, Any]:
    """Initialize a streaming anomaly detector.

    Args:
        detector_name: Name of the detector (e.g., loda, halfspacetrees, rshash,
            onlineisolationforest, localoutlierprobability, robustrandomcutforest,
            knncad, smartsifter, seasonalhybrid_esd, xstream, medianabsolutedeviation,
            and all other detectors from pysad.models).
        detector_kwargs: Additional detector arguments as JSON string.
        random_state: Random seed for reproducibility.
        ctx: MCP context for logging.

    Returns:
        Dictionary with initialization status and detector info.
    """
    if ctx:
        state = _get_state(ctx)
    else:
        state = SessionState()
    await _log(ctx, f"Initializing detector: {detector_name}")

    detectors = _build_detector_map()
    name_lower = detector_name.lower().replace("-", "_")
    if name_lower not in detectors:
        available = ", ".join(sorted(detectors.keys()))
        return {
            "success": False,
            "error": f"Unknown detector: {detector_name}. Available: {available}",
        }

    try:
        kwargs = json.loads(detector_kwargs)
        if random_state is not None and "random_state" not in kwargs:
            kwargs["random_state"] = random_state

        module_name, class_name = detectors[name_lower]
        module = __import__(module_name, fromlist=[class_name])
        detector_class = getattr(module, class_name)

        state.detector = detector_class(**kwargs)
        state.detector_name = class_name
        state.initialized = True

        await _log(ctx, f"Detector {class_name} initialized successfully")

        return {
            "success": True,
            "detector": class_name,
            "parameters": kwargs,
        }
    except Exception as e:
        await _log(ctx, f"Failed to initialize detector: {e}", level="error")
        return {"success": False, "error": str(e)}


async def initialize_calibrator(
    calibrator_name: str = "conformal",
    calibrator_kwargs: str = "{}",
    window_size: int = 300,
    ctx: Context[Any, Any] | None = None,
) -> dict[str, Any]:
    """Initialize a probability calibrator for anomaly scores.

    Args:
        calibrator_name: Name of the calibrator (conformal, spot, conformal_tsfm,
            gaussian_tail, anomaly_likelihood, group_conditional_conformal).
        calibrator_kwargs: Additional calibrator arguments as JSON string.
        window_size: Calibration window size.
        ctx: MCP context for logging.

    Returns:
        Dictionary with initialization status and calibrator info.
    """
    if ctx:
        state = _get_state(ctx)
    else:
        state = SessionState()
    await _log(ctx, f"Initializing calibrator: {calibrator_name}")

    calibrators = {
        "conformal": (_CALIBRATION_MODULE, "ConformalProbabilityCalibrator"),
        "spot": (_CALIBRATION_MODULE, "SPOTProbabilityCalibrator"),
        "conformal_tsfm": (_CALIBRATION_MODULE, "ConformalTSFMCalibrator"),
        "gaussian_tail": (_CALIBRATION_MODULE, "GaussianTailProbabilityCalibrator"),
        "anomaly_likelihood": (_CALIBRATION_MODULE, "AnomalyLikelihoodProbabilityCalibrator"),
        "group_conditional_conformal": (
            _CALIBRATION_MODULE,
            "GroupConditionalConformalCalibrator",
        ),
    }

    name_lower = calibrator_name.lower().replace("-", "_")
    if name_lower not in calibrators:
        return {
            "success": False,
            "error": f"Unknown calibrator: {calibrator_name}. Available: {list(calibrators.keys())}",
        }

    try:
        kwargs = json.loads(calibrator_kwargs)
        calibrators_with_window = {
            "conformal",
            "conformal_tsfm",
            "gaussian_tail",
            "anomaly_likelihood",
            "group_conditional_conformal",
        }
        if name_lower in calibrators_with_window and "window_size" not in kwargs:
            kwargs["window_size"] = window_size

        module_name, class_name = calibrators[name_lower]
        module = __import__(module_name, fromlist=[class_name])
        calibrator_class = getattr(module, class_name)

        state.calibrator = calibrator_class(**kwargs)
        state.calibrator_name = class_name

        await _log(ctx, f"Calibrator {class_name} initialized successfully")

        return {
            "success": True,
            "calibrator": class_name,
            "parameters": kwargs,
        }
    except Exception as e:
        await _log(ctx, f"Failed to initialize calibrator: {e}", level="error")
        return {"success": False, "error": str(e)}


def _load_csv_stream(
    state: SessionState,
    source: str,
    feature_columns: str | None,
    label_column: str | None,
    max_points: int | None,
) -> None:
    """Populate session stream state from a CSV file path."""
    csv_feature_columns = (
        [int(c.strip()) for c in feature_columns.split(",")] if feature_columns else None
    )
    csv_label_column = int(label_column) if label_column else None
    csv_streamer = CSVStreamer(
        feature_columns=csv_feature_columns,
        label_column=csv_label_column,
        has_header=True,
    )
    state.stream_data = []
    state.stream_labels = []
    for item in csv_streamer.iter(source):
        if max_points and len(state.stream_data) >= max_points:
            break
        if isinstance(item, tuple):
            x_arr, y_val = item
            state.stream_labels.append(int(y_val) if y_val is not None else 0)
        else:
            x_arr = item
            state.stream_labels.append(0)
        state.stream_data.append(x_arr)
    state.step = 0


def _populate_named_stream(state: SessionState, source: str, max_points: int | None) -> bool:
    """Populate session stream state from a named synthetic or bundled source."""
    if source == "synthetic":
        X_syn, y_syn = generate_stream(num_instances=max(max_points or 1000, 200))
        _set_stream_data(state, X_syn, y_syn)
        return True
    if source not in _BUNDLED_STREAM_SOURCES:
        return False
    path, drop_first = _BUNDLED_STREAM_SOURCES[source]
    X_bun, y_bun = _load_stream_data(path, max_points, drop_first_column=drop_first)
    _set_stream_data(state, X_bun, y_bun)
    return True


async def initialize_stream(
    source: str = "synthetic",
    max_points: int | None = None,
    feature_columns: str | None = None,
    label_column: str | None = None,
    ctx: Context[Any, Any] | None = None,
) -> dict[str, Any]:
    """Initialize a data stream for anomaly detection.

    Args:
        source: Data source (synthetic, nyc_taxi, arrhythmia, creditcard, or CSV file path).
        max_points: Maximum number of points to process.
        feature_columns: Feature column names for CSV files (comma-separated).
        label_column: Label column name for CSV files.
        ctx: MCP context for logging.

    Returns:
        Dictionary with initialization status and stream info.
    """
    state = _get_state(ctx) if ctx else SessionState()
    await _log(ctx, f"Initializing stream from source: {source}")

    try:
        if source.endswith(".csv"):
            _load_csv_stream(state, source, feature_columns, label_column, max_points)
        elif not _populate_named_stream(state, source, max_points):
            return {"success": False, "error": f"Unknown data source: {source}"}

        await _log(ctx, f"Stream initialized with {len(state.stream_data)} points")
        return {
            "success": True,
            "num_points": len(state.stream_data),
            "num_features": len(state.stream_data[0]) if state.stream_data else 0,
            "has_labels": bool(state.stream_labels),
        }
    except Exception as e:
        await _log(ctx, f"Failed to initialize stream: {e}", level="error")
        return {"success": False, "error": str(e)}


def _next_score_point(
    state: SessionState,
    features: list[float] | None,
    use_stream: bool,
) -> tuple[NDArray[Any], int | None]:
    """Resolve the next feature vector and optional ground-truth label for scoring."""
    if use_stream:
        if not state.stream_data or state.step >= len(state.stream_data):
            raise ValueError("No more data in stream")
        x = state.stream_data[state.step]
        label = state.stream_labels[state.step] if state.stream_labels else None
        return x, label
    if features is None:
        raise ValueError("Features must be provided when not using stream")
    return np.asarray(features, dtype=float), None


async def score_point(
    features: list[float] | None = None,
    use_stream: bool = False,
    threshold: float | None = None,
    ctx: Context[Any, Any] | None = None,
) -> dict[str, Any]:
    """Score a single data point for anomaly detection.

    Args:
        features: Feature vector as list of floats. If None and use_stream=True,
            uses the next point from the initialized stream.
        use_stream: Whether to use the next point from the initialized stream.
        threshold: Alert threshold for calibrated score (default: 0.95).
        ctx: MCP context for logging.

    Returns:
        Dictionary with anomaly score, calibrated score, alert status, and metadata.
    """
    state = _get_state(ctx) if ctx else SessionState()
    await _log(ctx, "Scoring data point")

    if state.detector is None:
        return {
            "success": False,
            "error": "Detector not initialized. Call initialize_detector first.",
        }

    try:
        x, label = _next_score_point(state, features, use_stream)

        # Score the point
        score = state.detector.fit_score_partial(x)

        result = {
            "success": True,
            "step": state.step,
            "features": x.tolist(),
            "score": float(score),
            "label": label,
        }

        # Apply calibrator if available
        if state.calibrator is not None:
            thresh = threshold if threshold is not None else state.threshold
            result.update(_calibrate(state, score, thresh))
            result["threshold"] = thresh

        state.step += 1

        cal_msg = (
            f", Calibrated: {result.get('calibrated_score', 'N/A'):.4f}"
            if "calibrated_score" in result
            else ""
        )
        await _log(ctx, f"Score: {score:.4f}{cal_msg}")

        return result
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        await _log(ctx, f"Failed to score point: {e}", level="error")
        return {"success": False, "error": str(e)}


async def score_batch(
    features_list: list[list[float]],
    threshold: float | None = None,
    ctx: Context[Any, Any] | None = None,
) -> dict[str, Any]:
    """Score a batch of data points for anomaly detection.

    Args:
        features_list: List of feature vectors.
        threshold: Alert threshold for calibrated score (default: 0.95).
        ctx: MCP context for logging.

    Returns:
        Dictionary with list of results for each point.
    """
    if ctx:
        state = _get_state(ctx)
    else:
        state = SessionState()
    await _log(ctx, f"Scoring batch of {len(features_list)} points")

    if state.detector is None:
        return {
            "success": False,
            "error": "Detector not initialized. Call initialize_detector first.",
        }

    try:
        results = []
        thresh = threshold if threshold is not None else state.threshold

        for features in features_list:
            x = np.asarray(features, dtype=float)
            score = state.detector.fit_score_partial(x)

            result = {
                "step": state.step,
                "features": x.tolist(),
                "score": float(score),
            }

            if state.calibrator is not None:
                result.update(_calibrate(state, score, thresh))

            results.append(result)
            state.step += 1

        await _log(
            ctx,
            f"Batch scoring complete. Alerts: {sum(1 for r in results if r.get('alert', False))}",
        )

        return {
            "success": True,
            "results": results,
            "threshold": thresh,
        }
    except Exception as e:
        await _log(ctx, f"Failed to score batch: {e}", level="error")
        return {"success": False, "error": str(e)}


async def get_status(ctx: Context[Any, Any] | None = None) -> dict[str, Any]:
    """Get the current status of the MCP session.

    Args:
        ctx: MCP context for logging.

    Returns:
        Dictionary with current session status.
    """
    if ctx:
        state = _get_state(ctx)
    else:
        state = SessionState()

    return {
        "success": True,
        "initialized": state.initialized,
        "detector": state.detector_name,
        "calibrator": state.calibrator_name,
        "step": state.step,
        "threshold": state.threshold,
        "stream_length": len(state.stream_data),
        "stream_position": state.step,
        "has_stream": len(state.stream_data) > 0,
    }


async def reset_session(ctx: Context[Any, Any] | None = None) -> dict[str, Any]:
    """Reset the MCP session to initial state.

    Args:
        ctx: MCP context for logging.

    Returns:
        Dictionary with reset status.
    """
    if ctx:
        state = _get_state(ctx)
    else:
        state = SessionState()
    await _log(ctx, "Resetting session")

    state.detector = None
    state.calibrator = None
    state.stream_data = []
    state.stream_labels = None
    state.step = 0
    state.detector_name = "LODA"
    state.calibrator_name = "ConformalProbabilityCalibrator"
    state.initialized = False

    return {"success": True, "message": "Session reset successfully"}


async def list_detectors(ctx: Context[Any, Any] | None = None) -> dict[str, Any]:
    """List available streaming anomaly detectors.

    Args:
        ctx: MCP context for logging.

    Returns:
        Dictionary with list of available detectors and their descriptions.
    """
    detectors: dict[str, str] = {}
    for key, (module_name, class_name) in _build_detector_map().items():
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            doc = cls.__doc__ or ""
            # Extract first line of docstring as description
            desc = (
                doc.strip().split("\n")[0]
                if doc.strip()
                else f"{class_name} streaming anomaly detector"
            )
        except Exception:
            desc = f"{class_name} streaming anomaly detector"
        detectors[key] = desc

    return {
        "success": True,
        "detectors": [
            {"name": name, "description": desc} for name, desc in sorted(detectors.items())
        ],
    }


async def list_calibrators(ctx: Context[Any, Any] | None = None) -> dict[str, Any]:
    """List available probability calibrators.

    Args:
        ctx: MCP context for logging.

    Returns:
        Dictionary with list of available calibrators and their descriptions.
    """
    calibrators = {
        "conformal": "Conformal prediction calibrator (Vovk et al., 2005)",
        "spot": "SPOT extreme value theory calibrator (Ren et al., 2019)",
        "conformal_tsfm": "Conformal time series forecasting calibrator",
        "gaussian_tail": "Gaussian tail probability calibrator",
        "anomaly_likelihood": "Anomaly likelihood calibrator (Numenta)",
        "group_conditional_conformal": "Group-conditional conformal calibrator",
    }

    return {
        "success": True,
        "calibrators": [{"name": name, "description": desc} for name, desc in calibrators.items()],
    }


async def list_data_sources(ctx: Context[Any, Any] | None = None) -> dict[str, Any]:
    """List available built-in data sources.

    Args:
        ctx: MCP context for logging.

    Returns:
        Dictionary with list of available data sources.
    """
    sources = {
        "synthetic": "Synthetic stream with injected anomalies",
        "nyc_taxi": "NYC Taxi passenger count (NAB dataset)",
        "arrhythmia": "Arrhythmia dataset (ODDS)",
        "creditcard": "Credit card fraud dataset (ULB)",
        "csv_file": "Custom CSV file (provide path as source)",
    }

    return {
        "success": True,
        "sources": [{"name": name, "description": desc} for name, desc in sources.items()],
    }


# Create the server instance
mcp_server = create_mcp_server()


def main() -> None:
    """Entry point for the pysad-mcp console script."""
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
