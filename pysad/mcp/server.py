"""MCP server for PySAD streaming anomaly detection.

This module provides an MCP (Model Context Protocol) server that exposes
PySAD's streaming anomaly detection capabilities as tools for LLM clients.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from mcp.server.fastmcp import Context, FastMCP

from pysad.utils import ArrayStreamer, Data, generate_stream


# Global state for the MCP server
@dataclass
class MCPState:
    """Global state for the MCP server session."""

    detector: Any = None
    calibrator: Any = None
    streamer: Any = None
    stream_data: list[np.ndarray] = field(default_factory=list)
    stream_labels: list[int] | None = field(default_factory=list)
    step: int = 0
    threshold: float = 0.95
    detector_name: str = "LODA"
    calibrator_name: str = "ConformalProbabilityCalibrator"
    initialized: bool = False


STATE = MCPState()


def create_mcp_server() -> FastMCP:
    """Create and configure the PySAD MCP server.

    Returns:
        Configured FastMCP server instance.
    """
    mcp = FastMCP(
        name="PySAD Streaming Anomaly Detection",
        instructions=(
            "PySAD MCP server for streaming anomaly detection. "
            "Provides tools to initialize detectors, calibrators, and data streams, "
            "and to score data points in real-time."
        ),
        host="127.0.0.1",
        port=8000,
    )

    # Register tools
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

    return mcp


# Tool definitions
async def initialize_detector(
    detector_name: str = "loda",
    detector_kwargs: str = "{}",
    random_state: int | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Initialize a streaming anomaly detector.

    Args:
        detector_name: Name of the detector (loda, half_space_trees, rs_hash,
            local_outlier_probability, robust_random_cut_forest, knncad,
            smart_sifter, seasonal_hybrid_esd, xstream, median_absolute_deviation).
        detector_kwargs: Additional detector arguments as JSON string.
        random_state: Random seed for reproducibility.
        ctx: MCP context for logging.

    Returns:
        Dictionary with initialization status and detector info.
    """
    if ctx:
        await ctx.info(f"Initializing detector: {detector_name}")

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

    name_lower = detector_name.lower().replace("-", "_")
    if name_lower not in detectors:
        return {
            "success": False,
            "error": f"Unknown detector: {detector_name}. Available: {list(detectors.keys())}",
        }

    try:
        kwargs = json.loads(detector_kwargs)
        if random_state is not None and "random_state" not in kwargs:
            kwargs["random_state"] = random_state

        module_name, class_name = detectors[name_lower]
        module = __import__(module_name, fromlist=[class_name])
        detector_class = getattr(module, class_name)

        STATE.detector = detector_class(**kwargs)
        STATE.detector_name = class_name
        STATE.initialized = True

        if ctx:
            await ctx.info(f"Detector {class_name} initialized successfully")

        return {
            "success": True,
            "detector": class_name,
            "parameters": kwargs,
        }
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to initialize detector: {e}")
        return {"success": False, "error": str(e)}


async def initialize_calibrator(
    calibrator_name: str = "conformal",
    calibrator_kwargs: str = "{}",
    window_size: int = 300,
    ctx: Context | None = None,
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
        await ctx.info(f"Initializing calibrator: {calibrator_name}")

    calibrators = {
        "conformal": ("pysad.transform.probability_calibration", "ConformalProbabilityCalibrator"),
        "spot": ("pysad.transform.probability_calibration", "SPOTProbabilityCalibrator"),
        "conformal_tsfm": ("pysad.transform.probability_calibration", "ConformalTSFMCalibrator"),
        "gaussian_tail": (
            "pysad.transform.probability_calibration",
            "GaussianTailProbabilityCalibrator",
        ),
        "anomaly_likelihood": (
            "pysad.transform.probability_calibration",
            "AnomalyLikelihoodProbabilityCalibrator",
        ),
        "group_conditional_conformal": (
            "pysad.transform.probability_calibration",
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

        STATE.calibrator = calibrator_class(**kwargs)
        STATE.calibrator_name = class_name

        if ctx:
            await ctx.info(f"Calibrator {class_name} initialized successfully")

        return {
            "success": True,
            "calibrator": class_name,
            "parameters": kwargs,
        }
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to initialize calibrator: {e}")
        return {"success": False, "error": str(e)}


async def initialize_stream(
    source: str = "synthetic",
    max_points: int | None = None,
    csv_target: str | None = None,
    csv_timestamp: str | None = None,
    feature_columns: str | None = None,
    label_column: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Initialize a data stream for anomaly detection.

    Args:
        source: Data source (synthetic, nyc_taxi, arrhythmia, creditcard, or CSV file path).
        max_points: Maximum number of points to process.
        csv_target: Target column name for CSV files.
        csv_timestamp: Timestamp column name for CSV files.
        feature_columns: Feature column names for CSV files (comma-separated).
        label_column: Label column name for CSV files.
        ctx: MCP context for logging.

    Returns:
        Dictionary with initialization status and stream info.
    """
    if ctx:
        await ctx.info(f"Initializing stream from source: {source}")

    try:
        # CSVStreamer uses integer column indices
        csv_feature_columns: list[int] | None = None
        csv_label_column: int | None = None
        csv_has_header: bool = True
        if feature_columns:
            csv_feature_columns = [int(c.strip()) for c in feature_columns.split(",")]
        if label_column:
            csv_label_column = int(label_column)
        # csv_target and csv_timestamp are not used by CSVStreamer directly
        # but kept for compatibility with terminal CLI

        if source == "synthetic":
            n_instances = max(max_points or 1000, 200)
            X, y = generate_stream(num_instances=n_instances)
            STATE.stream_data = [x for x in ArrayStreamer().iter(X)]
            STATE.stream_labels = y.tolist() if y is not None else None

        elif source == "nyc_taxi":
            data = Data()
            X, y = data.get_data("nab/realKnownCause/nyc_taxi.csv", download=True)
            X = X[:, 1:]  # Remove timestamp column
            if max_points:
                X = X[:max_points]
                y = y[:max_points] if y is not None else None
            STATE.stream_data = [x for x in ArrayStreamer().iter(X)]
            STATE.stream_labels = y.tolist() if y is not None else None

        elif source == "arrhythmia":
            data = Data()
            X, y = data.get_data("arrhythmia.mat", download=True)
            if max_points:
                X = X[:max_points]
                y = y[:max_points] if y is not None else None
            STATE.stream_data = [x for x in ArrayStreamer().iter(X)]
            STATE.stream_labels = y.tolist() if y is not None else None

        elif source == "creditcard":
            data = Data()
            X, y = data.get_data("creditcard/creditcard.arff", download=True)
            if max_points:
                X = X[:max_points]
                y = y[:max_points] if y is not None else None
            STATE.stream_data = [x for x in ArrayStreamer().iter(X)]
            STATE.stream_labels = y.tolist() if y is not None else None

        elif source.endswith(".csv"):
            from pysad.utils import CSVStreamer

            csv_streamer = CSVStreamer(
                feature_columns=csv_feature_columns,
                label_column=csv_label_column,
                has_header=csv_has_header,
            )
            count = 0
            STATE.stream_data = []
            STATE.stream_labels = []
            for item in csv_streamer.iter(source):
                if max_points and count >= max_points:
                    break
                if isinstance(item, tuple):
                    x_arr, y_val = item
                    STATE.stream_labels.append(int(y_val) if y_val is not None else 0)
                else:
                    x_arr = item
                    STATE.stream_labels.append(0)
                STATE.stream_data.append(x_arr)
                count += 1

        else:
            return {"success": False, "error": f"Unknown data source: {source}"}

        STATE.step = 0

        if ctx:
            await ctx.info(f"Stream initialized with {len(STATE.stream_data)} points")

        return {
            "success": True,
            "num_points": len(STATE.stream_data),
            "num_features": len(STATE.stream_data[0]) if STATE.stream_data else 0,
            "has_labels": STATE.stream_labels is not None and len(STATE.stream_labels) > 0,
        }
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to initialize stream: {e}")
        return {"success": False, "error": str(e)}


async def score_point(
    features: list[float] | None = None,
    use_stream: bool = False,
    threshold: float | None = None,
    ctx: Context | None = None,
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
    if STATE.detector is None:
        return {
            "success": False,
            "error": "Detector not initialized. Call initialize_detector first.",
        }

    if ctx:
        await ctx.info("Scoring data point")

    try:
        if use_stream:
            if not STATE.stream_data or STATE.step >= len(STATE.stream_data):
                return {"success": False, "error": "No more data in stream"}
            x = STATE.stream_data[STATE.step]
            label = STATE.stream_labels[STATE.step] if STATE.stream_labels else None
        else:
            if features is None:
                return {
                    "success": False,
                    "error": "Features must be provided when not using stream",
                }
            x = np.asarray(features, dtype=float)
            label = None

        # Score the point
        score = STATE.detector.fit_score_partial(x)

        result = {
            "success": True,
            "step": STATE.step,
            "features": x.tolist(),
            "score": float(score),
            "label": label,
        }

        # Apply calibrator if available
        if STATE.calibrator is not None:
            calibrated = STATE.calibrator.fit_transform_partial(score)
            result["calibrated_score"] = float(calibrated)

            # Check alert
            thresh = threshold if threshold is not None else STATE.threshold
            alert = calibrated > thresh
            result["alert"] = bool(alert)
            result["threshold"] = thresh

        STATE.step += 1

        if ctx:
            await ctx.info(
                f"Score: {score:.4f}"
                + (
                    f", Calibrated: {result.get('calibrated_score', 'N/A'):.4f}"
                    if "calibrated_score" in result
                    else ""
                )
            )

        return result
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to score point: {e}")
        return {"success": False, "error": str(e)}


async def score_batch(
    features_list: list[list[float]],
    threshold: float | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Score a batch of data points for anomaly detection.

    Args:
        features_list: List of feature vectors.
        threshold: Alert threshold for calibrated score (default: 0.95).
        ctx: MCP context for logging.

    Returns:
        Dictionary with list of results for each point.
    """
    if STATE.detector is None:
        return {
            "success": False,
            "error": "Detector not initialized. Call initialize_detector first.",
        }

    if ctx:
        await ctx.info(f"Scoring batch of {len(features_list)} points")

    try:
        results = []
        thresh = threshold if threshold is not None else STATE.threshold

        for features in features_list:
            x = np.asarray(features, dtype=float)
            score = STATE.detector.fit_score_partial(x)

            result = {
                "step": STATE.step,
                "features": x.tolist(),
                "score": float(score),
            }

            if STATE.calibrator is not None:
                calibrated = STATE.calibrator.fit_transform_partial(score)
                result["calibrated_score"] = float(calibrated)
                result["alert"] = bool(calibrated > thresh)

            results.append(result)
            STATE.step += 1

        if ctx:
            await ctx.info(
                f"Batch scoring complete. Alerts: {sum(1 for r in results if r.get('alert', False))}"
            )

        return {
            "success": True,
            "results": results,
            "threshold": thresh,
        }
    except Exception as e:
        if ctx:
            await ctx.error(f"Failed to score batch: {e}")
        return {"success": False, "error": str(e)}


async def get_status(ctx: Context | None = None) -> dict[str, Any]:
    """Get the current status of the MCP session.

    Args:
        ctx: MCP context for logging.

    Returns:
        Dictionary with current session status.
    """
    return {
        "success": True,
        "initialized": STATE.initialized,
        "detector": STATE.detector_name,
        "calibrator": STATE.calibrator_name,
        "step": STATE.step,
        "threshold": STATE.threshold,
        "stream_length": len(STATE.stream_data),
        "stream_position": STATE.step,
        "has_stream": len(STATE.stream_data) > 0,
    }


async def reset_session(ctx: Context | None = None) -> dict[str, Any]:
    """Reset the MCP session to initial state.

    Args:
        ctx: MCP context for logging.

    Returns:
        Dictionary with reset status.
    """
    if ctx:
        await ctx.info("Resetting session")

    STATE.detector = None
    STATE.calibrator = None
    STATE.stream_data = []
    STATE.stream_labels = None
    STATE.step = 0
    STATE.detector_name = "LODA"
    STATE.calibrator_name = "ConformalProbabilityCalibrator"
    STATE.initialized = False

    return {"success": True, "message": "Session reset successfully"}


async def list_detectors(ctx: Context | None = None) -> dict[str, Any]:
    """List available streaming anomaly detectors.

    Args:
        ctx: MCP context for logging.

    Returns:
        Dictionary with list of available detectors and their descriptions.
    """
    detectors = {
        "loda": "Lightweight on-line detector of anomalies (Pevný, 2016)",
        "half_space_trees": "Half-Space Trees (Tan et al., 2011)",
        "rs_hash": "RS-Hash subspace outlier detection (Sathe & Aggarwal, 2016)",
        "local_outlier_probability": "Local Outlier Probabilities (Kriegel et al., 2009)",
        "robust_random_cut_forest": "Robust Random Cut Forest (Guha et al., 2016)",
        "knncad": "Conformalized k-NN anomaly detection (Burnaev & Ishimtsev, 2016)",
        "smart_sifter": "Finite mixtures with discounting learning (Yamanishi et al., 2000)",
        "seasonal_hybrid_esd": "Seasonal Hybrid ESD (Hochenbaum et al., 2017)",
        "xstream": "xStream (Manzoor et al., 2018)",
        "median_absolute_deviation": "Median absolute deviation over sliding window",
    }

    return {
        "success": True,
        "detectors": [{"name": name, "description": desc} for name, desc in detectors.items()],
    }


async def list_calibrators(ctx: Context | None = None) -> dict[str, Any]:
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


async def list_data_sources(ctx: Context | None = None) -> dict[str, Any]:
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


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Lifespan context manager for the MCP server."""
    # Startup
    yield None
    # Shutdown
    pass


if __name__ == "__main__":
    # Run the server
    mcp_server.run(transport="stdio")
