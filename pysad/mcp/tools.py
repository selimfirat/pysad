"""MCP tool definitions for PySAD streaming anomaly detection.

This module provides auto-discovered tools for all PySAD detectors, drift meters,
and explainability methods. Tools are registered with the MCP server to expose
PySAD functionality to AI assistants.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, cast

import numpy as np
from mcp.server.fastmcp import FastMCP

from pysad.models import __all__ as MODEL_NAMES
from pysad.statistics import __all__ as STAT_NAMES

# Model parameter defaults for common detectors.
# These provide sensible defaults when users don't specify all parameters.
DETECTOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "LODA": {"num_bins": 10, "num_random_cuts": 100, "random_state": 42},
    "HalfSpaceTrees": {"num_trees": 25, "max_depth": 7, "window_size": 250, "random_state": 42},
    "RSHash": {"num_components": 32, "r": 1.0, "window_size": 1000, "random_state": 42},
    "LocalOutlierProbability": {
        "window_size": 100,
        "n_neighbors": 20,
        "extent": 3.0,
        "random_state": 42,
    },
    "RobustRandomCutForest": {"num_trees": 50, "tree_size": 256, "random_state": 42},
    "KNNCAD": {"window_size": 100, "n_neighbors": 5, "random_state": 42},
    "SmartSifter": {
        "num_components": 10,
        "learning_rate": 0.01,
        "forgetting_factor": 0.99,
        "random_state": 42,
    },
    "SeasonalHybridESD": {"seasonality": 7, "hybrid": True, "random_state": 42},
    "xStream": {"num_components": 10, "window_size": 100, "random_state": 42},
    "MedianAbsoluteDeviation": {"window_size": 100, "threshold": 3.0},
    "OnlineIsolationForest": {
        "num_trees": 100,
        "max_leaf_samples": 32,
        "window_size": 2048,
        "random_state": 42,
    },
    "DAMP": {"window_size": 100, "order": 2, "threshold": 3.0},
    "IncrementalLOF": {"window_size": 100, "n_neighbors": 20, "random_state": 42},
    "RSForest": {"num_trees": 50, "tree_size": 256, "random_state": 42},
    "MiLOF": {"window_size": 100, "n_neighbors": 20, "random_state": 42},
    "PEWMA": {"alpha": 0.2, "threshold": 3.0},
    "TEDA": {"threshold": 3.0},
    "D3": {"window_size": 100, "n_components": 2, "random_state": 42},
    "SAND": {"window_size": 100, "n_projections": 10, "random_state": 42},
    "SDOstream": {"window_size": 100, "num_clusters": 5, "random_state": 42},
    "KitNet": {"window_size": 100, "max_autoencoder_size": 10, "random_state": 42},
    "HTM": {"num_columns": 2048, "input_size": 10, "random_state": 42},
    "IDKS": {"window_size": 100, "num_kernels": 10, "random_state": 42},
    "FluxEV": {"window_size": 100, "n_projections": 10, "random_state": 42},
    "KELOS": {"window_size": 100, "n_components": 10, "random_state": 42},
    "MStream": {"window_size": 100, "num_grids": 10, "random_state": 42},
    "Inqmad": {"window_size": 100, "num_projections": 10, "random_state": 42},
    "Loop": {"window_size": 100, "n_neighbors": 20, "random_state": 42},
    "LocalPolynomialDetector": {"window_size": 100, "degree": 2, "random_state": 42},
    "StreamingAutoregressiveDetector": {"window_size": 100, "order": 5, "random_state": 42},
    "HoltWinters": {"alpha": 0.3, "beta": 0.1, "gamma": 0.1, "seasonal_period": 7},
    "SpectralResidual": {"window_size": 100, "saliency_window": 3, "threshold": 3.0},
    "MatrixProfile": {"window_size": 100, "subsequence_length": 10, "random_state": 42},
    "CPOD": {"window_size": 100, "n_components": 2, "random_state": 42},
    "ThreshLEAP": {"window_size": 100, "threshold": 3.0, "random_state": 42},
    "StreamKMPlusPlus": {"num_clusters": 5, "window_size": 1000, "random_state": 42},
    "ExactStorm": {"window_size": 100, "n_components": 2, "random_state": 42},
    "ApproxStorm": {"window_size": 100, "n_components": 2, "random_state": 42},
    "MCOD": {"window_size": 100, "n_neighbors": 5, "random_state": 42},
    "MIDAS": {"num_buckets": 1000, "factor": 0.5, "random_state": 42},
    "GNGA": {"max_nodes": 100, "learning_rate": 0.01, "random_state": 42},
    "ADeMS": {"window_size": 100, "n_components": 10, "random_state": 42},
    "AnyOut": {"window_size": 100, "n_projections": 10, "random_state": 42},
    "ABSAD": {"window_size": 100, "n_components": 2, "random_state": 42},
    "CostSensitiveStreamModel": {"window_size": 100, "cost_ratio": 10.0, "random_state": 42},
    "DriftAdaptiveDetector": {
        "window_size": 100,
        "base_detector": "HalfSpaceTrees",
        "random_state": 42,
    },
    "FeedbackHST": {"num_trees": 25, "max_depth": 7, "window_size": 250, "random_state": 42},
    "OeSNNUAD": {"window_size": 100, "similarity_threshold": 0.7, "random_state": 42},
    "OnlineOneClassSVM": {"window_size": 100, "nu": 0.05, "random_state": 42},
    "IForestASD": {"num_trees": 100, "max_samples": 256, "random_state": 42},
    "PCBIsolationForest": {"num_trees": 100, "max_samples": 256, "random_state": 42},
    "RelativeEntropy": {"window_size": 100, "n_bins": 10, "random_state": 42},
    "StandardAbsoluteDeviation": {"window_size": 100, "threshold": 3.0},
    "ZeroShotEmbeddingDetector": {
        "model_name": "all-MiniLM-L6-v2",
        "window_size": 100,
        "random_state": 42,
    },
}

# Drift meter parameter defaults
DRIFT_DEFAULTS: dict[str, dict[str, Any]] = {
    "ADWINMeter": {"delta": 0.002, "max_buckets": 5},
    "DDMMeter": {"warning_level": 2.0, "drift_level": 3.0, "min_num_items": 30},
    "EDDMMeter": {"warning_threshold": 0.95, "drift_threshold": 0.9, "min_num_errors": 30},
    "FHDDMMeter": {"window_size": 100, "delta": 1e-6},
    "FHDDMSMeter": {"short_window_size": 25, "long_window_size": 100, "delta": 1e-7},
    "FHDDMSAddMeter": {"short_window_size": 25, "stack_size": 4, "delta": 1e-7},
    "HDDMAMeter": {"drift_confidence": 0.001, "warning_confidence": 0.005, "two_sided": False},
    "HDDMWMeter": {
        "drift_confidence": 0.001,
        "warning_confidence": 0.005,
        "alpha": 0.05,
        "two_sided": False,
    },
    "KSWINMeter": {"alpha": 0.0001, "window_size": 300, "stat_size": 30, "random_state": 42},
    "PageHinkleyMeter": {"delta": 0.0, "threshold": 50.0, "min_instances": 30},
    "CUSUMMeter": {"threshold": 5.0, "drift": 0.0},
    "ExponentialMovingAverageMeter": {"alpha": 0.1},
    "ExponentialMovingVarianceMeter": {"alpha": 0.1},
    "KLLMeter": {"k": 256, "random_state": 42},
    "OnlinePeriodicityMeter": {"max_period": 100, "min_period": 2, "threshold": 0.5},
    "DDSketchMeter": {"alpha": 0.01, "max_bins": 1024},
    "MedianMeter": {"alpha": 0.01, "max_bins": 1024},
    "QuantileMeter": {"quantile": 0.5, "alpha": 0.01, "max_bins": 1024},
    "RelativeErrorQuantileMeter": {"quantile": 0.5, "epsilon": 0.01},
    "RunningStatistic": {},
    "AbsStatistic": {},
    "AverageMeter": {},
    "CountMeter": {},
    "MaxMeter": {},
    "MinMeter": {},
    "SumMeter": {},
    "SumSquaresMeter": {},
    "VarianceMeter": {},
}


def _get_model_class(name: str) -> type:
    """Get a model class by name from pysad.models."""
    import pysad.models as models_module

    if not hasattr(models_module, name):
        raise ValueError(f"Model {name} not found in pysad.models")
    return cast(type, getattr(models_module, name))


def _get_stat_class(name: str) -> type:
    """Get a statistic class by name from pysad.statistics."""
    import pysad.statistics as stats_module

    if not hasattr(stats_module, name):
        raise ValueError(f"Statistic {name} not found in pysad.statistics")
    return cast(type, getattr(stats_module, name))


def _instantiate_with_defaults(
    cls: type, user_kwargs: dict[str, Any], defaults: dict[str, Any]
) -> Any:
    """Instantiate a class with user kwargs merged with defaults."""
    merged = {**defaults, **user_kwargs}
    # Filter to only valid constructor parameters
    sig = inspect.signature(cls)
    valid_params = {k: v for k, v in merged.items() if k in sig.parameters}
    return cls(**valid_params)


def _serialize_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _serialize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_numpy(v) for v in obj]
    return obj


def _get_param_info(cls: type) -> dict[str, dict[str, Any]]:
    """Extract parameter info from a class constructor."""
    sig = inspect.signature(cls)
    params: dict[str, dict[str, Any]] = {}
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        ann = p.annotation
        if ann == inspect.Parameter.empty:
            type_name = "Any"
        elif isinstance(ann, str):
            type_name = ann
        elif hasattr(ann, "__name__"):
            type_name = ann.__name__
        else:
            type_name = str(ann)
        params[p.name] = {
            "type": type_name,
            "default": p.default if p.default != inspect.Parameter.empty else None,
        }
    return params


def _ann_from_str(ann_str: str) -> type:
    """Convert annotation string to type."""
    if ann_str == "int":
        return int
    if ann_str == "float":
        return float
    if ann_str == "str":
        return str
    if ann_str == "bool":
        return bool
    return Any


def _create_detector_tool(
    mcp: FastMCP, name: str, model_class: type, defaults: dict[str, Any]
) -> None:
    """Create and register a detector tool with explicit parameters."""
    param_info = _get_param_info(model_class)

    # Build the function signature dynamically
    params: list[inspect.Parameter] = [
        inspect.Parameter(
            "data",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=list[list[float]] | list[float],
        ),
        inspect.Parameter(
            "fit", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=bool, default=True
        ),
    ]
    for param_name, info in param_info.items():
        default = info["default"]
        ann = _ann_from_str(info["type"])
        params.append(
            inspect.Parameter(
                param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ann, default=default
            )
        )

    sig = inspect.Signature(params)

    async def detector_tool(
        data: list[list[float]] | list[float],
        fit: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a PySAD streaming anomaly detector on data."""
        try:
            detector = _instantiate_with_defaults(model_class, kwargs, defaults)

            # Convert input to numpy array
            X = np.array(data, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            scores = []
            for x in X:
                if fit:
                    score = detector.fit_score_partial(x)
                else:
                    score = detector.score_partial(x)
                scores.append(float(score))

            result = {
                "detector": name,
                "scores": scores,
                "num_instances": len(scores),
                "parameters": _serialize_numpy({**defaults, **kwargs}),
            }

            # Add model-specific info if available
            if hasattr(detector, "window_"):
                result["window_size"] = len(detector.window_)
            if hasattr(detector, "trees_"):
                result["num_trees"] = len(detector.trees_)

            return result
        except Exception as e:
            return {"error": str(e), "detector": name}

    # Set the signature on the function
    detector_tool.__signature__ = sig  # type: ignore[attr-defined]
    detector_tool.__doc__ = f"""Run PySAD {name} streaming anomaly detector.

Parameters:
{json.dumps(param_info, indent=2)}

Args:
    data: Input data as list of feature vectors or single vector.
    fit: Whether to fit the model (True) or only score (False).

Returns:
    Dictionary with anomaly scores and model info.
"""

    # Register the tool
    mcp.tool(name=f"pysad_{name.lower()}", description=f"PySAD {name} streaming anomaly detector")(
        detector_tool
    )


def _create_stat_tool(mcp: FastMCP, name: str, stat_class: type, defaults: dict[str, Any]) -> None:
    """Create and register a statistic tool with explicit parameters."""
    param_info = _get_param_info(stat_class)

    params: list[inspect.Parameter] = [
        inspect.Parameter("data", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=list[float]),
    ]
    for param_name, info in param_info.items():
        default = info["default"]
        ann = _ann_from_str(info["type"])
        params.append(
            inspect.Parameter(
                param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=ann, default=default
            )
        )

    sig = inspect.Signature(params)

    async def stat_tool(
        data: list[float],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a PySAD drift meter or statistic on streaming data."""
        try:
            stat = _instantiate_with_defaults(stat_class, kwargs, defaults)

            values = []
            drifts = []
            warnings = []

            for x in data:
                stat.update(x)
                values.append(stat.get())
                drifts.append(getattr(stat, "drift_detected", False))
                warnings.append(getattr(stat, "warning_detected", False))

            return {
                "statistic": name,
                "values": values,
                "drift_detected": drifts,
                "warning_detected": warnings,
                "final_value": values[-1] if values else None,
                "parameters": _serialize_numpy({**defaults, **kwargs}),
            }
        except Exception as e:
            return {"error": str(e), "statistic": name}

    stat_tool.__signature__ = sig  # type: ignore[attr-defined]
    stat_tool.__doc__ = f"""Run PySAD {name} drift meter/statistic.

Parameters:
{json.dumps(param_info, indent=2)}

Args:
    data: Input data as list of scalar values.

Returns:
    Dictionary with tracked values and drift/warning flags.
"""

    mcp.tool(name=f"pysad_{name.lower()}", description=f"PySAD {name} drift meter/statistic")(
        stat_tool
    )


def register_detector_tools(mcp: FastMCP) -> None:
    """Register all PySAD detectors as MCP tools."""
    for name in MODEL_NAMES:
        # Skip base classes and utilities
        if name in {"BaseModel", "StreamPipeline", "ModelEnsemble"}:
            continue
        if name.startswith("_"):
            continue
        try:
            model_class = _get_model_class(name)
            defaults = DETECTOR_DEFAULTS.get(name, {})
            _create_detector_tool(mcp, name, model_class, defaults)
        except Exception:
            # Skip models that can't be registered
            pass


def register_statistic_tools(mcp: FastMCP) -> None:
    """Register all PySAD statistics/drift meters as MCP tools."""
    for name in STAT_NAMES:
        if name.startswith("_"):
            continue
        try:
            stat_class = _get_stat_class(name)
            defaults = DRIFT_DEFAULTS.get(name, {})
            _create_stat_tool(mcp, name, stat_class, defaults)
        except Exception:
            pass


def register_explainability_tools(mcp: FastMCP) -> None:
    """Register explainability tools for detectors with explain_partial."""

    @mcp.tool(
        name="pysad_explain", description="Explain anomaly score using SHAP or feature attribution"
    )
    def explain_tool(
        detector_name: str,
        data: list[list[float]] | list[float],
        detector_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Explain anomaly scores using detector's explain_partial method.

        Args:
            detector_name: Name of the detector (e.g., "OnlineIsolationForest", "LODA", "HalfSpaceTrees").
            data: Input data as list of feature vectors or single vector.
            detector_kwargs: Optional parameters for detector initialization.

        Returns:
            Dictionary with explanations (SHAP values or feature attributions) per instance.
        """
        try:
            model_class = _get_model_class(detector_name)
            defaults = DETECTOR_DEFAULTS.get(detector_name, {})
            kwargs = detector_kwargs or {}
            detector = _instantiate_with_defaults(model_class, kwargs, defaults)

            X = np.array(data, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            # Fit on all data first
            for x in X:
                detector.fit_partial(x)

            # Then explain each instance
            explanations = []
            for x in X:
                if hasattr(detector, "explain_partial"):
                    expl = detector.explain_partial(x)
                    explanations.append(_serialize_numpy(expl))
                else:
                    explanations.append(None)

            return {
                "detector": detector_name,
                "explanations": explanations,
                "num_instances": len(explanations),
                "has_explain": all(e is not None for e in explanations),
            }
        except Exception as e:
            return {"error": str(e), "detector": detector_name}


def register_stream_tool(mcp: FastMCP) -> None:
    """Register a tool for running a full streaming pipeline."""

    @mcp.tool(name="pysad_stream", description="Run a streaming anomaly detection pipeline")
    def stream_tool(
        detector_name: str,
        data: list[list[float]] | list[float],
        calibrator_name: str | None = None,
        detector_kwargs: dict[str, Any] | None = None,
        calibrator_kwargs: dict[str, Any] | None = None,
        threshold: float = 0.95,
    ) -> dict[str, Any]:
        """Run a complete streaming anomaly detection pipeline with optional calibration.

        Args:
            detector_name: Name of the detector (e.g., "LODA", "HalfSpaceTrees").
            data: Input data as list of feature vectors.
            calibrator_name: Optional calibrator name (e.g., "ConformalProbabilityCalibrator", "SPOTProbabilityCalibrator").
            detector_kwargs: Optional detector parameters.
            calibrator_kwargs: Optional calibrator parameters.
            threshold: Alert threshold for calibrated scores (default 0.95).

        Returns:
            Dictionary with raw scores, calibrated scores, alerts, and metadata.
        """
        try:
            model_class = _get_model_class(detector_name)
            det_defaults = DETECTOR_DEFAULTS.get(detector_name, {})
            detector = _instantiate_with_defaults(model_class, detector_kwargs or {}, det_defaults)

            X = np.array(data, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            calibrator = None
            if calibrator_name:
                from pysad.transform.probability_calibration import (
                    AnomalyLikelihoodProbabilityCalibrator,
                    ConformalProbabilityCalibrator,
                    ConformalTSFMCalibrator,
                    GaussianTailProbabilityCalibrator,
                    GroupConditionalConformalCalibrator,
                    SPOTProbabilityCalibrator,
                )

                calibrators = {
                    "ConformalProbabilityCalibrator": ConformalProbabilityCalibrator,
                    "SPOTProbabilityCalibrator": SPOTProbabilityCalibrator,
                    "ConformalTSFMCalibrator": ConformalTSFMCalibrator,
                    "GaussianTailProbabilityCalibrator": GaussianTailProbabilityCalibrator,
                    "AnomalyLikelihoodProbabilityCalibrator": AnomalyLikelihoodProbabilityCalibrator,
                    "GroupConditionalConformalCalibrator": GroupConditionalConformalCalibrator,
                }
                if calibrator_name in calibrators:
                    calibrator = calibrators[calibrator_name](**(calibrator_kwargs or {}))

            results = []
            for x in X:
                score = detector.fit_score_partial(x)
                result = {"raw_score": float(score), "features": x.tolist()}

                if calibrator:
                    calibrated = calibrator.fit_transform_partial(score)
                    result["calibrated_score"] = float(calibrated)
                    result["alert"] = calibrated > threshold

                results.append(result)

            return {
                "detector": detector_name,
                "calibrator": calibrator_name,
                "threshold": threshold,
                "results": results,
                "num_alerts": sum(1 for r in results if r.get("alert", False)),
            }
        except Exception as e:
            return {"error": str(e), "detector": detector_name}


def register_all_tools(mcp: FastMCP) -> None:
    """Register all PySAD tools with the MCP server."""
    register_detector_tools(mcp)
    register_statistic_tools(mcp)
    register_explainability_tools(mcp)
    register_stream_tool(mcp)
