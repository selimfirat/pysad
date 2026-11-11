"""Shared contract tests auto-discovered from ``__all__`` across non-model PySAD subpackages."""

from __future__ import annotations

import inspect
import pickle
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

import pysad.evaluation
import pysad.statistics
import pysad.transform.ensemble
import pysad.transform.postprocessing
import pysad.transform.preprocessing
import pysad.transform.probability_calibration
import pysad.transform.projection
import pysad.utils
from pysad.evaluation import AUROCMetric
from pysad.statistics import AverageMeter
from pysad.utils import fix_seed


def _concrete_exports(module: Any) -> list[tuple[str, type]]:
    out: list[tuple[str, type]] = []
    for name in getattr(module, "__all__", ()):
        if name.startswith("_") or name.startswith("Base"):
            continue
        obj = getattr(module, name)
        if inspect.isclass(obj) and not inspect.isabstract(obj):
            out.append((name, obj))
    return out


PREPROCESSOR_CLASSES = _concrete_exports(pysad.transform.preprocessing)
PROJECTOR_CLASSES = _concrete_exports(pysad.transform.projection)
POSTPROCESSOR_CLASSES = _concrete_exports(pysad.transform.postprocessing)
CALIBRATOR_CLASSES = _concrete_exports(pysad.transform.probability_calibration)
ENSEMBLER_CLASSES = _concrete_exports(pysad.transform.ensemble)
STATISTIC_CLASSES = _concrete_exports(pysad.statistics)
METRIC_CLASSES = _concrete_exports(pysad.evaluation)

UNIVARIATE_PREPROCESSORS = frozenset(
    {"ModifiedSTLResidualTransformer", "OneShotSTL", "SeasonalTrendDecomposer"}
)
PREPROCESSOR_PARAMS: dict[str, dict[str, Any]] = {
    "ModifiedSTLResidualTransformer": {"period": 4, "window_size": 12},
    "OneShotSTL": {"period": 4, "num_init_periods": 3},
    "SeasonalTrendDecomposer": {"season_length": 4},
    "ShingleTransformer": {"shingle_size": 3},
}
PROJECTOR_PARAMS: dict[str, dict[str, Any]] = {
    "GaussianRandomProjector": {"num_components": 2, "random_state": 5},
    "SparseRandomProjector": {"num_components": 2, "random_state": 5},
    "StreamhashProjector": {"num_components": 2},
}
POSTPROCESSOR_PARAMS: dict[str, dict[str, Any]] = {
    "NonparametricDynamicThresholdPostprocessor": {"window_size": 12},
    "RunningAveragePostprocessor": {"window_size": 6},
    "RunningMaxPostprocessor": {"window_size": 6},
    "RunningMedianPostprocessor": {"window_size": 6},
    "RunningZScorePostprocessor": {"window_size": 6},
    "ThresholdFilter": {"threshold": 1.0},
}
CALIBRATOR_PARAMS: dict[str, dict[str, Any]] = {
    "AnomalyLikelihood": {"window_size": 20, "short_window_size": 5, "probationary_period": 10},
    "ConformalProbabilityCalibrator": {"window_size": 20},
    "GaussianTailProbabilityCalibrator": {"window_size": 20, "short_window_size": 5},
    "SPOT": {"init_length": 15},
}
ENSEMBLER_PARAMS: dict[str, dict[str, Any]] = {
    "AverageOfMaximumScoreEnsembler": {"n_buckets": 2, "random_state": 5},
    "MaximumOfAverageScoreEnsembler": {"n_buckets": 2, "random_state": 5},
}
STATISTIC_PARAMS: dict[str, dict[str, Any]] = {
    "AbsStatistic": {"statistic_cls": AverageMeter},
    "KSWINMeter": {"window_size": 20, "stat_size": 6, "random_state": 5},
    "OnlinePeriodicityMeter": {"window_size": 8},
    "RunningStatistic": {"statistic_cls": AverageMeter, "window_size": 6},
}
METRIC_PARAMS: dict[str, dict[str, Any]] = {
    "RollingMetric": {"metric_cls": AUROCMetric, "window_size": 8},
    "WindowedMetric": {"metric_cls": AUROCMetric, "window_size": 8},
}


def _make_preprocessor(name: str, cls: type) -> Any:
    return cls(**PREPROCESSOR_PARAMS.get(name, {}))


def _preprocessor_stream(name: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    if name in UNIVARIATE_PREPROCESSORS:
        return rng.normal(loc=2.0, scale=0.5, size=(24, 1))
    return rng.normal(loc=2.0, scale=0.5, size=(24, 4))


@pytest.mark.parametrize(
    ("name", "cls"), PREPROCESSOR_CLASSES, ids=[n for n, _ in PREPROCESSOR_CLASSES]
)
def test_preprocessors_partial_batch_state_and_pickle_contracts(name: str, cls: type) -> None:
    X = _preprocessor_stream(name)
    partial_t = _make_preprocessor(name, cls)
    batch_t = _make_preprocessor(name, cls)

    seq_out = np.vstack([partial_t.fit_transform_partial(x) for x in X])
    batch_out = batch_t.fit_transform(X)
    assert seq_out.ndim == 2 and seq_out.shape[0] == len(X)
    assert np.all(np.isfinite(seq_out))
    np.testing.assert_allclose(batch_out, seq_out, atol=1e-12)

    probe = X[-3:]
    np.testing.assert_allclose(
        partial_t.transform_partial(probe[0]), partial_t.transform_partial(probe[0])
    )
    np.testing.assert_allclose(
        batch_t.transform(probe),
        np.vstack([partial_t.transform_partial(x) for x in probe]),
        atol=1e-12,
    )

    restored = pickle.loads(pickle.dumps(partial_t))
    for x in probe:
        np.testing.assert_allclose(
            partial_t.fit_transform_partial(x),
            restored.fit_transform_partial(x),
            atol=1e-12,
        )


@pytest.mark.parametrize(("name", "cls"), PROJECTOR_CLASSES, ids=[n for n, _ in PROJECTOR_CLASSES])
def test_projectors_partial_batch_state_and_pickle_contracts(name: str, cls: type) -> None:
    X = np.random.default_rng(11).normal(size=(20, 6))
    partial_p = cls(**PROJECTOR_PARAMS[name])
    batch_p = cls(**PROJECTOR_PARAMS[name])

    seq_out = np.vstack([partial_p.fit_transform_partial(x) for x in X])
    batch_out = batch_p.fit_transform(X)
    assert seq_out.shape == (20, 2)
    assert np.all(np.isfinite(seq_out))
    np.testing.assert_allclose(batch_out, seq_out, atol=1e-12)

    probe = X[-4:]
    np.testing.assert_allclose(
        partial_p.transform_partial(probe[0]), partial_p.transform_partial(probe[0])
    )
    np.testing.assert_allclose(
        batch_p.transform(probe),
        np.vstack([partial_p.transform_partial(x) for x in probe]),
        atol=1e-12,
    )

    restored = pickle.loads(pickle.dumps(partial_p))
    for x in probe:
        np.testing.assert_allclose(
            partial_p.fit_transform_partial(x),
            restored.fit_transform_partial(x),
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("name", "cls"), POSTPROCESSOR_CLASSES, ids=[n for n, _ in POSTPROCESSOR_CLASSES]
)
def test_postprocessors_partial_batch_state_and_pickle_contracts(name: str, cls: type) -> None:
    scores = np.random.default_rng(13).uniform(0.1, 3.0, size=30)
    partial_p = cls(**POSTPROCESSOR_PARAMS.get(name, {}))
    batch_p = cls(**POSTPROCESSOR_PARAMS.get(name, {}))

    seq_out = np.array([partial_p.fit_transform_partial(float(s)) for s in scores])
    batch_out = batch_p.fit_transform(scores)
    assert seq_out.shape == (30,) and np.all(np.isfinite(seq_out))
    np.testing.assert_allclose(batch_out, seq_out, atol=1e-12)

    probe = scores[-5:]
    assert partial_p.transform_partial(float(probe[0])) == partial_p.transform_partial(
        float(probe[0])
    )
    np.testing.assert_allclose(
        batch_p.transform(probe),
        np.array([partial_p.transform_partial(float(s)) for s in probe]),
        atol=1e-12,
    )

    restored = pickle.loads(pickle.dumps(partial_p))
    for s in probe:
        np.testing.assert_allclose(
            partial_p.fit_transform_partial(float(s)),
            restored.fit_transform_partial(float(s)),
            atol=1e-12,
        )


@pytest.mark.parametrize(
    ("name", "cls"), CALIBRATOR_CLASSES, ids=[n for n, _ in CALIBRATOR_CLASSES]
)
def test_calibrators_probability_bounds_monotonicity_batch_and_pickle(name: str, cls: type) -> None:
    scores = np.random.default_rng(17).normal(loc=1.0, scale=0.2, size=40)
    partial_c = cls(**CALIBRATOR_PARAMS.get(name, {}))
    batch_c = cls(**CALIBRATOR_PARAMS.get(name, {}))

    seq_out = np.array([partial_c.fit_transform_partial(float(s)) for s in scores])
    batch_out = batch_c.fit_transform(scores)
    assert np.all((seq_out >= 0.0) & (seq_out <= 1.0))
    np.testing.assert_allclose(batch_out, seq_out, atol=1e-12)

    low_p = partial_c.transform_partial(1.0)
    high_p = partial_c.transform_partial(10.0)
    assert partial_c.transform_partial(1.0) == low_p
    assert high_p >= low_p

    probe = np.array([0.8, 1.0, 2.5, 8.0])
    np.testing.assert_allclose(
        batch_c.transform(probe),
        np.array([partial_c.transform_partial(float(s)) for s in probe]),
        atol=1e-12,
    )

    restored = pickle.loads(pickle.dumps(partial_c))
    for s in probe:
        np.testing.assert_allclose(
            partial_c.fit_transform_partial(float(s)),
            restored.fit_transform_partial(float(s)),
            atol=1e-12,
        )


@pytest.mark.parametrize(("name", "cls"), ENSEMBLER_CLASSES, ids=[n for n, _ in ENSEMBLER_CLASSES])
def test_ensemblers_monotonicity_batch_and_pickle_contracts(name: str, cls: type) -> None:
    S = np.random.default_rng(19).uniform(0.0, 2.0, size=(20, 4))
    partial_e = cls(**ENSEMBLER_PARAMS.get(name, {}))
    batch_e = cls(**ENSEMBLER_PARAMS.get(name, {}))

    seq_out = np.array([partial_e.fit_transform_partial(row) for row in S])
    batch_out = batch_e.fit_transform(S)
    assert seq_out.shape == (20,) and np.all(np.isfinite(seq_out))
    np.testing.assert_allclose(batch_out, seq_out, atol=1e-12)

    low_e = cls(**ENSEMBLER_PARAMS.get(name, {})).transform_partial(np.array([0.1, 0.2, 0.1, 0.2]))
    high_e = cls(**ENSEMBLER_PARAMS.get(name, {})).transform_partial(np.array([5.1, 5.2, 5.1, 5.2]))
    assert high_e > low_e

    restored = pickle.loads(pickle.dumps(partial_e))
    for row in S[-4:]:
        np.testing.assert_allclose(
            partial_e.fit_transform_partial(row),
            restored.fit_transform_partial(row),
            atol=1e-12,
        )


@pytest.mark.parametrize(("name", "cls"), STATISTIC_CLASSES, ids=[n for n, _ in STATISTIC_CLASSES])
def test_statistics_update_get_and_pickle_contracts(name: str, cls: type) -> None:
    stream = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0] * 5
    meter = cls(**STATISTIC_PARAMS.get(name, {}))
    for value in stream[:20]:
        returned = meter.update(value)
        assert returned is meter

    val = meter.get()
    assert isinstance(val, (int, float, np.floating)) and np.isfinite(float(val))

    restored = pickle.loads(pickle.dumps(meter))
    assert restored.get() == meter.get()
    for value in stream[20:]:
        np.testing.assert_allclose(restored.update(value).get(), meter.update(value).get())


@pytest.mark.parametrize(("name", "cls"), METRIC_CLASSES, ids=[n for n, _ in METRIC_CLASSES])
def test_metrics_update_batch_matches_sequential_and_pickles(name: str, cls: type) -> None:
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1], dtype=int)
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=int)

    seq_m = cls(**METRIC_PARAMS.get(name, {}))
    for yt, yp in zip(y_true, y_pred, strict=True):
        seq_m.update(int(yt), int(yp))

    batch_m = cls(**METRIC_PARAMS.get(name, {}))
    returned = batch_m.update_batch(y_true, y_pred)
    assert returned is batch_m
    np.testing.assert_allclose(batch_m.get(), seq_m.get(), atol=1e-12)

    restored = pickle.loads(pickle.dumps(seq_m))
    np.testing.assert_allclose(restored.get(), seq_m.get(), atol=1e-12)
    restored.update(1, 1)
    seq_m.update(1, 1)
    np.testing.assert_allclose(restored.get(), seq_m.get(), atol=1e-12)


def _discover_non_model_random_state_components() -> list[tuple[str, Any]]:
    discovered: list[tuple[str, Any]] = []
    for mod in (
        pysad.transform.projection,
        pysad.transform.ensemble,
        pysad.statistics,
        pysad.utils,
    ):
        for name, cls in _concrete_exports(mod):
            if "random_state" in inspect.signature(cls.__init__).parameters:
                discovered.append((name, cls))
    return discovered


RANDOM_STATE_COMPONENTS = _discover_non_model_random_state_components()


def _run_stochastic_component(name: str, cls: type, seed: int) -> np.ndarray:
    X = np.arange(80, dtype=np.float64).reshape(20, 4) + 0.25
    if issubclass(
        cls, pysad.transform.projection.GaussianRandomProjector.__bases__[0].__bases__[0]
    ):
        proj = cls(num_components=2, random_state=seed)
        return proj.fit_transform(X)
    if name in ENSEMBLER_PARAMS:
        base_kw = {k: v for k, v in ENSEMBLER_PARAMS[name].items() if k != "random_state"}
        ens = cls(**base_kw, random_state=seed)
        return np.array([ens.fit_transform_partial(row) for row in X])
    if name == "KSWINMeter":
        meter = cls(window_size=16, stat_size=6, random_state=seed)
        vals = np.sin(np.arange(40, dtype=np.float64))
        return np.array([meter.update(float(v)).get() for v in vals])
    streamer = cls(shuffle=True, random_state=seed)
    if name == "PandasStreamer":
        data: Any = pd.DataFrame(X)
    elif name == "PolarsStreamer":
        data = pl.DataFrame(X)
    else:
        data = X
    return np.vstack(list(streamer.iter(data)))


@pytest.mark.parametrize(
    ("name", "cls"),
    RANDOM_STATE_COMPONENTS,
    ids=[n for n, _ in RANDOM_STATE_COMPONENTS],
)
def test_non_model_random_state_determinism_and_global_isolation(name: str, cls: type) -> None:
    fix_seed(29)
    expected_global = np.random.rand(3)

    fix_seed(29)
    out_5a = _run_stochastic_component(name, cls, 5)
    np.testing.assert_array_equal(np.random.rand(3), expected_global)

    out_5b = _run_stochastic_component(name, cls, 5)
    out_6 = _run_stochastic_component(name, cls, 6)
    np.testing.assert_array_equal(out_5a, out_5b)
    if name != "KSWINMeter":
        assert not np.array_equal(out_5a, out_6)
