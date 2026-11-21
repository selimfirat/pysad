from __future__ import annotations

import pickle
from functools import partial

import numpy as np

from pysad.models import LODA
from pysad.transform.postprocessing import ThresholdFilter
from pysad.transform.preprocessing import FunctionTransformer
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator


def _project_first_two(x: np.ndarray) -> np.ndarray:
    return np.asarray(x[:2], dtype=np.float64)


def _scalar_norm(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def _encode_log_line(line: str, scale: float = 1.0) -> np.ndarray:
    digits = sum(ch.isdigit() for ch in line)
    is_error = float("ERROR" in line)
    return np.array([len(line) * scale, float(digits), is_error], dtype=np.float64)


class _RunningSumState:
    def __init__(self) -> None:
        self.total = 0.0

    def fit(self, x: np.ndarray) -> None:
        self.total += float(np.asarray(x, dtype=np.float64)[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        return np.array([arr[0] - self.total, arr[1]], dtype=np.float64)


def test_function_transformer_identity_and_shape_conversion():
    identity = FunctionTransformer()
    vec = np.array([1.5, -2.0, 3.0])
    out = identity.fit_transform_partial(vec)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, vec)

    scalar_t = FunctionTransformer(_scalar_norm)
    scalar_out = scalar_t.transform_partial(np.array([3.0, 4.0]))
    assert scalar_out.shape == (1,)
    assert scalar_out.dtype == np.float64
    np.testing.assert_allclose(scalar_out, [25.0])

    shrink_t = FunctionTransformer(_project_first_two)
    X = np.arange(12, dtype=np.float64).reshape(3, 4)
    batch_out = shrink_t.fit_transform(X)
    assert batch_out.shape == (3, 2)
    np.testing.assert_allclose(batch_out, X[:, :2])

    empty_fixed = FunctionTransformer(_project_first_two, output_dims=2).transform(
        np.empty((0, 4), dtype=np.float64)
    )
    assert empty_fixed.shape == (0, 2)
    empty_inferred = FunctionTransformer().transform(np.empty((0, 4), dtype=np.float64))
    assert empty_inferred.shape == (0, 4)
    empty_list = FunctionTransformer(_encode_log_line).transform([])
    assert empty_list.shape == (0, 0)


def test_function_transformer_stateful_partial_vs_batch_parity():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [2.0, -1.0]], dtype=np.float64)

    state_a = _RunningSumState()
    t_partial = FunctionTransformer(state_a.transform, fit_func=state_a.fit, output_dims=2)
    partial_out = np.vstack([t_partial.fit_transform_partial(xi) for xi in X])

    state_b = _RunningSumState()
    t_batch = FunctionTransformer(state_b.transform, fit_func=state_b.fit, output_dims=2)
    batch_out = t_batch.fit_transform(X)

    np.testing.assert_allclose(partial_out, batch_out)

    # Separate fit followed by frozen transform matches an independently fitted transformer.
    state_c = _RunningSumState()
    t_frozen = FunctionTransformer(state_c.transform, fit_func=state_c.fit, output_dims=2).fit(X)
    np.testing.assert_allclose(t_frozen.transform(X), t_batch.transform(X))


def test_function_transformer_pickle_and_pipeline_with_text_stream():
    logs = [
        "INFO user=1 ok",
        "INFO user=2 ok",
        "INFO user=3 ok",
        "INFO user=4 ok",
        "INFO user=5 ok",
        "ERROR db connection failed code=9999999",
    ]
    encoder = FunctionTransformer(partial(_encode_log_line, scale=0.5), output_dims=3)
    restored = pickle.loads(pickle.dumps(encoder))
    np.testing.assert_allclose(
        encoder.transform(logs),
        restored.transform(logs),
    )

    pipe = (
        FunctionTransformer(partial(_encode_log_line, scale=1.0), output_dims=3)
        | LODA(random_state=0)
        | ConformalProbabilityCalibrator(window_size=20)
        | ThresholdFilter(threshold=0.9)
    )
    preds_stream = [pipe.fit_predict_partial(msg) for msg in logs]
    assert all(type(p) is int and p in (0, 1) for p in preds_stream)

    pipe_batch = (
        FunctionTransformer(partial(_encode_log_line, scale=1.0), output_dims=3)
        | LODA(random_state=0)
        | ConformalProbabilityCalibrator(window_size=20)
        | ThresholdFilter(threshold=0.9)
    )
    preds_batch = pipe_batch.fit_predict(logs)
    assert preds_batch.dtype == np.int64
    np.testing.assert_array_equal(preds_batch, np.array(preds_stream, dtype=np.int64))

    frozen_partial = [pipe.predict_partial(msg) for msg in logs]
    np.testing.assert_array_equal(
        pipe_batch.predict(logs), np.array(frozen_partial, dtype=np.int64)
    )
