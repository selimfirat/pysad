from __future__ import annotations

import numpy as np
import pytest

from pysad.models import (
    FRAHST,
    GNGA,
    LODA,
    DriftAdaptiveDetector,
    FluxEV,
    OeSNNUAD,
    StreamPipeline,
)
from pysad.transform.postprocessing import (
    AveragePostprocessor,
    NonparametricDynamicThresholdPostprocessor,
    QuantileFilter,
    ThresholdFilter,
)
from pysad.transform.preprocessing import InstanceStandardScaler
from pysad.transform.probability_calibration import (
    AnomalyLikelihoodProbabilityCalibrator,
    ConformalProbabilityCalibrator,
    ConformalTSFMCalibrator,
    GaussianTailProbabilityCalibrator,
    GroupConditionalConformalCalibrator,
    SPOTProbabilityCalibrator,
)


def test_output_kind_across_models_calibrators_filters_and_pipelines():
    assert LODA().output_kind == "score"
    assert FRAHST().output_kind == "decision"
    assert FluxEV().output_kind == "score"
    assert GNGA().output_kind == "score"
    assert OeSNNUAD().output_kind == "score"

    for cal in (
        AnomalyLikelihoodProbabilityCalibrator(),
        ConformalProbabilityCalibrator(window_size=10),
        ConformalTSFMCalibrator(alpha=0.1, gamma=0.005, window_size=10),
        GaussianTailProbabilityCalibrator(window_size=10, short_window_size=3),
        GroupConditionalConformalCalibrator(window_size=10),
        SPOTProbabilityCalibrator(num_init=10),
    ):
        assert cal.output_kind == "probability"

    assert AveragePostprocessor(window_size=5).output_kind == "score"
    assert NonparametricDynamicThresholdPostprocessor(window_size=12).output_kind == "score"
    assert ThresholdFilter(threshold=0.5).output_kind == "decision"
    assert QuantileFilter(q=0.95).output_kind == "decision"

    assert (InstanceStandardScaler() | LODA()).output_kind == "score"
    assert (LODA() | ConformalProbabilityCalibrator(window_size=10)).output_kind == "probability"
    assert (
        LODA() | ConformalProbabilityCalibrator(window_size=10) | ThresholdFilter(0.9)
    ).output_kind == "decision"
    assert DriftAdaptiveDetector(LODA).output_kind == "score"
    assert DriftAdaptiveDetector(FRAHST).output_kind == "decision"


def test_invalid_decision_to_probability_compositions_raise_value_error():
    with pytest.raises(ValueError, match="probability calibrator"):
        _ = FRAHST() | SPOTProbabilityCalibrator(num_init=10)

    with pytest.raises(ValueError, match="probability calibrator"):
        _ = LODA() | ThresholdFilter(0.5) | ConformalProbabilityCalibrator(window_size=10)

    with pytest.raises(ValueError, match="probability calibrator"):
        _ = (
            LODA()
            | QuantileFilter(q=0.9)
            | ConformalTSFMCalibrator(alpha=0.1, gamma=0.005, window_size=10)
        )

    with pytest.raises(ValueError, match="probability calibrator"):
        _ = ThresholdFilter(0.5) | ConformalProbabilityCalibrator(window_size=10)

    with pytest.raises(ValueError, match="probability calibrator"):
        StreamPipeline(
            LODA(),
            postprocessors=[QuantileFilter(q=0.9), ConformalProbabilityCalibrator(window_size=10)],
        )

    # Valid compositions remain allowed.
    valid_cal_then_filter = (
        LODA(random_state=0)
        | ConformalProbabilityCalibrator(window_size=20)
        | ThresholdFilter(0.95)
    )
    assert valid_cal_then_filter.output_kind == "decision"

    valid_filter_then_smoother = (
        LODA(random_state=0) | ThresholdFilter(0.5) | AveragePostprocessor(window_size=5)
    )
    assert valid_filter_then_smoother.output_kind == "score"


def test_non_decision_models_and_pipelines_reject_predict_without_mutating_state():
    X = np.random.default_rng(0).normal(size=(15, 3))
    targets = [
        LODA(random_state=0),
        InstanceStandardScaler() | LODA(random_state=0),
        LODA(random_state=0) | ConformalProbabilityCalibrator(window_size=20),
        DriftAdaptiveDetector(lambda: LODA(random_state=0)),
    ]

    for model in targets:
        with pytest.raises(TypeError, match="ThresholdFilter"):
            model.predict_partial(X[0])
        with pytest.raises(TypeError, match="ThresholdFilter"):
            model.fit_predict_partial(X[0])
        with pytest.raises(TypeError, match="ThresholdFilter"):
            model.predict(X[:5])
        with pytest.raises(TypeError, match="ThresholdFilter"):
            model.fit_predict(X[:5])

    # State remained completely untouched after failed fit_predict calls.
    fresh = LODA(random_state=0)
    rejected = targets[0]
    np.testing.assert_allclose(rejected.fit_score(X), fresh.fit_score(X))

    fresh_pipe = InstanceStandardScaler() | LODA(random_state=0)
    np.testing.assert_allclose(targets[1].fit_score(X), fresh_pipe.fit_score(X))


def test_decision_models_predict_matches_paper_threshold_and_pre_fit_semantics():
    rng = np.random.default_rng(3)

    # FRAHST: output_kind == "decision", returns {0.0, 1.0} scores -> exact 0/1 ints.
    X_mv = rng.normal(size=(30, 4))
    X_mv[20:, :] += 8.0
    m_score = FRAHST(random_state=0)
    m_pred = FRAHST(random_state=0)
    scores = m_score.fit_score(X_mv)
    preds = m_pred.fit_predict(X_mv)
    assert preds.dtype == np.int64
    np.testing.assert_array_equal(preds, (scores > 0.0).astype(np.int64))
    assert type(m_pred.predict_partial(X_mv[0])) is int
    np.testing.assert_array_equal(
        m_pred.predict(X_mv[:5]),
        (m_score.score(X_mv[:5]) > 0.0).astype(np.int64),
    )

    # FluxEV: uses pre-fit threshold_ before updating state on fit_predict_partial.
    X_uni = rng.normal(loc=0.0, scale=0.2, size=(60, 1))
    X_uni[50] = 8.0
    f_ref = FluxEV(window_size=4, period=None, calibration_size=15)
    f_pred = FluxEV(window_size=4, period=None, calibration_size=15)
    expected_fluxev = []
    actual_fluxev = []
    for xi in X_uni:
        pre_thresh = f_ref.threshold_
        expected_fluxev.append(int(f_ref.fit_score_partial(xi) > pre_thresh))
        p = f_pred.fit_predict_partial(xi)
        assert type(p) is int
        actual_fluxev.append(p)
    assert actual_fluxev == expected_fluxev
    assert 1 in actual_fluxev

    # GNGA: positive anomaly margin triggers binary prediction 1.
    g_ref = GNGA(k=5, max_neurons=20, random_state=0)
    g_pred = GNGA(k=5, max_neurons=20, random_state=0)
    g_scores = g_ref.fit_score(X_mv)
    g_preds = g_pred.fit_predict(X_mv)
    np.testing.assert_array_equal(g_preds, (g_scores > 0.0).astype(np.int64))

    # OeSNNUAD: score exceeding epsilon triggers binary prediction 1.
    o_ref = OeSNNUAD(window_size=10, num_input_neurons=5, epsilon=2.0, random_state=0)
    o_pred = OeSNNUAD(window_size=10, num_input_neurons=5, epsilon=2.0, random_state=0)
    o_scores = o_ref.fit_score(X_uni)
    o_preds = o_pred.fit_predict(X_uni)
    np.testing.assert_array_equal(o_preds, (o_scores > o_ref.epsilon).astype(np.int64))


def test_decision_pipelines_and_drift_wrapper():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 3))
    X[-3:] += 6.0

    # ThresholdFilter and QuantileFilter pipelines (partial vs batch parity).
    for post in (ThresholdFilter(0.85), QuantileFilter(q=0.9)):
        p_partial = InstanceStandardScaler() | LODA(random_state=0) | post
        seq_preds = [p_partial.fit_predict_partial(xi) for xi in X]
        assert all(type(p) is int and p in (0, 1) for p in seq_preds)

        fresh_post = (
            ThresholdFilter(0.85) if isinstance(post, ThresholdFilter) else QuantileFilter(q=0.9)
        )
        p_batch = InstanceStandardScaler() | LODA(random_state=0) | fresh_post
        batch_preds = p_batch.fit_predict(X)
        assert batch_preds.dtype == np.int64
        np.testing.assert_array_equal(batch_preds, np.array(seq_preds, dtype=np.int64))

        frozen_seq = np.array([p_partial.predict_partial(xi) for xi in X[:6]], dtype=np.int64)
        np.testing.assert_array_equal(p_batch.predict(X[:6]), frozen_seq)

    # NonparametricDynamicThresholdPostprocessor requires nonnegative scores (e.g., calibrated probabilities).
    ndt_pipe = (
        LODA(random_state=0)
        | ConformalProbabilityCalibrator(window_size=15)
        | NonparametricDynamicThresholdPostprocessor(window_size=15)
    )
    ndt_preds = ndt_pipe.fit_predict(X)
    assert ndt_preds.dtype == np.int64
    assert set(np.unique(ndt_preds)).issubset({0, 1})

    # DriftAdaptiveDetector wrapping a decision-capable model uses pre-reset decision.
    drift_scores = DriftAdaptiveDetector(lambda: FRAHST(random_state=0), threshold=1.0)
    drift_preds = DriftAdaptiveDetector(lambda: FRAHST(random_state=0), threshold=1.0)
    s_out = drift_scores.fit_score(X)
    p_out = drift_preds.fit_predict(X)
    np.testing.assert_array_equal(p_out, (s_out > 0.0).astype(np.int64))
    assert drift_preds.num_resets_ == drift_scores.num_resets_


def test_predict_methods_reject_nan_without_mutating_state():
    rng = np.random.default_rng(7)
    X_clean = rng.normal(size=(10, 3))
    nan_x = np.array([1.0, np.nan, 0.5])
    X_nan = np.vstack([X_clean[:4], nan_x])

    for factory in (
        lambda: FRAHST(random_state=0),
        lambda: LODA(random_state=0) | ThresholdFilter(0.5),
    ):
        model = factory()
        ref = factory()
        model.fit(X_clean)
        ref.fit(X_clean)

        for method, arg in (
            ("predict_partial", nan_x),
            ("fit_predict_partial", nan_x),
            ("predict", X_nan),
            ("fit_predict", X_nan),
        ):
            with pytest.raises(ValueError, match="does not accept NaN values"):
                getattr(model, method)(arg)

        # Rejections leave state identical to reference model.
        np.testing.assert_allclose(model.fit_score(X_clean), ref.fit_score(X_clean))
