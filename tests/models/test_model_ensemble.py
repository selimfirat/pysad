from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike

from pysad.core import BaseScoreEnsembler
from pysad.evaluation import AUROCMetric, evaluate_stream
from pysad.models import LODA, ModelEnsemble, RandomModel, StreamPipeline, xStream
from pysad.transform.ensemble import MaximumScoreEnsembler
from pysad.transform.postprocessing import ZScorePostprocessor
from pysad.transform.preprocessing import ShingleTransformer
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.utils import fix_seed


class _RunningSumEnsembler(BaseScoreEnsembler):
    """Stateful test ensembler that weights detectors by their cumulative score sums."""

    def __init__(self) -> None:
        self.totals: np.ndarray | None = None

    def fit_partial(self, scores: ArrayLike) -> _RunningSumEnsembler:
        row = np.asarray(scores, dtype=np.float64)
        if self.totals is None:
            self.totals = np.abs(row).copy() + 1.0
        else:
            self.totals += np.abs(row)
        return self

    def transform_partial(self, scores: ArrayLike) -> float:
        row = np.asarray(scores, dtype=np.float64)
        if self.totals is None:
            return float(row.mean())
        return float((row @ self.totals) / self.totals.sum())


def test_ampersand_operator_flattens_default_ensembles_and_matches_manual_loop() -> None:
    X = np.random.RandomState(42).rand(120, 4)

    def make_submodels():
        return [
            LODA(random_state=0) | ZScorePostprocessor(window_size=25),
            xStream(random_state=1, window_size=25) | ZScorePostprocessor(window_size=25),
            LODA(random_state=2) | ZScorePostprocessor(window_size=25),
        ]

    m1, m2, m3 = make_submodels()
    ens = m1 & m2 & m3
    assert isinstance(ens, ModelEnsemble)
    assert ens.models == (m1, m2, m3)

    # Manual loop must match both partial and batch fit_score bit-for-bit.
    ref_models = make_submodels()
    expected = np.array(
        [np.mean([m.fit_score_partial(x) for m in ref_models]) for x in X], dtype=np.float64
    )

    a, b, c = make_submodels()
    ens_stream = a & b & c
    streamed = np.array([ens_stream.fit_score_partial(x) for x in X])
    np.testing.assert_array_equal(streamed, expected)
    np.testing.assert_array_equal(ens.fit_score(X), expected)

    # Non-default ensembler is preserved rather than flattened when combined with &.
    custom = ModelEnsemble([m1, m2], ensembler=MaximumScoreEnsembler())
    nested = custom & m3
    assert nested.models == (custom, m3)


def test_model_ensemble_composes_with_pipe_and_evaluate_stream() -> None:
    rng = np.random.RandomState(7)
    X = rng.normal(size=(150, 3))
    y = np.zeros(150, dtype=np.int32)
    y[[60, 90, 120, 140]] = 1
    X[y == 1] += 6.0

    pipeline = (
        ShingleTransformer(shingle_size=2)
        | (
            (LODA(random_state=0) | ZScorePostprocessor(window_size=30))
            & (xStream(random_state=1, window_size=30) | ZScorePostprocessor(window_size=30))
        )
        | ConformalProbabilityCalibrator(window_size=40)
    )
    assert isinstance(pipeline, StreamPipeline)
    results = evaluate_stream(pipeline, X, y, {"AUROC": AUROCMetric()})
    assert results["AUROC"] > 0.7


def test_model_ensemble_stateful_ensembler_and_global_rng_batch_parity() -> None:
    X = np.random.RandomState(11).rand(50, 3)

    # Stateful custom BaseScoreEnsembler agrees between partial and batch fit/score/fit_score.
    def make_stateful() -> ModelEnsemble:
        return ModelEnsemble(
            [LODA(random_state=0), xStream(random_state=1, window_size=15)],
            ensembler=_RunningSumEnsembler(),
        )

    e_batch = make_stateful()
    e_stream = make_stateful()
    np.testing.assert_allclose(
        e_batch.fit_score(X),
        [e_stream.fit_score_partial(x) for x in X],
    )

    f_batch = make_stateful().fit(X[:30])
    f_stream = make_stateful()
    for x in X[:30]:
        f_stream.fit_partial(x)
    np.testing.assert_allclose(
        f_batch.score(X[30:]),
        [f_stream.score_partial(x) for x in X[30:]],
    )

    # Multiple unseeded components sharing global RNG fall back cleanly so batch == partial.
    fix_seed(19)
    unseeded_a = RandomModel() & RandomModel()
    fix_seed(23)
    s_batch = unseeded_a.fit_score(X[:15])

    fix_seed(19)
    unseeded_b = RandomModel() & RandomModel()
    fix_seed(23)
    s_stream = np.array([unseeded_b.fit_score_partial(x) for x in X[:15]])
    np.testing.assert_array_equal(s_batch, s_stream)

    with pytest.raises(ValueError, match="at least one model"):
        ModelEnsemble([])
