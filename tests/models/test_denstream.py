"""Stateful property tests for the DenStream detector contract.

The tests model initialization, online updates, and read-only scoring as one state machine. They
also compare the model with a fresh replay of the same fitted stream.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

from pysad.models import DenStream

MAX_SCORE = 1.5
POINT_VALUES = st.one_of(
    st.sampled_from(
        [
            -1_000_000.0,
            -100.0,
            -10.0,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            10.0,
            100.0,
            1_000_000.0,
        ]
    ),
    st.floats(
        min_value=-1_000_000.0,
        max_value=1_000_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
POINT = st.tuples(POINT_VALUES, POINT_VALUES).map(
    lambda values: np.asarray(values, dtype=np.float64)
)
LABEL = st.integers(min_value=0, max_value=1)

STATEFUL_SETTINGS = settings(
    max_examples=10,
    stateful_step_count=30,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)


@STATEFUL_SETTINGS
class _DenStreamStateMachine(RuleBasedStateMachine):
    """Check DenStream while initialization and online updates change its state."""

    def __init__(self) -> None:
        super().__init__()
        self.model = DenStream(init_samples=2)
        self.params: dict[str, float | int] = {}
        self.history: list[tuple[np.ndarray, int]] = []
        self.fit_score_results: dict[int, float] = {}

    @initialize(
        init_samples=st.integers(min_value=2, max_value=5),
        stream_speed=st.integers(min_value=1, max_value=3),
        epsilon=st.sampled_from((0.5, 1.0, 1.5)),
        decaying_factor=st.sampled_from((0.05, 0.1, 0.2)),
        prune_interval=st.integers(min_value=1, max_value=3),
    )
    def setup(
        self,
        init_samples: int,
        stream_speed: int,
        epsilon: float,
        decaying_factor: float,
        prune_interval: int,
    ) -> None:
        """Create the model and complete its initialization phase."""
        self.params = {
            "init_samples": init_samples,
            "stream_speed": stream_speed,
            "epsilon": epsilon,
            "decaying_factor": decaying_factor,
            "beta": 0.5,
            "mu": 3.0,
            "prune_interval": prune_interval,
        }
        self.model = DenStream(**self.params)
        self.history = []
        self.fit_score_results = {}

        for index in range(init_samples):
            point = np.zeros(2, dtype=np.float64)
            # Later points differ, so a changed first point changes the learned state.
            if index > 0:
                point.fill(1_000_000.0)
            saved_point = point.copy()
            label = index % 2

            assert self.model.score_partial(point) == 0.0
            assert self.model.fit_partial(point, label) is self.model
            np.testing.assert_array_equal(point, saved_point)

            # The model must not keep the caller's array during initialization.
            point.fill(1_000_000.0)
            self.history.append((saved_point, label))
            assert self.model.initialized_ == (index + 1 >= init_samples)

        self._assert_lifecycle()
        self._assert_matches_replay(np.zeros(2, dtype=np.float64))

    def _replay_model(self) -> DenStream:
        """Return a fresh model fitted with the same points in the same order."""
        model = DenStream(**self.params)
        for point, label in self.history:
            model.fit_partial(point.copy(), label)
        return model

    def _state_summary(self) -> tuple[int, bool, int, int]:
        """Return the public state that scoring must not change."""
        return (
            self.model.timestamp_,
            self.model.initialized_,
            len(self.model.p_micro_clusters_),
            len(self.model.o_micro_clusters_),
        )

    def _assert_lifecycle(self) -> None:
        """Check the initialization and timestamp transitions."""
        num_fitted = len(self.history)
        init_samples = int(self.params["init_samples"])
        stream_speed = int(self.params["stream_speed"])

        assert self.model.initialized_ == (num_fitted >= init_samples)
        assert self.model.timestamp_ == num_fitted // stream_speed - 1

    def _assert_score(self, score: float) -> None:
        """Check the public score type, range, and stability."""
        assert type(score) is float
        assert np.isfinite(score)
        assert 0.0 <= score <= MAX_SCORE

    def _assert_matches_replay(self, probe: np.ndarray) -> None:
        """Check that the model has the same state and score as a fresh replay."""
        replay = self._replay_model()
        assert self._state_summary() == (
            replay.timestamp_,
            replay.initialized_,
            len(replay.p_micro_clusters_),
            len(replay.o_micro_clusters_),
        )

        score = self.model.score_partial(probe.copy())
        expected = replay.score_partial(probe.copy())
        assert score == expected

    @rule(point=POINT, label=LABEL)
    def fit_partial(self, point: np.ndarray, label: int) -> None:
        """Fit one instance without scoring it first."""
        point = np.asarray(point, dtype=np.float64).copy()
        saved_point = point.copy()

        assert self.model.fit_partial(point, label) is self.model
        np.testing.assert_array_equal(point, saved_point)

        point.fill(-1_000_000.0)
        self.history.append((saved_point, label))
        self._assert_lifecycle()
        self._assert_matches_replay(saved_point)

    @rule(point=POINT)
    def score_partial(self, point: np.ndarray) -> None:
        """Score one instance without changing the model state."""
        point = np.asarray(point, dtype=np.float64).copy()
        saved_point = point.copy()
        state_before = self._state_summary()

        score = self.model.score_partial(point)

        np.testing.assert_array_equal(point, saved_point)
        assert self._state_summary() == state_before
        self._assert_score(score)
        self._assert_matches_replay(saved_point)

    @rule(point=POINT, label=LABEL)
    def fit_score_partial(self, point: np.ndarray, label: int) -> None:
        """Score one instance before fitting it."""
        point = np.asarray(point, dtype=np.float64).copy()
        saved_point = point.copy()
        replay = self._replay_model()
        expected = replay.score_partial(point.copy())

        score = self.model.fit_score_partial(point, label)

        np.testing.assert_array_equal(point, saved_point)
        self._assert_score(score)
        assert score == expected

        index = len(self.history)
        self.fit_score_results[index] = score
        point.fill(-1_000_000.0)
        self.history.append((saved_point, label))
        self._assert_lifecycle()
        self._assert_matches_replay(saved_point)

    @rule()
    def check_batch_and_sequential_scores(self) -> None:
        """Check batch parity and the scores returned by earlier state transitions."""
        if not self.history:
            return

        X = np.vstack([point for point, _ in self.history])
        y = np.asarray([label for _, label in self.history], dtype=np.int64)

        batch_scores = DenStream(**self.params).fit_score(X, y)
        sequential_model = DenStream(**self.params)
        sequential_scores = [
            sequential_model.fit_score_partial(point, label) for point, label in self.history
        ]

        np.testing.assert_allclose(batch_scores, sequential_scores, rtol=0.0, atol=0.0)
        for index, score in self.fit_score_results.items():
            assert score == sequential_scores[index]

        self._assert_matches_replay(np.zeros(2, dtype=np.float64))


TestDenStreamStateMachine = _DenStreamStateMachine.TestCase
