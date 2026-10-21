import pickle

import numpy as np
import pytest

import pysad.statistics as statistics

SEED = 61
NUM_FIT = 60
NUM_NEXT = 20

# Wrappers that take another statistic class as a constructor argument, rather than being
# default-constructible themselves; covered separately below.
PICKLE_SKIP_STATISTICS = {"AbsStatistic", "RunningStatistic"}


def _values():
    rng = np.random.RandomState(SEED)
    return rng.rand(NUM_FIT + NUM_NEXT)


def _fitted_statistic(statistic_name, values):
    statistic_cls = getattr(statistics, statistic_name)
    statistic = statistic_cls()
    for value in values[:NUM_FIT]:
        statistic.update(value)

    return statistic


def _next_outputs(statistic, values):
    outputs = []
    for value in values[NUM_FIT:]:
        statistic.update(value)
        outputs.append(statistic.get())

    return np.asarray(outputs, dtype=np.float64)


@pytest.mark.parametrize(
    "statistic_name", [name for name in statistics.__all__ if name not in PICKLE_SKIP_STATISTICS]
)
def test_pickle_round_trip(statistic_name):
    values = _values()
    statistic = _fitted_statistic(statistic_name, values)

    restored = pickle.loads(pickle.dumps(statistic))

    assert type(restored) is type(statistic)
    np.testing.assert_allclose(_next_outputs(restored, values), _next_outputs(statistic, values))


def test_abs_statistic_pickle_round_trip():
    values = _values()
    statistic = statistics.AbsStatistic(statistics.AverageMeter)
    for value in values[:NUM_FIT]:
        statistic.update(value)

    restored = pickle.loads(pickle.dumps(statistic))

    np.testing.assert_allclose(_next_outputs(restored, values), _next_outputs(statistic, values))


def test_running_statistic_pickle_round_trip():
    values = _values()
    statistic = statistics.RunningStatistic(statistics.AverageMeter, window_size=16)
    for value in values[:NUM_FIT]:
        statistic.update(value)

    restored = pickle.loads(pickle.dumps(statistic))

    np.testing.assert_allclose(_next_outputs(restored, values), _next_outputs(statistic, values))
