import pickle

import numpy as np
import pytest


def _symbols(rng, names, num_columns=2048, num_active=40):
    return {name: rng.choice(num_columns, num_active, replace=False) for name in names}


def _train(tm, rng, symbols, sequences, num_repeats):
    for _ in range(num_repeats):
        for sequence in sequences:
            for name in sequence:
                tm.fit(symbols[name])
            tm.fit(rng.choice(2048, 40, replace=False))


def test_learns_high_order_sequences():
    from pysad.models.htm.temporal_memory import TemporalMemory

    rng = np.random.RandomState(0)
    symbols = _symbols(rng, "ABCDXY")
    tm = TemporalMemory(random_state=0)
    assert tm.score(symbols["A"]) == 1.0

    _train(tm, rng, symbols, ["ABCD", "XBCY"], num_repeats=30)

    # After A-B-C only D is predicted, and after X-B-C only Y.
    for context, expected, other in [("ABC", "D", "Y"), ("XBC", "Y", "D")]:
        for name in context:
            tm.fit(symbols[name])
        assert tm.score(symbols[expected]) == 0.0
        assert tm.score(symbols[other]) > 0.9
        tm.fit(rng.choice(2048, 40, replace=False))


def test_score_does_not_change_state():
    from pysad.models.htm.temporal_memory import TemporalMemory

    stream = [np.random.RandomState(i % 7).choice(2048, 40, replace=False) for i in range(100)]
    scored = TemporalMemory(random_state=1)
    reference = TemporalMemory(random_state=1)
    for columns in stream:
        scored.score(np.arange(2048))
        assert scored.score(columns) == reference.score(columns)
        scored.fit(columns)
        reference.fit(columns)


def test_pickled_copy_gives_the_same_scores():
    from pysad.models.htm.temporal_memory import TemporalMemory

    stream = [np.random.RandomState(i % 5).choice(2048, 40, replace=False) for i in range(60)]
    tm = TemporalMemory(random_state=2)
    for columns in stream[:30]:
        tm.fit(columns)
    copy = pickle.loads(pickle.dumps(tm))
    for columns in stream[30:]:
        assert tm.score(columns) == copy.score(columns)
        tm.fit(columns)
        copy.fit(columns)


def test_memory_stays_bounded():
    from pysad.models.htm.temporal_memory import TemporalMemory

    rng = np.random.RandomState(3)
    tm = TemporalMemory(
        num_columns=64,
        num_cells_per_column=4,
        activation_threshold=2,
        learning_threshold=3,
        max_segments_per_cell=2,
        max_synapses_per_segment=6,
        max_new_synapses=4,
        random_state=3,
    )
    sizes = []
    for _ in range(3):
        for _ in range(2000):
            columns = rng.choice(64, 4, replace=False)
            assert 0.0 <= tm.score(columns) <= 1.0
            tm.fit(columns)
        sizes.append(len(pickle.dumps(tm)))
    assert sizes[2] <= 1.1 * sizes[0]


def test_rejects_more_new_synapses_than_fit_on_a_segment():
    from pysad.models.htm.temporal_memory import TemporalMemory

    with pytest.raises(ValueError, match="max_new_synapses"):
        TemporalMemory(max_synapses_per_segment=16, max_new_synapses=32)
