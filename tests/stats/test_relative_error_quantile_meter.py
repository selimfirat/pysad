import pickle

import numpy as np
import pytest


def _fill(meter, values):
    for value in values:
        meter.update(float(value))
    return meter


@pytest.mark.parametrize("high_rank_accuracy", [False, True])
@pytest.mark.parametrize("order", ["ascending", "descending", "shuffled"])
def test_relative_rank_accuracy(high_rank_accuracy, order):
    from pysad.statistics import RelativeErrorQuantileMeter

    values = np.arange(30_000, dtype=float)
    if order == "descending":
        values = values[::-1]
    elif order == "shuffled":
        np.random.RandomState(61).shuffle(values)
    meter = _fill(
        RelativeErrorQuantileMeter(
            q=0.5, k=20, max_items=100_000, high_rank_accuracy=high_rank_accuracy, random_state=61
        ),
        values,
    )
    assert meter.num_items == len(values)
    for q in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
        meter.q = q
        estimate = meter.get()
        target = q * (len(values) - 1)
        tail_rank = (1 - q if high_rank_accuracy else q) * len(values)
        assert abs(estimate - target) <= max(1, 0.03 * tail_rank)


@pytest.mark.parametrize("high_rank_accuracy", [False, True])
def test_merging_compacted_streams_preserves_accuracy_and_source(high_rank_accuracy):
    from pysad.statistics import RelativeErrorQuantileMeter

    values = np.random.RandomState(61).permutation(40_000)
    meters = [
        _fill(
            RelativeErrorQuantileMeter(
                k=20, max_items=100_000, high_rank_accuracy=high_rank_accuracy, random_state=i
            ),
            part,
        )
        for i, part in enumerate(np.array_split(values, 8))
    ]
    while len(meters) > 1:
        merged = []
        for receiver, source in zip(meters[::2], meters[1::2], strict=True):
            source_before = pickle.dumps(source)
            assert receiver.merge(source) is receiver
            assert pickle.dumps(source) == source_before
            merged.append(receiver)
        meters = merged
    meter = meters[0]
    assert meter.num_items == len(values)
    for q in [0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]:
        meter.q = q
        tail_rank = (1 - q if high_rank_accuracy else q) * len(values)
        assert abs(meter.get() - q * (len(values) - 1)) <= max(1, 0.03 * tail_rank)
    _fill(meter, range(40_000, 50_000))
    assert meter.num_items == 50_000
    assert meter.get() == 49_999


@pytest.mark.parametrize("high_rank_accuracy", [False, True])
def test_random_compaction_keeps_each_parity(high_rank_accuracy):
    from pysad.statistics import RelativeErrorQuantileMeter

    estimates = [
        _fill(
            RelativeErrorQuantileMeter(
                q=0.1 if high_rank_accuracy else 0.9,
                k=2,
                max_items=16,
                high_rank_accuracy=high_rank_accuracy,
                random_state=seed,
            ),
            range(13),
        ).get()
        for seed in range(200)
    ]
    expected = {0.0, 1.0} if high_rank_accuracy else {10.0, 11.0}
    assert set(estimates) == expected
    assert abs(np.mean(estimates) - np.mean(list(expected))) < 0.1


def test_seed_and_pickle_preserve_future_estimates():
    from pysad.statistics import RelativeErrorQuantileMeter

    values = np.random.RandomState(61).normal(size=4_000)
    first = RelativeErrorQuantileMeter(k=10, max_items=100_000, random_state=61)
    second = RelativeErrorQuantileMeter(k=10, max_items=100_000, random_state=61)
    _fill(first, values[:2_000])
    _fill(second, values[:2_000])
    restored = pickle.loads(pickle.dumps(first))
    for meter in [first, second, restored]:
        _fill(meter, values[2_000:])
    for q in [0.01, 0.1, 0.5, 0.9]:
        for meter in [first, second, restored]:
            meter.q = q
        assert first.get() == second.get() == restored.get()


def test_memory_stays_small_after_many_updates():
    from pysad.statistics import RelativeErrorQuantileMeter

    meter = RelativeErrorQuantileMeter(k=20, max_items=1_000_000, random_state=61)
    _fill(meter, range(50_000))
    assert len(pickle.dumps(meter)) < 100_000
    meter.q = 0.01
    assert abs(meter.get() - 499.99) < 30


@pytest.mark.parametrize("q, expected", [(0, -5.0), (0.5, 3.0), (1, 10.0)])
def test_empty_small_and_self_merged_streams(q, expected):
    from pysad.statistics import RelativeErrorQuantileMeter

    meter = RelativeErrorQuantileMeter(q=q)
    assert meter.get() == 0.0
    for value in [10.0, 3.0, -5.0, 3.0, 7.0]:
        assert meter.update(value) is meter
    assert meter.get() == expected
    assert meter.merge(meter) is meter
    assert meter.num_items == 10
    assert meter.get() == expected
    assert meter.merge(RelativeErrorQuantileMeter()) is meter


@pytest.mark.parametrize("high_rank_accuracy", [False, True])
def test_repeated_values_after_compaction(high_rank_accuracy):
    from pysad.statistics import RelativeErrorQuantileMeter

    meter = RelativeErrorQuantileMeter(
        k=2, max_items=10_000, high_rank_accuracy=high_rank_accuracy, random_state=61
    )
    _fill(meter, [7.0] * 1000)
    assert meter.get() == 7.0


@pytest.mark.parametrize(
    "kwargs",
    [{"q": -0.1}, {"q": 1.1}, {"k": 0}, {"k": 3}, {"max_items": 0}],
)
def test_invalid_parameters_raise(kwargs):
    from pysad.statistics import RelativeErrorQuantileMeter

    with pytest.raises(ValueError):
        RelativeErrorQuantileMeter(**kwargs)


@pytest.mark.parametrize("kwargs", [{"k": 20}, {"max_items": 200}, {"high_rank_accuracy": True}])
def test_incompatible_merge_leaves_receiver_unchanged(kwargs):
    from pysad.statistics import RelativeErrorQuantileMeter

    meter = _fill(RelativeErrorQuantileMeter(k=10, max_items=100), range(10))
    parameters = {"k": 10, "max_items": 100, **kwargs}
    source = _fill(RelativeErrorQuantileMeter(**parameters), range(10))
    before = pickle.dumps(meter)
    with pytest.raises(ValueError):
        meter.merge(source)
    assert pickle.dumps(meter) == before


def test_stream_limit_rejects_update_and_merge_before_changing_state():
    from pysad.statistics import RelativeErrorQuantileMeter

    meter = _fill(RelativeErrorQuantileMeter(max_items=5), range(5))
    source = _fill(RelativeErrorQuantileMeter(max_items=5), [10])
    before = pickle.dumps(meter)
    with pytest.raises(ValueError):
        meter.update(100.0)
    assert pickle.dumps(meter) == before
    with pytest.raises(ValueError):
        meter.merge(source)
    assert pickle.dumps(meter) == before


def test_compaction_schedule_expands_the_sorted_tail():
    from pysad.statistics import RelativeErrorQuantileMeter

    meter = _fill(RelativeErrorQuantileMeter(q=0.66, k=2, max_items=64, random_state=0), range(26))
    # The second compaction selects 17 and 21 from the four-value tail.
    assert meter.get() == 17.0


def test_merge_combines_compaction_schedules():
    from pysad.statistics import RelativeErrorQuantileMeter

    receiver = _fill(
        RelativeErrorQuantileMeter(q=0.31, k=2, max_items=64, random_state=0), range(21)
    )
    source = _fill(RelativeErrorQuantileMeter(k=2, max_items=64, random_state=0), range(100, 127))
    # States 1 and 3 combine to 3. The next compaction includes three sections.
    receiver.merge(source)
    assert receiver.num_items == 48
    assert receiver.get() == 15.0


def test_merge_compacts_adjacent_overflowing_levels_bottom_to_top():
    from pysad.statistics import RelativeErrorQuantileMeter

    receiver = _fill(RelativeErrorQuantileMeter(k=2, max_items=128, random_state=0), range(40))
    source = _fill(RelativeErrorQuantileMeter(k=2, max_items=128, random_state=0), range(40, 80))
    # Algorithm 3 gives capacity 24 and merged states 5 and 0.
    # Seed 0 selects parity 1 for both merge compactions.
    # Level 0 promotes [35, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 73, 75, 79].
    # Level 1 then holds 30 values and promotes [66, 70, 75, 79] together.
    # Below 66, 20 unit weights and 22 double weights give cumulative weight 64.
    # The four promoted values raise that weight to 68, 72, 76, and 80.
    receiver.merge(source)
    for q, expected in [(0.8, 62.0), (0.85, 66.0), (0.9, 70.0), (0.95, 75.0)]:
        receiver.q = q
        assert receiver.get() == expected
