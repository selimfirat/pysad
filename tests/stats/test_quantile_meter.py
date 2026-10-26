def _paper_quantile_meter(q):
    """An independent transcription of the P² algorithm of Jain and Chlamtac (1985), zero-based.

    The test compares the meter against this function step by step, in the same way the model
    tests compare against a hand-worked reference.
    """
    import math

    state = {"heights": None, "positions": None, "desired": None}
    increments = [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0]

    def update(x):
        s = state
        if s["heights"] is None:
            if len(buffer) < 5:
                buffer.append(x)
            if len(buffer) < 5:
                return
            ordered = sorted(buffer)
            s["heights"] = ordered
            s["positions"] = [0, 1, 2, 3, 4]
            s["desired"] = [0.0, 2.0 * q, 4.0 * q, 2.0 + 2.0 * q, 4.0]
            return

        heights, positions, desired = s["heights"], s["positions"], s["desired"]
        for i in range(5):
            desired[i] += increments[i]
        if x < heights[0]:
            heights[0] = x
            k = 0
        elif x < heights[1]:
            k = 0
        elif x < heights[2]:
            k = 1
        elif x < heights[3]:
            k = 2
        elif x < heights[4]:
            k = 3
        else:
            heights[4] = x
            k = 3
        for i in range(k + 1, 5):
            positions[i] += 1
        for i in range(1, 4):
            d = desired[i] - positions[i]
            if (d >= 1 and positions[i + 1] - positions[i] > 1) or (
                d <= -1 and positions[i - 1] - positions[i] < -1
            ):
                d = 1 if d > 0 else -1
                predicted = heights[i] + d / (positions[i + 1] - positions[i - 1]) * (
                    (positions[i] - positions[i - 1] + d)
                    * (heights[i + 1] - heights[i])
                    / (positions[i + 1] - positions[i])
                    + (positions[i + 1] - positions[i] - d)
                    * (heights[i] - heights[i - 1])
                    / (positions[i] - positions[i - 1])
                )
                if not heights[i - 1] < predicted < heights[i + 1]:
                    neighbor = i + d
                    predicted = heights[i] + d * (heights[neighbor] - heights[i]) / (
                        positions[neighbor] - positions[i]
                    )
                heights[i] = predicted
                positions[i] += d

    buffer: list[float] = []

    def get():
        s = state
        if s["heights"] is None:
            ordered = sorted(buffer)
            if not ordered:
                return 0.0
            rank = (len(ordered) - 1) * q
            low = math.floor(rank)
            if low == rank:
                return ordered[low]
            return ordered[low] + (rank - low) * (ordered[low + 1] - ordered[low])

        return s["heights"][2]

    return update, get


def test_quantile_meter_matches_the_paper_algorithm_step_by_step():
    import numpy as np

    from pysad.statistics import QuantileMeter
    from pysad.utils import fix_seed

    for q in (0.05, 0.5, 0.95):
        fix_seed(61)
        X = np.random.uniform(-3.0, 3.0, 500)
        update_reference, get_reference = _paper_quantile_meter(q)
        meter = QuantileMeter(q)
        for x in X:
            update_reference(float(x))
            meter.update(float(x))
            assert meter.get() == get_reference()


def test_quantile_meter_converges_to_the_true_quantile():
    import numpy as np

    from pysad.statistics import QuantileMeter
    from pysad.utils import fix_seed

    fix_seed(61)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        X = np.random.uniform(0.0, 1.0, 10_000)
        meter = QuantileMeter(q)
        for x in X:
            meter.update(x)
        # The P² estimate of a smooth distribution is close to the exact quantile; a broken
        # marker update misses by far more than this on at least one of the quantiles above.
        assert abs(meter.get() - np.quantile(X, q)) < 0.02


def test_quantile_meter_is_exact_before_the_fifth_observation():
    import numpy as np

    from pysad.statistics import QuantileMeter
    from pysad.utils import fix_seed

    for q in (0.3, 0.5, 0.7):
        fix_seed(61)
        X = np.random.uniform(0.0, 5.0, 4)
        meter = QuantileMeter(q)
        for i, x in enumerate(X):
            if i == 0:
                assert meter.get() == 0.0
            else:
                # The interpolated quantile differs from numpy's by at most an ulp of rounding.
                assert np.isclose(meter.get(), np.quantile(X[:i], q))
            meter.update(x)
        assert np.isclose(meter.get(), np.quantile(X, q))


def test_quantile_meter_returns_zero_before_the_first_observation():
    from pysad.statistics import QuantileMeter

    assert QuantileMeter(0.9).get() == 0.0


def test_quantile_meter_keeps_the_markers_sorted():
    import numpy as np

    from pysad.statistics import QuantileMeter
    from pysad.utils import fix_seed

    fix_seed(61)
    # Duplicates, ties at the extremes and a moving regime, which pushes the markers around.
    X = np.concatenate(
        [
            np.random.uniform(0.0, 1.0, 500),
            np.zeros(100),
            np.full(100, 7.0),
            np.random.uniform(6.0, 8.0, 500),
        ]
    )
    meter = QuantileMeter(0.95)
    for x in X:
        meter.update(x)
        heights = meter._heights
        assert heights[0] <= heights[1] <= heights[2] <= heights[3] <= heights[4]


def test_quantile_meter_tracks_a_constant_stream():
    from pysad.statistics import QuantileMeter

    meter = QuantileMeter(0.7)
    for _ in range(200):
        meter.update(3.14)
    assert meter.get() == 3.14


def test_quantile_meter_refuses_to_remove():
    import pytest

    from pysad.statistics import QuantileMeter

    meter = QuantileMeter(0.5)
    meter.update(1.0)
    with pytest.raises(ValueError, match="cannot remove"):
        meter.remove(1.0)


def test_quantile_meter_rejects_quantiles_outside_the_open_unit_interval():
    import numpy as np
    import pytest

    from pysad.statistics import QuantileMeter

    for q in (0.0, 1.0, -0.1, 1.5, np.nan):
        with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
            QuantileMeter(q)
