"""Regression tests for issue #453: retained model inputs must be safe to overwrite.

Each test fits one model through a single reusable buffer and another model
through independent arrays, then checks that both models give the same probe
score. Overwriting the buffer after fitting must not change learned history.
"""


def _fit_reused(model, points):
    import numpy as np

    buf = np.empty(2, dtype=np.float64)
    for point in points:
        buf[:] = point
        model.fit_partial(buf)
    return model


def _fit_independent(model, points):
    import numpy as np

    for point in points:
        model.fit_partial(np.array(point, dtype=np.float64))
    return model


def test_exact_storm_reused_buffer_matches_independent():
    import numpy as np

    from pysad.models import ExactStorm

    points = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [5.0, 5.0]]
    probe = np.array([0.0, 0.0])

    reused = _fit_reused(ExactStorm(window_size=10, max_radius=1.0), points)
    independent = _fit_independent(ExactStorm(window_size=10, max_radius=1.0), points)

    assert reused.score_partial(probe) == independent.score_partial(probe)


def test_exact_storm_calibrated_radius_reused_buffer_matches_independent():
    import numpy as np

    from pysad.models import ExactStorm

    points = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [5.0, 5.0]]
    probe = np.array([0.0, 0.0])

    reused = _fit_reused(ExactStorm(window_size=10, calibration_size=5), points)
    independent = _fit_independent(ExactStorm(window_size=10, calibration_size=5), points)

    assert reused.radius_ == independent.radius_
    assert reused.score_partial(probe) == independent.score_partial(probe)


def test_iforest_asd_reused_buffer_matches_independent():
    import numpy as np

    from pysad.models import IForestASD

    kwargs = {"n_estimators": 10, "random_state": 61, "contamination": 0.1}
    points = [[float(i), float(i)] for i in range(8)]
    probe = np.array([0.0, 0.0])

    reused = _fit_reused(IForestASD(window_size=8, **kwargs), points)
    independent = _fit_independent(IForestASD(window_size=8, **kwargs), points)

    assert reused.score_partial(probe) == independent.score_partial(probe)


def test_reference_window_model_reused_buffer_matches_independent():
    import numpy as np

    from pysad.models.iforest_asd import _IsolationForest
    from pysad.models.integrations import ReferenceWindowModel

    kwargs = {"n_estimators": 10, "random_state": 61, "contamination": 0.1}
    points = [[float(i), float(i)] for i in range(8)]
    probe = np.array([0.0, 0.0])

    reused = _fit_reused(
        ReferenceWindowModel(_IsolationForest, window_size=8, sliding_size=8, **kwargs),
        points,
    )
    independent = _fit_independent(
        ReferenceWindowModel(_IsolationForest, window_size=8, sliding_size=8, **kwargs),
        points,
    )

    assert reused.score_partial(probe) == independent.score_partial(probe)


def test_reference_window_model_initial_window_is_copied():
    import numpy as np

    from pysad.models.iforest_asd import _IsolationForest
    from pysad.models.integrations import ReferenceWindowModel

    kwargs = {"n_estimators": 10, "random_state": 61, "contamination": 0.1}
    initial_points = [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [0.2, 0.0],
        [0.0, 0.2],
        [0.2, 0.2],
        [0.1, 0.2],
    ]
    stream_points = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]
    probe = np.array([0.0, 0.0])

    initial = np.array(initial_points, dtype=np.float64)
    copied = ReferenceWindowModel(
        _IsolationForest,
        window_size=8,
        sliding_size=4,
        initial_window_X=initial,
        **kwargs,
    )
    initial[:] = 999.0
    for point in stream_points:
        copied.fit_partial(np.array(point, dtype=np.float64))

    pristine = ReferenceWindowModel(
        _IsolationForest,
        window_size=8,
        sliding_size=4,
        initial_window_X=np.array(initial_points, dtype=np.float64),
        **kwargs,
    )
    for point in stream_points:
        pristine.fit_partial(np.array(point, dtype=np.float64))

    assert copied.score_partial(probe) == pristine.score_partial(probe)


def test_window_copies_array_items():
    import numpy as np

    from pysad.utils import Window
    from pysad.utils.window import UnlimitedWindow

    for window in (Window(window_size=10), UnlimitedWindow()):
        buf = np.array([1.0, 2.0])
        window.update(buf)
        buf[:] = 999.0
        np.testing.assert_array_equal(window.get()[0], np.array([1.0, 2.0]))
