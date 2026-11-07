import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.mark.parametrize("threshold", [0.5, np.full(5, 0.5)])
def test_plot_stream_marks_the_scores_above_the_threshold(threshold):
    import matplotlib.pyplot as plt

    from pysad.utils import plot_stream

    X = np.arange(10.0).reshape(5, 2)
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    y = np.array([0, 1, 0, 0, 1])

    fig = plot_stream(X, scores, threshold, y=y)

    features_ax, scores_ax, alerts_ax = fig.axes
    alerts, anomalies = (events.get_positions() for events in alerts_ax.collections)
    assert alerts == [1, 3]
    assert anomalies == [1, 4]
    assert [label.get_text() for label in alerts_ax.get_yticklabels()] == ["alert", "anomaly"]
    plt.close(fig)


def test_plot_stream_without_labels_shows_only_the_alerts():
    import matplotlib.pyplot as plt

    from pysad.utils import plot_stream

    fig = plot_stream(np.zeros((3, 1)), np.array([0.0, 1.0, 0.0]), threshold=0.5)

    alerts_ax = fig.axes[2]
    assert [events.get_positions() for events in alerts_ax.collections] == [[1]]
    assert [label.get_text() for label in alerts_ax.get_yticklabels()] == ["alert"]
    plt.close(fig)
