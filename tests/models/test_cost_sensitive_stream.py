import pickle

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


def test_cost_sensitive_stream_model_uses_cost_weighted_updates():
    from pysad.models import CostSensitiveStreamModel

    normal = CostSensitiveStreamModel(cost_ratio=10.0, learning_rate=0.1)
    normal.fit_partial(np.array([1.0, 0.0]), y=0)

    anomaly = CostSensitiveStreamModel(cost_ratio=10.0, learning_rate=0.1)
    anomaly.fit_partial(np.array([1.0, 0.0]), y=1)

    assert normal.score_partial(np.array([1.0, 0.0])) == pytest.approx(-0.1)
    assert anomaly.score_partial(np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_cost_sensitive_stream_model_scores_before_it_updates():
    from pysad.models import CostSensitiveStreamModel

    model = CostSensitiveStreamModel(cost_ratio=10.0, learning_rate=0.1)
    instance = np.array([1.0, 0.0])

    assert model.fit_score_partial(instance, y=1) == 0.0
    assert model.score_partial(instance) == pytest.approx(1.0)


def test_cost_sensitive_stream_model_treats_a_missing_label_as_normal():
    from pysad.models import CostSensitiveStreamModel

    missing = CostSensitiveStreamModel(learning_rate=0.1)
    explicit = CostSensitiveStreamModel(learning_rate=0.1)
    instance = np.array([1.0, 0.0])

    missing.fit_partial(instance)
    explicit.fit_partial(instance, y=0)

    assert missing.score_partial(instance) == explicit.score_partial(instance)


def test_cost_sensitive_stream_model_ranks_anomalies_above_chance():
    from pysad.models import CostSensitiveStreamModel

    normal = np.tile(np.array([1.0, 1.0]), (60, 1))
    anomalous = np.tile(np.array([-5.0, -5.0]), (20, 1))
    X = np.vstack([normal, anomalous])
    y = np.concatenate([np.zeros(60, dtype=np.int64), np.ones(20, dtype=np.int64)])

    scores = CostSensitiveStreamModel().fit_score(X)

    assert roc_auc_score(y, scores) > 0.5


def test_cost_sensitive_stream_model_rejects_an_invalid_label():
    from pysad.models import CostSensitiveStreamModel

    model = CostSensitiveStreamModel()

    with pytest.raises(ValueError):
        model.fit_partial(np.array([1.0, 0.0]), y=2)


def test_cost_sensitive_stream_model_keeps_bounded_memory():
    from pysad.models import CostSensitiveStreamModel

    model = CostSensitiveStreamModel()
    instance = np.array([1.0, 0.0])
    model.fit_partial(instance)
    initial_size = len(pickle.dumps(model))

    for _ in range(1_000):
        model.fit_partial(instance)

    assert len(pickle.dumps(model)) == initial_size
