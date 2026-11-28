import numpy as np
import pytest

from pysad.models import FeedbackHST


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 0},
        {"window_size": True},
        {"num_trees": 0},
        {"max_depth": 0},
        {"feedback_budget": 0},
        {"random_work_space": "yes"},
        {"check_kl_divergence": "no"},
        {"kl_alpha": 0.0},
        {"kl_alpha": 1.0},
        {"replace_frac": -0.1},
        {"replace_frac": 1.5},
        {"tau": 0.0},
        {"tau": 1.0},
        {"learning_rate": 0.0},
        {"max_epochs": 0},
        {"rmsprop_decay": -0.1},
        {"rmsprop_decay": 1.0},
        {"rmsprop_eps": 0.0},
        {"prior_influence": -1.0},
        {"score_transition_window": -1},
    ],
)
def test_feedback_hst_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        FeedbackHST(feature_mins=[0.0, 0.0], feature_maxes=[1.0, 1.0], **kwargs)


def test_feedback_hst_active_feedback_and_reset_cycle():
    rng = np.random.RandomState(42)
    X_init = rng.uniform(0.1, 0.9, size=(20, 2))
    model = FeedbackHST(
        feature_mins=[0.0, 0.0],
        feature_maxes=[1.0, 1.0],
        window_size=8,
        num_trees=4,
        max_depth=3,
        feedback_budget=2,
        tau=0.2,
        initial_window_X=X_init,
        max_epochs=15,
        score_transition_window=2,
        random_state=42,
    )

    assert model.should_query() is True
    assert model.get_feedback_used() == 0
    assert model.get_feedback_budget() == 2

    queried_flags = []
    for i in range(20):
        x = rng.uniform(0.1, 0.9, size=2)
        if i == 19:
            x = np.array([0.99, 0.99])
        model.fit_score_partial(x)
        queried_flags.append(model.request_feedback(x))

    # Buffer stays bounded at 2 * window_size after querying more than 16 items.
    assert any(queried_flags[11:])
    assert len(model._unlabeled_buffer) == 16

    with pytest.raises(ValueError, match="label must be 0 or 1"):
        model.provide_feedback(np.array([0.5, 0.5]), 2)

    # Provide two labels (one nominal, one anomaly) to exercise weight updates & blend transition.
    model.provide_feedback(np.array([0.2, 0.2]), 0)
    blended_0 = model.score_partial(np.array([0.99, 0.99]))
    model.provide_feedback(np.array([0.99, 0.99]), 1)
    blended_1 = model.score_partial(np.array([0.99, 0.99]))
    steady = model.score_partial(np.array([0.99, 0.99]))

    assert isinstance(blended_0, float)
    assert isinstance(blended_1, float)
    assert isinstance(steady, float)
    assert model.get_feedback_used() == 2
    assert model.should_query() is False
    assert model.request_feedback(np.array([0.99, 0.99])) is False

    # Further feedback beyond budget is a no-op.
    model.provide_feedback(np.array([0.1, 0.1]), 0)
    assert model.get_feedback_used() == 2

    attr = model.explain_partial(np.array([0.5, 0.5]))
    assert attr.shape == (2,)

    model.reset_feedback()
    assert model.get_feedback_used() == 0
    assert model.should_query() is True


@pytest.mark.parametrize("check_kl", [True, False])
def test_feedback_hst_drift_replacement_modes(check_kl):
    rng = np.random.RandomState(7)
    model = FeedbackHST(
        feature_mins=[0.0, 0.0],
        feature_maxes=[1.0, 1.0],
        window_size=10,
        num_trees=6,
        max_depth=3,
        feedback_budget=3,
        check_kl_divergence=check_kl,
        kl_alpha=0.25,
        replace_frac=0.5,
        max_epochs=10,
        score_transition_window=0,
        random_state=7,
    )

    # First window initializes reference; second window is shifted to trigger replacement.
    for x in rng.uniform(0.05, 0.25, size=(10, 2)):
        model.fit_score_partial(x)
        model.request_feedback(x)
    model.provide_feedback(np.array([0.95, 0.95]), 1)

    shifted = rng.uniform(0.75, 0.95, size=(10, 2))
    scores = [model.fit_score_partial(x) for x in shifted]
    assert len(scores) == 10
    assert all(np.isfinite(s) for s in scores)


def test_feedback_hst_window_size_one_and_seeded_reproducibility():
    def run_once():
        rng = np.random.RandomState(11)
        m = FeedbackHST(
            feature_mins=[0.0, 0.0],
            feature_maxes=[1.0, 1.0],
            window_size=1,
            num_trees=3,
            max_depth=2,
            feedback_budget=2,
            random_state=11,
        )
        return [m.fit_score_partial(rng.rand(2)) for _ in range(4)]

    assert run_once() == run_once()
