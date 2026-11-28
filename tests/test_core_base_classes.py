"""Tests for pysad.core base classes and concrete default methods."""

import numpy as np
import pytest

from pysad.core.base_metric import BaseMetric
from pysad.core.base_model import BaseModel
from pysad.core.base_postprocessor import BasePostprocessor
from pysad.core.base_statistic import BaseStatistic
from pysad.core.base_streamer import BaseStreamer
from pysad.core.base_transformer import BaseTransformer


class ConcreteModel(BaseModel):
    """Test implementation of BaseModel for testing purposes."""

    def __init__(self):
        self.fitted_data = []
        self.score_count = 0

    def fit_partial(self, X, y=None):
        self.fitted_data.append((X.copy(), y))
        return self

    def score_partial(self, X):
        self.score_count += 1
        return np.linalg.norm(X)


class ConcreteMetric(BaseMetric):
    """Test implementation of BaseMetric for testing purposes."""

    def __init__(self):
        self.values = []

    def update(self, y_true, y_pred):
        self.values.append((y_true, y_pred))

    def get(self):
        if not self.values:
            return 0.0
        correct = sum(1 for y_true, y_pred in self.values if y_true == (y_pred > 0.5))
        return correct / len(self.values)


class ConcretePostprocessor(BasePostprocessor):
    """Test implementation of BasePostprocessor for testing purposes."""

    def __init__(self, multiplier=2.0):
        self.multiplier = multiplier
        self.fitted_scores = []

    def fit_partial(self, score):
        self.fitted_scores.append(score)
        return self

    def transform_partial(self, score=None):
        return score * self.multiplier if score is not None else 0.0


class ConcreteTransformer(BaseTransformer):
    """Test implementation of BaseTransformer for testing purposes."""

    def __init__(self):
        self.fitted_data = []

    def fit_partial(self, X):
        self.fitted_data.append(X.copy())
        return self

    def transform_partial(self, X):
        return X + 1.0


class TestBaseModel:
    """Tests for BaseModel concrete batch/partial wrappers."""

    def test_fit_score_partial(self):
        """Test fit_score_partial method."""
        model = ConcreteModel()
        X = np.array([1.0, 2.0, 3.0])
        y = 1

        score = model.fit_score_partial(X, y)

        assert len(model.fitted_data) == 1
        assert model.score_count == 1
        np.testing.assert_array_equal(model.fitted_data[0][0], X)
        assert model.fitted_data[0][1] == y
        assert isinstance(score, float)

    def test_fit_method(self):
        """Test batch fitting with fit method."""
        model = ConcreteModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        fitted_model = model.fit(X, y)

        assert fitted_model is model
        assert len(model.fitted_data) == 3
        for i in range(3):
            np.testing.assert_array_equal(model.fitted_data[i][0], X[i])
            assert model.fitted_data[i][1] == y[i]

    def test_fit_method_without_labels(self):
        """Test batch fitting without labels."""
        model = ConcreteModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        fitted_model = model.fit(X)

        assert fitted_model is model
        assert len(model.fitted_data) == 2
        for i in range(2):
            np.testing.assert_array_equal(model.fitted_data[i][0], X[i])
            assert model.fitted_data[i][1] is None

    def test_score_method(self):
        """Test batch scoring with score method."""
        model = ConcreteModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        scores = model.score(X)

        assert len(scores) == 3
        assert model.score_count == 3
        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float64

    def test_fit_score_method(self):
        """Test batch fit and score with fit_score method."""
        model = ConcreteModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])

        scores = model.fit_score(X, y)

        assert len(scores) == 2
        assert len(model.fitted_data) == 2
        assert model.score_count == 2
        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float64

    def test_fit_score_method_without_labels(self):
        """Test batch fit and score without labels."""
        model = ConcreteModel()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        model.fit_score(X)

        assert len(model.fitted_data) == 2
        for fitted_data in model.fitted_data:
            assert fitted_data[1] is None


def test_base_transformer_fit_method():
    """Test that fit passes each instance, not an (instance, label) pair, to fit_partial."""
    transformer = ConcreteTransformer()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert transformer.fit(X) is transformer
    assert len(transformer.fitted_data) == 2
    np.testing.assert_array_equal(transformer.fitted_data[1], X[1])


@pytest.mark.parametrize(
    "base_cls",
    [BaseModel, BaseMetric, BasePostprocessor, BaseStatistic, BaseStreamer, BaseTransformer],
)
def test_abstract_base_classes_cannot_be_instantiated(base_cls):
    with pytest.raises(TypeError):
        base_cls()


def test_base_model_fit_score_partial_scores_before_it_fits():
    """The default fit_score_partial is prequential: each instance is scored by the model
    from the instances before it."""

    class LastValueModel(BaseModel):
        def __init__(self):
            self.last = 0.0

        def fit_partial(self, X, y=None):
            self.last = float(X[0])
            return self

        def score_partial(self, X):
            return abs(float(X[0]) - self.last)

    scores = LastValueModel().fit_score(np.array([[1.0], [3.0], [6.0]]))

    np.testing.assert_array_equal(scores, [1.0, 2.0, 3.0])


def test_pipe_operator_works_without_preimporting_models():
    """Concrete core subclasses support | even when pysad.models was not imported first."""
    t1 = ConcreteTransformer()
    t2 = ConcreteTransformer()
    m = ConcreteModel()
    p1 = ConcretePostprocessor(multiplier=2.0)
    p2 = ConcretePostprocessor(multiplier=3.0)

    chain_t = t1 | t2
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(chain_t.fit_transform(X), X + 2.0)

    chain_p = p1 | p2
    scores = np.array([1.0, 2.5, 4.0])
    np.testing.assert_array_equal(chain_p.fit_transform(scores), scores * 6.0)

    pipeline = ConcreteTransformer() | ConcreteModel() | ConcretePostprocessor(multiplier=2.0)
    out = pipeline.fit_score(X)
    expected = np.linalg.norm(X + 1.0, axis=1) * 2.0
    np.testing.assert_allclose(out, expected)

    with pytest.raises(TypeError):
        _ = m | ConcreteModel()
    with pytest.raises(TypeError):
        _ = m | t1
    with pytest.raises(TypeError):
        _ = p1 | m
    with pytest.raises(TypeError):
        _ = t1 | p1
    with pytest.raises(TypeError):
        _ = m | 42
    with pytest.raises(TypeError):
        _ = 42 | m


def test_base_metric_update_batch_default():
    metric = ConcreteMetric()
    assert metric.update_batch([1, 0, 1], [0.8, 0.2, 0.3]) is metric
    assert metric.get() == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError):
        metric.update_batch([1, 0], [0.5])


class DecisionPostprocessor(BasePostprocessor):
    """Binary decision postprocessor for exercising output_kind and decision chains."""

    _output_kind = "decision"

    def fit_partial(self, score):
        return self

    def transform_partial(self, score):
        return 1.0 if score > 0.5 else 0.0


class ProbabilityPostprocessor(BasePostprocessor):
    """Probability calibrator-like postprocessor for chain validation checks."""

    _output_kind = "probability"

    def fit_partial(self, score):
        return self

    def transform_partial(self, score):
        return min(max(float(score), 0.0), 1.0)


class _ShapedSeq:
    """Iterable container with `.shape` to exercise non-ndarray `_iterate` fallback paths."""

    def __init__(self, items, shape):
        self._items = items
        self.shape = shape

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


def test_base_postprocessor_batch_chain_and_decision_rules():
    p1 = ConcretePostprocessor(multiplier=2.0)
    p2 = ConcretePostprocessor(multiplier=3.0)
    assert p1.output_kind == "score"
    assert p1._can_predict() is False
    with pytest.raises(TypeError, match="continuous outputs"):
        p1._decide_from_score(0.5)

    assert p1.fit(np.array([1.0, 2.0])) is p1
    assert p1.fit(_ShapedSeq([3.0, 4.0], (2,))) is p1
    assert len(p1.fitted_scores) == 4
    np.testing.assert_allclose(p1.transform(_ShapedSeq([1.5, 2.5], (2,))), [3.0, 5.0])
    np.testing.assert_allclose(p1.transform(np.array([1.5, 2.5])), [3.0, 5.0])

    chain = p1 | p2
    assert chain.output_kind == "score"
    assert chain._can_predict() is False
    assert chain.fit_partial(2.0) is chain
    assert chain.transform_partial(2.0) == pytest.approx(12.0)
    assert chain.fit_transform_partial(2.0) == pytest.approx(12.0)
    assert chain.fit(np.array([1.0, 2.0])) is chain
    np.testing.assert_allclose(chain.transform(np.array([1.0, 2.0])), [6.0, 12.0])

    dec = DecisionPostprocessor()
    assert dec._can_predict() is True
    assert dec._decide_from_score(0.9) == 1
    assert dec._decide_from_score(0.0) == 0

    dec_chain = p1 | dec
    assert dec_chain.output_kind == "decision"
    assert dec_chain._can_predict() is True
    assert dec_chain._decide_from_score(1.0) == 1

    with pytest.raises(ValueError, match="Cannot apply probability calibrator"):
        _ = dec | ProbabilityPostprocessor()


def test_base_transformer_iterate_and_chain_methods():
    t1 = ConcreteTransformer()
    t2 = ConcreteTransformer()
    chain = (t1 | t2) | ConcreteTransformer()

    assert len(chain.steps) == 3
    x_seq = _ShapedSeq([np.array([1.0, 2.0]), np.array([3.0, 4.0])], (2, 2))
    assert chain.fit(x_seq) is chain
    assert len(t1.fitted_data) == 2
    np.testing.assert_allclose(chain.transform(x_seq), np.array([[4.0, 5.0], [6.0, 7.0]]))
    np.testing.assert_allclose(
        chain.fit_transform_partial(np.array([0.0, 1.0])), np.array([3.0, 4.0])
    )
    assert chain.fit_partial(np.array([2.0, 3.0])) is chain
    np.testing.assert_allclose(chain.transform_partial(np.array([2.0, 3.0])), np.array([5.0, 6.0]))
