"""
Tests for pysad.core base classes and abstract methods.
"""
import pytest
import numpy as np

from pysad.core.base_model import BaseModel
from pysad.core.base_metric import BaseMetric
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
        # Return a simple score based on L2 norm
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
        # Return accuracy for testing
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


class ConcreteStatistic(BaseStatistic):
    """Test implementation of BaseStatistic for testing purposes."""
    
    def __init__(self):
        self.values = []
        
    def update(self, num):
        self.values.append(num)
        
    def get(self):
        return np.mean(self.values) if self.values else 0.0


class ConcreteStreamer(BaseStreamer):
    """Test implementation of BaseStreamer for testing purposes."""
    
    def __init__(self, shuffle=False):
        self.shuffle = shuffle
        
    def iter(self, X, y=None):
        indices = np.arange(len(X))
        if self.shuffle:
            np.random.shuffle(indices)
            
        if y is None:
            for i in indices:
                yield X[i]
        else:
            for i in indices:
                yield X[i], y[i]


class ConcreteTransformer(BaseTransformer):
    """Test implementation of BaseTransformer for testing purposes."""
    
    def __init__(self):
        self.fitted_data = []
        
    def fit_partial(self, X):
        self.fitted_data.append(X.copy())
        return self
        
    def transform_partial(self, X):
        # Simple transformation: add 1.0
        return X + 1.0


class TestBaseModel:
    """Tests for BaseModel abstract class and its concrete methods."""
    
    def test_abstract_methods_exist(self):
        """Test that BaseModel has the required abstract methods."""
        model = ConcreteModel()
        assert hasattr(model, 'fit_partial')
        assert hasattr(model, 'score_partial')
        assert callable(model.fit_partial)
        assert callable(model.score_partial)
        
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
        
        assert fitted_model is model  # Should return self
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
        
        scores = model.fit_score(X)
        
        assert len(scores) == 2
        assert len(model.fitted_data) == 2
        for fitted_data in model.fitted_data:
            assert fitted_data[1] is None


class TestBaseMetric:
    """Tests for BaseMetric abstract class."""
    
    def test_abstract_methods_exist(self):
        """Test that BaseMetric has the required abstract methods."""
        metric = ConcreteMetric()
        assert hasattr(metric, 'update')
        assert hasattr(metric, 'get')
        assert callable(metric.update)
        assert callable(metric.get)
        
    def test_update_and_get(self):
        """Test basic update and get functionality."""
        metric = ConcreteMetric()
        
        # Initially should return default value
        assert metric.get() == 0.0
        
        # Add some values
        metric.update(1, 0.8)  # Correct
        metric.update(0, 0.2)  # Correct
        metric.update(1, 0.3)  # Incorrect
        
        accuracy = metric.get()
        assert 0.0 <= accuracy <= 1.0


class TestBasePostprocessor:
    """Tests for BasePostprocessor abstract class."""
    
    def test_abstract_methods_exist(self):
        """Test that BasePostprocessor has the required abstract methods."""
        processor = ConcretePostprocessor()
        assert hasattr(processor, 'fit_partial')
        assert hasattr(processor, 'transform_partial')
        assert callable(processor.fit_partial)
        assert callable(processor.transform_partial)
        
    def test_fit_transform_workflow(self):
        """Test the fit and transform workflow."""
        processor = ConcretePostprocessor(multiplier=3.0)
        
        # Fit on some scores
        processor.fit_partial(1.0)
        processor.fit_partial(2.0)
        
        assert len(processor.fitted_scores) == 2
        
        # Transform a score
        result = processor.transform_partial(5.0)
        assert result == 15.0  # 5.0 * 3.0
        
    def test_transform_without_score(self):
        """Test transform_partial with None score."""
        processor = ConcretePostprocessor()
        result = processor.transform_partial()
        assert result == 0.0


class TestBaseStatistic:
    """Tests for BaseStatistic abstract class."""
    
    def test_abstract_methods_exist(self):
        """Test that BaseStatistic has the required abstract methods."""
        stat = ConcreteStatistic()
        assert hasattr(stat, 'update')
        assert hasattr(stat, 'get')
        assert callable(stat.update)
        assert callable(stat.get)
        
    def test_update_and_get(self):
        """Test basic update and get functionality."""
        stat = ConcreteStatistic()
        
        # Initially should return default value
        assert stat.get() == 0.0
        
        # Add some values
        stat.update(1.0)
        stat.update(2.0)
        stat.update(3.0)
        
        mean_value = stat.get()
        assert mean_value == 2.0


class TestBaseStreamer:
    """Tests for BaseStreamer abstract class."""
    
    def test_abstract_methods_exist(self):
        """Test that BaseStreamer has the required abstract methods."""
        streamer = ConcreteStreamer()
        assert hasattr(streamer, 'iter')
        assert callable(streamer.iter)
        
    def test_iter_without_labels(self):
        """Test iteration without labels."""
        streamer = ConcreteStreamer(shuffle=False)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        results = list(streamer.iter(X))
        assert len(results) == 3
        for i, x in enumerate(results):
            np.testing.assert_array_equal(x, X[i])
            
    def test_iter_with_labels(self):
        """Test iteration with labels."""
        streamer = ConcreteStreamer(shuffle=False)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        results = list(streamer.iter(X, y))
        assert len(results) == 2
        for i, (x, y_val) in enumerate(results):
            np.testing.assert_array_equal(x, X[i])
            assert y_val == y[i]
            
    def test_iter_with_shuffle(self):
        """Test iteration with shuffle enabled."""
        np.random.seed(42)
        streamer = ConcreteStreamer(shuffle=True)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        results = list(streamer.iter(X))
        assert len(results) == 3
        # With shuffle, order should potentially be different


class TestBaseTransformer:
    """Tests for BaseTransformer abstract class."""
    
    def test_abstract_methods_exist(self):
        """Test that BaseTransformer has the required abstract methods."""
        transformer = ConcreteTransformer()
        assert hasattr(transformer, 'fit_partial')
        assert hasattr(transformer, 'transform_partial')
        assert callable(transformer.fit_partial)
        assert callable(transformer.transform_partial)
        
    def test_fit_transform_workflow(self):
        """Test the fit and transform workflow."""
        transformer = ConcreteTransformer()
        X = np.array([1.0, 2.0, 3.0])
        
        # Fit
        fitted_transformer = transformer.fit_partial(X)
        assert fitted_transformer is transformer
        assert len(transformer.fitted_data) == 1
        np.testing.assert_array_equal(transformer.fitted_data[0], X)
        
        # Transform
        result = transformer.transform_partial(X)
        expected = X + 1.0
        np.testing.assert_array_equal(result, expected)


class TestAbstractClassInstantiation:
    """Test that abstract base classes cannot be instantiated directly."""
    
    def test_base_model_abstract(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel()
            
    def test_base_metric_abstract(self):
        """Test that BaseMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMetric()
            
    def test_base_postprocessor_abstract(self):
        """Test that BasePostprocessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePostprocessor()
            
    def test_base_statistic_abstract(self):
        """Test that BaseStatistic can be instantiated but has no abstract methods."""
        # BaseStatistic is abstract but has no abstract methods, so it can be instantiated
        stat = BaseStatistic()
        assert isinstance(stat, BaseStatistic)
            
    def test_base_streamer_abstract(self):
        """Test that BaseStreamer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStreamer()
            
    def test_base_transformer_abstract(self):
        """Test that BaseTransformer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTransformer()
