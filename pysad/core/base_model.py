from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any

from pysad.utils import _iterate
import numpy as np


def _to_float_score(score: float | np.number | np.ndarray | list[float]) -> float:
    """Converts a single-instance score to a Python float.

    Models may compute a score as a Python number, a NumPy scalar or a one-element array. This helper maps all of them to a plain ``float``.

    Args:
        score (float, np.number or array-like with one element): The score to convert.

    Returns:
        float: The score as a Python float.
    """
    score = np.asarray(score)
    if score.size != 1:
        raise ValueError(
            "Expected a single score for one instance, got an array of shape {}.".format(score.shape))

    return float(score.reshape(-1)[0])


def _returns_float_score(method: Callable[..., Any]) -> Callable[..., float]:
    """Wraps a single-instance scoring method so that it returns a Python float."""
    @wraps(method)
    def wrapper(self, *args, **kwargs) -> float:
        return _to_float_score(method(self, *args, **kwargs))

    setattr(wrapper, "_returns_float_score", True)
    return wrapper


class BaseModel(ABC):
    """Abstract base class for the models.

    Single-instance methods (`score_partial` and `fit_score_partial`) always return a Python `float`, and batch methods (`score` and `fit_score`) return a `np.float64` array of shape (num_instances,). Subclasses may compute a score as a NumPy scalar or a one-element array; it is converted to a `float` automatically.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        for name in ("score_partial", "fit_score_partial"):
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "_returns_float_score", False):
                setattr(cls, name, _returns_float_score(method))

    @abstractmethod
    def fit_partial(self, X: np.ndarray, y: int | None = None) -> "BaseModel":
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            object: Returns the self.
        """
        pass

    @abstractmethod
    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        pass

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Applies fit_partial and score_partial to the next instance, respectively.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        return _to_float_score(self.fit_partial(X, y).score_partial(X))

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "BaseModel":
        """Fits the model to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances in order to fit.
            y (int): The labels of the instances in order to fit (Optional for unsupervised models, default=None).

        Returns:
            object: Fitted model.
        """
        for xi, yi in _iterate(X, y):
            self.fit_partial(xi, yi)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Scores all instances via score_partial iteratively.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances in order to score.

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.empty(X.shape[0], dtype=np.float64)
        for i, (xi, _) in enumerate(_iterate(X)):
            y_pred[i] = _to_float_score(self.score_partial(xi))

        return y_pred

    def fit_score(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """This helper method applies fit_score_partial to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances in order to fit.
            y (np.int32 array of shape (num_instances, )): The labels of the instances in order to fit (Optional for unsupervised models, default=None).

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.empty(X.shape[0], dtype=np.float64)
        for i, (xi, yi) in enumerate(_iterate(X, y)):
            y_pred[i] = _to_float_score(self.fit_score_partial(xi, yi))

        return y_pred
