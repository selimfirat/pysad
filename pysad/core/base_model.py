from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np

from pysad.utils import _iterate


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
            f"Expected a single score for one instance, got an array of shape {score.shape}."
        )

    return float(score.reshape(-1)[0])


def _returns_float_score(method: Callable[..., Any]) -> Callable[..., float]:
    """Wraps a single-instance scoring method so that it returns a Python float."""

    @wraps(method)
    def wrapper(self: BaseModel, *args: Any, **kwargs: Any) -> float:
        return _to_float_score(method(self, *args, **kwargs))

    # Marks the method as wrapped, so that __init_subclass__ doesn't wrap it twice.
    wrapper._returns_float_score = True  # type: ignore[attr-defined]
    return wrapper


def _rejects_nan_input(method: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps a single-instance method so that it rejects NaN input.

    Every numeric instance is checked for NaN before the method runs, and a ``ValueError`` that
    names the model is raised when it holds one, so the model is left unchanged. Input of a
    non-numeric dtype is passed through untouched.
    """

    @wraps(method)
    def wrapper(self: BaseModel, X: Any, *args: Any, **kwargs: Any) -> Any:
        x = np.asarray(X)
        if x.dtype.kind in "fc" and bool(np.isnan(x).any()):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")

        return method(self, X, *args, **kwargs)

    # Marks the method as wrapped, so that __init_subclass__ doesn't wrap it twice.
    wrapper._rejects_nan_input = True  # type: ignore[attr-defined]
    return wrapper


class BaseModel(ABC):
    """Abstract base class for the models.

    Single-instance methods (`score_partial` and `fit_score_partial`) always return a Python `float`, and batch methods (`score` and `fit_score`) return a `np.float64` array of shape (num_instances,). Subclasses may compute a score as a NumPy scalar or a one-element array; it is converted to a `float` automatically.

    Every model rejects an instance that holds NaN with a `ValueError` that names the model,
    from every public entry point, before any of its methods run; the model is left unchanged.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        for name in ("score_partial", "fit_score_partial"):
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "_returns_float_score", False):
                setattr(cls, name, _returns_float_score(method))

        # The batch names are here for subclasses that override them and bypass the wrapped
        # single-instance methods, such as the pyod integrations scoring a whole window at once.
        for name in (
            "fit",
            "score",
            "fit_score",
            "fit_partial",
            "score_partial",
            "fit_score_partial",
        ):
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "_rejects_nan_input", False):
                setattr(cls, name, _rejects_nan_input(method))

    @abstractmethod
    def fit_partial(self, X: np.ndarray, y: int | None = None) -> BaseModel:
        """Updates the model with the next instance.

        The model keeps no reference to ``X``, so the caller can change ``X`` after the call returns.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int or None): The label of the instance. Models that do not use labels ignore it (Default=None).

        Raises:
            ValueError: If `X` holds NaN.

        Returns:
            object: The model itself.
        """
        pass

    @abstractmethod
    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance. This method neither changes nor keeps ``X``.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.

        Raises:
            ValueError: If `X` holds NaN.
        """
        pass

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the next instance with the model from the earlier instances, and then updates the model with it.

        This order is prequential evaluation, also called test-then-train. The model never scores an instance that it learned from.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int or None): The label of the instance. Models that do not use labels ignore it (Default=None).

        Raises:
            ValueError: If `X` holds NaN.

        Returns:
            float: The anomalousness score of the input instance.
        """
        score = self.score_partial(X)
        self.fit_partial(X, y)

        return _to_float_score(score)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseModel:
        """Fits the model to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to fit, in order.
            y (np.int32 array of shape (num_instances,)): The labels of the instances, in order. Models that do not use labels ignore them (Default=None).

        Raises:
            ValueError: If any instance in `X` holds NaN, at the first instance that does.

        Returns:
            object: The model itself.
        """
        for xi, yi in _iterate(X, y):
            self.fit_partial(xi, yi)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Scores all instances in order with ``score_partial``.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to score, in order.

        Raises:
            ValueError: If any instance in `X` holds NaN, at the first instance that does.

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.empty(X.shape[0], dtype=np.float64)
        for i, (xi, _) in enumerate(_iterate(X)):
            y_pred[i] = _to_float_score(self.score_partial(xi))

        return y_pred

    def fit_score(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Applies ``fit_score_partial`` to each instance in order and returns the scores.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to fit, in order.
            y (np.int32 array of shape (num_instances, )): The labels of the instances, in order. Models that do not use labels ignore them (Default=None).

        Raises:
            ValueError: If any instance in `X` holds NaN, at the first instance that does.

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.empty(X.shape[0], dtype=np.float64)
        for i, (xi, yi) in enumerate(_iterate(X, y)):
            y_pred[i] = _to_float_score(self.fit_score_partial(xi, yi))

        return y_pred
