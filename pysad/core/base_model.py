from __future__ import annotations

import importlib
import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from pysad.core._iterate import _iterate
from pysad.core.base_postprocessor import BasePostprocessor, _PostprocessorChain
from pysad.core.base_transformer import BaseTransformer, _TransformerChain


def _to_float_score(score: float | np.number[Any] | NDArray[Any] | list[float]) -> float:
    """Converts a single-instance score to a Python float.

    Models may compute a score as a Python number, a NumPy scalar or a one-element array. This helper maps all of them to a plain ``float``.

    Args:
        score (float, np.number or array-like with one element): The score to convert.

    Returns:
        float: The score as a Python float.
    """
    if type(score) is float:
        return score
    if isinstance(score, int | np.floating | np.integer) and not isinstance(score, bool):
        return float(score)
    score = np.asarray(score)
    if score.size != 1:
        raise ValueError(
            f"Expected a single score for one instance, got an array of shape {score.shape}."
        )

    return float(score.reshape(-1)[0])


def _to_int_decision(label: int | bool | np.integer[Any] | NDArray[Any]) -> int:
    """Converts a binary decision label into a plain Python int (``0`` or ``1``)."""
    if type(label) is int and label in (0, 1):
        return label
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, np.integer):
        val = int(label)
        if val in (0, 1):
            return val
        raise ValueError(f"Expected binary decision label 0 or 1, got {label!r}.")
    arr = np.asarray(label)
    if arr.size != 1:
        raise ValueError(
            f"Expected a single decision for one instance, got an array of shape {arr.shape}."
        )
    scalar = arr.reshape(-1)[0].item()
    if isinstance(scalar, bool | int | np.integer) and int(scalar) in (0, 1):
        return int(scalar)
    raise ValueError(f"Expected binary decision label 0 or 1, got {scalar!r}.")


def _has_nan(X: Any) -> bool:
    """Returns whether `X` is a floating or complex value/array containing NaN."""
    if isinstance(X, np.ndarray):
        return X.dtype.kind in "fc" and bool(np.isnan(X).any())
    x = np.asarray(X)
    return x.dtype.kind in "fc" and bool(np.isnan(x).any())


def _clean_numeric_batch(X: Any) -> bool:
    """Returns whether `X` is a numeric NumPy array free of NaN values."""
    return isinstance(X, np.ndarray) and (
        X.dtype.kind in "biu" or (X.dtype.kind in "fc" and not bool(np.isnan(X).any()))
    )


def _run_separate_fit_score(
    score_fn: Any,
    fit_fn: Any,
    X: NDArray[Any],
    y: NDArray[Any] | None,
    y_pred: NDArray[Any],
) -> None:
    """Evaluates separate `score_partial` then `fit_partial` over a clean numeric batch."""
    if y is None:
        for i in range(len(X)):
            xi = X[i]
            raw = score_fn(xi)
            y_pred[i] = raw if type(raw) is float else _to_float_score(raw)
            fit_fn(xi, None)
        return
    for i in range(len(X)):
        xi = X[i]
        raw = score_fn(xi)
        y_pred[i] = raw if type(raw) is float else _to_float_score(raw)
        fit_fn(xi, y[i])


def _run_combined_fit_score(
    fit_score_fn: Any,
    X: NDArray[Any],
    y: NDArray[Any] | None,
    y_pred: NDArray[Any],
) -> None:
    """Evaluates an overridden `fit_score_partial` over a clean numeric batch."""
    if y is None:
        for i in range(len(X)):
            raw = fit_score_fn(X[i], None)
            y_pred[i] = raw if type(raw) is float else _to_float_score(raw)
        return
    for i in range(len(X)):
        raw = fit_score_fn(X[i], y[i])
        y_pred[i] = raw if type(raw) is float else _to_float_score(raw)


def _returns_float_score(method: Callable[..., Any]) -> Callable[..., float]:
    """Wraps a single-instance scoring method so that it returns a Python float."""

    @wraps(method)
    def wrapper(self: BaseModel, *args: Any, **kwargs: Any) -> float:
        score = method(self, *args, **kwargs)
        return score if type(score) is float else _to_float_score(score)

    # Marks the method as wrapped, so that __init_subclass__ doesn't wrap it twice.
    wrapper._returns_float_score = True  # type: ignore[attr-defined]
    return wrapper


def _returns_int_decision(method: Callable[..., Any]) -> Callable[..., int]:
    """Wraps a single-instance decision method so that it returns 0 or 1 as int."""

    @wraps(method)
    def wrapper(self: BaseModel, *args: Any, **kwargs: Any) -> int:
        label = method(self, *args, **kwargs)
        return label if type(label) is int and label in (0, 1) else _to_int_decision(label)

    # Marks the method as wrapped, so that __init_subclass__ doesn't wrap it twice.
    wrapper._returns_int_decision = True  # type: ignore[attr-defined]
    return wrapper


def _rejects_nan_input(method: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps a single-instance method so that it rejects NaN input.

    Every numeric instance is checked for NaN before the method runs, and a ``ValueError`` that
    names the model is raised when it holds one, so the model is left unchanged. Input of a
    non-numeric dtype is passed through untouched.
    """

    @wraps(method)
    def wrapper(self: BaseModel, X: Any, *args: Any, **kwargs: Any) -> Any:
        if _has_nan(X):
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

    __bibtex__: ClassVar[str | None] = None
    _input_kind: ClassVar[Literal["multivariate", "univariate", "timestamped", "edge"]] = (
        "multivariate"
    )
    _output_kind: ClassVar[Literal["score", "probability", "decision"]] = "score"
    _has_decision_rule: ClassVar[bool] = False
    _is_detector: ClassVar[bool] = True
    _detector_doc: ClassVar[tuple[str, str, str] | None] = None
    _pipeline_cls: ClassVar[Callable[..., BaseModel] | None] = None
    _ensemble_cls: ClassVar[Callable[..., BaseModel] | None] = None
    _raw_fit_partial: Callable[..., Any]
    _raw_score_partial: Callable[..., Any]
    _raw_fit_score_partial: Callable[..., Any] | None = None

    @classmethod
    def _effective_input_kind(cls) -> Literal["multivariate", "univariate", "timestamped", "edge"]:
        """Returns the model's effective input kind, treating modules using ``check_univariate`` as univariate."""
        kind = cls._input_kind
        if kind != "multivariate":
            return kind
        mod = sys.modules.get(cls.__module__)
        if mod is not None and hasattr(mod, "check_univariate"):
            return "univariate"
        return kind

    @classmethod
    def _required_init_params(cls) -> tuple[str, ...]:
        """Returns names of required constructor parameters without defaults."""
        sig = inspect.signature(cls.__init__)
        return tuple(
            p.name
            for p in sig.parameters.values()
            if p.name != "self"
            and p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        for name in ("score_partial", "fit_score_partial"):
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "_returns_float_score", False):
                setattr(cls, name, _returns_float_score(method))

        for name in ("predict_partial", "fit_predict_partial"):
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "_returns_int_decision", False):
                setattr(cls, name, _returns_int_decision(method))

        # The batch names are here for subclasses that override them and bypass the wrapped
        # single-instance methods, such as the pyod integrations scoring a whole window at once.
        for name in (
            "fit",
            "score",
            "fit_score",
            "predict",
            "fit_predict",
            "fit_partial",
            "score_partial",
            "fit_score_partial",
            "predict_partial",
            "fit_predict_partial",
        ):
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "_rejects_nan_input", False):
                setattr(cls, name, _rejects_nan_input(method))

        cls._raw_fit_partial = inspect.unwrap(cls.fit_partial)
        cls._raw_score_partial = inspect.unwrap(cls.score_partial)
        raw_fit_score = inspect.unwrap(cls.fit_score_partial)
        cls._raw_fit_score_partial = (
            None if raw_fit_score is BaseModel.fit_score_partial else raw_fit_score
        )

    @property
    def output_kind(self) -> Literal["score", "probability", "decision"]:
        """Output kind produced by this model."""
        return self._output_kind

    def _can_predict(self) -> bool:
        return self.output_kind == "decision" or self._has_decision_rule

    def _decide_from_score(self, score: float) -> int:
        if self.output_kind == "decision":
            return 1 if score > 0.0 else 0
        raise TypeError(
            f"{type(self).__name__} produces continuous outputs (output_kind={self.output_kind!r}) "
            "without an intrinsic decision rule. Pipe it into ThresholdFilter or QuantileFilter "
            "to call predict_partial."
        )

    @abstractmethod
    def fit_partial(self, X: NDArray[Any], y: int | None = None) -> BaseModel:
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
    def score_partial(self, X: NDArray[Any]) -> float:
        """Scores the anomalousness of the next instance. This method neither changes nor keeps ``X``.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.

        Raises:
            ValueError: If `X` holds NaN.
        """
        pass

    def fit_score_partial(self, X: NDArray[Any], y: int | None = None) -> float:
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
        if _has_nan(X):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")
        cls = type(self)
        raw_score = cls._raw_score_partial(self, X)
        score = raw_score if type(raw_score) is float else _to_float_score(raw_score)
        cls._raw_fit_partial(self, X, y)

        return score

    def fit(self, X: NDArray[Any], y: NDArray[Any] | None = None) -> BaseModel:
        """Fits the model to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to fit, in order.
            y (np.int32 array of shape (num_instances,)): The labels of the instances, in order. Models that do not use labels ignore them (Default=None).

        Raises:
            ValueError: If any instance in `X` holds NaN, at the first instance that does.

        Returns:
            object: The model itself.
        """
        if _clean_numeric_batch(X):
            if y is not None and len(X) != len(y):
                raise ValueError("X and y must have the same length.")
            fit_fn = type(self)._raw_fit_partial.__get__(self)
            if y is None:
                for i in range(len(X)):
                    fit_fn(X[i], None)
            else:
                for i in range(len(X)):
                    fit_fn(X[i], y[i])
            return self

        for xi, yi in _iterate(X, y):
            self.fit_partial(xi, yi)

        return self

    def score(self, X: NDArray[Any]) -> NDArray[Any]:
        """Scores all instances in order with ``score_partial``.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to score, in order.

        Raises:
            ValueError: If any instance in `X` holds NaN, at the first instance that does.

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.empty(X.shape[0], dtype=np.float64)
        if _clean_numeric_batch(X):
            score_fn = type(self)._raw_score_partial.__get__(self)
            for i in range(len(X)):
                raw = score_fn(X[i])
                y_pred[i] = raw if type(raw) is float else _to_float_score(raw)
            return y_pred

        for i, (xi, _) in enumerate(_iterate(X)):
            y_pred[i] = _to_float_score(self.score_partial(xi))

        return y_pred

    def fit_score(self, X: NDArray[Any], y: NDArray[Any] | None = None) -> NDArray[Any]:
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
        if not _clean_numeric_batch(X):
            for i, (xi, yi) in enumerate(_iterate(X, y)):
                y_pred[i] = _to_float_score(self.fit_score_partial(xi, yi))
            return y_pred

        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        cls = type(self)
        if cls._raw_fit_score_partial is None:
            _run_separate_fit_score(
                cls._raw_score_partial.__get__(self),
                cls._raw_fit_partial.__get__(self),
                X,
                y,
                y_pred,
            )
            return y_pred

        _run_combined_fit_score(cls._raw_fit_score_partial.__get__(self), X, y, y_pred)
        return y_pred

    def predict_partial(self, X: NDArray[Any]) -> int:
        """Predicts a binary anomaly label for one instance without learning.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to predict.

        Returns:
            int: 1 for anomaly or 0 for normal.

        Raises:
            TypeError: If this model has no binary output or decision rule.
            ValueError: If ``X`` holds NaN.
        """
        if _has_nan(X):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")
        if not self._can_predict():
            self._decide_from_score(0.0)
        score = self.score_partial(X)
        return _to_int_decision(self._decide_from_score(score))

    def fit_predict_partial(self, X: NDArray[Any], y: int | None = None) -> int:
        """Predicts a binary anomaly label for one instance, then updates the model.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to predict and fit.
            y (int or None): The label of the instance. Models that do not use labels ignore it
                (Default=None).

        Returns:
            int: 1 for anomaly or 0 for normal.

        Raises:
            TypeError: If this model has no binary output or decision rule.
            ValueError: If ``X`` holds NaN.
        """
        if _has_nan(X):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")
        if not self._can_predict():
            self._decide_from_score(0.0)
        cls = type(self)
        raw_score = cls._raw_score_partial(self, X)
        score = raw_score if type(raw_score) is float else _to_float_score(raw_score)
        decision = _to_int_decision(self._decide_from_score(score))
        cls._raw_fit_partial(self, X, y)
        return decision

    def predict(self, X: NDArray[Any]) -> NDArray[np.int64]:
        """Predicts binary anomaly labels for all instances in order without learning.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to predict.

        Returns:
            np.int64 array of shape (num_instances,): Predicted binary labels (0 or 1).

        Raises:
            TypeError: If this model has no binary output or decision rule.
            ValueError: If any instance in ``X`` holds NaN.
        """
        if _has_nan(X):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")
        if not self._can_predict():
            self._decide_from_score(0.0)
        if _clean_numeric_batch(X):
            preds = np.empty(len(X), dtype=np.int64)
            for i in range(len(X)):
                preds[i] = self.predict_partial(X[i])
            return preds
        return np.array([self.predict_partial(xi) for xi, _ in _iterate(X)], dtype=np.int64)

    def fit_predict(self, X: NDArray[Any], y: NDArray[Any] | None = None) -> NDArray[np.int64]:
        """Applies ``fit_predict_partial`` to each instance in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to fit and
                predict, in order.
            y (np.int32 array of shape (num_instances,) or None): Optional instance labels
                (Default=None).

        Returns:
            np.int64 array of shape (num_instances,): Predicted binary labels (0 or 1).

        Raises:
            TypeError: If this model has no binary output or decision rule.
            ValueError: If ``X`` and ``y`` lengths differ, or if any instance in ``X`` holds NaN.
        """
        if _has_nan(X):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if not self._can_predict():
            self._decide_from_score(0.0)
        if _clean_numeric_batch(X):
            preds = np.empty(len(X), dtype=np.int64)
            if y is None:
                for i in range(len(X)):
                    preds[i] = self.fit_predict_partial(X[i], None)
            else:
                for i in range(len(X)):
                    preds[i] = self.fit_predict_partial(X[i], y[i])
            return preds
        return np.array(
            [self.fit_predict_partial(xi, yi) for xi, yi in _iterate(X, y)],
            dtype=np.int64,
        )

    @staticmethod
    def _make_pipeline(
        model: BaseModel,
        preprocessors: tuple[BaseTransformer, ...] = (),
        postprocessors: tuple[BasePostprocessor, ...] = (),
    ) -> BaseModel:
        pipeline_cls = BaseModel._pipeline_cls
        if pipeline_cls is None:
            pipeline_cls = importlib.import_module("pysad.models.stream_pipeline").StreamPipeline
        return pipeline_cls(model, preprocessors=preprocessors, postprocessors=postprocessors)

    def __or__(self, other: BasePostprocessor) -> BaseModel:
        """Chains this model with a postprocessor or postprocessor chain into a StreamPipeline."""
        if isinstance(other, BasePostprocessor):
            p_steps = other.steps if isinstance(other, _PostprocessorChain) else (other,)
            return self._make_pipeline(self, postprocessors=p_steps)
        return NotImplemented

    def __ror__(self, other: BaseTransformer) -> BaseModel:
        """Chains a transformer or transformer chain before this model into a StreamPipeline."""
        if isinstance(other, BaseTransformer):
            t_steps = other.steps if isinstance(other, _TransformerChain) else (other,)
            return self._make_pipeline(self, preprocessors=t_steps)
        return NotImplemented

    @staticmethod
    def _make_ensemble(models: tuple[BaseModel, ...]) -> BaseModel:
        ensemble_cls = BaseModel._ensemble_cls
        if ensemble_cls is None:
            ensemble_cls = importlib.import_module("pysad.models.model_ensemble").ModelEnsemble
        return ensemble_cls(models)

    def __and__(self, other: BaseModel) -> BaseModel:
        """Combines this model with another model (or default ensemble) into a ModelEnsemble."""
        if isinstance(other, BaseModel):
            left = (
                self.models
                if getattr(self, "_default_ensembler", False) and hasattr(self, "models")
                else (self,)
            )
            right = (
                other.models
                if getattr(other, "_default_ensembler", False) and hasattr(other, "models")
                else (other,)
            )
            return self._make_ensemble((*left, *right))
        return NotImplemented
