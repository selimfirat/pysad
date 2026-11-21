from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_transformer import BaseTransformer


def _to_feature_vector(output: Any) -> NDArray[np.float64]:
    """Convert callable output into a 1-D float64 feature vector."""
    arr = np.asarray(output, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr


class FunctionTransformer(BaseTransformer):
    """Wraps a callable so custom feature extraction or encoders fit in a pipeline.

    ``transform_partial`` passes each incoming instance to ``func`` and converts
    the result into a 1-D ``np.float64`` feature vector. If ``func`` is ``None``,
    the instance passes through unchanged.

    Pass ``fit_func`` if the wrapped extractor keeps state that updates on each
    instance during ``fit_partial`` or ``fit_transform_partial``. For pickling,
    use a top-level function or ``functools.partial`` for ``func`` and
    ``fit_func``.

    Args:
        func (callable or None): Maps one instance to a numeric vector, scalar,
            or array. ``None`` uses the input instance unchanged (Default=None).
        fit_func (callable or None): Optional callable invoked with each
            instance in ``fit_partial``. ``None`` makes fitting a no-op
            (Default=None).
        output_dims (int): Expected number of output features. A non-positive
            value infers the width from the transformed output (Default=-1).
    """

    def __init__(
        self,
        func: Callable[[Any], Any] | None = None,
        fit_func: Callable[[Any], Any] | None = None,
        output_dims: int = -1,
    ) -> None:
        super().__init__(output_dims)
        self.func = func
        self.fit_func = fit_func

    def fit_partial(self, X: Any) -> FunctionTransformer:
        """Update the transformer state with one instance.

        Args:
            X (Any): The next stream instance.

        Returns:
            object: ``self``.
        """
        if self.fit_func is not None:
            self.fit_func(X)
        return self

    def transform_partial(self, X: Any) -> NDArray[np.float64]:
        """Transform one instance into a 1-D float64 feature vector.

        Args:
            X (Any): The instance to transform.

        Returns:
            np.float64 array of shape (output_dims,): Transformed feature vector.
        """
        out = X if self.func is None else self.func(X)
        return _to_feature_vector(out)

    def fit(self, X: Any) -> FunctionTransformer:
        """Fit the transformer on every instance in ``X`` in order.

        Args:
            X (Iterable[Any]): Sequence or iterable of stream instances.

        Returns:
            object: ``self``.
        """
        if self.fit_func is None:
            return self
        for xi in X:
            self.fit_partial(xi)
        return self

    def _transform_batch(
        self, X: Any, transform_fn: Callable[[Any], NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        rows = [transform_fn(xi) for xi in X]
        if not rows:
            width = (
                self.output_dims
                if self.output_dims > 0
                else (X.shape[1] if isinstance(X, np.ndarray) and X.ndim >= 2 else 0)
            )
            return np.empty((0, width), dtype=np.float64)
        return np.stack(rows, axis=0)

    def transform(self, X: Any) -> NDArray[np.float64]:
        """Transform all instances in ``X`` in order without fitting.

        Args:
            X (Iterable[Any]): Sequence or iterable of stream instances.

        Returns:
            np.float64 array of shape (num_instances, output_dims): Transformed
            feature matrix.
        """
        return self._transform_batch(X, self.transform_partial)

    def fit_transform(self, X: Any) -> NDArray[np.float64]:
        """Fit and transform all instances in ``X`` in order.

        Args:
            X (Iterable[Any]): Sequence or iterable of stream instances.

        Returns:
            np.float64 array of shape (num_instances, output_dims): Transformed
            feature matrix.
        """
        return self._transform_batch(X, self.fit_transform_partial)
