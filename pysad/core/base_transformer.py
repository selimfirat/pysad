from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysad.core._iterate import _iterate


class BaseTransformer(ABC):
    """Base class for transformers that map each feature vector to a new feature vector.

    Args:
        output_dims (int): The number of features of each transformed feature vector. A value of 0 or less means that the output has as many features as the input.
    """

    def __init__(self, output_dims: int) -> None:
        self.output_dims = output_dims

    @abstractmethod
    def fit_partial(self, X: NDArray[Any]) -> BaseTransformer:
        """Fits particular (next) timestep's features to train the transformer.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: self.
        """
        pass

    @abstractmethod
    def transform_partial(self, X: NDArray[Any]) -> NDArray[Any]:
        """Transforms particular (next) timestep's vector.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (output_dims,): Transformed feature vector.
        """
        pass

    def fit_transform_partial(self, X: NDArray[Any]) -> NDArray[Any]:
        """Updates the transformer with the next instance and transforms the instance.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (output_dims,): Transformed feature vector.
        """
        return self.fit_partial(X).transform_partial(X)

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        """Shortcut method that iteratively applies transform_partial to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): Input feature vectors.

        Returns:
            np.float64 array of shape (num_instances, output_dims): Transformed feature vectors.
        """
        return self._transform_all(X, self.transform_partial)

    def fit(self, X: NDArray[Any]) -> BaseTransformer:
        """Shortcut method that iteratively applies fit_partial to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): Input feature vectors.

        Returns:
            object: The fitted transformer.
        """
        if isinstance(X, np.ndarray) and X.ndim == 2:
            fit_fn = self.fit_partial
            for i in range(len(X)):
                fit_fn(X[i])
            return self

        for xi, _ in _iterate(X):
            self.fit_partial(xi)

        return self

    def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
        """Shortcut method that iteratively applies fit_transform_partial to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): Input feature vectors.

        Returns:
            np.float64 array of shape (num_instances, output_dims): Transformed feature vectors.
        """
        return self._transform_all(X, self.fit_transform_partial)

    def __or__(self, other: BaseTransformer) -> _TransformerChain:
        """Compose two transformers into a left-to-right processing chain.

        The ``|`` operator creates a sequential chain. Each transformer
        receives the output of the preceding transformer.

        Calling ``transform`` applies each step in order without fitting.
        Calling ``fit`` fits each step in order: all steps except the last
        are fitted and used to transform the data before the next step
        is fitted. The last step is fitted on the transformed data.

        Calling ``fit_transform`` fits and transforms each step in order,
        passing each transformed output to the next step.

        The partial methods follow the same left-to-right order for a
        single input vector. ``transform_partial`` transforms without
        fitting. ``fit_partial`` fits each preceding step and transforms
        its output before fitting the final step. ``fit_transform_partial``
        fits and transforms every step.

        For example, with existing transformer instances ``first`` and
        ``second``, ``(first | second).fit_transform(X)`` fits and
        transforms the data through both transformers in order.

        Args:
            other (BaseTransformer): Transformer to apply after this one.

        Returns:
            _TransformerChain: A sequential composition of transformers.
        """

        if isinstance(other, BaseTransformer):
            left = self.steps if isinstance(self, _TransformerChain) else (self,)
            right = other.steps if isinstance(other, _TransformerChain) else (other,)
            return _TransformerChain((*left, *right))
        return NotImplemented

    def _transform_all(
        self, X: NDArray[Any], transform_partial: Callable[[NDArray[Any]], NDArray[Any]]
    ) -> NDArray[Any]:
        raw_dims = getattr(self, "output_dims", -1)
        output_dims = (
            raw_dims
            if isinstance(raw_dims, int) and not isinstance(raw_dims, bool) and raw_dims > 0
            else X.shape[1]
        )
        transformed_X = np.empty((X.shape[0], output_dims), dtype=np.float64)
        if isinstance(X, np.ndarray) and X.ndim == 2:
            for i in range(len(X)):
                transformed_X[i] = transform_partial(X[i])
            return transformed_X

        for i, (xi, _) in enumerate(_iterate(X)):
            transformed_X[i] = transform_partial(xi)

        return transformed_X


class _TransformerChain(BaseTransformer):
    """Lightweight sequential chain of transformers created by ``|``."""

    def __init__(self, steps: tuple[BaseTransformer, ...]) -> None:
        raw_dims = getattr(steps[-1], "output_dims", -1) if steps else -1
        super().__init__(raw_dims if isinstance(raw_dims, int) else -1)
        self.steps = steps

    def fit_partial(self, X: NDArray[Any]) -> _TransformerChain:
        for step in self.steps[:-1]:
            X = step.fit_transform_partial(X)
        if self.steps:
            self.steps[-1].fit_partial(X)
            self.output_dims = getattr(self.steps[-1], "output_dims", -1)
        return self

    def transform_partial(self, X: NDArray[Any]) -> NDArray[Any]:
        for step in self.steps:
            X = step.transform_partial(X)
        return X

    def fit_transform_partial(self, X: NDArray[Any]) -> NDArray[Any]:
        for step in self.steps:
            X = step.fit_transform_partial(X)
        if self.steps:
            self.output_dims = getattr(self.steps[-1], "output_dims", -1)
        return X

    def fit(self, X: NDArray[Any]) -> _TransformerChain:
        for step in self.steps[:-1]:
            X = step.fit_transform(X)
        if self.steps:
            self.steps[-1].fit(X)
            self.output_dims = getattr(self.steps[-1], "output_dims", -1)
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        for step in self.steps:
            X = step.transform(X)
        if self.steps:
            self.output_dims = getattr(self.steps[-1], "output_dims", -1)
        return X

    def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
        for step in self.steps:
            X = step.fit_transform(X)
        if self.steps:
            self.output_dims = getattr(self.steps[-1], "output_dims", -1)
        return X
