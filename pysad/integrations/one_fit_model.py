from __future__ import annotations

from typing import Any

import numpy as np

from ._batch_model import _BatchModel


class OneFitModel(_BatchModel):
    """Runs a batch model on a stream by fitting it once on initial instances.

    The wrapper fits ``model_cls`` (such as a PyOD detector or scikit-learn outlier estimator) to the initial instances. Then it scores all incoming instances with this fitted model and does not learn from them. For scikit-learn :class:`~sklearn.base.OutlierMixin` estimators, ``decision_function`` is negated so higher scores mean more anomalous.

    Args:
        model_cls (class): The model class to be instantiated.
        initial_X (np.float64 array of shape (num_initial_instances, num_features)): Initial instances to fit.
        initial_y (np.int32 array of shape (num_initial_instances,)): Initial window's ground truth labels. Used if not None. Needs to be ``None`` for the unsupervised ``model_cls`` models (Default=None).
        **kwargs (Keyword arguments): Keyword arguments that are passed to the ``model_cls``.
    """

    def __init__(
        self,
        model_cls: type[Any],
        initial_X: np.ndarray,
        initial_y: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_cls, **kwargs)

        self.initial_y = initial_y
        self.initial_X = initial_X
        self._fit_model()

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> OneFitModel:
        """Does nothing, because the model is fitted only once on the initial instances.

        Args:
            X (np.float64 array of shape (num_features,)): Ignored.
            y (int or None): Ignored (Default=None).

        Returns:
            object: The model itself.
        """
        return self

    def _fit_model(self) -> None:
        self.reset_model()

        if self.initial_y is None:
            self.model.fit(self.initial_X)
        else:
            self.model.fit(self.initial_X, self.initial_y)

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        return self._score_one(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Scores all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to score, in order.

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)

        return self._score_batch(X)

    def fit_score(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Scores all instances in order without updating the model.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to score, in order.
            y (np.int32 array of shape (num_instances,)): Ignored (Default=None).

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        return self.score(X)
