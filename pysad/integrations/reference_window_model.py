from __future__ import annotations

from typing import Any, TypeVar

import numpy as np

from pysad.utils import _iterate

from ._batch_model import _BatchModel

# typing.Self needs Python 3.11; this TypeVar lets fit_partial return the subclass type.
_ReferenceWindowModelT = TypeVar("_ReferenceWindowModelT", bound="ReferenceWindowModel")


class ReferenceWindowModel(_BatchModel):
    """Runs a batch model on a stream by retraining it on a reference window at regular intervals :cite:`xstream`.

    This wrapper keeps track of the reference window of size ``window_size``. For every ``sliding_size`` instances, it resets the model by training a new ``model_cls`` instance (such as a PyOD detector or scikit-learn outlier estimator) on the reference window. For scikit-learn :class:`~sklearn.base.OutlierMixin` estimators, ``decision_function`` is negated so higher scores mean more anomalous. This implementation is based on the reference windowing described in :cite:`xstream`.

    Without ``initial_window_X``, there is no reference window at the start. Until the first window is full, the model trains on all the instances so far when their number is a power of two (1, 2, 4, 8 and so on).

    Args:
        model_cls (class): The model class to be instantiated.
        window_size (int): The size of each window.
        sliding_size (int): The sliding length of the windows.
        initial_window_X (np.float64 array of shape (num_initial_instances, num_features)): Initial instances to fit (Default=None).
        initial_window_y (np.int32 array of shape (num_initial_instances,)): Initial window's ground truth labels. Used if not None. Needs to be ``None`` for the unsupervised ``model_cls`` models (Default=None).
        **kwargs (Keyword arguments): Keyword arguments that are passed to the ``model_cls``.
    """

    def __init__(
        self,
        model_cls: type[Any],
        window_size: int,
        sliding_size: int,
        initial_window_X: np.ndarray | None = None,
        initial_window_y: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        super().__init__(model_cls, **kwargs)

        self.window_size = window_size
        self.sliding_size = sliding_size

        self.cur_window_X: list[np.ndarray] = []
        self.cur_window_y: list[int] = []

        # A list of instances until the first window is full, an array after that.
        self.reference_window_X: np.ndarray | list[np.ndarray] | None = (
            None if initial_window_X is None else np.array(initial_window_X)
        )
        self.reference_window_y: np.ndarray | list[int] | None = (
            None if initial_window_y is None else np.array(initial_window_y)
        )

        if self.reference_window_X is not None:
            self._fit_model()
            self.initial_ref_window = True
        else:
            self.initial_ref_window = False

    def fit_partial(
        self: _ReferenceWindowModelT, X: np.ndarray, y: int | None = None
    ) -> _ReferenceWindowModelT:
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            object: self.
        """
        self.cur_window_X.append(np.array(X))

        if y is not None:
            self.cur_window_y.append(y)

        if (
            not self.initial_ref_window
            and len(self.cur_window_X) < self.window_size
            and (self.reference_window_X is None or len(self.reference_window_X) < self.window_size)
        ):
            # Until the first window is full, the model trains on the instances so far each time
            # their number doubles, so that the first window costs about log2(window_size) fits.
            if len(self.cur_window_X) & (len(self.cur_window_X) - 1) == 0:
                self.reference_window_X = self.cur_window_X.copy()
                self.reference_window_y = self.cur_window_y.copy() if y is not None else None
                self._fit_model()
        elif len(self.cur_window_X) % self.sliding_size == 0:
            self.reference_window_X = np.concatenate(
                [self.reference_window_X, self.cur_window_X], axis=0
            )
            self.reference_window_X = self.reference_window_X[
                max(0, len(self.reference_window_X) - self.window_size) :
            ]

            if y is not None:
                self.reference_window_y = np.concatenate(
                    [self.reference_window_y, self.cur_window_y], axis=0
                )
                self.reference_window_y = self.reference_window_y[
                    max(0, len(self.reference_window_y) - self.window_size) :
                ]

            self.cur_window_X = []
            self.cur_window_y = []
            self._fit_model()

        return self

    def _fit_model(self) -> None:
        self.reset_model()

        if self.reference_window_y is None:
            self.model.fit(self.reference_window_X)
        else:
            self.model.fit(self.reference_window_X, self.reference_window_y)

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.model is None:  # there is no model before the first fit
            return 0.0

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

        if self.model is None:
            return np.zeros(len(X), dtype=np.float64)

        return self._score_batch(X)

    def fit_score(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Applies ``fit_score_partial`` to each instance in order and returns the scores.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to fit, in order.
            y (np.int32 array of shape (num_instances,)): The labels of the instances, in order (Default=None).

        Returns:
            np.float64 array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.empty(len(X), dtype=np.float64)
        start = 0
        for i, (xi, yi) in enumerate(_iterate(X, y)):
            next_len = len(self.cur_window_X) + 1
            if (
                not self.initial_ref_window
                and next_len < self.window_size
                and (
                    self.reference_window_X is None
                    or len(self.reference_window_X) < self.window_size
                )
            ):
                refit = (next_len & (next_len - 1)) == 0
            else:
                refit = next_len % self.sliding_size == 0

            if refit:
                y_pred[start : i + 1] = self.score(X[start : i + 1])
                start = i + 1

            self.fit_partial(xi, yi)

        if start < len(X):
            y_pred[start:] = self.score(X[start:])

        return y_pred
