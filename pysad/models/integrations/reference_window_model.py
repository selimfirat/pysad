import numpy as np

from .pyod_model import PYODModel


class ReferenceWindowModel(PYODModel):
    """Runs a batch model on a stream by retraining it on a reference window at regular intervals :cite:`xstream`.

    This wrapper keeps track of the reference window of size ``window_size``. For every ``sliding_size`` instances, it resets the model by training a new ``model_cls`` instance on the reference window. This implementation is based on the reference windowing described in :cite:`xstream`.

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
        model_cls,
        window_size,
        sliding_size,
        initial_window_X=None,
        initial_window_y=None,
        **kwargs,
    ):
        super().__init__(model_cls, **kwargs)

        self.window_size = window_size
        self.sliding_size = sliding_size

        self.cur_window_X = []
        self.cur_window_y = []

        self.reference_window_X = initial_window_X
        self.reference_window_y = initial_window_y

        if self.reference_window_X is not None:
            self._fit_model()
            self.initial_ref_window = True
        else:
            self.initial_ref_window = False

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            object: self.
        """
        self.cur_window_X.append(X)

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

    def fit_score_partial(self, X, y=None):
        """Scores the next instance with the current model, then fits the model to it.

        The current model never trained on the instance, as in the reference windowing of :cite:`xstream`. Before the first fit, there is no model, and the score is 0.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and fit.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            float: The anomalousness score of the input instance, as in `score_partial`.
        """
        score = 0.0 if self.model is None else self.score_partial(X)
        self.fit_partial(X, y)

        return score

    def _fit_model(self):
        self.reset_model()

        if self.reference_window_y is None:
            self.model.fit(self.reference_window_X)
        else:
            self.model.fit(self.reference_window_X, self.reference_window_y)

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        score = self.model.decision_function([X])[0]

        return score
