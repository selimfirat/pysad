from .pyod_model import PYODModel
import numpy as np


class ReferenceWindowModel(PYODModel):
    """This PyOD model wrapper wraps the batch anomaly detectors. This wrapper keeps track of the reference window of size `window_length`. For every `sliding_size` instnaces, it resets the model by training new `model_cls` instance with the reference window. This implementation is based on the reference windowing described in :cite:`xstream`.

        Args:
            model_cls (class): The model class to be instantiated.
            window_size (int): The size of each window.
            sliding_size (int): The sliding length of the windows.
            initial_X (np.float64 array of shape (num_initial_instances, num_features)): Initial instances to fit.
            initial_y (np.int32 array of shape (num_initial_instances,)): Initial window's ground truth labels. Used if not None. Needs to be `None` for the unsupervised `model_cls` models. (Default=None).
            **kwargs (Keyword arguments): Keyword arguments that is passed to the `model_cls`.
    """

    def __init__(
            self,
            model_cls,
            window_size,
            sliding_size,
            initial_window_X=None,
            initial_window_y=None,
            **kwargs):
        """

        Args:
            model_cls:
            window_size:
            sliding_size:
            initial_window_X:
            initial_window_y:
        """
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
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            object: self.
        """
        self.cur_window_X.append(X)

        if y is not None:
            self.cur_window_y.append(y)

        if not self.initial_ref_window and len(
                self.cur_window_X) < self.window_size:
            self.reference_window_X = self.cur_window_X
            self.reference_window_y = self.cur_window_y if y is not None else None
            self._fit_model()
        elif len(self.cur_window_X) % self.sliding_size == 0:
            self.reference_window_X = np.concatenate(
                [self.reference_window_X, self.cur_window_X], axis=0)
            self.reference_window_X = self.reference_window_X[max(
                0, len(self.reference_window_X) - self.window_size):]

            if y is not None:
                self.reference_window_y = self.reference_window_y[max(
                    0, len(self.reference_window_y) - self.window_size):]
                self.reference_window_y = np.concatenate(
                    [self.reference_window_y, self.cur_window_y], axis=0)

            self.cur_window = []
            self._fit_model()

        return self

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
