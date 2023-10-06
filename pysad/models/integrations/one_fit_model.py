from .pyod_model import PYODModel


class OneFitModel(PYODModel):
    """The wrapper model fits the `model_cls` to the initial instnaces. Then it scores all incoming instances, with this fitted model.

        Args:
            model_cls (class):     The model class to be instantiated.
            initial_X (np.float64 array of shape (num_initial_instances, num_features)): Initial instances to fit.
            initial_y (np.int32 array of shape (num_initial_instances,): Initial window's ground truth labels. Used if not None. Needs to be `None` for the unsupervised `model_cls` models. (Default=None).
            **kwargs (Keyword arguments): Keyword arguments that are passed to the `model_cls`.
    """

    def __init__(self, model_cls, initial_X, initial_y=None, **kwargs):
        super().__init__(model_cls, **kwargs)

        self.initial_y = initial_y
        self.initial_X = initial_X
        self._fit_model()

    def fit_partial(self, X, y=None):
        """This method is ignored. Added for convenience.

        Args:
            X: any
            y: any

        Returns:
            object: Returns the self.
        """
        return self

    def _fit_model(self):

        self.reset_model()

        if self.initial_y is None:
            self.model.fit(self.initial_X)
        else:
            self.model.fit(self.initial_X, self.initial_y)

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        score = self.model.decision_function([X])[0]

        return score
