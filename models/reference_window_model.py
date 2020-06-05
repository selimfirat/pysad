from models.base_model import BaseModel
from models.pyod_model import PYODModel


class ReferenceWindowModel(BaseModel, PYODModel):

    def __init__(self, model_cls, window_size, sliding_size, **kwargs):
        super().__init__(**kwargs)

        self.window_size = window_size
        self.sliding_size = sliding_size
        self.model_cls = model_cls
        self.model = None

        self.cur_window = []
        self.reference_window = None

    def fit_partial(self, X, y=None):

        self.cur_window.append((X, y))

        if len(self.cur_window) == self.sliding_size:
            self.reference_window = self.reference_window[self.sliding_size:] + self.cur_window
            self.cur_window = []
            self.model = self.reset_model()
            ref_window_X, ref_window_y = list(zip(*self.cur_window))
            if ref_window_y[0] is None:
                ref_window_y = None

            self.model.fit(ref_window_X, ref_window_y)

    def score_partial(self, X):

        if self.reference_window is None:
            ref_window_X, ref_window_y = list(zip(*self.cur_window))
            if ref_window_y[0] is None:
                ref_window_y = None

            self.model = self.reset_model()
            self.model.fit(ref_window_X, ref_window_y)

        score = self.model.decision_function([X])[0]

        return score
