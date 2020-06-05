from evaluation.base_evaluator import BaseEvaluator
from utils.mean_meter import MeanMeter


class WindowedEvaluator(BaseEvaluator):

    def __init__(self, window_length, evaluator, **kwargs):
        super().__init__(**kwargs)

        self.window_length = window_length
        self.evaluator_cls = evaluator

        self.evaluator = self.init_evaluator()

        self.score_meter = MeanMeter()

        self.step = 0

    def init_evaluator(self):
        return self.evaluator_cls(**self.kwargs)

    def update(self, y_true, y_pred):

        self.step += 1
        self.evaluator.update(y_true, y_pred)

        if self.step % self.window_length == 0:
            score = self.evaluator.get()
            self.score_meter.update(score)
            self.evaluator = self.init_evaluator()

        return self

    def get(self):

        return self.score_meter.get_mean()
