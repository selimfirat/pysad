"""Score a stream with MCOD, a distance-based detector over a sliding window."""

from pysad.evaluation import AUROCMetric
from pysad.models import MCOD
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("cardio.mat")

    # The score is the fraction of window instances that are not neighbors of the instance.
    # With max_radius=None, the first calibration_size (256) instances set the radius.
    model = MCOD(window_size=512)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
