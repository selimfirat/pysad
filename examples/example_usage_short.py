"""Score a benchmark dataset point by point and measure AUROC."""

from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.utils import Data

if __name__ == "__main__":
    model = LODA(random_state=0)
    metric = AUROCMetric()

    for x, label in Data().get_iterator("arrhythmia.mat", shuffle=False):
        score = model.fit_score_partial(x)  # learn from x, then score it
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
