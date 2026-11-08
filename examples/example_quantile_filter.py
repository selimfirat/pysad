"""Label each anomaly score 0 or 1 with a running quantile of the previous scores."""

from pysad.evaluation import PrecisionMetric, RecallMetric
from pysad.models import LODA
from pysad.transform.postprocessing import QuantileFilter
from pysad.utils import Data

if __name__ == "__main__":
    model = LODA(random_state=0)
    # A score gets the label 1 if it is at least the 0.9 quantile of the previous scores.
    quantile_filter = QuantileFilter(q=0.9)
    precision = PrecisionMetric()
    recall = RecallMetric()

    for x, label in Data().get_iterator("cardio.mat", shuffle=False):
        score = model.fit_score_partial(x)
        predicted = quantile_filter.fit_transform_partial(score)
        precision.update(label, predicted)
        recall.update(label, predicted)

    print(f"Precision: {precision.get():.3f}, recall: {recall.get():.3f}")
