"""Standardize streaming anomaly scores with a running z-score."""

from pysad.evaluation import AUROCMetric
from pysad.models import xStream
from pysad.transform.postprocessing import RunningZScorePostprocessor
from pysad.utils import Data

if __name__ == "__main__":
    model = xStream(random_state=0)
    postprocessor = RunningZScorePostprocessor(window_size=100)
    metric = AUROCMetric()

    for x, label in Data().get_iterator("arrhythmia.mat", shuffle=False):
        score = model.fit_score_partial(x)
        normalized_score = postprocessor.fit_transform_partial(score)
        metric.update(label, normalized_score)

    print(f"AUROC: {metric.get():.3f}")
