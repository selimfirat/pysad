"""Average the normalized scores of two detectors with an ensemble."""

from pysad.evaluation import AUROCMetric
from pysad.models import LODA, xStream
from pysad.transform.postprocessing import ZScorePostprocessor
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    # Raw scores of different detectors have different scales, so normalize each one first.
    ensemble = (xStream(random_state=0) | ZScorePostprocessor(window_size=200)) & (
        LODA(random_state=0) | ZScorePostprocessor(window_size=200)
    )
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        metric.update(label, ensemble.fit_score_partial(x))

    print(f"AUROC: {metric.get():.3f}")
