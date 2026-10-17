"""Average the scores of two detectors with an ensembler."""

import numpy as np

from pysad.evaluation import AUROCMetric
from pysad.models import LODA, xStream
from pysad.transform.ensemble import AverageScoreEnsembler
from pysad.transform.postprocessing import RunningZScorePostprocessor
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    models = [xStream(random_state=61), LODA(random_state=61)]
    # Raw scores of different detectors have different scales, so normalize each one first.
    normalizers = [RunningZScorePostprocessor(window_size=200) for _ in models]
    ensembler = AverageScoreEnsembler()
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        scores = np.array(
            [
                normalizer.fit_transform_partial(model.fit_score_partial(x))
                for model, normalizer in zip(models, normalizers, strict=True)
            ]
        )
        score = ensembler.fit_transform_partial(scores)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
