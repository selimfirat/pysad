"""Score a stream with Inqmad, a quantum measurement density detector."""

from pysad.evaluation import AUROCMetric
from pysad.models import Inqmad
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    # Inqmad scales the points with its first 256 points, so it takes raw features.
    model = Inqmad(input_shape=X.shape[1], random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
