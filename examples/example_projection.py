"""Project high-dimensional data and score with an online detector."""

from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.transform.projection import GaussianRandomProjector
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    projector = GaussianRandomProjector(num_components=20, random_state=0)
    model = LODA(random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        projected = projector.fit_transform_partial(x)
        score = model.fit_score_partial(projected)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
