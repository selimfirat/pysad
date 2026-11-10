"""Run a batch PyOD KNN on a stream after fitting it once on initial data."""

from pyod.models.knn import KNN

from pysad.evaluation import AUROCMetric
from pysad.models.integrations import OneFitModel
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    model = OneFitModel(KNN, initial_X=X[:100])
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X[100:], y[100:]):
        # fit_score_partial does not learn here: the model only scores each point.
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
