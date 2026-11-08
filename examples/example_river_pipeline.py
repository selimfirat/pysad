"""Run a PySAD detector in a River preprocessing pipeline, scoring before learning."""

from river import preprocessing

from pysad.evaluation import AUROCMetric
from pysad.integrations import RiverAnomalyDetector
from pysad.models import LODA
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")
    pipeline = preprocessing.StandardScaler() | RiverAnomalyDetector(LODA(random_state=0))
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        features = dict(enumerate(x))
        # Score before updating any stage with the current instance.
        score = pipeline.score_one(features)
        metric.update(label, score)
        pipeline.learn_one(features)

    print(f"AUROC: {metric.get():.3f}")
