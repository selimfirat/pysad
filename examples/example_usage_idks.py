"""Score a stream with IDK-S, the incremental Isolation Distributional Kernel.

IDK-S keeps a sliding window of recent instances as hypersphere partitions and scores each point by
how many window instances share its partition. Higher scores mean more anomalous instances.
"""

from pysad.evaluation import AUROCMetric
from pysad.models import IDK
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    model = IDK(input_shape=X.shape[1], random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
