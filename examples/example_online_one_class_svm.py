"""Score a stream with OnlineOneClassSVM, the online one-class SVM of NORMA."""

from pysad.evaluation import AUROCMetric
from pysad.models import OnlineOneClassSVM
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_stream("arrhythmia.mat", random_state=0, scale=True)

    # The kernel width of the paper, for features in [-1, 1], rescaled to features in [0, 1].
    model = OnlineOneClassSVM(gamma=8.0 / X.shape[1])
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
