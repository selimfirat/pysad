"""Score a stream with EXPoSE, the expected similarity to the kernel mean embedding."""

from pysad.evaluation import AUROCMetric
from pysad.models import EXPoSE
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    # Random Fourier features approximate the RBF kernel, so one point costs
    # O(num_random_features * num_features) and the model takes constant memory.
    model = EXPoSE(num_random_features=500, gamma=1.0, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
