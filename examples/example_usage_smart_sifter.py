"""Score a stream with SmartSifter, a Gaussian mixture that forgets old points."""

from pysad.evaluation import AUROCMetric
from pysad.models import SmartSifter
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    # A larger discount forgets old points faster, so the mixture follows drift faster.
    model = SmartSifter(num_components=2, discount=0.01, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)  # -log p(x) under the mixture, then learn x
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
