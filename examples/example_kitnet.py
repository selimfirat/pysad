"""Score a stream with KitNet, an ensemble of small autoencoders for network traffic."""

from pysad.evaluation import AUROCMetric
from pysad.models import KitNet
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("cardio.mat")

    model = KitNet(random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
