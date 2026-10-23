"""Score a univariate series from recent windows of points."""

from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.transform.preprocessing import ShingleTransformer
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("nab/artificialWithAnomaly/art_daily_jumpsup.csv")

    shingler = ShingleTransformer(shingle_size=24)
    model = LODA(random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        shingle = shingler.fit_transform_partial(x)
        score = model.fit_score_partial(shingle)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
