"""Score a stream with Half-Space Trees, whose trees need each feature's range before the stream starts."""

from pysad.evaluation import AUROCMetric
from pysad.models import HalfSpaceTrees
from pysad.utils import ArrayStreamer, Data, get_minmax_array

if __name__ == "__main__":
    X, y = Data().get_data("cardio.mat")

    # The trees split the feature space between these bounds.
    # Here they come from the first window_size points.
    # Later points can fall outside them.
    # If you know each feature range, use it instead.
    feature_mins, feature_maxes = get_minmax_array(X[:250])

    model = HalfSpaceTrees(feature_mins, feature_maxes, window_size=250, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
