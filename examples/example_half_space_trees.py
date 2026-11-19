"""Score a stream with HalfSpaceTrees, which compares each point to the previous window."""

from pysad.evaluation import AUROCMetric
from pysad.models import HalfSpaceTrees
from pysad.utils import ArrayStreamer, Data, get_minmax_array

if __name__ == "__main__":
    X, y = Data().get_data("cardio.mat")

    # HalfSpaceTrees needs the range of each feature. Compute it from the data.
    mins, maxes = get_minmax_array(X)

    model = HalfSpaceTrees(feature_mins=mins, feature_maxes=maxes, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
