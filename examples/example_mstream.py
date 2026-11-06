"""Score a stream of network connections with MStream, which finds bursts of similar records."""

from pysad.evaluation import AUROCMetric
from pysad.models import MStream
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    # The http connections of KDD Cup 1999 in time order. The attacks come in bursts.
    X, y = Data().get_data("kddcup99/http.csv")

    # Every 10 records form one time tick. The features are already log-transformed and go below
    # -1, so the model does not take log(1 + x) again.
    model = MStream(tick_size=10, log_transform=False, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
