"""Score a synthetic stream with StreamKM++, which scores each point by its distance to the nearest center of a streaming coreset clustering."""

from pysad.evaluation import AUROCMetric
from pysad.models import StreamKMPlusPlus
from pysad.utils import ArrayStreamer, generate_stream

if __name__ == "__main__":
    X, y = generate_stream(random_state=0)

    model = StreamKMPlusPlus(num_clusters=3, coreset_size=100, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
