"""Explain an xStream anomaly score with per-feature attribution weights."""

import numpy as np

from pysad.models import xStream
from pysad.utils import ArrayStreamer, generate_stream

if __name__ == "__main__":
    X, y = generate_stream(num_instances=2000, num_features=5, random_state=0)

    model = xStream(random_state=0)

    # Learn the normal distribution from the first part of the stream.
    split = 400
    for x in ArrayStreamer().iter(X[:split]):
        model.fit_partial(x)

    # Score the rest of the stream without learning from it, then explain the highest-scoring point.
    scores = model.score(X[split:])
    index = int(np.argmax(scores))
    point = X[split + index]

    weights = model.score_explain_partial(point)
    ranked = np.argsort(weights)[::-1]

    print(f"Explaining point {split + index} with score {scores[index]:.3f}")
    for feature in ranked:
        print(f"  feature {feature}: weight {weights[feature]:.3f}")
