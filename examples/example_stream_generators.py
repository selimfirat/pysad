"""Score a detector on normal and adversarial streams to measure resilience."""

import numpy as np

from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.transform.stream_generators import SlowPoisonStream, StealthBurstStream
from pysad.utils import ArrayStreamer

if __name__ == "__main__":
    X_normal = np.random.RandomState(0).standard_normal((2000, 5))

    for name, generator in (
        ("slow poison", SlowPoisonStream(random_state=0)),
        ("stealth burst", StealthBurstStream(random_state=0)),
    ):
        model = LODA(random_state=0)
        metric = AUROCMetric()
        for x_normal in ArrayStreamer().iter(X_normal):
            x = generator.fit_transform_partial(x_normal)
            label = int(generator.is_anomaly)
            score = model.fit_score_partial(x)
            metric.update(label, score)
        print(f"{name}: AUROC {metric.get():.3f}")
