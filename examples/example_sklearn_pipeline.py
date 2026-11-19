"""Score batches with fitted preprocessing and a fitted PySAD model."""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pysad.integrations import SklearnAnomalyDetector
from pysad.models import LODA
from pysad.utils import Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")
    history, batch = X[:200], X[200:]
    scaler = StandardScaler().fit(history)
    scaled_history = scaler.transform(history)
    model = LODA(random_state=0).fit(scaled_history)
    # Set the threshold in PySAD score units from historical scores.
    threshold = float(np.quantile(model.score(scaled_history), 0.95))
    pipeline = Pipeline(
        steps=[
            ("scale", scaler),
            ("detect", SklearnAnomalyDetector(model, threshold=threshold)),
        ]
    )
    scores = pipeline.score_samples(batch)
    labels = pipeline.predict(batch)
    print(f"Scored instances: {scores.size}")
    print(f"Outliers: {(labels == -1).sum()}")
