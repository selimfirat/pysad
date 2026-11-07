"""Raise alerts from the anomaly likelihood of HTM, which reacts to a sustained increase of the scores more than to single noisy scores."""

import numpy as np

from pysad.transform.probability_calibration import AnomalyLikelihoodProbabilityCalibrator

if __name__ == "__main__":
    # Raw scores between 0 and 1, as HTM gives: mostly low, with single noisy spikes.
    rng = np.random.default_rng(0)
    scores = rng.beta(0.5, 5, size=5000)
    spikes = rng.choice(5000, size=20, replace=False)
    scores[spikes] = 1.0
    scores[4000:4050] = rng.uniform(0.5, 1.0, size=50)  # the anomaly: a sustained increase

    calibrator = AnomalyLikelihoodProbabilityCalibrator()

    alerts = []
    for t, score in enumerate(scores):
        likelihood = calibrator.fit_transform_partial(score)
        # The paper reports an anomaly when the likelihood is at least 1 - 1e-5.
        if likelihood >= 1 - 1e-5:
            alerts.append(t)

    print(f"{int((scores >= 0.99).sum())} raw scores are at least 0.99")
    print(f"likelihood alerts at {alerts}, the anomaly starts at 4000")
