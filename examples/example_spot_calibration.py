"""Raise alerts at a chosen risk with extreme value theory (SPOT)."""

from pysad.models import LODA
from pysad.transform.probability_calibration import SPOTProbabilityCalibrator
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("cardio.mat")

    q = 1e-3  # the probability that a normal score raises an alert
    model = LODA(random_state=0)
    calibrator = SPOTProbabilityCalibrator(q=q, num_init=500)

    alerts = []
    for t, x in enumerate(ArrayStreamer().iter(X)):
        score = model.fit_score_partial(x)
        # The calibrated value is above 1 - q when the score is above calibrator.threshold_.
        # The calibrator returns 0 for the first 500 scores, while it collects them.
        if calibrator.fit_transform_partial(score) > 1 - q:
            alerts.append(t)

    print(f"{len(alerts)} alerts, {int(y[alerts].sum())} of them true anomalies")
    print(f"alarm threshold on the raw scores: {calibrator.threshold_:.3f}")
