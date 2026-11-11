"""Find the features that make a point anomalous with the explanations of LODA."""

import numpy as np

from pysad.models import LODA

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 8))
    model = LODA(random_state=0).fit(X)

    point = rng.normal(size=8)
    point[[2, 5]] += 6.0  # features 2 and 5 are anomalous

    # A higher t statistic means that the feature contributes more to the score.
    t = model.explain_partial(point)
    print(f"Score: {model.score_partial(point):.2f}")
    print(f"Features by contribution: {t.argsort()[::-1].tolist()}")
