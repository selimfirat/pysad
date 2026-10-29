"""Scale streaming feature vectors per instance with InstanceStandardScaler."""

from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.transform.preprocessing import InstanceStandardScaler
from pysad.utils import Data

if __name__ == "__main__":
    scaler = InstanceStandardScaler()
    model = LODA(random_state=0)
    metric = AUROCMetric()

    for x, label in Data().get_iterator("arrhythmia.mat", shuffle=False):
        scaled_x = scaler.fit_transform_partial(x)
        score = model.fit_score_partial(scaled_x)
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
