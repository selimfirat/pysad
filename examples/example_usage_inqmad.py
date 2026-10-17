"""Score a stream with Inqmad, a quantum measurement density detector."""

from pysad.evaluation import AUROCMetric
from pysad.models import Inqmad
from pysad.transform.preprocessing import InstanceUnitNormScaler
from pysad.utils import ArrayStreamer, Data

if __name__ == "__main__":
    X, y = Data().get_data("arrhythmia.mat")

    scaler = InstanceUnitNormScaler()
    # gamma is the kernel bandwidth. Tune it to the scale of the scaled points.
    model = Inqmad(input_shape=X.shape[1], dim_x=128, gamma=10, random_state=0)
    metric = AUROCMetric()

    for x, label in ArrayStreamer().iter(X, y):
        score = model.fit_score_partial(scaler.fit_transform_partial(x))
        metric.update(label, score)

    print(f"AUROC: {metric.get():.3f}")
