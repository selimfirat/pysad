from pysad.evaluation.metrics import AUROCMetric
from pysad.models.loda import LODA
from pysad.utils.data import Data

model = LODA()
metric = AUROCMetric()
streaming_data = Data().get_iterator("arrhythmia.mat")

for x, y_true in streaming_data:
    anomaly_score = model.fit_score_partial(x)

    metric.update(y_true, anomaly_score)

print(f"Area under ROC metric is {metric.get()}.")
