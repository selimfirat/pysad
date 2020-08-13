from sklearn.utils import shuffle
from pysad.evaluation import AUROCMetric
from pysad.models import xStream
from pysad.utils import ArrayStreamer
from pysad.transform.postprocessing import RunningAveragePostprocessor
from pysad.transform.preprocessing import InstanceUnitNormScaler
from pysad.utils import Data
from tqdm import tqdm
import numpy as np

# This example demonstrates the usage of the most modules in pysad framework.
if __name__ == "__main__":
    np.random.seed(61)
    data = Data("data")

    X_all, y_all = data.get_data("arrhythmia.mat")
    X_all, y_all = shuffle(X_all, y_all)

    iterator = ArrayStreamer(shuffle=False)
    model = xStream()
    preprocessor = InstanceUnitNormScaler()
    postprocessor = RunningAveragePostprocessor(window_size=5)
    auroc = AUROCMetric()

    y_pred = []
    for X, y in tqdm(iterator.iter(X_all[100:], y_all[100:])):
        X = preprocessor.fit_transform_partial(X)

        score = model.fit_score_partial(X)
        score = postprocessor.fit_transform_partial(score)

        y_pred.append(score)
        auroc.update(y, score)

    print("AUROC: ", auroc.get())
