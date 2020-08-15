# Import modules.
from sklearn.utils import shuffle
from pysad.evaluation import AUROCMetric
from pysad.models import xStream
from pysad.utils import ArrayStreamer
from pysad.transform.postprocessing import RunningAveragePostprocessor
from pysad.transform.preprocessing import InstanceUnitNormScaler
from pysad.utils import Data
from tqdm import tqdm
import numpy as np

# This example demonstrates the usage of the most modules in PySAD framework.
if __name__ == "__main__":
    np.random.seed(61)  # Fix random seed.

    # Get data to stream.
    data = Data("data")
    X_all, y_all = data.get_data("arrhythmia.mat")
    X_all, y_all = shuffle(X_all, y_all)

    iterator = ArrayStreamer(shuffle=False)  # Init streamer to simulate streaming data.

    model = xStream()  # Init xStream anomaly detection model.
    preprocessor = InstanceUnitNormScaler()  # Init normalizer.
    postprocessor = RunningAveragePostprocessor(window_size=5)  # Init running average postprocessor.
    auroc = AUROCMetric()  # Init area under receiver-operating- characteristics curve metric.

    for X, y in tqdm(iterator.iter(X_all[100:], y_all[100:])):  # Stream data.
        X = preprocessor.fit_transform_partial(X)  # Fit preprocessor to and transform the instance.

        score = model.fit_score_partial(X)  # Fit model to and score the instance.
        score = postprocessor.fit_transform_partial(score)  # Apply running averaging to the score.

        auroc.update(y, score)  # Update AUROC metric.

    # Output resulting AUROCS metric.
    print("AUROC: ", auroc.get())
