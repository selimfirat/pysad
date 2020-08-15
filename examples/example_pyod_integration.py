# Import modules.
from pyod.models.iforest import IForest
from sklearn.utils import shuffle
from pysad.evaluation import AUROCMetric
from pysad.models.integrations import ReferenceWindowModel
from pysad.utils import ArrayStreamer
from pysad.utils import Data
from tqdm import tqdm
import numpy as np

# This example demonstrates the integration of a PyOD model via ReferenceWindowModel.
if __name__ == "__main__":
    np.random.seed(61)  # Fix seed.

    # Get data to stream.
    data = Data("data")
    X_all, y_all = data.get_data("arrhythmia.mat")
    X_all, y_all = shuffle(X_all, y_all)
    iterator = ArrayStreamer(shuffle=False)

    # Fit reference window integration to first 100 instances initially.
    model = ReferenceWindowModel(model_cls=IForest, window_size=240, sliding_size=30, initial_window_X=X_all[:100])

    auroc = AUROCMetric()  # Init area under receiver-operating-characteristics curve metric tracker.

    for X, y in tqdm(iterator.iter(X_all[100:], y_all[100:])):

        model.fit_partial(X)  # Fit to the instance.
        score = model.score_partial(X)  # Score the instance.

        auroc.update(y, score)  # Update the metric.

    # Output AUROC metric.
    print("AUROC: ", auroc.get())
