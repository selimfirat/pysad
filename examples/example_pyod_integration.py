from pyod.models.iforest import IForest
from sklearn.utils import shuffle
from pysad.evaluation.metrics import AUROCMetric
from pysad.models.integrations.reference_window_model import ReferenceWindowModel
from pysad.utils.array_streamer import ArrayStreamer
from pysad.utils.data import Data
from tqdm import tqdm
import numpy as np

# This example demonstrates the integration of a PYOD model via ReferenceWindowModel.
if __name__ == "__main__":
    np.random.seed(61)
    data = Data("data")
    X_all, y_all = data.get_data("arrhythmia.mat")
    X_all, y_all = shuffle(X_all, y_all)

    model = ReferenceWindowModel(model_cls=IForest, window_size=240, sliding_size=30, initial_window_X=X_all[:100])

    iterator = ArrayStreamer(shuffle=False)

    auroc = AUROCMetric()

    y_pred = []
    for X, y in tqdm(iterator.iter(X_all[100:], y_all[100:])):
        model.fit_partial(X)
        score = model.score_partial(X)

        y_pred.append(score)

        auroc.update(y, score)

    print("AUROC: ", auroc.get())
