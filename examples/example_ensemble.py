from sklearn.utils import shuffle
from tqdm import tqdm
from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.models import xStream
from pysad.utils import ArrayStreamer
from pysad.transform.ensemble import AverageScoreEnsembler
from pysad.utils import Data
import numpy as np

# This example demonstrates the usage of an ensembling method.
if __name__ == '__main__':
    np.random.seed(61)

    data = Data("data")
    X_all, y_all = data.get_data("arrhythmia.mat")
    X_all, y_all = shuffle(X_all, y_all)
    iterator = ArrayStreamer(shuffle=False)
    auroc = AUROCMetric()

    models = [
        xStream(),
        LODA()
    ]
    ensembler = AverageScoreEnsembler()

    for X, y in tqdm(iterator.iter(X_all[100:], y_all[100:])):
        model_scores = np.empty(len(models), dtype=np.float)
        for i, model in enumerate(models):
            model.fit_partial(X)
            model_scores[i] = model.score_partial(X)

        score = ensembler.fit_transform_partial(model_scores)
        auroc.update(y, score)

    print("AUROC: ", auroc.get())
