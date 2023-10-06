# Import modules.
from pysad.evaluation import AUROCMetric
from pysad.models import LODA
from pysad.models import xStream
from pysad.utils import ArrayStreamer
from pysad.transform.ensemble import AverageScoreEnsembler
from pysad.utils import Data
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np

# This example demonstrates the usage of an ensembling method.
if __name__ == '__main__':
    np.random.seed(61)  # Fix random seed.

    data = Data("data")
    X_all, y_all = data.get_data("arrhythmia.mat")  # Load Aryhytmia data.
    X_all, y_all = shuffle(X_all, y_all)  # Shuffle data.
    iterator = ArrayStreamer(shuffle=False)  # Create streamer to simulate streaming data.
    auroc = AUROCMetric()  # Tracker of area under receiver-operating- characteristics curve metric.

    models = [  # Models to be ensembled.
        xStream(),
        LODA()
    ]
    ensembler = AverageScoreEnsembler()  # Ensembler module.

    for X, y in tqdm(iterator.iter(X_all, y_all)):  # Iterate over examples.
        model_scores = np.empty(len(models), dtype=np.float64)

        # Fit & Score via for each model.
        for i, model in enumerate(models):
            model.fit_partial(X)
            model_scores[i] = model.score_partial(X)

        score = ensembler.fit_transform_partial(model_scores)  # fit to ensembler model and get ensembled score.

        auroc.update(y, score)  # update AUROC metric.

    # Output score.
    print("AUROC: ", auroc.get())
