import pyod
from sklearn.utils import shuffle

from models.reference_window_model import ReferenceWindowModel
from streaming.array_iterator import ArrayIterator
from utils.data import Data

data = Data("data")

X_all, y_all = data.get_data("arrhythmia.mat")
X_all, y_all = shuffle(X_all, y_all)

model = ReferenceWindowModel(model_cls=pyod.models.iforest.IForest, window_size=240, sliding_size=30, initial_window_X=X_all[:100])

iterator = ArrayIterator()

for X, y in iterator.iter(X_all[100:], y_all[100:]):
    model.fit_partial(X)
    score = model.score_partial(X)

    print(score, y)
