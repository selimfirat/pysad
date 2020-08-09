from pysad.models.one_fit_model import OneFitModel


def test_reference_window(test_path):
    from sklearn.utils import shuffle
    from pysad.models.reference_window_model import ReferenceWindowModel
    from pysad.utils.data import Data
    from pysad.evaluation.metrics import AUROCMetric
    from pysad.utils.array_streamer import ArrayStreamer
    import os
    from pyod.models.iforest import IForest

    data = Data(os.path.join(test_path,"../../examples/data"))

    X_all, y_all = data.get_data("arrhythmia.mat")
    X_all, y_all = shuffle(X_all, y_all)

    model = ReferenceWindowModel(model_cls=IForest, window_size=240, sliding_size=30,
                                 initial_window_X=X_all[:100])

    iterator = ArrayStreamer(shuffle=False)

    auroc = AUROCMetric()

    y_pred = []
    for X, y in iterator.iter(X_all[100:], y_all[100:]):
        model.fit_partial(X)
        score = model.score_partial(X)

        y_pred.append(score)

        auroc.update(y, score)

    print("AUROC: ", auroc.get())


def test_one_fit(test_path):
    from sklearn.utils import shuffle
    from pysad.utils.data import Data
    from pysad.evaluation.metrics import AUROCMetric
    from pysad.utils.array_streamer import ArrayStreamer
    import os
    from pyod.models.iforest import IForest

    data = Data(os.path.join(test_path, "../../examples/data"))

    X_all, y_all = data.get_data("arrhythmia.mat")
    print(X_all, y_all)
    X_all, y_all = shuffle(X_all, y_all)

    model = OneFitModel(model_cls=IForest, initial_X=X_all[:100])

    iterator = ArrayStreamer(shuffle=False)

    auroc = AUROCMetric()

    y_pred = []
    for X, y in iterator.iter(X_all[100:], y_all[100:]):
        model.fit_partial(X)
        score = model.score_partial(X)

        y_pred.append(score)

        auroc.update(y, score)

    print("AUROC: ", auroc.get())
