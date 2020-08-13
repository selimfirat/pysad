

def test_streamhash_projector(test_path):
    from sklearn.utils import shuffle
    from pysad.utils import Data
    import os
    from pysad.transform.projection import StreamhashProjector

    for num_components in [2, 50, 250]:
        data = Data(os.path.join(test_path, "../../../examples/data"))

        X_all, y_all = data.get_data("arrhythmia.mat")
        X_all, y_all = shuffle(X_all, y_all)

        projector = StreamhashProjector(num_components=num_components)

        projected_X = projector.fit_transform(X_all)

        assert projected_X.shape == (X_all.shape[0], num_components)
