

def test_gaussian_random_projector(test_path):
    from pysad.transform.projection import GaussianRandomProjector

    for num_components in [2, 50, 250]:

        projector = GaussianRandomProjector(num_components=num_components)

        helper_test_projector(test_path, projector, num_components)


def test_sparse_random_projector(test_path):
    from pysad.transform.projection import SparseRandomProjector

    for num_components in [2, 50, 250]:

        projector = SparseRandomProjector(num_components=num_components)

        helper_test_projector(test_path, projector, num_components)


def helper_test_projector(test_path, projector, num_components):
    import os
    from sklearn.utils import shuffle
    from pysad.utils import Data

    data = Data(os.path.join(test_path, "../../../examples/data"))

    X_all, y_all = data.get_data("arrhythmia.mat")
    X_all, y_all = shuffle(X_all, y_all)
    projected_X = projector.fit_transform(X_all)

    assert projected_X.shape == (X_all.shape[0], num_components)
