

def test_instance_standard_scaler():
    import numpy as np
    from pysad.transform.preprocessing import InstanceStandardScaler

    X = np.random.rand(100, 25)
    scaler = InstanceStandardScaler()

    scaled_X = scaler.fit_transform(X)
    assert np.all(np.isclose(scaled_X, (X - X.mean(1, keepdims=True))/X.std(1, keepdims=True)))

    scaler = scaler.fit(X)
    scaled_X = scaler.transform(X)
    assert np.all(np.isclose(scaled_X, (X - X.mean(1, keepdims=True))/X.std(1, keepdims=True)))
