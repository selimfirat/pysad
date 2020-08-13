

def test_instance_unit_norm_scaler():
    import numpy as np
    from pysad.transform.preprocessing import InstanceUnitNormScaler

    X = np.random.rand(100, 25)
    scaler = InstanceUnitNormScaler()

    scaled_X = scaler.fit_transform(X)
    assert np.all(np.isclose(np.linalg.norm(scaled_X, axis=1), 1.0))

    scaler = scaler.fit(X)
    scaled_X = scaler.transform(X)
    assert np.all(np.isclose(np.linalg.norm(scaled_X, axis=1), 1.0))
