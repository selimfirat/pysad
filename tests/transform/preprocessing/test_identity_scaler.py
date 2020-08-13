

def test_identity_scaler():
    import numpy as np
    from pysad.transform.preprocessing import IdentityScaler

    X = np.random.rand(100, 25)
    scaler = IdentityScaler()

    scaled_X = scaler.fit_transform(X)
    assert np.all(np.isclose(scaled_X, X))

    scaler = scaler.fit(X)
    scaled_X = scaler.transform(X)
    assert np.all(np.isclose(scaled_X, X))
