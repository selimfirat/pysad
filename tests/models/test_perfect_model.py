def test_perfect_model():
    import numpy as np
    from pysad.models.perfect_model import PerfectModel
    np.random.seed(61)
    model = PerfectModel()
    y1 = np.random.randint(0, 2, 100)
    y = np.random.randint(0, 2, 100)
    y_pred = model.fit_score([None]*100, y)
    print(y_pred)
    assert np.all(np.isclose(y, y_pred))
    assert not np.all(np.isclose(y1, y_pred))
