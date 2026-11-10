"""The metrics of benchmarks/benchmark.py."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from tests.test_dimensionality import load_benchmark


def test_last_bit_differences_give_the_same_auroc():
    benchmark = load_benchmark()
    y = np.array([0, 1, 0, 1, 0, 0])
    scores = np.array([0.3, 0.3, 0.7, 0.7, 0.1, 1e-5])
    # another platform: the scores of the anomalies are one bit higher
    other = np.where(y == 1, np.nextafter(scores, np.inf), scores)
    assert roc_auc_score(y, scores) != roc_auc_score(y, other)
    assert roc_auc_score(y, benchmark.significant(scores)) == roc_auc_score(
        y, benchmark.significant(other)
    )


def test_sand_registration_and_fixed_configuration():
    from pysad.models import SAND

    benchmark = load_benchmark()
    assert "SAND" in benchmark.NAB_DETECTORS
    assert "SAND" not in benchmark.DETECTORS
    assert "SAND" not in benchmark.SEASONAL + benchmark.TIMESTAMPED
    assert benchmark.series_of("SAND") == benchmark.nab_series()
    for n, period in [(1000, None), (2000, 24), (20_000, 288)]:
        model = benchmark.UNIVARIATE["SAND"](n, period)
        assert isinstance(model, SAND)
        assert model.subsequence_length == 10
        assert model.centroid_length == 40
        assert model.batch_size == 256
        assert model.random_state == 0


@pytest.mark.parametrize("archive", ["nab", "ucr"])
def test_sand_benchmark_scores_before_update_and_reproduces_metrics(monkeypatch, archive):
    benchmark = load_benchmark()
    rng = np.random.default_rng(7)
    X = rng.normal(size=(768, 1))
    X[500:530, 0] += 4.0
    y = np.zeros(len(X), dtype=int)
    y[500:540] = 1
    scaled = (X - X.min()) / (X.max() - X.min())

    # Explicit score-then-fit calls are the expected protocol, including batch boundaries.
    model = benchmark.UNIVARIATE["SAND"](len(X), None)
    scores = []
    for x in scaled:
        scores.append(model.score_partial(x))
        model.fit_partial(x)
    scores = np.array(scores)
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores[:256], 0.0)
    assert np.any(scores[256:] != 0.0)
    repeated = benchmark.UNIVARIATE["SAND"](len(X), None).fit_score(scaled)
    np.testing.assert_array_equal(scores, repeated)

    monkeypatch.setattr(benchmark.Data, "get_data", lambda self, name: (X.copy(), y.copy()))
    if archive == "nab":
        row = benchmark.run_univariate("nab/synthetic.csv", "SAND")
        cutoff = benchmark.warmup(len(X))
        transform = benchmark.ConformalProbabilityCalibrator(window_size=len(X))
        np.testing.assert_array_equal(
            row["_nab"][0], transform.fit_transform(benchmark.significant(scores))
        )
        np.testing.assert_array_equal(row["_nab"][1], y)
    else:
        row = benchmark.run_ucr("ucr/synthetic_256_500_540", "SAND")
        cutoff = 256
        top = cutoff + benchmark.significant(scores[cutoff:]).argmax()
        assert row["solved"] == int(500 - benchmark.UCR_MARGIN <= top <= 539 + benchmark.UCR_MARGIN)
    expected = benchmark.significant(scores)[cutoff:]
    assert row["seed"] == 0
    assert row["auroc"] == round(roc_auc_score(y[cutoff:], expected), 4)
    assert row["aupr"] == round(average_precision_score(y[cutoff:], expected), 4)


def test_write_ranking_marks_chance_level_and_fraction_column(tmp_path, monkeypatch):
    """The tables mark values at or below chance, and the summary shows Mean AUPR / fraction."""
    benchmark = load_benchmark()

    def fake_load(dataset, seed=0, points=None):
        if dataset == "B":
            n = 16
            y = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            n = 8
            y = np.array([0, 0, 1, 1, 1, 0, 0, 0])
        return np.zeros((n, 1)), y

    monkeypatch.setattr(benchmark, "OUT", tmp_path)
    monkeypatch.setattr(benchmark, "load", fake_load)

    rows = [
        ("A", "LODA", 0.80, 0.50, 1.0),
        ("A", "RSHash", 0.90, 0.70, 1.0),
        ("A", "TEDA", 0.50, 0.25, 1.0),
        ("B", "LODA", 0.40, 0.10, 1.0),
        ("B", "RSHash", 0.60, 0.40, 1.0),
        ("B", "TEDA", 0.50, 0.25, 1.0),
        ("C", "LODA", 0.50, 0.50, 1.0),
        ("C", "RSHash", 0.50, 0.50, 1.0),
        ("C", "TEDA", 0.50, 0.50, 1.0),
    ]
    results = {}
    for dataset, detector, auroc, aupr, ms in rows:
        results[(dataset, detector, 0)] = {
            "dataset": dataset,
            "detector": detector,
            "seed": 0,
            "auroc": f"{auroc:.3f}",
            "aupr": f"{aupr:.3f}",
            "ms_per_point": f"{ms:.2f}",
            "commit": "abc123",
        }

    benchmark.write_ranking(results, {"A": "A", "B": "B", "C": "C"}, "")

    summary = (tmp_path / "summary.csv").read_text().splitlines()
    assert summary[0] == (
        "Detector,Mean rank,Mean AUROC,Mean AUPR,Mean AUPR / fraction,Time per point (ms),Commit"
    )
    assert summary[1] == (
        ":class:`~pysad.models.RSHash`,1.3,0.667,0.533,1.333,1.00,"
        "`abc123 <https://github.com/selimfirat/pysad/commit/abc123>`__"
    )
    assert summary[2] == (
        ":class:`~pysad.models.LODA`,2.3,0.567,0.367,0.800,1.00,"
        "`abc123 <https://github.com/selimfirat/pysad/commit/abc123>`__"
    )
    assert summary[3] == (
        ":class:`~pysad.models.TEDA`,2.3,0.500,0.333,0.833,1.00,"
        "`abc123 <https://github.com/selimfirat/pysad/commit/abc123>`__"
    )

    aupr = (tmp_path / "aupr.csv").read_text().splitlines()
    assert aupr[0] == "Dataset,Fraction of anomalies,RSHash,LODA,TEDA"
    assert aupr[1] == "A,0.500,**0.700**,*0.500*,*0.250*"
    assert aupr[2] == "B,0.250,**0.400**,*0.100*,*0.250*"
    assert aupr[3] == "C,0.500,***0.500***,***0.500***,***0.500***"

    auroc = (tmp_path / "auroc.csv").read_text().splitlines()
    assert auroc[0] == "Dataset,RSHash,LODA,TEDA"
    assert auroc[1] == "A,**0.900**,0.800,*0.500*"
    assert auroc[2] == "B,**0.600**,*0.400*,*0.500*"
    assert auroc[3] == "C,***0.500***,***0.500***,***0.500***"
