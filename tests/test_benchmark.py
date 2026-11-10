"""The metrics of benchmarks/benchmark.py."""

import csv

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


def test_summary_has_no_mean_aupr(tmp_path):
    benchmark = load_benchmark()
    datasets = ["ds1", "ds2"]
    results = {}
    for ds in datasets:
        # A: best AUROC, C: worst AUROC but best AUPR. AUPR must not drive the order.
        for detector, auroc, aupr, ms in [
            ("LODA", 0.9, 0.4, 1.0),
            ("RSHash", 0.8, 0.5, 2.0),
            ("xStream", 0.7, 0.9, 1.5),
        ]:
            results[(ds, detector, 0)] = {
                "dataset": ds,
                "detector": detector,
                "seed": 0,
                "auroc": auroc,
                "aupr": aupr,
                "ms_per_point": ms,
                "commit": "abc",
            }
    benchmark.OUT = tmp_path
    benchmark.write_ranking(results, {ds: ds for ds in datasets}, "")
    with open(tmp_path / "summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == [
        "Detector",
        "Mean rank",
        "Mean AUROC",
        "Time per point (ms)",
        "Commit",
    ]
    assert [
        row["Detector"] for row in rows
    ] == [f":class:`~pysad.models.{d}`" for d in ["LODA", "RSHash", "xStream"]]
    # The per-dataset AUPR table still exists, with one value per dataset and detector.
    with open(tmp_path / "aupr.csv", newline="") as f:
        aupr_rows = list(csv.DictReader(f))
    assert [row["Dataset"] for row in aupr_rows] == datasets
    assert float(aupr_rows[0]["xStream"].replace("*", "")) == 0.9  # the best cell is bolded
