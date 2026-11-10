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


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.95, 0.995, 0.99, 0.2], [1 / np.sqrt(3), 0, -1 / np.sqrt(3), 1 / np.sqrt(3)]),
        ([0.1, 0.1, 0.1, 0.1], [0, 0, 0, 0]),
        ([1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0]),
    ],
)
def test_alert_mcc_uses_each_threshold(scores, expected):
    benchmark = load_benchmark()
    row = benchmark.alerts("conformal", np.array(scores), np.array([1, 0, 1, 0]), True)
    for threshold, value in zip([0.95, 0.99, 0.995, "rate"], expected, strict=True):
        assert row[f"conformal_{threshold}_mcc"] == round(value, 4)


def test_alert_mcc_excludes_invalid_runs():
    benchmark = load_benchmark()
    row = benchmark.alerts("conformal", np.array([0.1, np.nan]), np.array([0, 1]), False)
    assert np.isnan(row["conformal_0.95_mcc"])


def test_alert_mcc_survives_csv_and_averages_runs(tmp_path, monkeypatch):
    import csv

    benchmark = load_benchmark()
    monkeypatch.setattr(benchmark, "OUT", tmp_path)
    rows = []
    for seed, scores in enumerate([[0.95, 0.995, 0.99, 0.2], [0.1, 0.1, 0.1, 0.1]]):
        row = {
            "dataset": "synthetic",
            "detector": "LODA",
            "seed": seed,
            "auroc": 0.5,
            "aupr": 0.5,
            "ms_per_point": 0.1,
        }
        for name in benchmark.CALIBRATORS:
            row.update(benchmark.alerts(name, np.array(scores), np.array([1, 0, 1, 0]), True))
        rows.append(row)
    benchmark.write_csv(
        "results.csv",
        benchmark.FIELDS,
        [[row.get(f, "") for f in benchmark.FIELDS] for row in rows],
    )
    results = benchmark.read_results(tmp_path / "results.csv")
    monkeypatch.setattr(benchmark, "DATASETS", ["synthetic"])
    monkeypatch.setattr(benchmark, "DRIFT", {})
    monkeypatch.setattr(
        benchmark.Data,
        "describe",
        lambda self, name: {"points": 4, "features": 1, "anomaly_rate": 0.5},
    )
    benchmark.write_tables(results)
    with (tmp_path / "alerts.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert [row["MCC"] for row in table] == ["0.289", "0.000", "-0.289", "0.289"] * 2
