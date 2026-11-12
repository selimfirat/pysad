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


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "labels",
    [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 0, 0],
    ],
)
def test_drift_metrics_check_labels_after_warmup(monkeypatch, labels):
    benchmark = load_benchmark()
    dataset = next(iter(benchmark.DRIFT))
    X = np.arange(len(labels), dtype=float).reshape(-1, 1)
    y = np.array(labels)
    monkeypatch.setattr(benchmark, "load", lambda *args: (X, y))

    row = benchmark.run((dataset, "TEDA", 0, len(X)))
    cutoff = benchmark.warmup(len(X))
    evaluated = y[cutoff:]
    if evaluated.min() == evaluated.max():
        assert np.isnan(row["auroc"])
        assert np.isnan(row["aupr"])
    else:
        scores = benchmark.significant(benchmark.DETECTORS["TEDA"](X, 0).fit_score(X))[cutoff:]
        assert row["auroc"] == round(roc_auc_score(evaluated, scores), 4)
        assert row["aupr"] == round(average_precision_score(evaluated, scores), 4)


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
    benchmark = load_benchmark()
    monkeypatch.setattr(benchmark, "OUT", tmp_path)
    labels = np.array([1, 0, 1, 0])
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
            row.update(benchmark.alerts(name, np.array(scores), labels, True))
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
        benchmark, "load", lambda dataset, seed=0, points=None: (np.zeros((4, 1)), labels)
    )
    monkeypatch.setattr(
        benchmark.Data,
        "describe",
        lambda self, name: {"points": 4, "features": 1, "anomaly_rate": 0.5},
    )
    benchmark.write_tables(results)
    with (tmp_path / "alerts.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert [row["MCC"] for row in table] == ["0.289", "0.000", "-0.289", "0.289"] * 2


@pytest.fixture
def ranking_tables(tmp_path, monkeypatch):
    """Write tables with values above, below, and at chance."""
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
    return tmp_path


def test_write_ranking_marks_chance_level_and_fraction_column(ranking_tables):
    """Mark values at or below chance and show the fraction of anomalies."""
    tmp_path = ranking_tables

    summary = (tmp_path / "summary.csv").read_text().splitlines()
    assert summary[0] == "Detector,Mean rank,Mean AUROC,Time per point (ms),Commit"
    assert summary[1] == (
        ":class:`~pysad.models.RSHash`,1.3,0.667,1.00,"
        "`abc123 <https://github.com/selimfirat/pysad/commit/abc123>`__"
    )
    assert summary[2] == (
        ":class:`~pysad.models.LODA`,2.3,0.567,1.00,"
        "`abc123 <https://github.com/selimfirat/pysad/commit/abc123>`__"
    )
    assert summary[3] == (
        ":class:`~pysad.models.TEDA`,2.3,0.500,1.00,"
        "`abc123 <https://github.com/selimfirat/pysad/commit/abc123>`__"
    )

    aupr = (tmp_path / "aupr.csv").read_text().splitlines()
    assert aupr[0] == "Dataset,Fraction of anomalies,RSHash,LODA,TEDA"
    assert aupr[1] == "A,0.500,**0.700**,*0.500*,*0.250*"
    assert aupr[2] == "B,0.250,**0.400**,*0.100*,*0.250*"

    auroc = (tmp_path / "auroc.csv").read_text().splitlines()
    assert auroc[0] == "Dataset,RSHash,LODA,TEDA"
    assert auroc[1] == "A,**0.900**,0.800,*0.500*"
    assert auroc[2] == "B,**0.600**,*0.400*,*0.500*"


def test_best_values_at_chance_render_in_bold_and_italics(ranking_tables):
    """Render both styles without visible markup in the documentation."""
    from io import StringIO
    from pathlib import Path

    pytest.importorskip("sphinx")
    from sphinx.application import Sphinx

    tmp_path = ranking_tables
    (tmp_path / "index.rst").write_text(
        "Chance values\n============="
        "\n\n.. csv-table:: AUPR\n   :file: aupr.csv\n   :header-rows: 1\n"
        "\n.. csv-table:: AUROC\n   :file: auroc.csv\n   :header-rows: 1\n"
    )
    warnings = StringIO()
    app = Sphinx(
        srcdir=str(tmp_path),
        confdir=str(Path(__file__).resolve().parents[1] / "docs"),
        outdir=str(tmp_path / "html"),
        doctreedir=str(tmp_path / "doctrees"),
        buildername="html",
        confoverrides={
            "extensions": [],
            "html_theme": "basic",
            "html_theme_options": {},
            "html_static_path": [],
            "html_css_files": [],
            "html_js_files": [],
            "html_favicon": None,
        },
        status=StringIO(),
        warning=warnings,
    )
    app.build()
    assert warnings.getvalue() == ""
    html = (tmp_path / "html" / "index.html").read_text()
    assert html.count("<strong><em>0.500</em></strong>") == 6
    assert "*0.500*" not in html


@pytest.mark.parametrize("aupr, marked", [("0.0434", True), ("0.0435", True), ("0.0436", False)])
def test_write_ranking_compares_chance_at_stored_precision(tmp_path, monkeypatch, aupr, marked):
    """Stored AUPR values compare against the rounded anomaly fraction."""
    benchmark = load_benchmark()
    # The last 23 labels contain one anomaly. The first seven labels are warmup.
    y = np.zeros(30, dtype=int)
    y[-1] = 1
    monkeypatch.setattr(benchmark, "OUT", tmp_path)
    monkeypatch.setattr(benchmark, "load", lambda *args: (np.zeros((30, 1)), y))
    results = {
        ("glass", detector, 0): {
            "auroc": "0.5",
            "aupr": value,
            "ms_per_point": "1.0",
        }
        for detector, value in [("LODA", "0.8"), ("RSForest", aupr)]
    }
    benchmark.write_ranking(results, {"glass": "glass"}, "")
    with (tmp_path / "aupr.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["RSForest"] == ("*0.043*" if marked else "0.044")


def test_changed_benchmark_detectors_isolates_table_edits():
    """Editing a detector entry in benchmark.py only flags that detector, not the whole suite."""
    from pathlib import Path

    benchmark = load_benchmark()
    source = (Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark.py").read_text(
        encoding="utf-8"
    )

    assert benchmark._changed_benchmark_detectors(source, source) == set()

    added = source.replace(
        "DETECTOR_OVERRIDES = {}",
        'DETECTOR_OVERRIDES = {"NewModel": lambda X, seed: models.LODA(random_state=seed)}',
    )
    assert benchmark._changed_benchmark_detectors(source, added) == {"NewModel"}

    modified = source.replace(
        "batch_size=WARMUP_POINTS, random_state=0",
        "batch_size=128, random_state=0",
    )
    assert benchmark._changed_benchmark_detectors(source, modified) == {"SAND"}

    shared_change = source.replace("WARMUP_POINTS = 256", "WARMUP_POINTS = 128")
    assert benchmark._changed_benchmark_detectors(source, shared_change) is None


def test_benchmark_auto_discovers_detectors_and_sub_registries(monkeypatch):
    """Detectors added to pysad.models.__all__ populate DETECTORS/UNIVARIATE/SEASONAL/TIMESTAMPED."""
    from pysad import models
    from pysad.core import BaseModel

    class DummyMultivariate(BaseModel):
        def __init__(self, feature_mins, feature_maxes, random_state=None):
            self.feature_mins = feature_mins
            self.feature_maxes = feature_maxes
            self.random_state = random_state

        def fit_partial(self, X, y=None):
            return self

        def score_partial(self, X):
            return 0.0

    class DummySeasonal(BaseModel):
        _input_kind = "univariate"

        def __init__(self, period, window_size):
            self.period = period
            self.window_size = window_size

        def fit_partial(self, X, y=None):
            return self

        def score_partial(self, X):
            return 0.0

    class DummyTimestamped(BaseModel):
        _input_kind = "timestamped"

        def __init__(self, min_val, max_val, random_state=None):
            self.min_val = min_val
            self.max_val = max_val
            self.random_state = random_state

        def fit_partial(self, X, y=None):
            return self

        def score_partial(self, X):
            return 0.0

    monkeypatch.setattr(models, "DummyMultivariate", DummyMultivariate, raising=False)
    monkeypatch.setattr(models, "DummySeasonal", DummySeasonal, raising=False)
    monkeypatch.setattr(models, "DummyTimestamped", DummyTimestamped, raising=False)
    monkeypatch.setattr(
        models,
        "__all__",
        [*models.__all__, "DummyMultivariate", "DummySeasonal", "DummyTimestamped"],
    )

    benchmark = load_benchmark()
    assert "DummyMultivariate" in benchmark.DETECTORS
    mv = benchmark.DETECTORS["DummyMultivariate"](np.zeros((10, 4)), 42)
    assert mv.feature_mins.shape == (4,) and mv.random_state == 42

    assert "DummySeasonal" in benchmark.UNIVARIATE
    assert "DummySeasonal" in benchmark.SEASONAL
    seas = benchmark.UNIVARIATE["DummySeasonal"](1000, 24)
    assert seas.period == 24 and seas.window_size == 73

    assert "DummyTimestamped" in benchmark.UNIVARIATE
    assert "DummyTimestamped" in benchmark.TIMESTAMPED
    ts = benchmark.UNIVARIATE["DummyTimestamped"](1000, None)
    assert ts.min_val == 0.0 and ts.max_val == 1.0 and ts.random_state == 0


def test_pysad_files_excludes_unrelated_package_reexports():
    """Detector dependency sets do not pull in unrelated modules through package __init__.py files."""
    benchmark = load_benchmark()
    loda_files = benchmark.pysad_files([benchmark.models.LODA])
    assert "pysad/models/loda.py" in loda_files
    assert "pysad/core/base_model.py" in loda_files
    assert "pysad/utils/polars_streamer.py" not in loda_files
    assert "pysad/utils/pandas_streamer.py" not in loda_files
    assert "pysad/utils/synthetic.py" not in loda_files
    assert "pysad/utils/validation.py" not in loda_files


def test_runtime_toml_ignores_tool_and_dev_sections():
    """Tool and test configuration edits in pyproject.toml do not trigger a full benchmark run."""
    from pathlib import Path

    benchmark = load_benchmark()
    source = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")

    tool_edited = source.replace("disallow_untyped_defs = true", "disallow_untyped_defs = false")
    assert benchmark._runtime_toml(source) == benchmark._runtime_toml(tool_edited)

    runtime_edited = source.replace('"scipy>=1.13.0",', '"scipy>=1.14.0",')
    assert benchmark._runtime_toml(source) != benchmark._runtime_toml(runtime_edited)


def test_example_pysad_files_isolates_example_imports():
    """Example dependency tracing only includes the pysad modules imported by that example."""
    from pathlib import Path

    from tests.test_examples import _example_pysad_files

    code = (Path(__file__).resolve().parents[1] / "examples" / "example_usage_short.py").read_text(
        encoding="utf-8"
    )
    files = _example_pysad_files(code)
    assert "pysad/models/loda.py" in files
    assert "pysad/evaluation/metrics.py" in files
    assert "pysad/utils/data.py" in files
    assert "pysad/utils/_datasets.py" in files
    assert "pysad/models/rshash.py" not in files
    assert "pysad/models/sand.py" not in files


def test_union_merged_files_have_no_duplicates():
    """Files configured with merge=union keep unique entries and sorted bibliography keys."""
    import ast
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    for path in sorted((root / "pysad").rglob("__init__.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                exports = ast.literal_eval(node.value)
                assert len(exports) == len(set(exports)), f"Duplicate in {path}: {exports}"

    bib = (root / "docs" / "refs.bib").read_text(encoding="utf-8")
    keys = re.findall(r"^@\w+\{([^,\s]+),", bib, flags=re.M)
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys, key=str.lower)

    examples_rst = (root / "docs" / "examples.rst").read_text(encoding="utf-8")
    includes = re.findall(r"^\.\. literalinclude::\s+(\S+)", examples_rst, flags=re.M)
    assert len(includes) == len(set(includes))


def test_summary_has_no_mean_aupr(tmp_path, monkeypatch):
    benchmark = load_benchmark()
    monkeypatch.setattr(
        benchmark, "load", lambda *args: (np.zeros((10, 1)), np.zeros(10, dtype=int))
    )
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
    assert [row["Detector"] for row in rows] == [
        f":class:`~pysad.models.{d}`" for d in ["LODA", "RSHash", "xStream"]
    ]
    # The per-dataset AUPR table still exists, with one value per dataset and detector.
    with open(tmp_path / "aupr.csv", newline="") as f:
        aupr_rows = list(csv.DictReader(f))
    assert [row["Dataset"] for row in aupr_rows] == datasets
    assert float(aupr_rows[0]["xStream"].replace("*", "")) == 0.9  # the best cell is bolded
