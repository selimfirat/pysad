"""The metrics of benchmarks/benchmark.py."""

import csv

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from pysad.core.base_model import BaseModel
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


@pytest.mark.parametrize("detector", ["HoltWinters", "FluxEV"])
@pytest.mark.parametrize("period", [24, 288])
def test_seasonal_benchmark_keeps_default_window(detector, period):
    from pysad import models

    benchmark = load_benchmark()
    X = np.sin(np.arange(4 * period) * 2 * np.pi / period).reshape(-1, 1)
    X[3 * period : 3 * period + 10] += 4.0
    model = benchmark.UNIVARIATE[detector](len(X), period)
    expected = getattr(models, detector)(period=period)
    assert model.window_size == expected.window_size
    np.testing.assert_array_equal(model.fit_score(X), expected.fit_score(X))


@pytest.mark.parametrize("detector", ["SeasonalESD", "SeasonalHybridESD"])
@pytest.mark.parametrize("period", [24, 288])
def test_seasonal_esd_benchmark_window_covers_three_periods(detector, period):
    benchmark = load_benchmark()
    model = benchmark.UNIVARIATE[detector](2000, period)
    assert model.period == period
    assert model.window_size == 3 * period + 1


def test_sdostream_registration_and_seeding():
    from pysad.models import SDOstream

    benchmark = load_benchmark()
    assert "SDOstream" in benchmark.DETECTORS
    assert "SDOstream" not in benchmark.UNIVARIATE
    X = np.random.default_rng(0).random((200, 3))
    model = benchmark.DETECTORS["SDOstream"](X, 7)
    assert isinstance(model, SDOstream)
    assert model.random_state == 7
    scores = benchmark.DETECTORS["SDOstream"](X, 7).fit_score(X)
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, benchmark.DETECTORS["SDOstream"](X, 7).fit_score(X))


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
    assert [row["MCC"] for row in table] == ["0.289", "0.000", "-0.289", "0.289"] * 3


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
    import re
    from pathlib import Path

    benchmark = load_benchmark()
    source = (Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark.py").read_text(
        encoding="utf-8"
    )

    assert benchmark._changed_benchmark_detectors(source, source) == set()

    added, count = re.subn(
        r"DETECTOR_OVERRIDES\s*=\s*\{",
        'DETECTOR_OVERRIDES = {\n    "NewModel": lambda X, seed: models.LODA(random_state=seed),',
        source,
        count=1,
    )
    assert count == 1
    assert benchmark._changed_benchmark_detectors(source, added) == {"NewModel"}

    synthetic_base = (
        "WARMUP_POINTS = 256\n"
        'DETECTOR_OVERRIDES = {"Demo": lambda X, seed: models.LODA(random_state=seed)}\n'
        'UNIVARIATE_OVERRIDES = {"Uni": lambda n, period: models.SAND(batch_size=256)}\n'
    )
    synthetic_mod = synthetic_base.replace("batch_size=256", "batch_size=128")
    assert benchmark._changed_benchmark_detectors(synthetic_base, synthetic_mod) == {"Uni"}

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
        # LODA and RSHash have the same AUROC on both datasets, so they have the same mean rank and
        # the same mean AUROC. RSHash has the higher AUPR, but LODA is faster, so LODA comes first.
        for detector, auroc, aupr, ms in [
            ("LODA", 0.9, 0.4, 1.0),
            ("RSHash", 0.9, 0.9, 2.0),
            ("xStream", 0.5, 0.2, 1.5),
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
    # LODA and RSHash tie on the mean rank and the mean AUROC, so the faster one comes first.
    assert rows[0]["Mean rank"] == rows[1]["Mean rank"] == "1.5"
    assert rows[0]["Mean AUROC"] == rows[1]["Mean AUROC"] == "0.900"
    assert rows[0]["Time per point (ms)"] == "1.00"
    assert rows[1]["Time per point (ms)"] == "2.00"
    # The per-dataset AUPR table still exists, with one value per dataset and detector.
    with open(tmp_path / "aupr.csv", newline="") as f:
        aupr_rows = list(csv.DictReader(f))
    assert [row["Dataset"] for row in aupr_rows] == datasets
    assert float(aupr_rows[0]["RSHash"].replace("*", "")) == 0.9  # the best cell is bolded


def test_ci_mode_skips_calibrators_and_changed_detectors_isolates_calibrator_edits(monkeypatch):
    benchmark = load_benchmark()
    monkeypatch.setattr(
        benchmark, "git", lambda *args: "pysad/transform/probability_calibration/gaussian_tail.py\n"
    )
    assert benchmark.changed_detectors("origin/master", ci=True) == []
    assert benchmark.changed_detectors("origin/master", ci=False) == [
        *benchmark.DETECTORS,
        *benchmark.UNIVARIATE,
    ]

    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 2))
    y = np.zeros(300, dtype=int)
    y[290:] = 1
    monkeypatch.setattr(benchmark, "load", lambda *args: (X.copy(), y.copy()))
    row = benchmark.run(("synthetic", "LODA", 0, 300, True))
    assert set(row) == {"dataset", "detector", "seed", "auroc", "aupr", "ms_per_point"}

    monkeypatch.setattr(benchmark.Data, "get_data", lambda self, name: (X[:, :1].copy(), y.copy()))
    uni_row = benchmark.run(("nab/synthetic.csv", "StandardAbsoluteDeviation", 0, 300, True))
    assert "_nab" not in uni_row
    assert set(uni_row) == {"dataset", "detector", "seed", "auroc", "aupr", "ms_per_point"}


def test_ci_plan_shards_and_ignore_unknown(monkeypatch):
    import json
    import sys

    benchmark = load_benchmark()
    monkeypatch.setattr(
        benchmark,
        "changed_detectors",
        lambda base, path=None, ci=False: ["LODA", "xStream", "StandardAbsoluteDeviation"],
    )
    monkeypatch.setattr(
        benchmark,
        "git",
        lambda *args: (
            "100644 blob aaa\tpysad/models/loda.py\n100644 blob bbb\tpysad/models/xstream.py\n"
        ),
    )
    plan = benchmark.ci_plan("origin/master")
    assert plan["has_jobs"] is True
    assert [(row["part"], row["shard"], row["has_base"]) for row in plan["include"]] == [
        ("odds", "LODA,xStream", True),
        ("drift", "LODA,xStream", True),
        ("univariate", "StandardAbsoluteDeviation", False),
    ]
    assert all(len(row["base_hash"]) == 16 for row in plan["include"])

    monkeypatch.setattr(benchmark, "changed_detectors", lambda base, path=None, ci=False: [])
    empty_plan = benchmark.ci_plan("origin/master")
    assert empty_plan == {"has_jobs": False, "include": []}

    shards = benchmark._pack_shards(
        ["A", "B", "C", "D"], 2, {"A": 10.0, "B": 9.0, "C": 2.0, "D": 1.0}
    )
    assert len(shards) == 2
    assert {d for s in shards for d in s} == {"A", "B", "C", "D"}

    # --ignore-unknown silently filters unknown detectors without error.
    ran = []
    monkeypatch.setattr(
        benchmark,
        "run",
        lambda job: (
            ran.append(job)
            or {"dataset": "d", "detector": job[1], "seed": 0, "auroc": 0.5, "aupr": 0.5}
        ),
    )
    monkeypatch.setattr(benchmark.Data, "get_data", lambda self, name: None)
    out_csv = str(__import__("pathlib").Path(__file__).resolve().parent / "__tmp_ignored.csv")
    try:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "benchmark.py",
                "--ci",
                out_csv,
                "--parts",
                "odds",
                "--detectors",
                "NonExistentDetector",
                "--ignore-unknown",
            ],
        )
        benchmark.main()
        assert ran == []
    finally:
        __import__("pathlib").Path(out_csv).unlink(missing_ok=True)
        __import__("sys")
    assert isinstance(json.dumps(plan), str)


def test_ci_plan_fallback_weights_and_det_hashes(monkeypatch):
    import json

    benchmark = load_benchmark()
    _, pw = benchmark._detector_weights()
    assert pw["odds"]["PCBIsolationForest"] > pw["odds"].get("LODA", 0.0)
    shards = benchmark._pack_shards(
        ["PCBIsolationForest", "OnlineIsolationForest", "RobustRandomCutForest", "LODA"],
        3,
        pw["odds"],
    )
    heavy_buckets = [
        i
        for i, s in enumerate(shards)
        for name in s
        if name in {"PCBIsolationForest", "OnlineIsolationForest", "RobustRandomCutForest"}
    ]
    assert len(set(heavy_buckets)) == 3

    monkeypatch.setattr(
        benchmark,
        "git",
        lambda *args: "100644 blob aaa\tpysad/models/loda.py\n",
    )
    plan = benchmark.ci_plan("HEAD", all_detectors=True)
    assert plan["has_jobs"] is True
    for row in plan["include"]:
        det_hashes = json.loads(row["det_hashes"])
        assert sorted(det_hashes) == sorted(row["shard"].split(","))


def test_base_cache_dir_and_compare_dirs(tmp_path, monkeypatch, capsys):
    import json
    import sys

    benchmark = load_benchmark()
    cache_dir = tmp_path / "cache"
    out1 = tmp_path / "run1.csv"
    out2 = tmp_path / "run2.csv"
    calls = []

    def fake_run(job):
        ds, det, seed, points, ci = job
        calls.append(det)
        return {
            "dataset": ds,
            "detector": det,
            "seed": seed,
            "auroc": 0.9 if det == "LODA" else 0.8,
            "aupr": 0.7,
        }

    class DummyFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, arg):
            return DummyFuture(fn(arg))

    monkeypatch.setattr(benchmark, "run", fake_run)
    monkeypatch.setattr(benchmark, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(benchmark, "as_completed", list)
    monkeypatch.setattr(benchmark, "DRIFT", {"kddcup99/http.csv": "KDD http"})
    monkeypatch.setattr(benchmark.Data, "get_data", lambda self, name: None)

    hashes1 = json.dumps({"LODA": "aaaa1111aaaa1111", "xStream": "bbbb2222bbbb2222"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--ci",
            str(out1),
            "--parts",
            "drift",
            "--detectors",
            "LODA,xStream",
            "--base-cache-dir",
            str(cache_dir),
            "--det-hashes",
            hashes1,
            "--no-cache",
        ],
    )
    benchmark.main()
    assert sorted(calls) == ["LODA", "xStream"]
    assert (cache_dir / "LODA-aaaa1111aaaa1111.csv").is_file()
    assert (cache_dir / "xStream-bbbb2222bbbb2222.csv").is_file()

    # Re-run when only xStream hash changed: LODA is served from cache, xStream recomputes.
    calls.clear()
    hashes2 = json.dumps({"LODA": "aaaa1111aaaa1111", "xStream": "cccc3333cccc3333"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--ci",
            str(out2),
            "--parts",
            "drift",
            "--detectors",
            "LODA,xStream",
            "--base-cache-dir",
            str(cache_dir),
            "--det-hashes",
            hashes2,
            "--no-cache",
        ],
    )
    benchmark.main()
    assert calls == ["xStream"]
    assert set(benchmark.read_results(out2)) == {
        ("kddcup99/http.csv", "LODA", 0),
        ("kddcup99/http.csv", "xStream", 0),
    }

    base_dir = tmp_path / "base_artifacts"
    head_dir = tmp_path / "head_artifacts"
    (base_dir / "s0").mkdir(parents=True)
    (head_dir / "s0").mkdir(parents=True)
    (base_dir / "s0" / "base.csv").write_text(out1.read_text(encoding="utf-8"), encoding="utf-8")
    (head_dir / "s0" / "head.csv").write_text(out2.read_text(encoding="utf-8"), encoding="utf-8")
    benchmark.compare_ci_dirs(base_dir, head_dir)
    assert "No run moved, out of 2." in capsys.readouterr().out

    # Perturb head AUROC and verify drift detection via --compare-dirs CLI.
    (head_dir / "s0" / "head.csv").write_text(
        out2.read_text(encoding="utf-8").replace(",0.9,", ",0.5,"), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys, "argv", ["benchmark.py", "--compare-dirs", str(base_dir), str(head_dir)]
    )
    benchmark.main()
    out = capsys.readouterr().out
    assert "1 of 2 runs moved." in out and "LODA" in out

    # Empty base directory leaves comparison graceful.
    empty_base = tmp_path / "empty_base"
    empty_base.mkdir()
    benchmark.compare_ci_dirs(empty_base, head_dir)
    assert "No base benchmark results to compare" in capsys.readouterr().out


class NotAModel:
    """A class that does not inherit from BaseModel."""

    def __init__(self, random_state=None):
        self.random_state = random_state


def _write_custom_model(tmp_path, class_name, source):
    path = tmp_path / "custom_model.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_load_custom_model_loads_valid_model(tmp_path):
    benchmark = load_benchmark()
    source = (
        "import numpy as np\n"
        "from pysad.core.base_model import BaseModel\n\n"
        "class CustomMultivariateDetector(BaseModel):\n"
        "    def __init__(self, window_size=50, random_state=None):\n"
        "        self.window_size = window_size\n"
        "        self.random_state = random_state\n"
        "    def fit_partial(self, X, y=None):\n"
        "        return self\n"
        "    def score_partial(self, X):\n"
        "        return 0.0\n"
    )
    path = _write_custom_model(tmp_path, "CustomMultivariateDetector", source)
    cls = benchmark._load_custom_model(f"{path}:CustomMultivariateDetector")
    assert cls is not None
    assert cls.__name__ == "CustomMultivariateDetector"
    assert issubclass(cls, benchmark.BaseModel)


@pytest.mark.parametrize("spec", ["no_colon", "just_a_path.py", "a:b:c"])
def test_load_custom_model_rejects_bad_spec(spec):
    benchmark = load_benchmark()
    try:
        benchmark._load_custom_model(spec)
    except ValueError:
        pass
    else:
        raise AssertionError(f"expected ValueError for spec {spec!r}")


def test_load_custom_model_rejects_missing_file(tmp_path):
    benchmark = load_benchmark()
    try:
        benchmark._load_custom_model(str(tmp_path / "does_not_exist.py:MyModel"))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing file")


def test_load_custom_model_rejects_missing_class(tmp_path):
    benchmark = load_benchmark()
    path = _write_custom_model(
        tmp_path,
        "OtherClass",
        "from pysad.core.base_model import BaseModel\nclass OtherClass(BaseModel):\n    def fit_partial(self, X, y=None): return self\n    def score_partial(self, X): return 0.0\n",
    )
    try:
        benchmark._load_custom_model(f"{path}:NotInFile")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing class")


def test_load_custom_model_rejects_non_model(tmp_path):
    benchmark = load_benchmark()
    path = _write_custom_model(
        tmp_path, "NotAModel", "class NotAModel:\n    def __init__(self): pass\n"
    )
    try:
        benchmark._load_custom_model(f"{path}:NotAModel")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-BaseModel class")


def test_custom_factory_multivariate(tmp_path):
    benchmark = load_benchmark()
    path = _write_custom_model(
        tmp_path,
        "CustomMultivariateDetector",
        "import numpy as np\n"
        "from pysad.core.base_model import BaseModel\n\n"
        "class CustomMultivariateDetector(BaseModel):\n"
        "    def __init__(self, window_size=50, random_state=None):\n"
        "        self.window_size = window_size\n"
        "        self.random_state = random_state\n"
        "    def fit_partial(self, X, y=None):\n"
        "        return self\n"
        "    def score_partial(self, X):\n"
        "        return 0.0\n",
    )
    cls = benchmark._load_custom_model(f"{path}:CustomMultivariateDetector")
    factory = benchmark._make_custom_factory(cls, "multivariate")
    model = factory(np.zeros((20, 3)), 7)
    assert isinstance(model, BaseModel)
    assert model.window_size == 50
    assert model.random_state == 7


def test_custom_factory_univariate(tmp_path):
    benchmark = load_benchmark()
    path = _write_custom_model(
        tmp_path,
        "CustomUnivariateDetector",
        "import numpy as np\n"
        "from pysad.core.base_model import BaseModel\n\n"
        "class CustomUnivariateDetector(BaseModel):\n"
        "    _input_kind = 'univariate'\n"
        "    def __init__(self, window_size=50, random_state=None):\n"
        "        self.window_size = window_size\n"
        "        self.random_state = random_state\n"
        "    def fit_partial(self, X, y=None):\n"
        "        return self\n"
        "    def score_partial(self, X):\n"
        "        return 0.0\n",
    )
    cls = benchmark._load_custom_model(f"{path}:CustomUnivariateDetector")
    factory = benchmark._make_custom_factory(cls, "univariate")
    model = factory(200, 24)
    assert isinstance(model, BaseModel)
    assert model.window_size == 50
    assert model.random_state == 0


def test_custom_factory_uses_defaults_for_optional_params(tmp_path):
    benchmark = load_benchmark()
    path = _write_custom_model(
        tmp_path,
        "CustomMultivariateDetector",
        "import numpy as np\n"
        "from pysad.core.base_model import BaseModel\n\n"
        "class CustomMultivariateDetector(BaseModel):\n"
        "    def __init__(self, window_size=50, random_state=None):\n"
        "        self.window_size = window_size\n"
        "        self.random_state = random_state\n"
        "    def fit_partial(self, X, y=None):\n"
        "        return self\n"
        "    def score_partial(self, X):\n"
        "        return 0.0\n",
    )
    cls = benchmark._load_custom_model(f"{path}:CustomMultivariateDetector")
    factory = benchmark._make_custom_factory(cls, "multivariate")
    model = factory(np.zeros((20, 3)), 7)
    assert model.window_size == 50


def test_write_latex_summary_ranks_and_marks(tmp_path):
    benchmark = load_benchmark()
    results = {
        ("arrhythmia", "LODA", 0): {
            "dataset": "arrhythmia",
            "detector": "LODA",
            "seed": 0,
            "auroc": 0.7722,
        },
        ("arrhythmia", "RSHash", 0): {
            "dataset": "arrhythmia",
            "detector": "RSHash",
            "seed": 0,
            "auroc": 0.7366,
        },
        ("cardio", "LODA", 0): {
            "dataset": "cardio",
            "detector": "LODA",
            "seed": 0,
            "auroc": 0.93,
        },
        ("cardio", "RSHash", 0): {
            "dataset": "cardio",
            "detector": "RSHash",
            "seed": 0,
            "auroc": 0.9261,
        },
    }
    out_path = tmp_path / "summary.tex"
    benchmark._write_latex_summary(results, ["LODA", "RSHash"], out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "\\begin{table}[htbp]" in text
    assert "\\caption{Benchmark AUROC summary (higher is better).}" in text
    assert "arrhythmia" in text and "cardio" in text
    assert "LODA" in text and "RSHash" in text
    assert "\\textbf{0.772}" in text
    assert "\\begin{tabular}" in text and "\\end{tabular}" in text


def test_write_latex_summary_handles_nan_and_chance(tmp_path):
    benchmark = load_benchmark()
    results = {
        ("arrhythmia", "LODA", 0): {
            "dataset": "arrhythmia",
            "detector": "LODA",
            "seed": 0,
            "auroc": 0.5,
        },
        ("arrhythmia", "RSHash", 0): {
            "dataset": "arrhythmia",
            "detector": "RSHash",
            "seed": 0,
            "auroc": 0.8,
        },
        ("cardio", "LODA", 0): {
            "dataset": "cardio",
            "detector": "LODA",
            "seed": 0,
            "auroc": float("nan"),
        },
        ("cardio", "RSHash", 0): {
            "dataset": "cardio",
            "detector": "RSHash",
            "seed": 0,
            "auroc": 0.8,
        },
    }
    out_path = tmp_path / "summary.tex"
    benchmark._write_latex_summary(results, ["LODA", "RSHash"], out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "\\textit{0.500}" in text
    assert "--" in text


def test_custom_model_runs_on_stream(tmp_path, monkeypatch):
    benchmark = load_benchmark()
    path = _write_custom_model(
        tmp_path,
        "CustomMultivariateDetector",
        "import numpy as np\n"
        "from pysad.core.base_model import BaseModel\n\n"
        "class CustomMultivariateDetector(BaseModel):\n"
        "    def __init__(self, window_size=50, random_state=None):\n"
        "        self.window_size = window_size\n"
        "        self.random_state = random_state\n"
        "        self._seen = 0\n"
        "    def fit_partial(self, X, y=None):\n"
        "        self._seen += 1\n"
        "        return self\n"
        "    def score_partial(self, X):\n"
        "        return float(self._seen)\n",
    )
    cls = benchmark._load_custom_model(f"{path}:CustomMultivariateDetector")
    factory = benchmark._make_custom_factory(cls, "multivariate")
    X = np.zeros((50, 2))
    model = factory(X, 0)
    scores, ms = benchmark.score_stream(model, X)
    assert len(scores) == 50
    assert ms > 0
    assert float(scores[-1]) == 49.0


@pytest.fixture
def pair_benchmark(tmp_path, monkeypatch):
    import sys

    benchmark = load_benchmark()
    monkeypatch.setattr(benchmark, "OUT", tmp_path / "output")
    monkeypatch.setattr(benchmark, "DATASETS", ["synthetic"])
    monkeypatch.setattr(benchmark, "DRIFT", {"drift/test.csv": "Synthetic"})
    monkeypatch.setattr(benchmark, "nab_series", lambda: ["nab/synthetic.csv"])
    X = np.arange(600, dtype=float).reshape(-1, 1) % 7
    y = np.tile([0, 0, 1], 200)
    monkeypatch.setattr(benchmark.Data, "get_data", lambda *args, **kwargs: (X.copy(), y.copy()))
    monkeypatch.setattr(
        benchmark, "load", lambda ds, seed, points=None: (X[:points].copy(), y[:points].copy())
    )
    monkeypatch.setattr(benchmark, "commit", lambda: "abc123")
    monkeypatch.setattr(benchmark, "warmup", lambda n: 5)
    monkeypatch.setattr(benchmark, "series_of", lambda d: benchmark.nab_series())
    # Run the real scoring code in this process, so each run uses the synthetic data.
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(benchmark, "ProcessPoolExecutor", ThreadPoolExecutor)

    def invoke(*extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "benchmark.py",
                "--detectors",
                "LODA",
                "--jobs",
                "1",
                "--cache-dir",
                str(tmp_path / "cache"),
                *extra,
            ],
        )
        benchmark.main()

    return benchmark, invoke


def test_pair_cache_reuses_ci_results_and_separates_settings(pair_benchmark, tmp_path, monkeypatch):
    benchmark, invoke = pair_benchmark
    output = tmp_path / "ci.csv"
    invoke("--ci", str(output), "--parts", "drift")
    expected = output.read_text()
    original_run = benchmark.run

    def fail_run(job):
        pytest.fail("An unchanged pair ran again.")

    monkeypatch.setattr(benchmark, "run", fail_run)
    invoke("--ci", str(output), "--parts", "drift")
    assert output.read_text() == expected
    monkeypatch.setattr(benchmark, "run", original_run)
    monkeypatch.setattr(benchmark, "CI_POINTS", 30)
    invoke("--ci", str(output), "--parts", "drift")
    assert output.read_text() != expected
    invoke("--ci", str(output), "--parts", "drift", "--no-cache")


def test_pair_cache_restores_full_rows_and_nab_scores(pair_benchmark, monkeypatch):
    benchmark, invoke = pair_benchmark
    invoke("--parts", "odds,nab,shingles")
    expected = {p.name: p.read_bytes() for p in benchmark.OUT.glob("*.csv")}
    assert {
        r["detector"] for r in benchmark.read_results(benchmark.OUT / "results.csv").values()
    } == {
        "LODA",
        "LODA" + benchmark.SHINGLED,
    }
    for p in benchmark.OUT.glob("*.csv"):
        p.unlink()

    def fail_run(job):
        pytest.fail("An unchanged pair ran again.")

    monkeypatch.setattr(benchmark, "run", fail_run)
    invoke("--parts", "odds,nab,shingles")
    assert {p.name: p.read_bytes() for p in benchmark.OUT.glob("*.csv")} == expected


def test_pair_cache_invalidates_only_changed_detector(pair_benchmark, tmp_path, monkeypatch):
    benchmark, invoke = pair_benchmark
    package = tmp_path / "package" / "pysad"
    package.mkdir(parents=True)
    for name in ["common", "LODA", "xStream"]:
        (package / f"{name}.py").write_text("original")
    monkeypatch.setattr(benchmark.pysad, "__file__", str(package / "__init__.py"))

    def dependencies(objects):
        if len(objects) == 1 and objects[0].__name__ in {"LODA", "xStream"}:
            return {f"pysad/{objects[0].__name__}.py"}
        return {"pysad/common.py"}

    monkeypatch.setattr(benchmark, "pysad_files", dependencies)
    output = tmp_path / "ci.csv"
    options = ["--ci", str(output), "--parts", "drift", "--detectors", "LODA,xStream"]
    invoke(*options)
    before = benchmark.read_results(output)
    # Change the stream too, so reused metrics differ from metrics that run again.
    X = np.arange(600, dtype=float).reshape(-1, 1) % 7
    y = (X[:, 0] == 6).astype(int)
    monkeypatch.setattr(benchmark, "load", lambda *args: (X.copy(), y.copy()))
    (package / "xStream.py").write_text("changed")
    invoke(*options)
    after = benchmark.read_results(output)
    assert after[("drift/test.csv", "LODA", 0)] == before[("drift/test.csv", "LODA", 0)]
    assert after[("drift/test.csv", "xStream", 0)] != before[("drift/test.csv", "xStream", 0)]
    (package / "common.py").write_text("changed")
    invoke(*options)
    assert (
        benchmark.read_results(output)[("drift/test.csv", "LODA", 0)]
        != before[("drift/test.csv", "LODA", 0)]
    )


def test_pair_cache_recovers_damaged_entry(pair_benchmark, tmp_path):
    benchmark, invoke = pair_benchmark
    output = tmp_path / "ci.csv"
    options = ["--ci", str(output), "--parts", "drift"]
    invoke(*options)
    expected = output.read_text()
    paths = list((tmp_path / "cache").glob("*.json"))
    assert len(paths) == 1
    for path in paths:
        path.write_text("{broken")
    invoke(*options)
    assert output.read_text() == expected
    assert benchmark.read_results(output)


def test_no_cache_bypasses_reads_and_writes(pair_benchmark, tmp_path, monkeypatch):
    benchmark, invoke = pair_benchmark
    output = tmp_path / "ci.csv"
    options = ["--ci", str(output), "--parts", "drift"]
    invoke(*options)
    expected = output.read_text()
    X = np.arange(600, dtype=float).reshape(-1, 1) % 7
    y = (X[:, 0] == 6).astype(int)
    monkeypatch.setattr(benchmark, "load", lambda *args: (X.copy(), y.copy()))
    invoke(*options, "--no-cache")
    assert output.read_text() != expected
    invoke(*options)
    assert output.read_text() == expected
