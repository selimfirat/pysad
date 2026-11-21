"""Tests for dependency-aware incremental test selection and sharding."""

from __future__ import annotations

import functools
import types

import pytest

import scripts.ci_affected as ci_affected
import tests.conftest as root_conftest
from pysad.evaluation import BrierScoreMetric
from pysad.statistics import AverageMeter
from tests.models.conftest import _changed_files_since


def _selected(
    test_paths: list[str], changed: frozenset[str], active_models: tuple[str, ...]
) -> set[str]:
    return {
        path
        for path in test_paths
        if root_conftest._is_test_file_selected(path, changed, active_models)
    }


def test_single_detector_edit_isolates_unrelated_tests() -> None:
    changed = frozenset({"pysad/models/loda.py"})
    candidates = [
        "tests/models/test_loda.py",
        "tests/models/test_sand.py",
        "tests/models/test_properties.py",
        "tests/evaluation/test_metrics.py",
        "tests/stats/test_statistics.py",
        "tests/test_public_contracts.py",
        "tests/utils/test_pandas_streamer.py",
    ]
    selected = _selected(candidates, changed, ("LODA",))
    assert "tests/models/test_loda.py" in selected
    assert "tests/models/test_properties.py" in selected
    assert "tests/test_public_contracts.py" in selected
    assert "tests/models/test_sand.py" not in selected
    assert "tests/evaluation/test_metrics.py" not in selected
    assert "tests/stats/test_statistics.py" not in selected
    assert "tests/utils/test_pandas_streamer.py" not in selected


def test_lazy_integration_import_resolves_to_its_test() -> None:
    changed = frozenset({"pysad/integrations/pysad_to_river_adapter.py"})
    candidates = [
        "tests/models/test_pysad_to_river_adapter.py",
        "tests/models/test_pysad_to_sklearn_adapter.py",
        "tests/models/test_score_types.py",
        "tests/models/test_properties.py",
    ]
    selected = _selected(candidates, changed, ())
    assert selected == {"tests/models/test_pysad_to_river_adapter.py"}


def test_readme_edit_runs_only_readme_and_doc_dependent_tests() -> None:
    changed = frozenset({"README.md"})
    candidates = [
        "tests/utils/test_data.py",
        "tests/test_performance_badges.py",
        "tests/models/test_loda.py",
        "tests/models/test_properties.py",
        "tests/evaluation/test_metrics.py",
    ]
    selected = _selected(candidates, changed, ())
    assert selected == {"tests/utils/test_data.py", "tests/test_performance_badges.py"}


def test_editing_a_test_file_selects_that_test_file() -> None:
    changed = frozenset({"tests/evaluation/test_brier_score.py"})
    candidates = [
        "tests/evaluation/test_brier_score.py",
        "tests/evaluation/test_f1_metric.py",
        "tests/models/test_properties.py",
    ]
    selected = _selected(candidates, changed, ())
    assert selected == {"tests/evaluation/test_brier_score.py"}


def test_global_triggers_and_empty_diff_cache(monkeypatch) -> None:
    assert ".github/workflows/coverage.yml" in root_conftest._GLOBAL_TRIGGERS
    assert "tests/conftest.py" in root_conftest._GLOBAL_TRIGGERS
    assert "tests/models/conftest.py" in root_conftest._GLOBAL_TRIGGERS
    _changed_files_since.cache_clear()
    monkeypatch.setenv("PYSAD_TESTS_SINCE", "HEAD")
    assert _changed_files_since("HEAD") >= frozenset()


def test_ci_affected_plans_for_tests_examples_and_codeql(monkeypatch) -> None:
    def with_changed(files: set[str]):
        monkeypatch.setattr(ci_affected, "_changed_files_since", lambda base: frozenset(files))
        monkeypatch.setattr(ci_affected, "_deps_changed_since", lambda base: False)
        monkeypatch.setattr(ci_affected, "_runtime_toml_changed_since", lambda base: False)

    with_changed({"docs/index.rst"})
    assert set(ci_affected.tests_plan("origin/master")["files"]) == {
        "tests/test_performance_badges.py",
        "tests/test_public_contracts.py",
    }
    assert ci_affected.tests_plan("origin/master")["needs_compat"] is False
    assert ci_affected.examples_plan("origin/master")["count"] == 0
    assert ci_affected.codeql_plan("origin/master") == {
        "python": False,
        "actions": False,
        "languages": [],
    }

    with_changed({"LICENSE"})
    assert ci_affected.tests_plan("origin/master")["count"] == 0

    with_changed({"pysad/evaluation/metrics.py"})
    tplan = ci_affected.tests_plan("origin/master")
    assert tplan["all"] is False
    assert tplan["needs_compat"] is True
    assert "tests/evaluation/test_brier_score.py" in tplan["files"]
    assert "tests/test_component_contracts.py" in tplan["files"]
    assert ci_affected.codeql_plan("origin/master")["languages"] == ["python"]

    with_changed({".github/workflows/benchmark.yml"})
    assert ci_affected.codeql_plan("origin/master")["languages"] == ["actions"]
    assert ci_affected.tests_plan("origin/master")["needs_compat"] is False


def test_pytest_ignore_collect_and_contract_filtering(monkeypatch) -> None:
    monkeypatch.setenv("PYSAD_TESTS_SINCE", "origin/master")
    monkeypatch.setattr(
        root_conftest, "_changed_files_since", lambda base: frozenset({"README.md"})
    )
    monkeypatch.setattr(root_conftest, "_deps_changed_since", lambda base: False)
    monkeypatch.setattr(root_conftest, "active_model_names", lambda: ())

    assert (
        root_conftest.pytest_ignore_collect(
            root_conftest.ROOT / "tests" / "models" / "test_loda.py", config=object()
        )
        is True
    )
    assert (
        root_conftest.pytest_ignore_collect(
            root_conftest.ROOT / "tests" / "utils" / "test_data.py", config=object()
        )
        is None
    )

    brier_item = types.SimpleNamespace(
        originalname="test_metrics_update_batch_matches_sequential_and_pickles",
        name="test_metrics_update_batch_matches_sequential_and_pickles[BrierScoreMetric]",
        callspec=types.SimpleNamespace(params={"cls": BrierScoreMetric}),
    )
    avg_item = types.SimpleNamespace(
        originalname="test_nonpositive_window_size_is_rejected",
        name="test_nonpositive_window_size_is_rejected[AverageMeter-0]",
        callspec=types.SimpleNamespace(params={"component": functools.partial(AverageMeter)}),
    )
    changed_brier = frozenset({"pysad/evaluation/metrics.py"})
    assert root_conftest._contract_item_selected(brier_item, changed_brier) is True
    assert root_conftest._contract_item_selected(avg_item, changed_brier) is False


def test_collection_sharding_partitions_all_items(monkeypatch) -> None:
    monkeypatch.delenv("PYSAD_TESTS_SINCE", raising=False)
    items = [
        types.SimpleNamespace(
            path=root_conftest.ROOT / "tests" / "evaluation" / f"test_fake_{i}.py",
            name=f"test_{i}",
        )
        for i in range(5)
    ]
    deselected_records: list[list[object]] = []
    config = types.SimpleNamespace(
        hook=types.SimpleNamespace(
            pytest_deselected=lambda items: deselected_records.append(list(items))
        )
    )

    monkeypatch.setenv("PYSAD_TEST_SHARD", "0/2")
    shard0 = list(items)
    root_conftest.pytest_collection_modifyitems(config, shard0)
    monkeypatch.setenv("PYSAD_TEST_SHARD", "1/2")
    shard1 = list(items)
    root_conftest.pytest_collection_modifyitems(config, shard1)

    names0 = [x.name for x in shard0]
    names1 = [x.name for x in shard1]
    assert set(names0) & set(names1) == set()
    assert sorted(names0 + names1) == sorted(x.name for x in items)

    with pytest.raises(ValueError, match="PYSAD_TEST_SHARD"):
        root_conftest._parse_shard_spec("2/2")
