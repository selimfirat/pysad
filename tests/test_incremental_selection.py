"""Tests for dependency-aware incremental test selection in tests/conftest.py."""

from __future__ import annotations

import tests.conftest as root_conftest
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
