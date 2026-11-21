"""Root pytest configuration for dependency-aware incremental test selection and sharding."""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import pytest
from hypothesis import settings

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ci_affected import (  # noqa: E402
    _GLOBAL_TRIGGERS,
    _SHARED_MODEL_FILES,
    _changed_files_since,
    _deps_changed_since,
    _file_dependencies,
    _mod_to_rel_file,
    matches_dependencies,
)

from tests.models.conftest import active_model_names  # noqa: E402

settings.register_profile("ci", database=None, derandomize=True, deadline=None)
if os.environ.get("CI") == "true":
    settings.load_profile("ci")

_CONTRACT_FILES = frozenset(
    {
        "tests/test_component_contracts.py",
        "tests/test_window_size_validation.py",
    }
)
_BOUNDED_CONTRACT_PREFIXES = (
    "pysad/statistics/",
    "pysad/transform/postprocessing/",
    "pysad/transform/probability_calibration/",
    "pysad/core/",
    "pysad/utils/",
)


def _is_test_file_selected(
    rel_path: str,
    changed: frozenset[str],
    active_models: tuple[str, ...],
) -> bool:
    if rel_path in changed:
        return True
    if rel_path in _SHARED_MODEL_FILES:
        if active_models:
            return True
        return matches_dependencies(rel_path, changed, ignore_model_prefix=True)
    return matches_dependencies(rel_path, changed)


def _contract_item_selected(item: pytest.Item, changed: frozenset[str]) -> bool:
    if (
        getattr(item, "originalname", item.name)
        == "test_bounded_components_do_not_grow_memory_after_warmup"
    ):
        return any(f.startswith(_BOUNDED_CONTRACT_PREFIXES) for f in changed)
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return True
    params = callspec.params
    target = params.get("cls")
    if target is None:
        target = params.get("component")
    if target is None:
        return True
    if isinstance(target, functools.partial):
        target = target.func
    rel_mod = _mod_to_rel_file(getattr(target, "__module__", ""))
    if rel_mod is None:
        return True
    exact_files, dir_prefixes = _file_dependencies(rel_mod)
    if exact_files & changed:
        return True
    return any(f.startswith(prefix) for prefix in dir_prefixes for f in changed)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip importing unaffected test modules when ``PYSAD_TESTS_SINCE`` is set."""
    base = os.environ.get("PYSAD_TESTS_SINCE")
    if not base or collection_path.is_dir() or not collection_path.name.startswith("test_"):
        return None
    if collection_path.suffix != ".py":
        return None
    try:
        rel_path = collection_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None
    changed = _changed_files_since(base)
    if (_GLOBAL_TRIGGERS & changed) or _deps_changed_since(base):
        return None
    if not _is_test_file_selected(rel_path, changed, active_model_names()):
        return True
    return None


def _parse_shard_spec(spec: str) -> tuple[int, int]:
    idx_s, total_s = spec.split("/", 1)
    shard_idx, shard_total = int(idx_s), int(total_s)
    if shard_total < 1 or not 0 <= shard_idx < shard_total:
        raise ValueError(f"Invalid PYSAD_TEST_SHARD={spec!r}")
    return shard_idx, shard_total


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Filter unaffected tests/contract cases and optionally shard items cleanly."""
    base = os.environ.get("PYSAD_TESTS_SINCE")
    deselected: list[pytest.Item] = []

    if base:
        changed = _changed_files_since(base)
        if not ((_GLOBAL_TRIGGERS & changed) or _deps_changed_since(base)):
            active_models = active_model_names()
            inits_changed = any(f.endswith("/__init__.py") for f in changed)
            file_cache: dict[str, bool] = {}
            kept: list[pytest.Item] = []
            for item in items:
                try:
                    rel_path = item.path.resolve().relative_to(ROOT).as_posix()
                except ValueError:
                    kept.append(item)
                    continue
                selected = file_cache.get(rel_path)
                if selected is None:
                    selected = _is_test_file_selected(rel_path, changed, active_models)
                    file_cache[rel_path] = selected
                if not selected:
                    deselected.append(item)
                    continue
                if (
                    rel_path in _CONTRACT_FILES
                    and rel_path not in changed
                    and not inits_changed
                    and not _contract_item_selected(item, changed)
                ):
                    deselected.append(item)
                    continue
                kept.append(item)
            items[:] = kept

    shard_spec = os.environ.get("PYSAD_TEST_SHARD", "").strip()
    if shard_spec:
        shard_idx, shard_total = _parse_shard_spec(shard_spec)
        sharded: list[pytest.Item] = []
        for idx, item in enumerate(items):
            if idx % shard_total == shard_idx:
                sharded.append(item)
            else:
                deselected.append(item)
        items[:] = sharded

    if deselected:
        config.hook.pytest_deselected(items=deselected)
