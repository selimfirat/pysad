"""Root pytest configuration for dependency-aware incremental test selection."""

from __future__ import annotations

import ast
import functools
import os
from pathlib import Path

import pytest

from tests.models.conftest import (
    _changed_files_since,
    _deps_changed_since,
    active_model_names,
)

ROOT = Path(__file__).resolve().parent.parent

_GLOBAL_TRIGGERS = frozenset(
    {
        ".github/workflows/coverage.yml",
        "tests/conftest.py",
        "tests/models/conftest.py",
    }
)

_SHARED_MODEL_FILES = frozenset(
    {
        "tests/models/test_discrimination.py",
        "tests/models/test_models.py",
        "tests/models/test_nan_input.py",
        "tests/models/test_pickling.py",
        "tests/models/test_properties.py",
        "tests/models/test_random_state.py",
        "tests/models/test_score_types.py",
    }
)

_LOCAL_PACKAGES = frozenset({"pysad", "tests", "scripts", "benchmarks", "spaces"})

_EXTRA_DEPENDENCIES: dict[str, tuple[set[str], set[str]]] = {
    "tests/test_benchmark.py": (
        {"benchmarks/benchmark.py"},
        {
            "docs/benchmark/",
            "pysad/models/",
            "pysad/transform/probability_calibration/",
            "pysad/utils/",
        },
    ),
    "tests/test_dimensionality.py": (
        {"benchmarks/benchmark.py"},
        {"pysad/models/"},
    ),
    "tests/test_public_contracts.py": (
        {"scripts/sync_exports.py", "benchmarks/benchmark.py"},
        {"docs/", "pysad/"},
    ),
    "tests/test_component_contracts.py": (set(), {"pysad/"}),
    "tests/test_window_size_validation.py": (set(), {"pysad/"}),
    "tests/test_core_base_classes.py": (set(), {"pysad/core/"}),
    "tests/test_examples.py": (
        {"docs/examples.rst", "docs/_autodiscover.py"},
        {"examples/"},
    ),
    "tests/test_playground_space.py": (
        set(),
        {
            "spaces/playground/",
            "pysad/models/",
            "pysad/transform/probability_calibration/",
            "pysad/evaluation/",
            "pysad/utils/",
        },
    ),
    "tests/test_context7_config.py": ({"context7.json"}, set()),
    "tests/test_comparison_chart.py": (
        set(),
        {"benchmarks/", "docs/benchmark/", "docs/images/", "scripts/"},
    ),
    "tests/test_performance_badges.py": (
        {"README.md"},
        {"docs/", "scripts/", "pysad/models/"},
    ),
    "tests/utils/test_data.py": ({"README.md"}, {"pysad/utils/"}),
    "tests/models/test_online_isolation_forest.py": (
        {"tests/models/data/online_isolation_forest_reference.json"},
        set(),
    ),
    "tests/transform/projection/test_sklearn_projectors.py": (set(), {"examples/data/"}),
}


@functools.cache
def _repo_py_files() -> frozenset[str]:
    """Case-sensitive set of tracked/local Python relative paths under the repository root."""
    files: set[str] = set()
    for top in _LOCAL_PACKAGES:
        top_dir = ROOT / top
        if top_dir.is_dir():
            for path in top_dir.rglob("*.py"):
                files.add(path.relative_to(ROOT).as_posix())
    return frozenset(files)


@functools.cache
def _parse_ast(rel_path: str) -> ast.Module | None:
    if rel_path not in _repo_py_files():
        return None
    return ast.parse((ROOT / rel_path).read_text(encoding="utf-8"))


def _resolve_relative_module(from_rel_path: str, module: str | None, level: int) -> str:
    if level == 0:
        return module or ""
    parts = list(Path(from_rel_path).parent.parts)
    up = level - 1
    if up > 0:
        parts = parts[:-up] if up < len(parts) else []
    if module:
        parts.extend(module.split("."))
    return ".".join(parts)


def _mod_to_rel_file(mod_name: str) -> str | None:
    if not mod_name:
        return None
    top = mod_name.split(".")[0]
    if top not in _LOCAL_PACKAGES:
        return None
    py_files = _repo_py_files()
    parts = mod_name.split(".")
    py_rel = "/".join(parts) + ".py"
    if py_rel in py_files:
        return py_rel
    init_rel = "/".join(parts) + "/__init__.py"
    if init_rel in py_files:
        return init_rel
    return None


@functools.cache
def _init_info(init_rel: str) -> tuple[dict[str, str], frozenset[str], bool]:
    """Return (symbol -> target_module_name, inline_defined_names, has_wildcard) for an __init__.py."""
    tree = _parse_ast(init_rel)
    if tree is None:
        return {}, frozenset(), False
    reexports: dict[str, str] = {}
    has_wildcard = False
    defs = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target_mod = _resolve_relative_module(init_rel, node.module, node.level)
            if not target_mod or target_mod.split(".")[0] not in _LOCAL_PACKAGES:
                continue
            for alias in node.names:
                if alias.name == "*":
                    has_wildcard = True
                else:
                    reexports[alias.asname or alias.name] = target_mod
    return reexports, frozenset(defs), has_wildcard


def _resolve_symbol_from_package(
    pkg_mod: str, init_rel: str, name: str, seen: set[str]
) -> tuple[str | None, bool]:
    """Resolve `name` imported from package `pkg_mod` (`init_rel`).

    Returns `(target_rel_file, needs_dir_prefix)`.
    """
    py_files = _repo_py_files()
    pkg_dir = pkg_mod.replace(".", "/")
    sub_py = f"{pkg_dir}/{name}.py"
    if sub_py in py_files:
        return sub_py, False
    sub_init = f"{pkg_dir}/{name}/__init__.py"
    if sub_init in py_files:
        return sub_init, True

    reexports, defs, has_wildcard = _init_info(init_rel)
    if name in reexports:
        target_mod = reexports[name]
        target_rel = _mod_to_rel_file(target_mod)
        if target_rel is None:
            return None, True
        if (
            target_rel.endswith("/__init__.py")
            and target_rel != init_rel
            and target_rel not in seen
        ):
            seen.add(target_rel)
            return _resolve_symbol_from_package(target_mod, target_rel, name, seen)
        return target_rel, False
    if name in defs:
        return None, False
    if has_wildcard:
        return None, True
    return None, True


@functools.cache
def _file_dependencies(rel_path: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return `(exact_files, dir_prefixes)` transitively depended upon by `rel_path`."""
    exact_files: set[str] = {rel_path}
    dir_prefixes: set[str] = set()
    extra_files, extra_prefixes = _EXTRA_DEPENDENCIES.get(rel_path, (set(), set()))
    exact_files.update(extra_files)
    dir_prefixes.update(extra_prefixes)

    visited: set[str] = set()
    queue: list[str] = [rel_path]
    while queue:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        tree = _parse_ast(cur)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    rel = _mod_to_rel_file(alias.name)
                    if rel is None:
                        continue
                    exact_files.add(rel)
                    if rel in _GLOBAL_TRIGGERS:
                        continue
                    if rel.endswith("/__init__.py"):
                        if "." in alias.name:
                            dir_prefixes.add(alias.name.replace(".", "/") + "/")
                    elif rel not in visited:
                        queue.append(rel)
            elif isinstance(node, ast.ImportFrom):
                mod_name = _resolve_relative_module(cur, node.module, node.level)
                if not mod_name or mod_name.split(".")[0] not in _LOCAL_PACKAGES:
                    continue
                rel = _mod_to_rel_file(mod_name)
                if rel is None:
                    continue
                exact_files.add(rel)
                if rel in _GLOBAL_TRIGGERS:
                    continue
                if not rel.endswith("/__init__.py"):
                    if rel not in visited:
                        queue.append(rel)
                    continue
                pkg_prefix = mod_name.replace(".", "/") + "/"
                for alias in node.names:
                    if alias.name == "*":
                        dir_prefixes.add(pkg_prefix)
                        continue
                    target_rel, needs_prefix = _resolve_symbol_from_package(
                        mod_name, rel, alias.name, {rel}
                    )
                    if needs_prefix:
                        dir_prefixes.add(
                            f"{pkg_prefix}{alias.name}/"
                            if target_rel and target_rel.endswith("/__init__.py")
                            else pkg_prefix
                        )
                    if target_rel is not None:
                        exact_files.add(target_rel)
                        if not target_rel.endswith("/__init__.py") and target_rel not in visited:
                            queue.append(target_rel)

    return frozenset(exact_files), frozenset(dir_prefixes)


def _matches_dependencies(
    rel_path: str,
    changed: frozenset[str],
    *,
    ignore_model_prefix: bool = False,
) -> bool:
    exact_files, dir_prefixes = _file_dependencies(rel_path)
    if exact_files & changed:
        return True
    for prefix in dir_prefixes:
        if ignore_model_prefix and prefix == "pysad/models/":
            continue
        if any(f.startswith(prefix) for f in changed):
            return True
    return False


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
        return _matches_dependencies(rel_path, changed, ignore_model_prefix=True)
    return _matches_dependencies(rel_path, changed)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect tests unaffected by changes since ``PYSAD_TESTS_SINCE``."""
    base = os.environ.get("PYSAD_TESTS_SINCE")
    if not base:
        return
    changed = _changed_files_since(base)
    if (_GLOBAL_TRIGGERS & changed) or _deps_changed_since(base):
        return

    active_models = active_model_names()
    file_cache: dict[str, bool] = {}
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

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
        if selected:
            kept.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
