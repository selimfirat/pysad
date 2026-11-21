"""Pure-stdlib change impact analyzer for incremental pytest, examples, and CodeQL selection."""

from __future__ import annotations

import argparse
import ast
import functools
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_GLOBAL_TRIGGERS = frozenset(
    {
        ".github/workflows/coverage.yml",
        "scripts/ci_affected.py",
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

_SHARED_MODEL_CONFTEST_TRIGGERS = frozenset(
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
def _changed_files_since(base: str) -> frozenset[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", base],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    return frozenset(out.split())


@functools.cache
def _deps_changed_since(base: str) -> bool:
    if "pyproject.toml" not in _changed_files_since(base):
        return False
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{base}:pyproject.toml"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return True
    after = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return proc.stdout.split("\ndocs = [", 1)[0] != after.split("\ndocs = [", 1)[0]


def _runtime_toml_changed_since(base: str) -> bool:
    if "pyproject.toml" not in _changed_files_since(base):
        return False
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{base}:pyproject.toml"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return True
    after = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return proc.stdout.split("\ntest = [", 1)[0] != after.split("\ntest = [", 1)[0]


@functools.cache
def _repo_py_files() -> frozenset[str]:
    """Case-sensitive set of local Python relative paths under the repository root."""
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
    """Return `(symbol -> target_module_name, inline_defined_names, has_wildcard)` for an `__init__.py`."""
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


def source_dependencies(source: str, rel_path: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return `(exact_files, dir_prefixes)` transitively imported from `source` at `rel_path`."""
    exact_files: set[str] = {rel_path}
    dir_prefixes: set[str] = set()
    try:
        root_tree: ast.Module | None = ast.parse(source)
    except SyntaxError:
        return frozenset(exact_files), frozenset(dir_prefixes)

    visited: set[str] = set()
    queue: list[tuple[str, ast.Module | None]] = [(rel_path, root_tree)]
    while queue:
        cur, pre_parsed = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        tree = pre_parsed if pre_parsed is not None else _parse_ast(cur)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pysad" or alias.name.startswith("pysad."):
                        exact_files.add("pysad/__init__.py")
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
                        queue.append((rel, None))
            elif isinstance(node, ast.ImportFrom):
                mod_name = _resolve_relative_module(cur, node.module, node.level)
                if not mod_name or mod_name.split(".")[0] not in _LOCAL_PACKAGES:
                    continue
                if mod_name == "pysad" or mod_name.startswith("pysad."):
                    exact_files.add("pysad/__init__.py")
                rel = _mod_to_rel_file(mod_name)
                if rel is None:
                    continue
                exact_files.add(rel)
                if rel in _GLOBAL_TRIGGERS:
                    continue
                if not rel.endswith("/__init__.py"):
                    if rel not in visited:
                        queue.append((rel, None))
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
                            queue.append((target_rel, None))

    return frozenset(exact_files), frozenset(dir_prefixes)


@functools.cache
def _file_dependencies(rel_path: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return `(exact_files, dir_prefixes)` transitively depended upon by `rel_path`."""
    exact_files: set[str] = {rel_path}
    dir_prefixes: set[str] = set()
    extra_files, extra_prefixes = _EXTRA_DEPENDENCIES.get(rel_path, (set(), set()))
    exact_files.update(extra_files)
    dir_prefixes.update(extra_prefixes)

    tree = _parse_ast(rel_path)
    if tree is not None:
        src_exact, src_prefixes = source_dependencies(
            (ROOT / rel_path).read_text(encoding="utf-8"), rel_path
        )
        exact_files.update(src_exact)
        dir_prefixes.update(src_prefixes)
    return frozenset(exact_files), frozenset(dir_prefixes)


def matches_dependencies(
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


@functools.cache
def _all_model_dependency_files() -> frozenset[str]:
    """Union of transitive repository dependencies across all modules in `pysad/models/`."""
    deps: set[str] = set()
    models_dir = ROOT / "pysad" / "models"
    if models_dir.is_dir():
        for py_path in models_dir.rglob("*.py"):
            rel = py_path.relative_to(ROOT).as_posix()
            exact_files, _ = _file_dependencies(rel)
            deps.update(exact_files)
    return frozenset(deps)


def _models_active_conservative(changed: frozenset[str]) -> bool:
    if _SHARED_MODEL_CONFTEST_TRIGGERS & changed:
        return True
    if any(f.startswith("pysad/models/") for f in changed):
        return True
    return bool(_all_model_dependency_files() & changed)


def is_test_file_selected(
    rel_path: str,
    changed: frozenset[str],
    *,
    models_active: bool,
) -> bool:
    if rel_path in changed:
        return True
    if rel_path in _SHARED_MODEL_FILES:
        if models_active:
            return True
        return matches_dependencies(rel_path, changed, ignore_model_prefix=True)
    return matches_dependencies(rel_path, changed)


def tests_plan(base: str) -> dict[str, object]:
    """Return test selection summary `{count, all, needs_compat, files}` relative to `base`."""
    changed = _changed_files_since(base)
    run_all = bool((_GLOBAL_TRIGGERS & changed) or _deps_changed_since(base))
    needs_compat = bool(
        run_all or "pyproject.toml" in changed or any(f.startswith("pysad/") for f in changed)
    )
    all_test_files = sorted(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").rglob("test_*.py")
    )
    if run_all:
        return {
            "count": len(all_test_files),
            "all": True,
            "needs_compat": True,
            "files": all_test_files,
        }
    models_active = _models_active_conservative(changed)
    selected = [
        f for f in all_test_files if is_test_file_selected(f, changed, models_active=models_active)
    ]
    return {
        "count": len(selected),
        "all": False,
        "needs_compat": needs_compat,
        "files": selected,
    }


def _parse_documented_example_outputs(text: str) -> dict[str, tuple[str, str]]:
    outputs: dict[str, tuple[str, str]] = {}
    for section in re.split(r"\n(?=[^\n]+\n-{3,}\n)", text):
        inc = re.search(
            r"^\.\.\s+literalinclude::\s+\.\./examples/(example_\w+\.py)", section, re.M
        )
        if inc is None:
            continue
        match = re.search(
            r"(It prints:|The script ends with:)\n\n\.\.\s+code-block::\s+text\n\n((?:   .+\n?)+)",
            section,
        )
        if match is None:
            continue
        mode = "endswith" if match.group(1).startswith("The script") else "exact"
        expected = "\n".join(line[3:] for line in match.group(2).rstrip("\n").splitlines())
        outputs[inc.group(1)] = (mode, expected)
    return outputs


def _notebook_code(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    lines = [
        line
        for cell in cells
        if cell.get("cell_type") == "code"
        for line in "".join(cell.get("source", ())).splitlines()
        if not line.lstrip().startswith(("%", "!"))
    ]
    return "\n".join(lines)


def _code_matches(code: str, rel_path: str, changed: frozenset[str]) -> bool:
    if rel_path in changed:
        return True
    exact_files, dir_prefixes = source_dependencies(code, rel_path)
    if exact_files & changed:
        return True
    return any(f.startswith(prefix) for prefix in dir_prefixes for f in changed)


def examples_plan(base: str) -> dict[str, object]:
    """Return `{count, files}` of examples/notebooks affected by changes since `base`."""
    changed = _changed_files_since(base)
    examples_dir = ROOT / "examples"
    example_scripts = sorted(p.name for p in examples_dir.glob("example_*.py"))
    notebooks = ["quickstart.ipynb", "nyc_taxi_alerts.ipynb"]
    all_targets = [*example_scripts, *notebooks]

    if (
        {
            "tests/test_examples.py",
            ".github/workflows/examples.yml",
            "docs/_autodiscover.py",
            "scripts/ci_affected.py",
        }
        & changed
    ) or _runtime_toml_changed_since(base):
        return {"count": len(all_targets), "files": all_targets}

    head_outputs = _parse_documented_example_outputs(
        (ROOT / "docs" / "examples.rst").read_text(encoding="utf-8")
    )
    base_outputs: dict[str, tuple[str, str]] = {}
    if "docs/examples.rst" in changed:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{base}:docs/examples.rst"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if proc.returncode == 0:
            base_outputs = _parse_documented_example_outputs(proc.stdout)

    affected: list[str] = []
    for name in example_scripts:
        if "docs/examples.rst" in changed and head_outputs.get(name) != base_outputs.get(name):
            affected.append(name)
            continue
        code = (examples_dir / name).read_text(encoding="utf-8")
        if _code_matches(code, f"examples/{name}", changed):
            affected.append(name)
    for nb in notebooks:
        code = _notebook_code(examples_dir / nb)
        if _code_matches(code, f"examples/{nb}", changed):
            affected.append(nb)
    return {"count": len(affected), "files": affected}


def codeql_plan(base: str) -> dict[str, object]:
    """Return `{python, actions, languages}` needing CodeQL analysis relative to `base`."""
    changed = _changed_files_since(base)
    need_python = any(f.endswith((".py", ".ipynb")) or f == "pyproject.toml" for f in changed)
    need_actions = any(f.startswith(".github/") for f in changed)
    languages = [
        lang for lang, enabled in (("python", need_python), ("actions", need_actions)) if enabled
    ]
    return {"python": need_python, "actions": need_actions, "languages": languages}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    for cmd in ("tests", "examples", "codeql"):
        p = sub.add_parser(cmd)
        p.add_argument("base", help="Git ref to diff against (e.g., origin/master)")
    args = parser.parse_args(argv)
    if args.kind == "tests":
        result = tests_plan(args.base)
    elif args.kind == "examples":
        result = examples_plan(args.base)
    else:
        result = codeql_plan(args.base)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
