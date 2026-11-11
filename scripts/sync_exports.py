"""Synchronizes public imports and ``__all__`` lists across PySAD subpackage ``__init__.py`` files."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Subpackages whose __init__.py consists solely of a docstring, relative imports, and __all__.
PURE_REEXPORT_PACKAGES: tuple[str, ...] = (
    "pysad/core",
    "pysad/evaluation",
    "pysad/models",
    "pysad/models/integrations",
    "pysad/statistics",
    "pysad/transform/ensemble",
    "pysad/transform/postprocessing",
    "pysad/transform/preprocessing",
    "pysad/transform/probability_calibration",
    "pysad/transform/projection",
)

# Public-named internal base/helper classes intentionally omitted from subpackage __all__.
EXCLUDED_SYMBOLS: dict[str, frozenset[str]] = {
    "pysad/core": frozenset({"UnivariateStatistic"}),
    "pysad/transform/ensemble": frozenset({"PYODScoreEnsembler"}),
    "pysad/utils": frozenset({"UnlimitedWindow"}),
}

# Nested single-detector modules exported by a parent package.
NESTED_MODULES: dict[str, tuple[tuple[str, str], ...]] = {
    "pysad/models": (("htm.htm", "pysad/models/htm/htm.py"),),
}

# Private helpers intentionally re-exported by a subpackage.
EXTRA_MODULE_EXPORTS: dict[str, dict[str, tuple[str, ...]]] = {
    "pysad/utils": {"array_streamer": ("_iterate",)},
}

LAZY_EXPORTS: dict[str, dict[str, str]] = {
    "pysad/integrations": {"RiverAnomalyDetector": "river_anomaly_detector"},
}


def _ruff_module_sort_key(mod_name: str) -> tuple[str, str]:
    return (mod_name.lower(), mod_name)


def _ruff_symbol_sort_key(name: str) -> tuple[int, str, str]:
    stripped = name.lstrip("_")
    if stripped.isupper():
        kind = 0
    elif stripped[:1].isupper():
        kind = 1
    else:
        kind = 2
    return (kind, name.lower(), name)


def _discover_module_exports(pkg: str) -> dict[str, list[str]]:
    pkg_dir = ROOT / pkg
    excluded = EXCLUDED_SYMBOLS.get(pkg, frozenset())
    by_module: dict[str, list[str]] = {}

    sources: list[tuple[str, Path]] = [
        (py_file.stem, py_file)
        for py_file in sorted(pkg_dir.glob("*.py"))
        if py_file.name != "__init__.py" and not py_file.name.startswith("_")
    ]
    for mod_name, rel_path in NESTED_MODULES.get(pkg, ()):
        sources.append((mod_name, ROOT / rel_path))

    for mod_name, py_file in sources:
        tree = ast.parse(py_file.read_text())
        names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef | ast.FunctionDef)
            and not node.name.startswith("_")
            and node.name not in excluded
        ]
        names.extend(EXTRA_MODULE_EXPORTS.get(pkg, {}).get(mod_name, ()))
        if names:
            by_module[mod_name] = sorted(set(names), key=_ruff_symbol_sort_key)

    return dict(sorted(by_module.items(), key=lambda item: _ruff_module_sort_key(item[0])))


def _format_import(mod_name: str, names: list[str]) -> str:
    joined = ", ".join(names)
    single = f"from .{mod_name} import {joined}"
    if len(single) <= 100:
        return single
    items = "\n".join(f"    {name}," for name in names)
    return f"from .{mod_name} import (\n{items}\n)"


def _format_all(names: list[str]) -> str:
    sorted_names = sorted(names, key=lambda s: (s.lower(), s))
    joined = ", ".join(f'"{name}"' for name in sorted_names)
    single = f"__all__ = [{joined}]"
    if len(single) <= 100 and len(sorted_names) <= 4:
        return single
    items = "\n".join(f'    "{name}",' for name in sorted_names)
    return f"__all__ = [\n{items}\n]"


def _extract_docstring_block(text: str) -> str:
    lines = text.splitlines()
    tree = ast.parse(text)
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        end_lineno = tree.body[0].end_lineno
        assert end_lineno is not None
        return "\n".join(lines[:end_lineno])
    return ""


def _render_pure_package(pkg: str) -> str:
    init_path = ROOT / pkg / "__init__.py"
    doc_block = _extract_docstring_block(init_path.read_text())
    by_module = _discover_module_exports(pkg)
    imports = "\n".join(_format_import(mod, names) for mod, names in by_module.items())
    all_names = [name for names in by_module.values() for name in names]
    all_block = _format_all(all_names)
    parts = [part for part in (doc_block, imports, all_block) if part]
    return "\n\n".join(parts) + "\n"


def _render_utils_package() -> str:
    pkg = "pysad/utils"
    init_path = ROOT / pkg / "__init__.py"
    text = init_path.read_text()
    lines = text.splitlines()
    tree = ast.parse(text)

    by_module = _discover_module_exports(pkg)
    inline_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and not node.name.startswith("_")
    ]
    first_rel_idx: int | None = None
    all_end_lineno: int | None = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and first_rel_idx is None:
            first_rel_idx = node.lineno - 1
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            all_end_lineno = node.end_lineno

    assert first_rel_idx is not None and all_end_lineno is not None
    prefix = "\n".join(lines[:first_rel_idx]).rstrip()
    suffix = "\n".join(lines[all_end_lineno:]).lstrip("\n")
    imports = "\n".join(_format_import(mod, names) for mod, names in by_module.items())
    all_names = [*(name for names in by_module.values() for name in names), *inline_names]
    all_block = _format_all(all_names)
    return f"{prefix}\n\n{imports}\n\n{all_block}\n\n\n{suffix}\n"


def _render_integrations_package() -> str:
    pkg = "pysad/integrations"
    init_path = ROOT / pkg / "__init__.py"
    text = init_path.read_text()
    lines = text.splitlines()
    tree = ast.parse(text)

    lazy = LAZY_EXPORTS[pkg]
    by_module = _discover_module_exports(pkg)
    eager_by_module = {
        mod: [n for n in names if n not in lazy]
        for mod, names in by_module.items()
        if any(n not in lazy for n in names)
    }
    first_rel_idx: int | None = None
    all_end_lineno: int | None = None
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and first_rel_idx is None:
            first_rel_idx = node.lineno - 1
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            all_end_lineno = node.end_lineno

    assert first_rel_idx is not None and all_end_lineno is not None
    prefix = "\n".join(lines[:first_rel_idx]).rstrip()
    suffix = "\n".join(lines[all_end_lineno:]).lstrip("\n")
    imports = "\n".join(_format_import(mod, names) for mod, names in eager_by_module.items())
    all_names = [name for names in by_module.values() for name in names]
    all_block = _format_all(all_names)
    return f"{prefix}\n\n{imports}\n\n{all_block}\n\n\n{suffix}\n"


def sync_all(check: bool = False) -> list[str]:
    changed: list[str] = []
    targets: list[tuple[Path, str]] = [
        *(
            (ROOT / pkg / "__init__.py", _render_pure_package(pkg))
            for pkg in PURE_REEXPORT_PACKAGES
        ),
        (ROOT / "pysad/utils/__init__.py", _render_utils_package()),
        (ROOT / "pysad/integrations/__init__.py", _render_integrations_package()),
    ]
    for path, expected in targets:
        current = path.read_text()
        if current != expected:
            rel = path.relative_to(ROOT).as_posix()
            changed.append(rel)
            if not check:
                path.write_text(expected)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="return a non-zero exit code instead of modifying files when exports are out of sync",
    )
    args = parser.parse_args()
    changed = sync_all(check=args.check)
    if changed:
        action = "Out of sync" if args.check else "Updated"
        print(f"{action}: {', '.join(changed)}", file=sys.stderr)
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
