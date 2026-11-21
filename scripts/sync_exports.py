"""Synchronizes public imports, ``__all__`` lists, and README detector tables in PySAD."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Subpackages whose __init__.py consists solely of a docstring, relative imports, and __all__.
PURE_REEXPORT_PACKAGES: tuple[str, ...] = (
    "pysad/core",
    "pysad/evaluation",
    "pysad/models",
    "pysad/statistics",
    "pysad/transform/ensemble",
    "pysad/transform/postprocessing",
    "pysad/transform/preprocessing",
    "pysad/transform/probability_calibration",
    "pysad/transform/projection",
)

# Public-named internal base/helper classes intentionally omitted from subpackage __all__.
EXCLUDED_SYMBOLS: dict[str, frozenset[str]] = {
    "pysad/core": frozenset({"RemovableStatistic"}),
    "pysad/transform/ensemble": frozenset({"PYODScoreEnsembler"}),
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
    "pysad/integrations": {
        "PySADToRiverAdapter": "pysad_to_river_adapter",
        "RiverDriftDetector": "river_drift_detector",
    },
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
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
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
    public_names = [n for n in names if not n.startswith("_")]
    private_names = [n for n in names if n.startswith("_")]
    lines: list[str] = []
    if public_names:
        joined = ", ".join(public_names)
        single = f"from .{mod_name} import {joined}"
        if len(single) <= 100:
            lines.append(single)
        else:
            items = "\n".join(f"    {name}," for name in public_names)
            lines.append(f"from .{mod_name} import (\n{items}\n)")
    for name in private_names:
        lines.append(f"from .{mod_name} import {name} as {name}")
    return "\n".join(lines)


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
    doc_block = _extract_docstring_block(init_path.read_text(encoding="utf-8"))
    by_module = _discover_module_exports(pkg)
    imports = "\n".join(_format_import(mod, names) for mod, names in by_module.items())
    all_names = [name for names in by_module.values() for name in names]
    all_block = _format_all(all_names)
    parts = [part for part in (doc_block, imports, all_block) if part]
    return "\n\n".join(parts) + "\n"


def _render_utils_package() -> str:
    pkg = "pysad/utils"
    init_path = ROOT / pkg / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
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
    all_names = [
        *(name for names in by_module.values() for name in names if not name.startswith("_")),
        *inline_names,
    ]
    all_block = _format_all(all_names)
    return f"{prefix}\n\n{imports}\n\n{all_block}\n\n\n{suffix}\n"


def _render_integrations_package() -> str:
    pkg = "pysad/integrations"
    init_path = ROOT / pkg / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
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


_AUTOGEN_START = "<!-- BEGIN:AUTOGEN_DETECTORS -->"
_AUTOGEN_END = "<!-- END:AUTOGEN_DETECTORS -->"


def _extract_bibtex_entries(raw: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", raw):
        key = match.group(1)
        brace_start = raw.index("{", match.start())
        depth = 0
        end = len(raw)
        for idx in range(brace_start, len(raw)):
            ch = raw[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        entries.append((key, raw[match.start() : end].strip()))
    return entries


def _iter_bibtex_literals(tree: ast.AST) -> list[str]:
    found: list[str] = []
    nodes = list(getattr(tree, "body", ()))
    for node in list(nodes):
        if isinstance(node, ast.ClassDef):
            nodes.extend(node.body)
    for node in nodes:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__bibtex__" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            found.append(node.value.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__bibtex__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            found.append(node.value.value)
    return found


def _bib_field(entry: str, field: str) -> str:
    match = re.search(rf"(?:^|[,\n])\s*{field}\s*=\s*", entry, re.I)
    if match is None:
        return ""
    pos = match.end()
    while pos < len(entry) and entry[pos].isspace():
        pos += 1
    if pos >= len(entry):
        return ""
    opener = entry[pos]
    if opener == "{":
        depth = 0
        for idx in range(pos, len(entry)):
            if entry[idx] == "{":
                depth += 1
            elif entry[idx] == "}":
                depth -= 1
                if depth == 0:
                    return entry[pos + 1 : idx]
        return entry[pos + 1 :]
    if opener == '"':
        end = entry.find('"', pos + 1)
        return entry[pos + 1 :] if end == -1 else entry[pos + 1 : end]
    rest = re.match(r"[^,}\s]+", entry[pos:])
    return rest.group(0) if rest else ""


def _clean_latex(text: str) -> str:
    text = re.sub(r"\\[\"'.^~=`Huvck]\{?([A-Za-z])\}?", r"\1", text)
    return " ".join(text.replace("{", "").replace("}", "").split())


def _author_surname(author: str) -> str:
    cleaned = _clean_latex(author)
    if "," in cleaned:
        return cleaned.split(",", 1)[0].strip()
    parts = cleaned.split()
    return parts[-1] if parts else cleaned


def _collect_bib_citations() -> dict[str, str]:
    sources: list[str] = []
    refs_bib = ROOT / "docs" / "refs.bib"
    if refs_bib.exists():
        sources.append(refs_bib.read_text(encoding="utf-8"))
    for py_path in sorted((ROOT / "pysad").rglob("*.py")):
        text = py_path.read_text(encoding="utf-8")
        if "__bibtex__" in text:
            sources.extend(_iter_bibtex_literals(ast.parse(text)))

    citations: dict[str, str] = {}
    for raw in sources:
        for key, entry in _extract_bibtex_entries(raw):
            if key in citations:
                continue
            raw_author = _bib_field(entry, "author")
            year = _clean_latex(_bib_field(entry, "year"))
            authors = [a.strip() for a in re.split(r"\s+and\s+", raw_author) if a.strip()]
            surnames = [_author_surname(a) for a in authors]
            if len(surnames) == 1:
                auth_text = surnames[0]
            elif len(surnames) == 2:
                auth_text = f"{surnames[0]} & {surnames[1]}"
            elif len(surnames) >= 3:
                auth_text = f"{surnames[0]} et al."
            else:
                auth_text = key
            citations[key] = f"{auth_text}, {year}" if year else auth_text
    return citations


def _class_attr_const(node: ast.ClassDef, attr_name: str) -> ast.AST | None:
    for stmt in node.body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == attr_name for t in stmt.targets
        ):
            return stmt.value
        if (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == attr_name
        ):
            return stmt.value
    return None


def _discover_detectors() -> dict[str, tuple[bool, str, str]]:
    """Returns ``{class_name: (is_detector, input_kind, raw_summary)}`` for models in AST order."""
    pkg = "pysad/models"
    pkg_dir = ROOT / pkg
    sources: list[Path] = [
        py_file
        for py_file in sorted(pkg_dir.glob("*.py"))
        if py_file.name != "__init__.py" and not py_file.name.startswith("_")
    ]
    sources.extend(ROOT / rel_path for _, rel_path in NESTED_MODULES.get(pkg, ()))

    detectors: dict[str, tuple[bool, str, str]] = {}
    for py_file in sources:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        has_check_univariate = any(
            isinstance(stmt, ast.ImportFrom)
            and any(alias.name == "check_univariate" for alias in stmt.names)
            for stmt in tree.body
        )
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            is_det_val = _class_attr_const(node, "_is_detector")
            is_detector = (
                bool(is_det_val.value)
                if isinstance(is_det_val, ast.Constant) and isinstance(is_det_val.value, bool)
                else True
            )
            kind_val = _class_attr_const(node, "_input_kind")
            input_kind = (
                str(kind_val.value)
                if isinstance(kind_val, ast.Constant) and isinstance(kind_val.value, str)
                else "multivariate"
            )
            if input_kind == "multivariate" and has_check_univariate:
                input_kind = "univariate"

            doc_val = _class_attr_const(node, "_detector_doc")
            raw_summary = ""
            if (
                isinstance(doc_val, ast.Tuple)
                and len(doc_val.elts) == 3
                and isinstance(doc_val.elts[0], ast.Constant)
                and isinstance(doc_val.elts[0].value, str)
            ):
                raw_summary = doc_val.elts[0].value
            else:
                doc = ast.get_docstring(node) or node.name
                raw_summary = doc.splitlines()[0].strip()

            detectors[node.name] = (is_detector, input_kind, raw_summary)
    return detectors


def _format_readme_method(name: str, raw_summary: str, citations: dict[str, str]) -> str:
    text = raw_summary.strip()
    if text.startswith(f"{name}: "):
        text = text[len(name) + 2 :].strip()

    def _replace_cite(match: re.Match[str]) -> str:
        refs = [citations.get(k.strip(), k.strip()) for k in match.group(1).split(",")]
        return f"({'; '.join(refs)})"

    text = re.sub(r":cite:`([^`]+)`", _replace_cite, text)
    text = re.sub(r":class:`~?(?:[^`]+\.)?([A-Za-z_]\w*)`", r"`\1`", text)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    text = " ".join(text.rstrip(".").split())
    return text


def _render_readme_detectors() -> str:
    readme_path = ROOT / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    if _AUTOGEN_START not in text or _AUTOGEN_END not in text:
        return text

    start_idx = text.index(_AUTOGEN_START)
    end_idx = text.index(_AUTOGEN_END) + len(_AUTOGEN_END)
    block = text[start_idx + len(_AUTOGEN_START) : text.index(_AUTOGEN_END)]

    multi_order: list[str] = []
    uni_order: list[str] = []
    existing_desc: dict[str, str] = {}
    section = "multi"
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("**Multivariate"):
            section = "multi"
        elif stripped.startswith("**Univariate"):
            section = "uni"
        else:
            row_match = re.match(r"^\|\s*`(\w+)`\s*\|\s*(.+?)\s*\|$", stripped)
            if row_match:
                cls_name, desc = row_match.group(1), row_match.group(2)
                existing_desc[cls_name] = desc
                (uni_order if section == "uni" else multi_order).append(cls_name)

    detectors = _discover_detectors()
    citations = _collect_bib_citations()
    active_multi = {
        name
        for name, (is_det, kind, _) in detectors.items()
        if is_det and kind in ("multivariate", "edge")
    }
    active_uni = {
        name
        for name, (is_det, kind, _) in detectors.items()
        if is_det and kind in ("univariate", "timestamped")
    }

    multi_names = [n for n in multi_order if n in active_multi] + sorted(
        active_multi - set(multi_order), key=lambda s: (s.lower(), s)
    )
    uni_names = [n for n in uni_order if n in active_uni] + sorted(
        active_uni - set(uni_order), key=lambda s: (s.lower(), s)
    )

    def _rows(names: list[str]) -> list[str]:
        out = ["| Class | Method |", "|---|---|"]
        for name in names:
            desc = existing_desc.get(name) or _format_readme_method(
                name, detectors[name][2], citations
            )
            out.append(f"| `{name}` | {desc} |")
        return out

    rendered = "\n".join(
        [
            _AUTOGEN_START,
            "<summary><b>All streaming detectors</b></summary>",
            "",
            "<br>",
            "",
            "**Multivariate**",
            "",
            *_rows(multi_names),
            "",
            "**Univariate**",
            "",
            *_rows(uni_names),
            _AUTOGEN_END,
        ]
    )
    return text[:start_idx] + rendered + text[end_idx:]


def sync_all(check: bool = False) -> list[str]:
    changed: list[str] = []
    targets: list[tuple[Path, str]] = [
        *(
            (ROOT / pkg / "__init__.py", _render_pure_package(pkg))
            for pkg in PURE_REEXPORT_PACKAGES
        ),
        (ROOT / "pysad/utils/__init__.py", _render_utils_package()),
        (ROOT / "pysad/integrations/__init__.py", _render_integrations_package()),
        (ROOT / "README.md", _render_readme_detectors()),
    ]
    for path, expected in targets:
        current = path.read_text(encoding="utf-8")
        if current != expected:
            rel = path.relative_to(ROOT).as_posix()
            changed.append(rel)
            if not check:
                path.write_text(expected, encoding="utf-8")
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
