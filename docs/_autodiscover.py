"""Auto-discovery helpers for Sphinx documentation, API stubs, BibTeX, and example galleries."""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
from pathlib import Path

import pysad.models
from pysad.utils import Data

NON_DETECTOR_MODELS = frozenset({"NullModel", "PerfectModel", "RandomModel", "StreamPipeline"})
_EXAMPLE_ACRONYMS = {
    "adems": "ADeMS",
    "adwin": "ADWIN",
    "aupr": "AUPR",
    "auroc": "AUROC",
    "cpod": "CPOD",
    "d3": "D3",
    "damp": "DAMP",
    "ddm": "DDM",
    "ddsketch": "DDSketch",
    "dilof": "DILOF",
    "eddm": "EDDM",
    "esd": "ESD",
    "expose": "EXPoSE",
    "fhddm": "FHDDM",
    "hddm": "HDDM",
    "htm": "HTM",
    "kelos": "KELOS",
    "knn": "KNN",
    "knncad": "KNNCAD",
    "kswin": "KSWIN",
    "loda": "LODA",
    "lof": "LOF",
    "mcod": "MCOD",
    "midas": "MIDAS",
    "mstream": "MStream",
    "nets": "NETS",
    "pewma": "PEWMA",
    "pyod": "PyOD",
    "rrcf": "RRCF",
    "rshash": "RSHash",
    "sand": "SAND",
    "spot": "SPOT",
    "stl": "STL",
    "svm": "SVM",
    "teda": "TEDA",
    "xstream": "xStream",
}


def _extract_bibtex_entries(raw: str) -> list[tuple[str, str]]:
    """Parses `(key, entry_text)` pairs from a BibTeX string."""
    entries: list[tuple[str, str]] = []
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", raw):
        key = match.group(1)
        start = match.start()
        brace_start = raw.index("{", start)
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
        entries.append((key, raw[start:end].strip()))
    return entries


def _iter_bibtex_Literals(tree: ast.AST) -> list[str]:
    """Extracts module-level and class-level `__bibtex__` string constants from an AST."""
    found: list[str] = []
    nodes_to_scan = list(getattr(tree, "body", ()))
    for node in list(nodes_to_scan):
        if isinstance(node, ast.ClassDef):
            nodes_to_scan.extend(node.body)
    for node in nodes_to_scan:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__bibtex__" for t in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
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


def collect_bibtex(root: Path) -> tuple[str, set[str]]:
    """Merges `docs/refs.bib` with all `__bibtex__` attributes in `pysad/`."""
    sources: list[str] = [(root / "docs" / "refs.bib").read_text(encoding="utf-8")]
    for py_path in sorted((root / "pysad").rglob("*.py")):
        if "kitnet_model" in py_path.parts:
            continue
        text = py_path.read_text(encoding="utf-8")
        if "__bibtex__" not in text:
            continue
        sources.extend(_iter_bibtex_Literals(ast.parse(text)))

    by_key: dict[str, str] = {}
    normalized_by_key: dict[str, str] = {}
    for raw in sources:
        for key, entry in _extract_bibtex_entries(raw):
            norm = " ".join(entry.split())
            if key in by_key:
                if normalized_by_key[key] != norm:
                    raise ValueError(f"Conflicting BibTeX entries for key {key!r}.")
                continue
            by_key[key] = entry
            normalized_by_key[key] = norm

    ordered = [by_key[k] for k in sorted(by_key, key=str.lower)]
    return "\n\n".join(ordered) + "\n", set(by_key)


def expand_api_autosummary(text: str) -> str:
    """Populates empty `.. autosummary::` directives in an API `.rst` page from `__all__`."""
    mod_match = re.search(r"^\.\.\s+(?:current)?module::\s+(\S+)", text, flags=re.M)
    if mod_match is None:
        return text

    module = importlib.import_module(mod_match.group(1))
    public_exports = [name for name in getattr(module, "__all__", ()) if not name.startswith("_")]

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != ".. autosummary::":
            out.append(lines[i])
            i += 1
            continue

        out.append(lines[i])
        i += 1
        kept_options: list[str] = []
        kind: str | None = None
        tail_names: list[str] = []
        while i < len(lines) and re.match(r"^\s+:[A-Za-z0-9_-]+:", lines[i]):
            opt_match = re.match(r"^\s+:([A-Za-z0-9_-]+):\s*(.*)$", lines[i])
            assert opt_match is not None
            opt_name, opt_val = opt_match.group(1), opt_match.group(2).strip()
            if opt_name == "kind":
                kind = opt_val
            elif opt_name == "tail":
                tail_names = [item.strip() for item in opt_val.split(",") if item.strip()]
            else:
                kept_options.append(lines[i])
            i += 1

        out.extend(kept_options)
        body_start = i
        has_explicit_items = False
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped == "":
                i += 1
                continue
            if lines[i].startswith("   "):
                has_explicit_items = True
                i += 1
                continue
            break

        if has_explicit_items:
            out.extend(lines[body_start:i])
            continue

        selected = list(public_exports)
        if kind == "classes":
            selected = [n for n in selected if inspect.isclass(getattr(module, n))]
        elif kind == "functions":
            selected = [n for n in selected if not inspect.isclass(getattr(module, n))]
        if tail_names:
            tail_set = set(tail_names)
            selected = [
                *(n for n in selected if n not in tail_set),
                *(n for n in tail_names if n in set(selected)),
            ]

        out.append("")
        out.extend(f"   {name}" for name in selected)
        if i < len(lines):
            out.append("")

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _infer_detector_doc(cls: type) -> tuple[str, str, str]:
    custom = getattr(cls, "_detector_doc", None)
    if isinstance(custom, tuple) and len(custom) == 3:
        return custom
    doc = inspect.getdoc(cls) or cls.__name__
    first_line = doc.splitlines()[0].rstrip(".")
    sig = inspect.signature(cls.__init__)
    required = [
        f"``{p.name}``"
        for p in sig.parameters.values()
        if p.name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    needs = ", ".join(required) if required else "Nothing"
    memory = (
        "Fixed: a window of ``window_size`` points" if "window_size" in sig.parameters else "Fixed"
    )
    return (first_line, needs, memory)


def _is_univariate_detector(cls: type) -> bool:
    kind = getattr(cls, "_input_kind", None)
    if kind in ("univariate", "timestamped"):
        return True
    if kind in ("multivariate", "edge"):
        return False
    return hasattr(sys.modules[cls.__module__], "check_univariate")


def expand_detectors_tables(text: str) -> str:
    """Appends rows to `docs/detectors.rst` for any detector in `pysad.models.__all__` not yet listed."""
    documented = set(re.findall(r":class:`~pysad\.models\.([A-Za-z_]\w*)`", text))
    missing = [
        name
        for name in pysad.models.__all__
        if name not in NON_DETECTOR_MODELS and name not in documented
    ]
    if not missing:
        return text

    multi_rows: list[str] = []
    uni_rows: list[str] = []
    for name in missing:
        cls = getattr(pysad.models, name)
        method, needs, memory = _infer_detector_doc(cls)
        row = (
            f"   * - :class:`~pysad.models.{name}`\n"
            f"     - {method}\n"
            f"     - {needs}\n"
            f"     - {memory}"
        )
        if _is_univariate_detector(cls):
            uni_rows.append(row)
        else:
            multi_rows.append(row)

    if multi_rows:
        marker = "\nUnivariate detectors\n--------------------"
        text = text.replace(marker, "\n" + "\n".join(multi_rows) + "\n" + marker, 1)
    if uni_rows:
        marker = "\nIntegrations\n------------"
        text = text.replace(marker, "\n" + "\n".join(uni_rows) + "\n" + marker, 1)
    return text


def _humanize_example_stem(stem: str) -> str:
    raw_words = stem.removeprefix("example_").split("_")
    words = [_EXAMPLE_ACRONYMS.get(w.lower(), w) for w in raw_words]
    if words and words[0] not in _EXAMPLE_ACRONYMS.values():
        words[0] = words[0].capitalize()
    return " ".join(words)


def expand_examples_gallery(text: str, root: Path) -> str:
    """Replaces `.. auto-examples::` with sections for any `examples/example_*.py` not yet included."""
    marker = ".. auto-examples::"
    if marker not in text:
        return text

    included = set(
        re.findall(r"^\.\.\s+literalinclude::\s+\.\./examples/(example_\w+\.py)", text, re.M)
    )
    blocks: list[str] = []
    for py_path in sorted((root / "examples").glob("example_*.py")):
        if py_path.name in included:
            continue
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        doc = ast.get_docstring(tree)
        start_line = 1
        if (
            doc is not None
            and tree.body
            and isinstance(tree.body[0], ast.Expr)
            and tree.body[0].end_lineno is not None
        ):
            src_lines = source.splitlines()
            start_line = tree.body[0].end_lineno + 1
            while start_line <= len(src_lines) and not src_lines[start_line - 1].strip():
                start_line += 1

        if doc and "\n\n" in doc:
            first, rest = doc.split("\n\n", 1)
            title = first.strip().rstrip(".")
            prose = rest.strip()
        else:
            title = _humanize_example_stem(py_path.stem)
            prose = (doc or "").strip()

        underline = "-" * len(title)
        section_parts = [f"{title}\n{underline}"]
        if prose:
            section_parts.append(prose)
        inc = f".. literalinclude:: ../examples/{py_path.name}\n   :language: python"
        if start_line > 1:
            inc += f"\n   :lines: {start_line}-"
        section_parts.append(inc)
        blocks.append("\n\n".join(section_parts))

    return text.replace(marker, "\n\n".join(blocks))


def parse_documented_example_outputs(text: str) -> dict[str, tuple[str, str]]:
    """Extracts `{example_name: ('exact' | 'endswith', expected_stdout)}` from `docs/examples.rst`."""
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


def doc_counts(root: Path) -> dict[str, int]:
    """Computes live counts exposed as reStructuredText substitutions across the docs."""
    num_detectors = sum(1 for name in pysad.models.__all__ if name not in NON_DETECTOR_MODELS)
    num_drift = 0
    bench_tree = ast.parse((root / "benchmarks" / "benchmark.py").read_text(encoding="utf-8"))
    for node in bench_tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DRIFT"
            and isinstance(node.value, ast.Dict)
        ):
            num_drift = len(node.value.keys)
            break
    return {
        "num_datasets": len(Data.datasets()),
        "num_detectors": num_detectors,
        "num_drift_streams": num_drift,
    }
