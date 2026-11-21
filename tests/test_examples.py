import ast
import contextlib
import functools
import importlib
import io
import json
import os
import re
import runpy
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import pysad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))
from _autodiscover import expand_examples_gallery, parse_documented_example_outputs  # noqa: E402

EXAMPLES_DIR = str(ROOT / "examples")
EXAMPLES = sorted(
    name
    for name in os.listdir(EXAMPLES_DIR)
    if name.startswith("example_") and name.endswith(".py")
)
_SLOW_EXAMPLES = (
    "example_htm.py",
    "example_mstream.py",
    "example_usage_smart_sifter.py",
    "example_usage_inqmad.py",
    "example_sand.py",
    "example_xstream.py",
    "example_creditcard.py",
    "example_windowed_metric.py",
    "example_streaming_ensemble.py",
    "example_concept_drift.py",
    "example_usage_rrcf.py",
    "example_usage_iforest_asd.py",
)
EXAMPLES_EXECUTION_ORDER = [e for e in _SLOW_EXAMPLES if e in EXAMPLES] + [
    e for e in EXAMPLES if e not in _SLOW_EXAMPLES
]
DOCUMENTED_OUTPUTS = parse_documented_example_outputs(
    (ROOT / "docs" / "examples.rst").read_text(encoding="utf-8")
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?")


def _example_pysad_files(code: str) -> set[str]:
    """Returns the repository-relative pysad source files transitively imported by `code`."""
    todo: list[str] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pysad"):
            mod = importlib.import_module(node.module)
            for alias in node.names:
                obj = getattr(mod, alias.name, mod)
                mod_name = (
                    obj.__name__ if isinstance(obj, ModuleType) else getattr(obj, "__module__", "")
                )
                if isinstance(mod_name, str) and mod_name.startswith("pysad"):
                    todo.append(mod_name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("pysad"):
                    importlib.import_module(alias.name)
                    todo.append(alias.name)

    files: set[str] = set()
    pysad_root = Path(pysad.__file__).parents[1]
    while todo:
        module = sys.modules.get(todo.pop())
        if module is None or module.__file__ is None:
            continue
        path = Path(module.__file__).relative_to(pysad_root).as_posix()
        if path in files:
            continue
        files.add(path)
        if path.endswith("/__init__.py"):
            continue
        for value in vars(module).values():
            name = (
                value.__name__
                if isinstance(value, ModuleType)
                else getattr(value, "__module__", "")
            )
            if isinstance(name, str) and name.split(".")[0] == "pysad":
                todo.append(name)
        for node in ast.walk(ast.parse(Path(module.__file__).read_text(encoding="utf-8"))):
            target = sys.modules.get(node.module) if isinstance(node, ast.ImportFrom) else None
            if (
                target is not None
                and target.__name__.split(".")[0] == "pysad"
                and target.__file__ is not None
                and not target.__file__.endswith("__init__.py")
            ):
                todo.append(target.__name__)
    return files


@functools.cache
def _changed_files_since(base: str) -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", base],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    return set(out.split())


@functools.cache
def _documented_outputs_at(base: str) -> dict[str, tuple[str, str]]:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{base}:docs/examples.rst"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return {}
    return parse_documented_example_outputs(proc.stdout)


def _should_run_example(rel_path: str, code: str, base: str | None = None) -> bool:
    if base is None:
        base = os.environ.get("PYSAD_EXAMPLES_SINCE")
    if not base:
        return True
    changed = _changed_files_since(base)
    if {
        "tests/test_examples.py",
        ".github/workflows/examples.yml",
        "docs/_autodiscover.py",
    } & changed:
        return True
    if "docs/examples.rst" in changed and rel_path.startswith("examples/example_"):
        name = Path(rel_path).name
        if DOCUMENTED_OUTPUTS.get(name) != _documented_outputs_at(base).get(name):
            return True
    if "pyproject.toml" in changed:
        before = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{base}:pyproject.toml"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
        after = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        if before.split("\ntest = [", 1)[0] != after.split("\ntest = [", 1)[0]:
            return True
    return rel_path in changed or bool(_example_pysad_files(code) & changed)


def affected_examples(base: str) -> list[str]:
    """Return example scripts and notebooks affected by changes since ``base``."""
    affected: list[str] = []
    for example in EXAMPLES_EXECUTION_ORDER:
        code = (ROOT / "examples" / example).read_text(encoding="utf-8")
        if _should_run_example(f"examples/{example}", code, base=base):
            affected.append(example)
    for nb in ("quickstart.ipynb", "nyc_taxi_alerts.ipynb"):
        code = notebook_code(os.path.join(EXAMPLES_DIR, nb))
        if _should_run_example(f"examples/{nb}", code, base=base):
            affected.append(nb)
    return affected


def _assert_output_matches_docs(
    example: str,
    actual_stdout: str,
    mode: str,
    expected: str,
) -> None:
    actual_lines = actual_stdout.rstrip("\n").splitlines()
    expected_lines = expected.splitlines()
    if mode == "endswith":
        assert len(actual_lines) >= len(expected_lines), (
            f"{example} printed {len(actual_lines)} lines, expected at least {len(expected_lines)}:\n"
            f"{actual_stdout}"
        )
        actual_lines = actual_lines[-len(expected_lines) :]
    else:
        assert len(actual_lines) == len(expected_lines), (
            f"{example} printed {len(actual_lines)} lines, docs/examples.rst expects "
            f"{len(expected_lines)}:\nactual:\n{actual_stdout}\nexpected:\n{expected}"
        )

    for act_line, exp_line in zip(actual_lines, expected_lines, strict=True):
        act_matches = list(_NUMBER_RE.finditer(act_line))
        exp_matches = list(_NUMBER_RE.finditer(exp_line))
        act_template = _NUMBER_RE.sub("{}", act_line)
        exp_template = _NUMBER_RE.sub("{}", exp_line)
        assert act_template == exp_template and len(act_matches) == len(exp_matches), (
            f"{example} output structure differs from docs/examples.rst:\n"
            f"  actual:   {act_line!r}\n"
            f"  expected: {exp_line!r}"
        )
        for act_m, exp_m in zip(act_matches, exp_matches, strict=True):
            act_s, exp_s = act_m.group(0), exp_m.group(0)
            if "." in act_s or "." in exp_s:
                assert abs(float(act_s) - float(exp_s)) <= 0.01, (
                    f"{example} numeric output drifted from docs/examples.rst:\n"
                    f"  actual:   {act_line!r}\n"
                    f"  expected: {exp_line!r}"
                )
            else:
                suffix = exp_line[exp_m.end() :]
                tol = 1 if suffix.startswith(" alert") else 0
                assert abs(int(act_s) - int(exp_s)) <= tol, (
                    f"{example} integer output drifted from docs/examples.rst:\n"
                    f"  actual:   {act_line!r}\n"
                    f"  expected: {exp_line!r}"
                )


def _run_in_process(
    path: Path | None = None, code: str | None = None, name: str = "<example>"
) -> str:
    """Executes an example script or notebook snippet in-process and captures stdout."""
    prev_cwd = os.getcwd()
    prev_path = sys.path[:]
    prev_argv = sys.argv[:]
    prev_filters = warnings.filters[:]
    prev_err = np.seterr()
    prev_backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].switch_backend("Agg")
    elif "matplotlib" in sys.modules:
        sys.modules["matplotlib"].use("Agg")
    buf = io.StringIO()
    try:
        os.chdir(EXAMPLES_DIR)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        sys.argv = [str(path) if path is not None else name]
        with contextlib.redirect_stdout(buf):
            if path is not None:
                runpy.run_path(str(path), run_name="__main__")
            else:
                exec(compile(code or "", name, "exec"), {"__name__": "__main__"})
    finally:
        os.chdir(prev_cwd)
        sys.path[:] = prev_path
        sys.argv[:] = prev_argv
        warnings.filters[:] = prev_filters
        np.seterr(**prev_err)
        if prev_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = prev_backend
        if "matplotlib.pyplot" in sys.modules:
            sys.modules["matplotlib.pyplot"].close("all")
    return buf.getvalue()


def test_examples_gallery_autodiscovers_all_scripts_and_parses_outputs():
    raw_rst = (ROOT / "docs" / "examples.rst").read_text(encoding="utf-8")
    expanded = expand_examples_gallery(raw_rst, ROOT)
    included = re.findall(
        r"^\.\.\s+literalinclude::\s+\.\./examples/(example_\w+\.py)", expanded, re.M
    )
    assert sorted(included) == EXAMPLES
    assert len(included) == len(set(included))
    assert len(DOCUMENTED_OUTPUTS) >= 25

    _assert_output_matches_docs("example_usage.py", "AUROC: 0.768\n", "exact", "AUROC: 0.768")
    _assert_output_matches_docs("example_usage.py", "AUROC: 0.775\n", "exact", "AUROC: 0.768")
    _assert_output_matches_docs("example_prob.py", "26 alerts\n", "exact", "27 alerts")
    with pytest.raises(AssertionError):
        _assert_output_matches_docs("example_usage.py", "AUROC: 0.700\n", "exact", "AUROC: 0.768")
    with pytest.raises(AssertionError):
        _assert_output_matches_docs("example_idx.py", "Step 10\n", "exact", "Step 11")


@pytest.mark.examples
@pytest.mark.parametrize("example", EXAMPLES_EXECUTION_ORDER)
def test_example_runs(example):
    path = ROOT / "examples" / example
    code = path.read_text(encoding="utf-8")
    if not _should_run_example(f"examples/{example}", code):
        pytest.skip(f"No changes affecting {example} since {os.environ['PYSAD_EXAMPLES_SINCE']}")

    stdout = _run_in_process(path=path, name=example)

    if example in DOCUMENTED_OUTPUTS:
        mode, expected = DOCUMENTED_OUTPUTS[example]
        _assert_output_matches_docs(example, stdout, mode, expected)


def notebook_code(path):
    """Joins the code cells of a notebook, skipping shell and magic lines such as ``%pip install``."""
    with open(path, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    lines = [
        line
        for cell in cells
        if cell["cell_type"] == "code"
        for line in "".join(cell["source"]).splitlines()
        if not line.lstrip().startswith(("%", "!"))
    ]
    return "\n".join(lines)


@pytest.mark.examples
def test_quickstart_notebook_runs():
    code = notebook_code(os.path.join(EXAMPLES_DIR, "quickstart.ipynb"))
    if not _should_run_example("examples/quickstart.ipynb", code):
        pytest.skip(
            f"No changes affecting quickstart.ipynb since {os.environ['PYSAD_EXAMPLES_SINCE']}"
        )

    _run_in_process(code=code, name="quickstart.ipynb")


@pytest.mark.examples
def test_nyc_taxi_alerts_notebook_runs():
    code = notebook_code(os.path.join(EXAMPLES_DIR, "nyc_taxi_alerts.ipynb"))
    if not _should_run_example("examples/nyc_taxi_alerts.ipynb", code):
        pytest.skip(
            f"No changes affecting nyc_taxi_alerts.ipynb since {os.environ['PYSAD_EXAMPLES_SINCE']}"
        )

    _run_in_process(code=code, name="nyc_taxi_alerts.ipynb")
