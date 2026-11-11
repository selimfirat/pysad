import ast
import functools
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import pysad

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = str(ROOT / "examples")
EXAMPLES = sorted(
    name
    for name in os.listdir(EXAMPLES_DIR)
    if name.startswith("example_") and name.endswith(".py")
)


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
    return files


@functools.cache
def _changed_files_since(base: str) -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", base],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return set(out.split())


def _should_run_example(rel_path: str, code: str) -> bool:
    base = os.environ.get("PYSAD_EXAMPLES_SINCE")
    if not base:
        return True
    changed = _changed_files_since(base)
    if {"tests/test_examples.py", ".github/workflows/examples.yml"} & changed:
        return True
    if "pyproject.toml" in changed:
        before = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{base}:pyproject.toml"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        after = (ROOT / "pyproject.toml").read_text()
        if before.split("\ntest = [", 1)[0] != after.split("\ntest = [", 1)[0]:
            return True
    return rel_path in changed or bool(_example_pysad_files(code) & changed)


def run_python(*args):
    """Runs the Python interpreter from the examples directory, with a non-interactive plot backend."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, *args],
        cwd=EXAMPLES_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


@pytest.mark.examples
@pytest.mark.parametrize("example", EXAMPLES)
def test_example_runs(example):
    code = (ROOT / "examples" / example).read_text()
    if not _should_run_example(f"examples/{example}", code):
        pytest.skip(f"No changes affecting {example} since {os.environ['PYSAD_EXAMPLES_SINCE']}")

    result = run_python(example)

    assert result.returncode == 0, f"{example} failed:\n{result.stderr[-3000:]}"


def notebook_code(path):
    """Joins the code cells of a notebook, skipping shell and magic lines such as ``%pip install``."""
    with open(path) as f:
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

    result = run_python("-c", code)

    assert result.returncode == 0, f"quickstart.ipynb failed:\n{result.stderr[-3000:]}"


@pytest.mark.examples
def test_nyc_taxi_alerts_notebook_runs():
    code = notebook_code(os.path.join(EXAMPLES_DIR, "nyc_taxi_alerts.ipynb"))
    if not _should_run_example("examples/nyc_taxi_alerts.ipynb", code):
        pytest.skip(
            f"No changes affecting nyc_taxi_alerts.ipynb since {os.environ['PYSAD_EXAMPLES_SINCE']}"
        )

    result = run_python("-c", code)

    assert result.returncode == 0, f"nyc_taxi_alerts.ipynb failed:\n{result.stderr[-3000:]}"
