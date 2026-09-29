import json
import os
import subprocess
import sys

import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
EXAMPLES = sorted(
    name
    for name in os.listdir(EXAMPLES_DIR)
    if name.startswith("example_") and name.endswith(".py")
)
OPTIONAL_DEPENDENCIES = {"example_usage_inqmad.py": "jax"}


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
    if example in OPTIONAL_DEPENDENCIES:
        pytest.importorskip(OPTIONAL_DEPENDENCIES[example])

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

    result = run_python("-c", code)

    assert result.returncode == 0, f"quickstart.ipynb failed:\n{result.stderr[-3000:]}"
