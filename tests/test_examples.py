import os
import subprocess
import sys

import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
EXAMPLES = sorted(
    name for name in os.listdir(EXAMPLES_DIR) if name.startswith("example_") and name.endswith(".py")
)
OPTIONAL_DEPENDENCIES = {"example_usage_inqmad.py": "jax"}


@pytest.mark.examples
@pytest.mark.parametrize("example", EXAMPLES)
def test_example_runs(example):
    if example in OPTIONAL_DEPENDENCIES:
        pytest.importorskip(OPTIONAL_DEPENDENCIES[example])

    result = subprocess.run(
        [sys.executable, example],
        cwd=EXAMPLES_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    assert result.returncode == 0, f"{example} failed:\n{result.stderr[-3000:]}"
