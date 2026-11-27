"""Tests for the bundled agent skills and plugin."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("args", [[], ["--skills"]])
def test_bundle_path_without_site_packages(args):
    """The command locates usable resources without third-party packages."""
    result = subprocess.run(
        [sys.executable, "-S", "-m", "pysad.agent", *args],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        check=True,
    )
    path = Path(result.stdout.strip())
    assert path.is_absolute()
    if args:
        skills = sorted(path.glob("*/SKILL.md"))
        assert len(skills) == 2
        for skill in skills:
            metadata = skill.read_text().split("---", 2)[1]
            assert f"name: {skill.parent.name}\n" in metadata
            assert "description: " in metadata
    else:
        manifest = json.loads((path / ".claude-plugin" / "plugin.json").read_text())
        assert manifest["name"] == "pysad"
        assert (path / "skills").is_dir()
        servers = json.loads((path / ".mcp.json").read_text())["mcpServers"]
        assert servers["pysad"]["command"] == "pysad-mcp"
        assert servers["pysad"]["args"] == []


def test_agent_subpackage():
    """The package exposes the agent bundle through its public namespace."""
    import pysad
    import pysad.agent

    assert pysad.agent is not None
    assert "agent" in dir(pysad)
    assert "agent" in pysad.__all__
