"""Validate release automation wiring before it runs on a real release."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "pysad"
PYPI_PROJECT_URL = f"https://pypi.org/project/{PROJECT_NAME}/"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PROJECT_NAME}/json"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_pattern(text: str, pattern: str, message: str) -> None:
    require(re.search(pattern, text, re.MULTILINE) is not None, message)


def package_version() -> str:
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", read("pysad/version.py"), re.MULTILINE)
    require(match is not None, "Could not read __version__ from pysad/version.py.")
    return match.group(1)


def check_release_please() -> None:
    workflow = read(".github/workflows/release-please.yml")
    config = json.loads(read("release-please-config.json"))
    manifest = json.loads(read(".release-please-manifest.json"))

    require_pattern(
        workflow,
        r"^\s*token:\s*\$\{\{\s*secrets\.RELEASE_PLEASE_TOKEN\s*\}\}\s*$",
        "Release Please must use RELEASE_PLEASE_TOKEN so release events trigger Publish.",
    )
    require_pattern(
        workflow,
        r"^\s*target-branch:\s*develop\s*$",
        "Release Please must target develop.",
    )
    require(config["packages"]["."]["release-type"] == "python", "Release Please must use python releases.")
    require(
        config["packages"]["."]["changelog-path"] == "CHANGES.txt",
        "Release Please must update CHANGES.txt.",
    )
    require(
        "pysad/version.py" in config["packages"]["."]["extra-files"],
        "Release Please must update pysad/version.py.",
    )
    require(manifest["."] == package_version(), "Release Please manifest must match pysad/version.py.")


def check_publish_workflow() -> None:
    workflow = read(".github/workflows/publish.yml")

    require_pattern(workflow, r"^\s*release:\s*$", "Publish must trigger from GitHub release events.")
    require_pattern(workflow, r"^\s*-\s*published\s*$", "Publish must run when releases are published.")
    require_pattern(workflow, r"^\s*needs:\s*build\s*$", "Publish must reuse artifacts from the build job.")
    require_pattern(workflow, r"^\s*runs-on:\s*ubuntu-latest\s*$", "Publish must run on a GitHub Ubuntu runner.")
    require_pattern(workflow, r"^\s*id-token:\s*write\s*$", "Publish must grant OIDC id-token permission.")
    require_pattern(workflow, r"^\s*name:\s*pypi\s*$", "Publish must use the pypi environment.")
    require_pattern(
        workflow,
        rf"^\s*url:\s*{re.escape(PYPI_PROJECT_URL)}\s*$",
        "Publish environment URL must point at the PyPI project.",
    )
    require_pattern(
        workflow,
        r"^\s*uses:\s*pypa/gh-action-pypi-publish@release/v1\s*$",
        "Publish must use the PyPA publish action.",
    )
    require("username:" not in workflow and "password:" not in workflow, "Trusted publishing must not use credentials.")
    require("repository-url:" not in workflow, "PyPI publish should use the default PyPI API endpoint.")


def check_pypi_project() -> None:
    with urllib.request.urlopen(PYPI_JSON_URL, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    require(payload["info"]["name"].lower() == PROJECT_NAME, "PyPI JSON API returned the wrong project.")
    require(payload["info"]["project_url"] == PYPI_PROJECT_URL, "PyPI project URL does not match workflow config.")
    require(payload["info"]["package_url"] == PYPI_PROJECT_URL, "PyPI package URL does not match workflow config.")


def main() -> int:
    try:
        check_release_please()
        check_publish_workflow()
        check_pypi_project()
    except urllib.error.URLError as exc:
        print(f"PyPI API check failed: {exc}", file=sys.stderr)
        return 1
    except AssertionError as exc:
        print(f"Release workflow check failed: {exc}", file=sys.stderr)
        return 1

    print("Release workflow and PyPI API checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
