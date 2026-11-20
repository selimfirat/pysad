"""Validation tests for repository Context7 configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_KEYS = {
    "$schema",
    "projectTitle",
    "description",
    "branch",
    "folders",
    "excludeFolders",
    "excludeFiles",
    "includeExtensions",
    "rules",
    "disallow",
    "redirect",
    "previousVersions",
    "url",
    "public_key",
}


def test_context7_json_conforms_to_official_schema():
    """context7.json obeys schema fields, length bounds, filename rules, and existing folders."""
    config = json.loads((ROOT / "context7.json").read_text(encoding="utf-8"))

    assert set(config) <= ALLOWED_KEYS
    assert config["$schema"] == "https://context7.com/schema/context7.json"
    assert 1 <= len(config["projectTitle"]) <= 100
    assert 10 <= len(config["description"]) <= 500
    assert config["branch"] == "master"

    folders = config["folders"]
    assert 1 <= len(folders) <= 50
    assert len(folders) == len(set(folders))
    for folder in folders:
        assert (ROOT / folder).is_dir(), f"Missing documented folder in context7.json: {folder}"

    exclude_folders = config["excludeFolders"]
    assert len(exclude_folders) <= 50
    assert len(exclude_folders) == len(set(exclude_folders))

    exclude_files = config["excludeFiles"]
    assert len(exclude_files) <= 100
    assert len(exclude_files) == len(set(exclude_files))
    for filename in exclude_files:
        assert re.match(r"^[^/\\]+$", filename), f"excludeFiles must be basenames only: {filename}"

    rules = config["rules"]
    assert 1 <= len(rules) <= 50
    for rule in rules:
        assert 1 <= len(rule) <= 255, f"Rule exceeds 255-char Context7 schema limit: {rule!r}"
