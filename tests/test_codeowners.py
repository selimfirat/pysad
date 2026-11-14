"""Static checks for the CODEOWNERS file that the branch ruleset review rule reads."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAINTAINER = "@selimfirat"
# GitHub reads the first CODEOWNERS file in this order and ignores the rest.
CODEOWNERS_PATHS = ("CODEOWNERS", ".github/CODEOWNERS", "docs/CODEOWNERS")


def _rules() -> list[str]:
    for relative_path in CODEOWNERS_PATHS:
        path = ROOT / relative_path
        if path.is_file():
            return [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
    raise AssertionError("Add a CODEOWNERS file at the repository root, in .github/, or in docs/.")


def test_every_path_has_the_maintainer_as_its_code_owner():
    """The last rule is a catch-all, so GitHub asks @selimfirat to review every pull request."""
    rules = _rules()
    assert rules[-1] == f"* {MAINTAINER}"


def test_every_rule_names_an_owner():
    """Each rule names at least one owner, so GitHub can request a review for the paths it matches."""
    for rule in _rules():
        assert len(rule.split()) > 1, f"The rule names no owner: {rule!r}"
