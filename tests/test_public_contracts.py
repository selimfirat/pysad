"""Fast static checks for public API documentation, citations, and benchmark registration."""

import ast
import re
import sys
from pathlib import Path

import pytest

import pysad.models
from tests.test_dimensionality import load_benchmark

ROOT = Path(__file__).resolve().parents[1]
for _extra_dir in (ROOT / "docs", ROOT / "scripts"):
    if str(_extra_dir) not in sys.path:
        sys.path.insert(0, str(_extra_dir))

from _autodiscover import (  # noqa: E402
    NON_DETECTOR_MODELS,
    collect_bibtex,
    doc_counts,
    expand_api_autosummary,
    expand_detectors_tables,
)
from sync_exports import sync_all  # noqa: E402

NON_BENCHMARK_MODELS = frozenset(
    {"MIDAS", "NullModel", "PerfectModel", "RandomModel", "StreamPipeline"}
)


def _package_exports(package: str) -> set[str]:
    init_py = ROOT.joinpath(*package.split("."), "__init__.py")
    tree = ast.parse(init_py.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return {name for name in ast.literal_eval(node.value) if not name.startswith("_")}
    return set()


def test_subpackage_exports_are_synced():
    """Every PySAD subpackage's imports and __all__ match scripts/sync_exports.py."""
    out_of_sync = sync_all(check=True)
    assert out_of_sync == [], f"Run `python scripts/sync_exports.py`: {out_of_sync}"


def test_public_exports_match_api_reference_pages():
    """Every public name in each documented subpackage's __all__ appears on its docs/api/ page."""
    documented_by_module: dict[str, set[str]] = {}
    for rst_path in sorted((ROOT / "docs" / "api").glob("*.rst")):
        text = expand_api_autosummary(rst_path.read_text(encoding="utf-8"))
        match = re.search(r"^\.\.\s+(?:current)?module::\s+(\S+)", text, flags=re.M)
        assert match is not None, f"Missing module directive in {rst_path}"
        module = match.group(1)
        names = set(re.findall(r"^\s{3}([A-Za-z_]\w*)\s*$", text, flags=re.M))
        names |= set(
            re.findall(r"^\.\.\s+auto(?:class|function)::\s+([A-Za-z_]\w*)", text, flags=re.M)
        )
        documented_by_module.setdefault(module, set()).update(names)

    for module, documented in documented_by_module.items():
        exports = _package_exports(module)
        assert exports == documented, (
            f"API docs mismatch for {module}: "
            f"undocumented={sorted(exports - documented)}, "
            f"not_in___all__={sorted(documented - exports)}"
        )


def test_docstring_citations_exist_in_bibliography(tmp_path):
    """Every :cite: key in pysad docstrings exists in docs/refs.bib or a __bibtex__ attribute."""
    _, bib_keys = collect_bibtex(ROOT)
    assert "angelov2014anomaly" in bib_keys

    cited_keys: set[str] = set()
    for py_path in (ROOT / "pysad").rglob("*.py"):
        if "kitnet_model" in py_path.parts:
            continue
        for group in re.findall(r":cite:`([^`]+)`", py_path.read_text(encoding="utf-8")):
            cited_keys.update(key.strip() for key in group.split(","))

    missing = sorted(cited_keys - bib_keys)
    assert not missing, f"Citations missing from docs/refs.bib or __bibtex__: {missing}"

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "refs.bib").write_text(
        "@article{dup2020, title={One}}\n", encoding="utf-8"
    )
    (tmp_path / "pysad").mkdir()
    (tmp_path / "pysad" / "bad.py").write_text(
        '__bibtex__ = "@article{dup2020, title={Two}}"\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Conflicting BibTeX"):
        collect_bibtex(tmp_path)


def test_detectors_rst_covers_all_streaming_detectors():
    """Expanding docs/detectors.rst lists every streaming detector in pysad.models.__all__."""
    expanded = expand_detectors_tables(
        (ROOT / "docs" / "detectors.rst").read_text(encoding="utf-8")
    )
    documented = set(re.findall(r":class:`~pysad\.models\.([A-Za-z_]\w*)`", expanded))
    expected = set(pysad.models.__all__) - NON_DETECTOR_MODELS
    assert documented - NON_DETECTOR_MODELS == expected

    counts = doc_counts(ROOT)
    assert counts["num_detectors"] == len(expected)
    assert counts["num_drift_streams"] > 0


def test_streaming_detectors_are_registered_in_benchmark():
    """Every streaming detector in pysad.models.__all__ is registered in benchmarks/benchmark.py."""
    benchmark = load_benchmark()
    benchmarked = set(benchmark.DETECTORS) | set(benchmark.UNIVARIATE)
    expected = set(pysad.models.__all__) - NON_BENCHMARK_MODELS
    assert expected == benchmarked


def test_model_pysad_files_isolates_detector_dependencies():
    """Incremental test selection maps each detector to its own module and shared core only."""
    from tests.models.conftest import _model_pysad_files

    sand_files = _model_pysad_files("SAND")
    assert "pysad/models/sand.py" in sand_files
    assert "pysad/core/base_model.py" in sand_files
    assert "pysad/models/loda.py" not in sand_files
    assert "pysad/models/exact_storm.py" not in sand_files
