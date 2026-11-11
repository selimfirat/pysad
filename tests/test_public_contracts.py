"""Fast static checks for public API documentation, citations, and benchmark registration."""

import ast
import re
from pathlib import Path

import pysad.models
from tests.test_dimensionality import load_benchmark

ROOT = Path(__file__).resolve().parents[1]
NON_BENCHMARK_MODELS = frozenset(
    {"MIDAS", "NullModel", "PerfectModel", "RandomModel", "StreamPipeline"}
)


def _package_exports(package: str) -> set[str]:
    init_py = ROOT.joinpath(*package.split("."), "__init__.py")
    tree = ast.parse(init_py.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return {name for name in ast.literal_eval(node.value) if not name.startswith("_")}
    return set()


def test_public_exports_match_api_reference_pages():
    """Every public name in each documented subpackage's __all__ appears on its docs/api/ page."""
    documented_by_module: dict[str, set[str]] = {}
    for rst_path in sorted((ROOT / "docs" / "api").glob("*.rst")):
        text = rst_path.read_text()
        match = re.search(r"^\.\.\s+(?:current)?module::\s+(\S+)", text, flags=re.M)
        assert match is not None, f"Missing module directive in {rst_path}"
        module = match.group(1)
        names = set(re.findall(r"^\s{3}([A-Za-z_]\w*)\s*$", text, flags=re.M))
        names |= set(re.findall(r"^\.\.\s+auto(?:class|function)::\s+([A-Za-z_]\w*)", text, flags=re.M))
        documented_by_module.setdefault(module, set()).update(names)

    for module, documented in documented_by_module.items():
        exports = _package_exports(module)
        assert exports == documented, (
            f"API docs mismatch for {module}: "
            f"undocumented={sorted(exports - documented)}, "
            f"not_in___all__={sorted(documented - exports)}"
        )


def test_docstring_citations_exist_in_bibliography():
    """Every :cite: key in pysad docstrings exists in docs/refs.bib."""
    bib_text = (ROOT / "docs" / "refs.bib").read_text()
    bib_keys = set(re.findall(r"^@\w+\{([^,\s]+),", bib_text, flags=re.M))

    cited_keys: set[str] = set()
    for py_path in (ROOT / "pysad").rglob("*.py"):
        if "kitnet_model" in py_path.parts:
            continue
        for group in re.findall(r":cite:`([^`]+)`", py_path.read_text()):
            cited_keys.update(key.strip() for key in group.split(","))

    missing = sorted(cited_keys - bib_keys)
    assert not missing, f"Citations missing from docs/refs.bib: {missing}"


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
