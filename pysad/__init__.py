"""
An open-source python framework for anomaly detection on streaming multivariate data.
"""

import importlib
from typing import Any

from .version import __version__

_SUBPACKAGES = frozenset(
    {"core", "evaluation", "explainability", "models", "statistics", "transform", "utils"}
)

__all__ = [
    "__version__",
    "core",
    "evaluation",
    "explainability",
    "models",
    "statistics",
    "transform",
    "utils",
]


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBPACKAGES)
