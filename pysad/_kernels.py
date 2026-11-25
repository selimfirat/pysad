"""Load optional PyO3 kernels for streaming detectors."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import cast

import numpy as np
from numpy.typing import NDArray

HalfSpaceTreesPathFn = Callable[
    [NDArray[np.float64], NDArray[np.uint16], NDArray[np.float64], int],
    NDArray[np.int64],
]
RRCFRouteToLeafFn = Callable[
    [
        NDArray[np.float64],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.float64],
        NDArray[np.int32],
        NDArray[np.int32],
    ],
    NDArray[np.int32],
]
RRCFCodispSumFn = Callable[
    [
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
    ],
    float,
]
LODADensitiesFn = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int],
    NDArray[np.float64],
]
StreamKMLloydFn = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int],
    NDArray[np.float64],
]


def _load_module() -> ModuleType | None:
    try:
        return importlib.import_module("pysad_kernels")
    except ImportError:  # pragma: no cover - optional native extension
        return None


def _load_kernel(module: ModuleType | None, name: str) -> object | None:
    if module is None:
        return None
    return getattr(module, name, None)


_mod = _load_module()

half_space_trees_path = cast(
    HalfSpaceTreesPathFn | None,
    _load_kernel(_mod, "half_space_trees_path"),
)
rrcf_route_to_leaf = cast(
    RRCFRouteToLeafFn | None,
    _load_kernel(_mod, "rrcf_route_to_leaf"),
)
rrcf_codisp_sum = cast(
    RRCFCodispSumFn | None,
    _load_kernel(_mod, "rrcf_codisp_sum"),
)
loda_neg_log_densities = cast(
    LODADensitiesFn | None,
    _load_kernel(_mod, "loda_neg_log_densities"),
)
streamkm_lloyd = cast(
    StreamKMLloydFn | None,
    _load_kernel(_mod, "streamkm_lloyd"),
)
