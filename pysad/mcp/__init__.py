"""
The :mod:`pysad.mcp` module provides a Model Context Protocol (MCP) server
for streaming anomaly detection, exposing PySAD capabilities as tools for
LLM clients.
"""

from __future__ import annotations

from typing import Any

_server: Any
try:
    from . import server as _mod

    _server = _mod
except ImportError:  # pragma: no cover - mcp is an optional dependency
    _server = None

SessionState: Any = getattr(_server, "SessionState", None)
create_mcp_server: Any = getattr(_server, "create_mcp_server", None)
get_status: Any = getattr(_server, "get_status", None)
initialize_calibrator: Any = getattr(_server, "initialize_calibrator", None)
initialize_detector: Any = getattr(_server, "initialize_detector", None)
initialize_stream: Any = getattr(_server, "initialize_stream", None)
list_calibrators: Any = getattr(_server, "list_calibrators", None)
list_data_sources: Any = getattr(_server, "list_data_sources", None)
list_detectors: Any = getattr(_server, "list_detectors", None)
main: Any = getattr(_server, "main", None)
mcp_server: Any = getattr(_server, "mcp_server", None)
reset_session: Any = getattr(_server, "reset_session", None)
score_batch: Any = getattr(_server, "score_batch", None)
score_point: Any = getattr(_server, "score_point", None)

__all__ = [
    "SessionState",
    "create_mcp_server",
    "get_status",
    "initialize_calibrator",
    "initialize_detector",
    "initialize_stream",
    "list_calibrators",
    "list_data_sources",
    "list_detectors",
    "main",
    "mcp_server",
    "reset_session",
    "score_batch",
    "score_point",
]
