"""
The :mod:`pysad.mcp` module provides a Model Context Protocol (MCP) server
for streaming anomaly detection, exposing PySAD capabilities as tools for
LLM clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

mcp_server: MCPServer | None

try:
    from .server import (
        SessionState,
        create_mcp_server,
        get_status,
        initialize_calibrator,
        initialize_detector,
        initialize_stream,
        list_calibrators,
        list_data_sources,
        list_detectors,
        main,
        mcp_server,
        reset_session,
        score_batch,
        score_point,
    )
except ImportError:  # pragma: no cover - mcp is an optional dependency
    SessionState = None  # type: ignore[assignment, misc]
    create_mcp_server = None  # type: ignore[assignment]
    get_status = None  # type: ignore[assignment]
    initialize_calibrator = None  # type: ignore[assignment]
    initialize_detector = None  # type: ignore[assignment]
    initialize_stream = None  # type: ignore[assignment]
    list_calibrators = None  # type: ignore[assignment]
    list_data_sources = None  # type: ignore[assignment]
    list_detectors = None  # type: ignore[assignment]
    main = None  # type: ignore[assignment]
    mcp_server = None
    reset_session = None  # type: ignore[assignment]
    score_batch = None  # type: ignore[assignment]
    score_point = None  # type: ignore[assignment]

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
    "reset_session",
    "score_batch",
    "score_point",
]
