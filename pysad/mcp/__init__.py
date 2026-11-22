"""
The :mod:`pysad.mcp` module provides an MCP (Model Context Protocol) server
for streaming anomaly detection, exposing PySAD capabilities as tools for
LLM clients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

mcp_server: FastMCP | None

try:
    from .server import mcp_server
except ImportError:  # pragma: no cover - mcp is an optional dependency
    mcp_server = None

__all__ = ["mcp_server"]
