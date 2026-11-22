"""
The :mod:`pysad.mcp` module provides an MCP (Model Context Protocol) server
for streaming anomaly detection, exposing PySAD capabilities as tools for
LLM clients.
"""

try:
    from .server import mcp_server
except ImportError:  # pragma: no cover - mcp is an optional dependency
    mcp_server = None  # type: ignore[assignment]

__all__ = ["mcp_server"]
