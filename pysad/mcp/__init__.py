"""
The :mod:`pysad.mcp` module provides an MCP (Model Context Protocol) server
for streaming anomaly detection, exposing PySAD capabilities as tools for
LLM clients.
"""

from .server import mcp_server

__all__ = ["mcp_server"]
