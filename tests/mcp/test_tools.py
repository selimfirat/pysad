"""Tests for auto-discovered MCP tools in pysad.mcp.tools."""

from __future__ import annotations

import json
from typing import Any

import pytest

try:
    from mcp.server.mcpserver import MCPServer

    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - mcp is an optional dependency
    MCP_AVAILABLE = False
    MCPServer = None  # type: ignore[assignment,misc]

try:
    from pysad.mcp.tools import (
        register_all_tools,
        register_detector_tools,
        register_explainability_tools,
        register_statistic_tools,
        register_stream_tool,
    )
except ImportError:  # pragma: no cover - mcp is an optional dependency
    register_all_tools = None  # type: ignore[assignment,misc]
    register_detector_tools = None  # type: ignore[assignment,misc]
    register_explainability_tools = None  # type: ignore[assignment,misc]
    register_statistic_tools = None  # type: ignore[assignment,misc]
    register_stream_tool = None  # type: ignore[assignment,misc]

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp package not installed"),
]


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests with asyncio only."""
    return "asyncio"


def _extract_payload(raw_result: Any) -> dict[str, Any]:
    """Extract the JSON tool result dictionary returned by MCPServer.call_tool."""
    structured = getattr(raw_result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    content = getattr(raw_result, "content", None)
    if content is None and isinstance(raw_result, tuple):
        content = raw_result[0]
    if content is None:
        content = raw_result
    if isinstance(content, dict):
        return content
    text = getattr(content[0], "text", "{}")
    return json.loads(text)


async def test_register_individual_and_all_tools() -> None:
    """Detector, statistic, explainability, and stream registration functions populate the server."""
    det_server = MCPServer("det-test")
    register_detector_tools(det_server)
    det_names = {tool.name for tool in await det_server.list_tools()}
    assert "pysad_loda" in det_names
    assert "pysad_basemodel" not in det_names

    stat_server = MCPServer("stat-test")
    register_statistic_tools(stat_server)
    stat_names = {tool.name for tool in await stat_server.list_tools()}
    assert "pysad_averagemeter" in stat_names
    assert "pysad_adwinmeter" in stat_names

    expl_server = MCPServer("expl-test")
    register_explainability_tools(expl_server)
    assert "pysad_explain" in {tool.name for tool in await expl_server.list_tools()}

    stream_server = MCPServer("stream-test")
    register_stream_tool(stream_server)
    assert "pysad_stream" in {tool.name for tool in await stream_server.list_tools()}

    full_server = MCPServer("full-test")
    register_all_tools(full_server)
    full_names = {tool.name for tool in await full_server.list_tools()}
    assert {"pysad_loda", "pysad_averagemeter", "pysad_explain", "pysad_stream"}.issubset(
        full_names
    )


async def test_registered_detector_and_statistic_tools_execute() -> None:
    """Calling registered detector and statistic tools returns structured results."""
    server = MCPServer("tools-exec")
    register_detector_tools(server)
    register_statistic_tools(server)

    det_res = _extract_payload(
        await server.call_tool(
            "pysad_loda",
            {"data": [[0.0, 0.0], [0.1, -0.1], [5.0, 5.0]], "fit": True, "random_state": 0},
        )
    )
    assert det_res["detector"] == "LODA"
    assert det_res["num_instances"] == 3
    assert len(det_res["scores"]) == 3

    stat_res = _extract_payload(
        await server.call_tool("pysad_averagemeter", {"data": [1.0, 3.0, 5.0]})
    )
    assert stat_res["statistic"] == "AverageMeter"
    assert stat_res["final_value"] == pytest.approx(3.0)


async def test_explain_and_stream_tools_handle_success_and_errors() -> None:
    """Explainability and pipeline stream tools return attributions, alerts, and clean error payloads."""
    server = MCPServer("pipeline-exec")
    register_explainability_tools(server)
    register_stream_tool(server)

    expl_res = _extract_payload(
        await server.call_tool(
            "pysad_explain",
            {
                "detector_name": "OnlineIsolationForest",
                "data": [[0.0, 0.0], [0.2, 0.1], [4.0, 4.0]],
                "detector_kwargs": {"num_trees": 4, "window_size": 16, "random_state": 0},
            },
        )
    )
    assert expl_res["detector"] == "OnlineIsolationForest"
    assert expl_res["has_explain"] is True
    assert expl_res["num_instances"] == 3
    assert len(expl_res["explanations"][0]) == 2

    loda_expl = _extract_payload(
        await server.call_tool(
            "pysad_explain",
            {"detector_name": "LODA", "data": [1.0, 2.0]},
        )
    )
    assert loda_expl["has_explain"] is True
    assert loda_expl["num_instances"] == 1

    mad_expl = _extract_payload(
        await server.call_tool(
            "pysad_explain",
            {"detector_name": "MedianAbsoluteDeviation", "data": [1.0]},
        )
    )
    assert mad_expl["has_explain"] is False

    err_expl = _extract_payload(
        await server.call_tool(
            "pysad_explain",
            {"detector_name": "DoesNotExist", "data": [[1.0, 2.0]]},
        )
    )
    assert "error" in err_expl

    stream_res = _extract_payload(
        await server.call_tool(
            "pysad_stream",
            {
                "detector_name": "LODA",
                "calibrator_name": "ConformalProbabilityCalibrator",
                "data": [[0.0, 0.0], [0.1, 0.1], [8.0, 8.0]],
                "detector_kwargs": {"random_state": 0},
                "calibrator_kwargs": {"window_size": 10},
                "threshold": 0.9,
            },
        )
    )
    assert stream_res["detector"] == "LODA"
    assert stream_res["calibrator"] == "ConformalProbabilityCalibrator"
    assert len(stream_res["results"]) == 3
    assert "calibrated_score" in stream_res["results"][0]

    err_stream = _extract_payload(
        await server.call_tool(
            "pysad_stream",
            {"detector_name": "DoesNotExist", "data": [[1.0, 2.0]]},
        )
    )
    assert "error" in err_stream
