"""Tests for the MCP server tools."""

from __future__ import annotations

import pytest

try:
    from mcp.server.fastmcp import Context

    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover - mcp is an optional dependency
    MCP_AVAILABLE = False
    Context = None  # type: ignore[assignment,misc]

try:
    from pysad.mcp.server import (
        SessionState,
        create_mcp_server,
        get_status,
        initialize_calibrator,
        initialize_detector,
        initialize_stream,
        list_calibrators,
        list_data_sources,
        list_detectors,
        reset_session,
        score_batch,
        score_point,
    )
except ImportError:  # pragma: no cover - mcp is an optional dependency
    SessionState = None  # type: ignore[assignment,misc]
    create_mcp_server = None  # type: ignore[assignment,misc]
    get_status = None  # type: ignore[assignment,misc]
    initialize_calibrator = None  # type: ignore[assignment,misc]
    initialize_detector = None  # type: ignore[assignment,misc]
    initialize_stream = None  # type: ignore[assignment,misc]
    list_calibrators = None  # type: ignore[assignment,misc]
    list_data_sources = None  # type: ignore[assignment,misc]
    list_detectors = None  # type: ignore[assignment,misc]
    reset_session = None  # type: ignore[assignment,misc]
    score_batch = None  # type: ignore[assignment,misc]
    score_point = None  # type: ignore[assignment,misc]

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp package not installed"),
]


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests with asyncio only."""
    return "asyncio"


@pytest.fixture
def mock_context() -> Context:
    """Create a mock context with session state."""
    ctx = Context()
    ctx._pysad_state = SessionState()
    return ctx


async def test_server_registers_all_tools() -> None:
    """The server exposes session-based tools and auto-discovered tools."""
    server = create_mcp_server()
    tools = await server.list_tools()
    tool_names = {tool.name for tool in tools}

    # Session-based tools (backward compatible)
    session_tools = {
        "initialize_detector",
        "initialize_calibrator",
        "initialize_stream",
        "score_point",
        "score_batch",
        "get_status",
        "reset_session",
        "list_detectors",
        "list_calibrators",
        "list_data_sources",
    }

    # Check that all session tools are present
    assert session_tools.issubset(tool_names)

    # Check that auto-discovered detector tools are present (at least key ones)
    detector_tools = {name for name in tool_names if name.startswith("pysad_")}
    assert "pysad_loda" in detector_tools
    assert "pysad_halfspacetrees" in detector_tools
    assert "pysad_onlineisolationforest" in detector_tools

    # Check that auto-discovered statistic tools are present
    assert "pysad_adwinmeter" in detector_tools
    assert "pysad_ddmmeter" in detector_tools

    # Check that explain and stream tools are present
    assert "pysad_explain" in detector_tools
    assert "pysad_stream" in detector_tools


async def test_initialize_detector_returns_public_configuration(mock_context) -> None:
    """Detector initialization returns the selected class and parameters."""
    result = await initialize_detector(
        detector_name="loda",
        detector_kwargs='{"num_bins": 20}',
        random_state=42,
        ctx=mock_context,
    )

    assert result == {
        "success": True,
        "detector": "LODA",
        "parameters": {"num_bins": 20, "random_state": 42},
    }


async def test_initialize_detector_rejects_unknown_name(mock_context) -> None:
    """An unknown detector returns an error without raising an exception."""
    result = await initialize_detector(detector_name="unknown", ctx=mock_context)

    assert result["success"] is False
    assert "Unknown detector" in result["error"]


async def test_initialize_detector_reports_invalid_configuration(mock_context) -> None:
    """Invalid detector JSON returns an error without raising an exception."""
    result = await initialize_detector(detector_kwargs="not-json", ctx=mock_context)

    assert result["success"] is False
    assert result["error"]


async def test_initialize_calibrator_returns_public_configuration(mock_context) -> None:
    """Calibrator initialization returns the selected class and parameters."""
    result = await initialize_calibrator(
        calibrator_name="conformal",
        calibrator_kwargs="{}",
        window_size=50,
        ctx=mock_context,
    )

    assert result == {
        "success": True,
        "calibrator": "ConformalProbabilityCalibrator",
        "parameters": {"window_size": 50},
    }


async def test_initialize_calibrator_rejects_unknown_name(mock_context) -> None:
    """An unknown calibrator returns an error without raising an exception."""
    result = await initialize_calibrator(calibrator_name="unknown", ctx=mock_context)

    assert result["success"] is False
    assert "Unknown calibrator" in result["error"]


async def test_initialize_calibrator_reports_invalid_configuration(mock_context) -> None:
    """Invalid calibrator JSON returns an error without raising an exception."""
    result = await initialize_calibrator(calibrator_kwargs="not-json", ctx=mock_context)

    assert result["success"] is False
    assert result["error"]


async def test_initialize_synthetic_stream(mock_context) -> None:
    """Synthetic stream initialization returns its size and feature count."""
    result = await initialize_stream(source="synthetic", max_points=300, ctx=mock_context)

    assert result["success"] is True
    assert result["num_points"] == 300
    assert result["num_features"] == 2
    assert result["has_labels"] is True


async def test_initialize_synthetic_stream_keeps_minimum_size(mock_context) -> None:
    """Synthetic streams always contain at least 200 points."""
    result = await initialize_stream(source="synthetic", max_points=10, ctx=mock_context)

    assert result["success"] is True
    assert result["num_points"] == 200


async def test_initialize_csv_stream(mock_context, tmp_path) -> None:
    """CSV stream initialization reads selected feature columns."""
    path = tmp_path / "stream.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    result = await initialize_stream(
        source=str(path),
        max_points=1,
        feature_columns="0,1",
        ctx=mock_context,
    )

    assert result["success"] is True
    assert result["num_points"] == 1
    assert result["num_features"] == 2


async def test_initialize_stream_rejects_unknown_source(mock_context) -> None:
    """An unknown data source returns an error without raising an exception."""
    result = await initialize_stream(source="unknown", ctx=mock_context)

    assert result["success"] is False
    assert "Unknown data source" in result["error"]


async def test_score_point_requires_a_detector(mock_context) -> None:
    """Point scoring fails when no detector is active."""
    result = await score_point(features=[1.0, 2.0], ctx=mock_context)

    assert result["success"] is False
    assert "Detector not initialized" in result["error"]


async def test_score_point_returns_score_and_advances_session(mock_context) -> None:
    """Point scoring returns the score and advances the stream position."""
    await initialize_detector(ctx=mock_context)

    result = await score_point(features=[1.0, 2.0], ctx=mock_context)

    assert result["success"] is True
    assert isinstance(result["score"], float)
    assert result["features"] == [1.0, 2.0]
    assert result["step"] == 0
    assert (await get_status(ctx=mock_context))["step"] == 1


async def test_score_point_requires_features_without_a_stream(mock_context) -> None:
    """Point scoring fails when no features or stream are available."""
    await initialize_detector(ctx=mock_context)

    result = await score_point(ctx=mock_context)

    assert result["success"] is False
    assert "Features must be provided" in result["error"]


async def test_score_point_applies_calibration_and_alert_threshold(mock_context) -> None:
    """Point scoring returns a calibrated score and alert decision."""
    await initialize_detector(ctx=mock_context)
    await initialize_calibrator(ctx=mock_context)

    result = await score_point(features=[1.0, 2.0], threshold=0.5, ctx=mock_context)

    assert result["success"] is True
    assert isinstance(result["calibrated_score"], float)
    assert isinstance(result["alert"], bool)
    assert result["threshold"] == 0.5


async def test_score_point_consumes_initialized_stream(mock_context) -> None:
    """Point scoring uses the next point from an initialized stream."""
    await initialize_detector(ctx=mock_context)
    await initialize_stream(source="synthetic", max_points=200, ctx=mock_context)

    first = await score_point(use_stream=True, ctx=mock_context)
    second = await score_point(use_stream=True, ctx=mock_context)

    assert first["success"] is True
    assert second["success"] is True
    assert first["step"] == 0
    assert second["step"] == 1


async def test_score_point_reports_exhausted_stream(mock_context) -> None:
    """Point scoring fails after the initialized stream is exhausted."""
    await initialize_detector(ctx=mock_context)
    await initialize_stream(source="synthetic", max_points=200, ctx=mock_context)
    for _ in range(200):
        await score_point(use_stream=True, ctx=mock_context)

    result = await score_point(use_stream=True, ctx=mock_context)

    assert result["success"] is False
    assert "No more data in stream" in result["error"]


async def test_score_batch_requires_a_detector(mock_context) -> None:
    """Batch scoring fails when no detector is active."""
    result = await score_batch(features_list=[[1.0, 2.0]], ctx=mock_context)

    assert result["success"] is False
    assert "Detector not initialized" in result["error"]


async def test_score_batch_returns_one_result_per_point(mock_context) -> None:
    """Batch scoring returns scores for every input point."""
    await initialize_detector(ctx=mock_context)

    result = await score_batch(features_list=[[1.0, 2.0], [3.0, 4.0]], ctx=mock_context)

    assert result["success"] is True
    assert len(result["results"]) == 2
    assert all(isinstance(item["score"], float) for item in result["results"])
    assert [item["step"] for item in result["results"]] == [0, 1]


async def test_score_batch_applies_calibration(mock_context) -> None:
    """Batch scoring returns calibrated scores and alert decisions."""
    await initialize_detector(ctx=mock_context)
    await initialize_calibrator(ctx=mock_context)

    result = await score_batch(features_list=[[1.0, 2.0]], threshold=0.5, ctx=mock_context)

    assert result["success"] is True
    assert isinstance(result["results"][0]["calibrated_score"], float)
    assert isinstance(result["results"][0]["alert"], bool)
    assert result["threshold"] == 0.5


async def test_get_status_reports_session_state(mock_context) -> None:
    """Status reports the active detector, calibrator, and stream position."""
    await initialize_detector(ctx=mock_context)
    await initialize_calibrator(ctx=mock_context)
    await initialize_stream(source="synthetic", max_points=200, ctx=mock_context)

    result = await get_status(ctx=mock_context)

    assert result["success"] is True
    assert result["initialized"] is True
    assert result["detector"] == "LODA"
    assert result["calibrator"] == "ConformalProbabilityCalibrator"
    assert result["stream_length"] == 200
    assert result["stream_position"] == 0
    assert result["has_stream"] is True


async def test_reset_session_clears_active_components(mock_context) -> None:
    """Reset removes the detector, calibrator, and stream."""
    await initialize_detector(ctx=mock_context)
    await initialize_calibrator(ctx=mock_context)
    await initialize_stream(source="synthetic", max_points=200, ctx=mock_context)
    await score_point(features=[1.0, 2.0], ctx=mock_context)

    result = await reset_session(ctx=mock_context)
    status = await get_status(ctx=mock_context)

    assert result["success"] is True
    assert status["initialized"] is False
    assert status["stream_length"] == 0
    assert status["stream_position"] == 0


async def test_list_detectors_returns_names_and_descriptions(mock_context) -> None:
    """The detector list includes the documented detector choices."""
    result = await list_detectors(ctx=mock_context)

    assert result["success"] is True
    names = {item["name"] for item in result["detectors"]}
    assert {"loda", "half_space_trees", "rs_hash"} <= names
    assert all(item["description"] for item in result["detectors"])


async def test_list_calibrators_returns_names_and_descriptions(mock_context) -> None:
    """The calibrator list includes the documented calibrator choices."""
    result = await list_calibrators(ctx=mock_context)

    assert result["success"] is True
    names = {item["name"] for item in result["calibrators"]}
    assert {"conformal", "spot", "gaussian_tail"} <= names
    assert all(item["description"] for item in result["calibrators"])


async def test_list_data_sources_returns_names_and_descriptions(mock_context) -> None:
    """The data source list includes the documented source choices."""
    result = await list_data_sources(ctx=mock_context)

    assert result["success"] is True
    names = {item["name"] for item in result["sources"]}
    assert {"synthetic", "nyc_taxi", "arrhythmia", "creditcard", "csv_file"} <= names
    assert all(item["description"] for item in result["sources"])


async def test_session_state_isolation() -> None:
    """Different contexts have independent session state."""
    ctx1 = Context()
    ctx1._pysad_state = SessionState()  # type: ignore[attr-defined]
    ctx2 = Context()
    ctx2._pysad_state = SessionState()  # type: ignore[attr-defined]

    await initialize_detector(detector_name="loda", ctx=ctx1)
    await initialize_detector(
        detector_name="half_space_trees",
        detector_kwargs='{"feature_mins": [0, 0], "feature_maxes": [1, 1]}',
        ctx=ctx2,
    )

    status1 = await get_status(ctx=ctx1)
    status2 = await get_status(ctx=ctx2)

    assert status1["detector"] == "LODA"
    assert status2["detector"] == "HalfSpaceTrees"
