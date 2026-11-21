"""Tests for the MCP server tools."""

from __future__ import annotations

import pytest

from pysad.mcp.server import (
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

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests with asyncio only."""
    return "asyncio"


@pytest.fixture(autouse=True)
async def reset_server_state() -> None:
    """Start each test with an empty MCP session."""
    await reset_session()


async def test_server_registers_all_tools() -> None:
    """The server exposes every documented tool."""
    server = create_mcp_server()
    tools = await server.list_tools()

    assert {tool.name for tool in tools} == {
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


async def test_initialize_detector_returns_public_configuration() -> None:
    """Detector initialization returns the selected class and parameters."""
    result = await initialize_detector(
        detector_name="loda",
        detector_kwargs='{"num_bins": 20}',
        random_state=42,
    )

    assert result == {
        "success": True,
        "detector": "LODA",
        "parameters": {"num_bins": 20, "random_state": 42},
    }


async def test_initialize_detector_rejects_unknown_name() -> None:
    """An unknown detector returns an error without raising an exception."""
    result = await initialize_detector(detector_name="unknown")

    assert result["success"] is False
    assert "Unknown detector" in result["error"]


async def test_initialize_detector_reports_invalid_configuration() -> None:
    """Invalid detector JSON returns an error without raising an exception."""
    result = await initialize_detector(detector_kwargs="not-json")

    assert result["success"] is False
    assert result["error"]


async def test_initialize_calibrator_returns_public_configuration() -> None:
    """Calibrator initialization returns the selected class and parameters."""
    result = await initialize_calibrator(
        calibrator_name="conformal",
        calibrator_kwargs="{}",
        window_size=50,
    )

    assert result == {
        "success": True,
        "calibrator": "ConformalProbabilityCalibrator",
        "parameters": {"window_size": 50},
    }


async def test_initialize_calibrator_rejects_unknown_name() -> None:
    """An unknown calibrator returns an error without raising an exception."""
    result = await initialize_calibrator(calibrator_name="unknown")

    assert result["success"] is False
    assert "Unknown calibrator" in result["error"]


async def test_initialize_calibrator_reports_invalid_configuration() -> None:
    """Invalid calibrator JSON returns an error without raising an exception."""
    result = await initialize_calibrator(calibrator_kwargs="not-json")

    assert result["success"] is False
    assert result["error"]


async def test_initialize_synthetic_stream() -> None:
    """Synthetic stream initialization returns its size and feature count."""
    result = await initialize_stream(source="synthetic", max_points=300)

    assert result["success"] is True
    assert result["num_points"] == 300
    assert result["num_features"] == 2
    assert result["has_labels"] is True


async def test_initialize_synthetic_stream_keeps_minimum_size() -> None:
    """Synthetic streams always contain at least 200 points."""
    result = await initialize_stream(source="synthetic", max_points=10)

    assert result["success"] is True
    assert result["num_points"] == 200


async def test_initialize_csv_stream(tmp_path) -> None:
    """CSV stream initialization reads selected feature columns."""
    path = tmp_path / "stream.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    result = await initialize_stream(
        source=str(path),
        max_points=1,
        feature_columns="0,1",
    )

    assert result["success"] is True
    assert result["num_points"] == 1
    assert result["num_features"] == 2


async def test_initialize_stream_rejects_unknown_source() -> None:
    """An unknown data source returns an error without raising an exception."""
    result = await initialize_stream(source="unknown")

    assert result["success"] is False
    assert "Unknown data source" in result["error"]


async def test_score_point_requires_a_detector() -> None:
    """Point scoring fails when no detector is active."""
    result = await score_point(features=[1.0, 2.0])

    assert result["success"] is False
    assert "Detector not initialized" in result["error"]


async def test_score_point_returns_score_and_advances_session() -> None:
    """Point scoring returns the score and advances the stream position."""
    await initialize_detector()

    result = await score_point(features=[1.0, 2.0])

    assert result["success"] is True
    assert isinstance(result["score"], float)
    assert result["features"] == [1.0, 2.0]
    assert result["step"] == 0
    assert (await get_status())["step"] == 1


async def test_score_point_requires_features_without_a_stream() -> None:
    """Point scoring fails when no features or stream are available."""
    await initialize_detector()

    result = await score_point()

    assert result["success"] is False
    assert "Features must be provided" in result["error"]


async def test_score_point_applies_calibration_and_alert_threshold() -> None:
    """Point scoring returns a calibrated score and alert decision."""
    await initialize_detector()
    await initialize_calibrator()

    result = await score_point(features=[1.0, 2.0], threshold=0.5)

    assert result["success"] is True
    assert isinstance(result["calibrated_score"], float)
    assert isinstance(result["alert"], bool)
    assert result["threshold"] == 0.5


async def test_score_point_consumes_initialized_stream() -> None:
    """Point scoring uses the next point from an initialized stream."""
    await initialize_detector()
    await initialize_stream(source="synthetic", max_points=200)

    first = await score_point(use_stream=True)
    second = await score_point(use_stream=True)

    assert first["success"] is True
    assert second["success"] is True
    assert first["step"] == 0
    assert second["step"] == 1


async def test_score_point_reports_exhausted_stream() -> None:
    """Point scoring fails after the initialized stream is exhausted."""
    await initialize_detector()
    await initialize_stream(source="synthetic", max_points=200)
    for _ in range(200):
        await score_point(use_stream=True)

    result = await score_point(use_stream=True)

    assert result["success"] is False
    assert "No more data in stream" in result["error"]


async def test_score_batch_requires_a_detector() -> None:
    """Batch scoring fails when no detector is active."""
    result = await score_batch(features_list=[[1.0, 2.0]])

    assert result["success"] is False
    assert "Detector not initialized" in result["error"]


async def test_score_batch_returns_one_result_per_point() -> None:
    """Batch scoring returns scores for every input point."""
    await initialize_detector()

    result = await score_batch(features_list=[[1.0, 2.0], [3.0, 4.0]])

    assert result["success"] is True
    assert len(result["results"]) == 2
    assert all(isinstance(item["score"], float) for item in result["results"])
    assert [item["step"] for item in result["results"]] == [0, 1]


async def test_score_batch_applies_calibration() -> None:
    """Batch scoring returns calibrated scores and alert decisions."""
    await initialize_detector()
    await initialize_calibrator()

    result = await score_batch(features_list=[[1.0, 2.0]], threshold=0.5)

    assert result["success"] is True
    assert isinstance(result["results"][0]["calibrated_score"], float)
    assert isinstance(result["results"][0]["alert"], bool)
    assert result["threshold"] == 0.5


async def test_get_status_reports_session_state() -> None:
    """Status reports the active detector, calibrator, and stream position."""
    await initialize_detector()
    await initialize_calibrator()
    await initialize_stream(source="synthetic", max_points=200)

    result = await get_status()

    assert result["success"] is True
    assert result["initialized"] is True
    assert result["detector"] == "LODA"
    assert result["calibrator"] == "ConformalProbabilityCalibrator"
    assert result["stream_length"] == 200
    assert result["stream_position"] == 0
    assert result["has_stream"] is True


async def test_reset_session_clears_active_components() -> None:
    """Reset removes the detector, calibrator, and stream."""
    await initialize_detector()
    await initialize_calibrator()
    await initialize_stream(source="synthetic", max_points=200)
    await score_point(features=[1.0, 2.0])

    result = await reset_session()
    status = await get_status()

    assert result["success"] is True
    assert status["initialized"] is False
    assert status["stream_length"] == 0
    assert status["stream_position"] == 0


async def test_list_detectors_returns_names_and_descriptions() -> None:
    """The detector list includes the documented detector choices."""
    result = await list_detectors()

    assert result["success"] is True
    names = {item["name"] for item in result["detectors"]}
    assert {"loda", "half_space_trees", "rs_hash"} <= names
    assert all(item["description"] for item in result["detectors"])


async def test_list_calibrators_returns_names_and_descriptions() -> None:
    """The calibrator list includes the documented calibrator choices."""
    result = await list_calibrators()

    assert result["success"] is True
    names = {item["name"] for item in result["calibrators"]}
    assert {"conformal", "spot", "gaussian_tail"} <= names
    assert all(item["description"] for item in result["calibrators"])


async def test_list_data_sources_returns_names_and_descriptions() -> None:
    """The data source list includes the documented source choices."""
    result = await list_data_sources()

    assert result["success"] is True
    names = {item["name"] for item in result["sources"]}
    assert {"synthetic", "nyc_taxi", "arrhythmia", "creditcard", "csv_file"} <= names
    assert all(item["description"] for item in result["sources"])
