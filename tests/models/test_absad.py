from __future__ import annotations

import pytest

from pysad.models.absad import ABSAD


def test_absad_default_initialization() -> None:
    model = ABSAD()
    assert model.window_size == 750
    assert model.num_neighbors == 100
    assert model.num_references == 100
    assert model.theta == 0.4
    assert model.confidence == 0.99
    assert model.control_limit_ is None
    assert len(model._window) == 0


def test_absad_custom_initialization() -> None:
    model = ABSAD(
        window_size=200,
        num_neighbors=30,
        num_references=20,
        theta=0.5,
        confidence=0.95,
    )
    assert model.window_size == 200
    assert model.num_neighbors == 30
    assert model.num_references == 20
    assert model.theta == 0.5
    assert model.confidence == 0.95


@pytest.mark.parametrize(
    ("kwargs", "expected_msg"),
    [
        ({"window_size": 0}, "window_size must be positive"),
        ({"window_size": -10}, "window_size must be positive"),
        ({"window_size": 100, "num_neighbors": 0}, "num_neighbors must be between 1 and window_size"),
        ({"window_size": 100, "num_neighbors": 101}, "num_neighbors must be between 1 and window_size"),
        ({"window_size": 100, "num_neighbors": 50, "num_references": 0}, "num_references must be between 1 and num_neighbors"),
        ({"window_size": 100, "num_neighbors": 50, "num_references": 51}, "num_references must be between 1 and num_neighbors"),
        ({"theta": -0.1}, "theta must be non-negative"),
        ({"confidence": 0.0}, r"confidence must be in \(0, 1\)"),
        ({"confidence": 1.0}, r"confidence must be in \(0, 1\)"),
        ({"confidence": 1.5}, r"confidence must be in \(0, 1\)"),
    ],
)
def test_absad_invalid_parameters(kwargs: dict[str, float | int], expected_msg: str) -> None:
    with pytest.raises(ValueError, match=expected_msg):
        ABSAD(**kwargs)
