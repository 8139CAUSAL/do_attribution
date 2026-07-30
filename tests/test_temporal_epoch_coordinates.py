"""Representative epoch-coordinate validation for CAL-036."""

from datetime import date

import pytest

from do_attribution.calibration import _normalize_epoch_coordinates


def _windows():
    return {
        "E1": (date(2026, 1, 1), date(2026, 2, 1)),
        "E2": (date(2026, 2, 1), date(2026, 3, 1)),
    }


def test_coordinates_are_optional_and_never_inferred_from_windows() -> None:
    assert _normalize_epoch_coordinates(
        None,
        epochs=("E1", "E2"),
        epoch_windows=_windows(),
    ) is None


def test_coordinates_follow_epoch_order_and_are_immutable() -> None:
    source = {
        "E2": date(2026, 3, 1),
        "E1": date(2026, 1, 15),
    }
    normalized = _normalize_epoch_coordinates(
        source,
        epochs=("E1", "E2"),
        epoch_windows=_windows(),
        required_epochs=("E2", "E1"),
    )
    source["E1"] = date(2030, 1, 1)

    assert normalized is not None
    assert tuple(normalized) == ("E1", "E2")
    assert normalized["E1"] == date(2026, 1, 15)
    assert normalized["E2"] == date(2026, 3, 1)
    with pytest.raises(TypeError):
        normalized["E1"] = date(2026, 1, 20)


def test_required_affected_epoch_coverage_is_explicit() -> None:
    with pytest.raises(ValueError, match=r"missing=\['E2'\]"):
        _normalize_epoch_coordinates(
            {"E1": date(2026, 1, 15)},
            epochs=("E1", "E2"),
            epoch_windows=_windows(),
            required_epochs=("E1", "E2"),
        )
    with pytest.raises(ValueError, match=r"missing=\['E1'\]"):
        _normalize_epoch_coordinates(
            None,
            epochs=("E1", "E2"),
            epoch_windows=_windows(),
            required_epochs=("E1",),
        )


@pytest.mark.parametrize(
    "coordinates",
    [
        {"E1": date(2026, 2, 1)},
        {"E2": date(2026, 1, 15)},
        {"E1": date(2025, 12, 31)},
        {"E1": 0.5},
        {"E1": "2026-01-15"},
    ],
)
def test_coordinates_must_use_window_type_and_lie_in_their_epoch(coordinates) -> None:
    with pytest.raises((TypeError, ValueError), match="coordinate|inside"):
        _normalize_epoch_coordinates(
            coordinates,
            epochs=("E1", "E2"),
            epoch_windows=_windows(),
        )


def test_coordinates_require_windows_and_known_epochs() -> None:
    with pytest.raises(ValueError, match="epoch_windows are required"):
        _normalize_epoch_coordinates(
            {"E1": 0.5},
            epochs=("E1",),
            epoch_windows=None,
        )
    with pytest.raises(ValueError, match="invalid epoch coverage"):
        _normalize_epoch_coordinates(
            {"E1": 0.5, "E2": 1.5},
            epochs=("E1",),
            epoch_windows={"E1": (0.0, 1.0)},
        )
    with pytest.raises(ValueError, match="unknown=.*'E2'"):
        _normalize_epoch_coordinates(
            {"E1": 0.5},
            epochs=("E1",),
            epoch_windows={"E1": (0.0, 1.0)},
            required_epochs=("E2",),
        )
