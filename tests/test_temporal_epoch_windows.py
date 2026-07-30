"""Authoritative temporal epoch-window validation for CAL-034."""

from datetime import date, datetime

import pytest

from do_attribution.calibration import _normalize_authoritative_epoch_windows


def test_windows_follow_authoritative_epoch_order_and_are_immutable() -> None:
    windows = {
        "E2": (date(2026, 2, 1), date(2026, 3, 1)),
        "E1": (date(2026, 1, 1), date(2026, 2, 1)),
    }

    normalized = _normalize_authoritative_epoch_windows(
        windows,
        epochs=("E1", "E2"),
    )
    windows["E1"] = (date(2030, 1, 1), date(2030, 2, 1))

    assert normalized is not None
    assert tuple(normalized) == ("E1", "E2")
    assert normalized["E1"] == (date(2026, 1, 1), date(2026, 2, 1))
    with pytest.raises(TypeError):
        normalized["E3"] = (date(2026, 3, 1), date(2026, 4, 1))


def test_windows_are_optional_but_never_fabricated() -> None:
    assert (
        _normalize_authoritative_epoch_windows(None, epochs=("E1", "E2"))
        is None
    )


@pytest.mark.parametrize(
    "windows",
    [
        {"E1": (0.0, 2.0), "E2": (1.0, 3.0)},
        {"E1": (0.0, 0.0), "E2": (0.0, 1.0)},
        {"E1": (1.0, 2.0), "E2": (0.0, 1.0)},
    ],
)
def test_overlapping_empty_or_unordered_windows_raise(windows) -> None:
    with pytest.raises(ValueError, match="nonempty|ordered and nonoverlapping"):
        _normalize_authoritative_epoch_windows(windows, epochs=("E1", "E2"))


@pytest.mark.parametrize(
    "windows",
    [
        {"E1": (date(2026, 1, 1), datetime(2026, 2, 1))},
        {"E1": (date(2026, 1, 1), date(2026, 2, 1)), "E2": (2.0, 3.0)},
        {"E1": (False, True)},
    ],
)
def test_incompatible_window_coordinate_types_raise(windows) -> None:
    with pytest.raises(TypeError, match="compatible|finite number"):
        _normalize_authoritative_epoch_windows(windows, epochs=tuple(windows))


@pytest.mark.parametrize(
    "windows",
    [
        {"E1": (0.0, 1.0)},
        {"E1": (0.0, 1.0), "E2": (1.0, 2.0), "E3": (2.0, 3.0)},
    ],
)
def test_window_keys_must_exactly_match_the_authoritative_grid(windows) -> None:
    with pytest.raises(ValueError, match="invalid epoch coverage"):
        _normalize_authoritative_epoch_windows(windows, epochs=("E1", "E2"))
