"""Within-epoch coordinate-weight validation for CAL-038."""

import math

import pytest

from do_attribution.calibration import _normalize_epoch_coordinate_weights


def _windows():
    return {"E1": (0.0, 1.0), "E2": (1.0, 2.0)}


def test_coordinate_weights_are_optional_but_required_for_affected_epochs() -> None:
    assert _normalize_epoch_coordinate_weights(
        None,
        epochs=("E1", "E2"),
        epoch_windows=_windows(),
    ) is None
    with pytest.raises(ValueError, match=r"missing=\['E1'\]"):
        _normalize_epoch_coordinate_weights(
            None,
            epochs=("E1", "E2"),
            epoch_windows=_windows(),
            required_epochs=("E1",),
        )


def test_coordinate_weights_preserve_epoch_and_pair_order_immutably() -> None:
    source = {
        "E2": [(1.8, 3), (2.0, 1)],
        "E1": [(0.8, 4), (0.2, 2)],
    }
    normalized = _normalize_epoch_coordinate_weights(
        source,
        epochs=("E1", "E2"),
        epoch_windows=_windows(),
        required_epochs=("E1", "E2"),
    )
    source["E1"].append((0.5, 100))

    assert normalized is not None
    assert tuple(normalized) == ("E1", "E2")
    assert normalized["E1"] == ((0.8, 4.0), (0.2, 2.0))
    assert normalized["E2"] == ((1.8, 3.0), (2.0, 1.0))
    with pytest.raises(TypeError):
        normalized["E1"] = ()


def test_duplicate_coordinates_raise_instead_of_hidden_aggregation() -> None:
    with pytest.raises(ValueError, match="duplicate key 0.5"):
        _normalize_epoch_coordinate_weights(
            {"E1": [(0.5, 1.0), (0.5, 2.0)]},
            epochs=("E1",),
            epoch_windows={"E1": (0.0, 1.0)},
        )


@pytest.mark.parametrize(
    "pairs",
    [
        [(0.5, -1.0)],
        [(0.5, 0.0)],
        [(0.5, math.nan)],
        [(0.5, math.inf)],
        [(0.5, True)],
    ],
)
def test_weights_must_be_finite_nonnegative_with_positive_total(pairs) -> None:
    with pytest.raises((TypeError, ValueError), match="epoch_coordinate_weights"):
        _normalize_epoch_coordinate_weights(
            {"E1": pairs},
            epochs=("E1",),
            epoch_windows={"E1": (0.0, 1.0)},
        )


@pytest.mark.parametrize(
    "pairs",
    [
        [(1.0, 1.0)],
        [(-0.1, 1.0)],
        [("0.5", 1.0)],
        [(True, 1.0)],
    ],
)
def test_weight_coordinates_must_match_and_lie_inside_their_epoch(pairs) -> None:
    with pytest.raises((TypeError, ValueError), match="coordinate|inside"):
        _normalize_epoch_coordinate_weights(
            {"E1": pairs},
            epochs=("E1", "E2"),
            epoch_windows=_windows(),
        )


def test_coordinate_weights_require_windows_known_keys_and_nonempty_pairs() -> None:
    with pytest.raises(ValueError, match="epoch_windows are required"):
        _normalize_epoch_coordinate_weights(
            {"E1": [(0.5, 1.0)]},
            epochs=("E1",),
            epoch_windows=None,
        )
    with pytest.raises(ValueError, match="invalid epoch coverage"):
        _normalize_epoch_coordinate_weights(
            {"E2": [(1.5, 1.0)]},
            epochs=("E1",),
            epoch_windows={"E1": (0.0, 1.0)},
        )
    with pytest.raises(ValueError, match="must not be empty"):
        _normalize_epoch_coordinate_weights(
            {"E1": []},
            epochs=("E1",),
            epoch_windows={"E1": (0.0, 1.0)},
        )


def test_scalar_support_cannot_satisfy_coordinate_weight_coverage() -> None:
    scalar_support = {"E1": 100.0}
    with pytest.raises(ValueError, match="epoch_coordinate_weights are required"):
        _normalize_epoch_coordinate_weights(
            None,
            epochs=tuple(scalar_support),
            epoch_windows={"E1": (0.0, 1.0)},
            required_epochs=("E1",),
        )
