"""Global-signal temporal epoch recommendation.

The functions in this module deliberately sit *beside* the calendar-frequency
temporal API.  They recommend a global set of left-closed, right-open epochs;
the temporal attribution classes then dispatch one independent static fit for
each epoch.  No attribution state or path fragment is shared across a boundary.

``ruptures`` and ``statsmodels`` are optional dependencies.  They are imported
only while the corresponding recommendation operation is running so importing
the core package remains lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isnan, lgamma, log, pi, sqrt  # MODIFIED: + isnan
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from .markov import MarkovAttribution
from .reporting import (
    TemporalReportingSnapshot,
    build_epoch_metrics,
    render_temporal_report,
)
from .utils import DEFAULT_SEPARATOR, to_string_array


SignalBuilder = Callable[[Sequence[str], Sequence[int], Sequence[pd.Timestamp]], pd.Series]


# ``BaseCost`` is deliberately guarded.  A module-level fallback makes
# ``import do_attribution`` work without the temporal extra, while a process
# with ruptures installed still receives a genuine BaseCost subclass.
try:  # pragma: no cover - exercised in temporal-extra environments
    from ruptures.base import BaseCost as _RupturesBaseCost
except ImportError:  # pragma: no cover - exercised by optional-dependency test
    class _RupturesBaseCost:  # type: ignore[too-many-ancestors]
        pass


class TemporalRecommendationError(ValueError):
    """Base class for temporal-recommendation validation failures."""


class SignalShapeError(TemporalRecommendationError):
    """Raised when a custom scalar signal violates its data contract."""


class SignalContinuityError(TemporalRecommendationError):
    """Raised when a conversion-rate calendar gap exceeds the allowed limit."""


class SignalSufficiencyError(TemporalRecommendationError):
    """Raised when a detector signal has insufficient observations."""


class EpochWidthError(TemporalRecommendationError):
    """Raised when immutable boundaries make the width floor impossible."""


class EpochSufficiencyError(TemporalRecommendationError):
    """Raised when immutable boundaries make the sample floor impossible."""


class PinOverlapError(TemporalRecommendationError):
    """Raised when pin intervals overlap or lie outside the recommendation."""


class NonCalendarBoundaryError(TemporalRecommendationError):
    """Raised when arbitrary boundaries cannot be represented as ``freq``."""


@dataclass(frozen=True)
class DecompositionResult:
    """Audit record for calendar preparation and optional decomposition."""

    observed: pd.Series
    prepared: pd.Series
    seasonal: pd.Series | pd.DataFrame | None
    trend: pd.Series
    residual: pd.Series | None
    robust_weights: pd.Series | None
    method: Literal["none", "STL", "MSTL"]
    periods: tuple[int, ...]


@dataclass(frozen=True)
class SegmentFit:
    """One disconnected weighted affine fit over a final epoch."""

    start: pd.Timestamp
    end: pd.Timestamp
    midpoint_value: float
    slope: float
    weighted_rss: float
    n_observed: int
    n_paths: int


@dataclass(frozen=True)
class MergeAction:
    """A deterministic boundary removal made to satisfy a final floor."""

    affected_epoch: pd.Interval
    reason: Literal["min_epoch_width", "min_samples", "post_snap_conflict"]
    removed_boundary: pd.Timestamp
    competing_scores: tuple[tuple[pd.Timestamp, float], ...] = ()


@dataclass(frozen=True)
class PinAction:
    """An auditable pin expansion or data-driven-cut removal."""

    interval: pd.Interval
    action: str
    boundary: pd.Timestamp | None = None
    detail: str = ""


@dataclass(frozen=True)
class EpochRecommendation:
    """Immutable output of :func:`recommend_epochs`.

    ``signal`` is the raw conversion-rate series for scalar recommendations
    and the adjacent-Frobenius diagnostic for transition recommendations.
    ``detector_signal`` is always the data passed to PELT (a scalar Series or
    aligned transition-matrix DataFrame).
    """

    boundaries: pd.DatetimeIndex
    raw_breakpoints: pd.DatetimeIndex
    signal_name: str | Callable[..., Any]
    signal: pd.Series
    detector_signal: pd.Series | pd.DataFrame
    detector_weights: pd.Series | None
    decomposition: DecompositionResult | None
    segment_fits: tuple[SegmentFit, ...]
    boundary_scores: pd.DataFrame
    path_counts: pd.Series
    epoch_metrics: pd.DataFrame
    min_samples: int
    merge_log: tuple[MergeAction, ...]
    pin_log: tuple[PinAction, ...]
    warnings: tuple[str, ...]
    streaming: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reporting_snapshot: TemporalReportingSnapshot | None = None

    def summary(
        self,
        output_format: Literal["summary", "report"] = "summary",
    ) -> str:
        """Render stored recommendation diagnostics without fitting a model."""

        snapshot = self.reporting_snapshot
        if snapshot is None:
            if isinstance(self.signal_name, str):
                signal_name = self.signal_name
            else:
                signal_name = getattr(self.signal_name, "__name__", "custom_signal")
            source = (
                "streaming:conversion_rate"
                if self.streaming
                else f"recommended:{signal_name}"
            )
            snapshot = TemporalReportingSnapshot(
                model_name=None,
                epoch_source=source,
                epoch_labels=tuple(self.epoch_metrics["epoch_label"].astype(str)),
                boundaries=pd.DatetimeIndex(self.boundaries),
                epoch_metrics=self.epoch_metrics.copy(deep=True),
                attribution_matrix=None,
                sample_floor=int(self.min_samples),
            )
        return render_temporal_report(snapshot, output_format)

    def as_freq_mapping(self) -> dict[int, str]:
        """Return a calendar-frequency representation when it is exact.

        Arbitrary recommender knots cannot safely be converted into the old
        ``Mapping[int, str]`` API.  This method is intentionally conservative:
        it succeeds only when every boundary is generated by one supported
        calendar grid over the covered window.
        """

        bounds = pd.DatetimeIndex(self.boundaries)
        for freq, start_freq in (("D", "D"), ("W", "W-MON"), ("M", "MS"), ("Q", "QS"), ("Y", "YS")):
            expected = pd.date_range(bounds[0], bounds[-1], freq=start_freq)
            if expected.equals(bounds):
                years = range(bounds[0].year, max(bounds[0].year, (bounds[-1] - pd.Timedelta(days=1)).year) + 1)
                return {int(year): freq for year in years}
        raise NonCalendarBoundaryError(
            "Recommended boundaries are not an exact supported calendar grid; "
            "use fit_by_epoch or fit_recommended instead."
        )


def _temporal_import_error(package: str) -> ImportError:
    return ImportError(
        f"Temporal epoch recommendation requires optional dependency {package!r}. "
        "Install it with pip install \"do-attribution[temporal]\"."
    )


def _require_ruptures():
    try:
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _temporal_import_error("ruptures") from exc
    return rpt


def _require_statsmodels():
    try:
        from statsmodels.tsa.seasonal import MSTL, STL
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _temporal_import_error("statsmodels>=0.14") from exc
    return STL, MSTL


class WeightedAffineCost(_RupturesBaseCost):
    """Constant-time weighted residual cost for discontinuous affine pieces.

    The cost stores prefix sufficient statistics for ``w, wt, wt², wx,
    wtx, wx²``.  It is compatible with ``ruptures.Pelt(custom_cost=...)``
    and intentionally returns infinity for segments with fewer than three
    positive-weight observations or a rank-deficient time design.
    """

    model = "weighted_affine"
    min_size = 3

    def __init__(self, times: Sequence[float], weights: Sequence[float]) -> None:
        self.times = np.asarray(times, dtype=float).reshape(-1)
        self.weights = np.asarray(weights, dtype=float).reshape(-1)
        if self.times.size == 0 or self.times.size != self.weights.size:
            raise ValueError("times and weights must be non-empty arrays of equal length")
        if not np.isfinite(self.times).all() or not np.isfinite(self.weights).all():
            raise ValueError("times and weights must be finite")
        if (self.weights < 0).any():
            raise ValueError("weights must be non-negative")
        self.signal: np.ndarray | None = None
        self._prefix: dict[str, np.ndarray] = {}
        self._positive_prefix: np.ndarray | None = None

    def fit(self, signal: np.ndarray) -> "WeightedAffineCost":
        values = np.asarray(signal, dtype=float)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError("WeightedAffineCost requires a one-dimensional signal")
        if len(values) != len(self.times):
            raise ValueError("signal length must equal times and weights length")
        if not np.isfinite(values).all():
            raise ValueError("WeightedAffineCost signal must be finite")
        self.signal = values
        w, t, x = self.weights, self.times, values
        self._prefix = {
            "w": _prefix_sum(w),
            "wt": _prefix_sum(w * t),
            "wt2": _prefix_sum(w * t * t),
            "wx": _prefix_sum(w * x),
            "wtx": _prefix_sum(w * t * x),
            "wx2": _prefix_sum(w * x * x),
        }
        self._positive_prefix = _prefix_sum((w > 0).astype(float))
        return self

    def error(self, start: int, end: int) -> float:
        if self.signal is None:
            raise RuntimeError("WeightedAffineCost.fit must be called before error")
        if start < 0 or end > len(self.signal) or start >= end:
            raise ValueError(f"invalid segment [{start}, {end})")
        stats = self._stats(start, end)
        if stats["n_positive"] < self.min_size:
            return float("inf")
        denominator = stats["w"] * stats["wt2"] - stats["wt"] ** 2
        tolerance = np.finfo(float).eps * max(1.0, abs(stats["w"] * stats["wt2"]))
        if not np.isfinite(denominator) or denominator <= tolerance:
            return float("inf")
        explained = (
            stats["wx"] ** 2 * stats["wt2"]
            - 2.0 * stats["wx"] * stats["wtx"] * stats["wt"]
            + stats["wtx"] ** 2 * stats["w"]
        ) / denominator
        rss = stats["wx2"] - explained
        if rss < 0 and abs(rss) <= 1e-10 * max(1.0, stats["wx2"]):
            rss = 0.0
        return float(rss) if np.isfinite(rss) and rss >= 0 else float("inf")

    def segment_parameters(self, start: int, end: int) -> tuple[float, float, float, int]:
        """Return ``(intercept, slope, rss, n_positive)`` for a segment."""

        stats = self._stats(start, end)
        rss = self.error(start, end)
        denominator = stats["w"] * stats["wt2"] - stats["wt"] ** 2
        if not np.isfinite(rss) or denominator <= 0:
            return (float("nan"), float("nan"), rss, int(stats["n_positive"]))
        slope = (stats["w"] * stats["wtx"] - stats["wt"] * stats["wx"]) / denominator
        intercept = (stats["wx"] - slope * stats["wt"]) / stats["w"]
        return float(intercept), float(slope), float(rss), int(stats["n_positive"])

    def _stats(self, start: int, end: int) -> dict[str, float]:
        if self._positive_prefix is None:
            raise RuntimeError("WeightedAffineCost.fit must be called before statistics")
        stats = {name: _interval_sum(values, start, end) for name, values in self._prefix.items()}
        stats["n_positive"] = _interval_sum(self._positive_prefix, start, end)
        return stats


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=float)))


def _interval_sum(prefix: np.ndarray, start: int, end: int) -> float:
    return float(prefix[end] - prefix[start])


def _normalize_anchors(anchor_timestamps: Sequence[Any]) -> pd.DatetimeIndex:
    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime(anchor_timestamps))
    except (TypeError, ValueError) as exc:
        raise TemporalRecommendationError("anchor_timestamps must be datetime-like") from exc
    if len(timestamps) == 0:
        raise TemporalRecommendationError("anchor_timestamps is empty")
    if timestamps.isna().any():
        bad = int(np.flatnonzero(timestamps.isna())[0])
        raise TemporalRecommendationError(f"anchor_timestamps[{bad}] is null")
    # Calendar recommendations use local calendar days.  Convert tz-aware
    # anchors to their wall-clock date rather than comparing mixed tz offsets.
    if timestamps.tz is not None:
        timestamps = timestamps.tz_localize(None)
    return timestamps.normalize()


def _validate_inputs(
    paths: Sequence[str], conversions: Sequence[int], anchor_timestamps: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    paths_arr = to_string_array(paths)
    raw_conversions = list(conversions)
    if len(paths_arr) == 0:
        raise TemporalRecommendationError("paths is empty")
    if len(raw_conversions) != len(paths_arr):
        raise TemporalRecommendationError(
            f"conversions length ({len(raw_conversions)}) does not match paths length ({len(paths_arr)})"
        )
    if len(anchor_timestamps) != len(paths_arr):
        raise TemporalRecommendationError(
            f"anchor_timestamps length ({len(anchor_timestamps)}) does not match paths length ({len(paths_arr)})"
        )
    for index, value in enumerate(raw_conversions):
        try:
            valid = bool(value == 0) or bool(value == 1)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise TemporalRecommendationError(
                f"conversion at index {index} must be binary 0 or 1, got {value!r}"
            )
    return paths_arr, np.asarray(raw_conversions, dtype=int), _normalize_anchors(anchor_timestamps)


def _positive_timedelta(value: str | pd.Timedelta, parameter: str) -> pd.Timedelta:
    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise TemporalRecommendationError(f"{parameter} must be a valid positive timedelta") from exc
    if not pd.notna(delta) or delta <= pd.Timedelta(0):
        raise TemporalRecommendationError(f"{parameter} must be positive, got {value!r}")
    return delta


def _daily_conversion_rate(
    conversions: np.ndarray, anchors: pd.DatetimeIndex
) -> tuple[pd.Series, pd.Series, pd.Series]:
    calendar = pd.date_range(anchors.min(), anchors.max(), freq="D")
    counts = pd.Series(1, index=anchors).groupby(level=0).sum().reindex(calendar, fill_value=0).astype(int)
    converted = pd.Series(conversions, index=anchors).groupby(level=0).sum().reindex(calendar, fill_value=0).astype(int)
    rate = converted.div(counts.where(counts > 0)).astype(float)
    rate.name = "conversion_rate"
    counts.name = "path_count"
    converted.name = "conversion_count"
    return rate, counts, converted


def _prepare_missing_rate(
    observed: pd.Series, max_gap: pd.Timedelta
) -> tuple[pd.Series, pd.Series]:
    missing = observed.isna()
    if missing.any():
        positions = np.flatnonzero(missing.to_numpy())
        start = 0
        while start < len(positions):
            stop = start
            while stop + 1 < len(positions) and positions[stop + 1] == positions[stop] + 1:
                stop += 1
            length = stop - start + 1
            gap = pd.Timedelta(days=length)
            if gap > max_gap:
                left = observed.index[positions[start]]
                right = observed.index[positions[stop]]
                raise SignalContinuityError(
                    f"missing conversion-rate span [{left.date()}, {right.date()}] is {gap}, "
                    f"exceeding max_imputation_gap={max_gap}"
                )
            start = stop + 1
    prepared = observed.interpolate(method="time").ffill().bfill()
    prepared.name = observed.name
    return prepared, missing


def _decompose(
    observed: pd.Series,
    prepared: pd.Series,
    *,
    stl_period: int | tuple[int, ...] | Literal["auto"] | None,
    n_observed: int,
    notices: list[str],
) -> DecompositionResult:
    n = len(prepared)
    periods: tuple[int, ...]
    mode: Literal["none", "STL", "MSTL"] = "none"
    if stl_period is None:
        periods = ()
    elif stl_period == "auto":
        if n_observed < 14:
            notices.append("automatic decomposition disabled: fewer than two observed weekly cycles remain")
            periods = ()
        elif n < 14:
            periods = ()
        elif n <= 730:
            mode, periods = "STL", (7,)
        else:
            mode, periods = "MSTL", (7, 365)
    elif isinstance(stl_period, int) and not isinstance(stl_period, bool):
        if stl_period <= 1 or n < 2 * stl_period + 1:
            raise SignalSufficiencyError(
                f"stl_period={stl_period!r} requires at least {2 * stl_period + 1} regular observations; got {n}"
            )
        mode, periods = "STL", (int(stl_period),)
    elif isinstance(stl_period, tuple) and all(isinstance(p, int) and p > 1 for p in stl_period):
        periods = tuple(int(p) for p in stl_period)
        if not periods or any(2 * p >= n for p in periods):
            raise SignalSufficiencyError(
                f"MSTL periods {periods!r} must each be strictly less than half of {n} observations"
            )
        mode = "MSTL"
    else:
        raise TemporalRecommendationError("stl_period must be 'auto', None, an int, or a tuple of ints")

    if mode == "none":
        return DecompositionResult(observed, prepared, None, prepared.copy(), None, None, "none", ())

    STL, MSTL = _require_statsmodels()
    if mode == "STL":
        result = STL(prepared, period=periods[0], robust=True).fit()
        seasonal = pd.Series(result.seasonal, index=prepared.index, name="seasonal")
        trend = pd.Series(result.trend, index=prepared.index, name="trend")
        residual = pd.Series(result.resid, index=prepared.index, name="residual")
        robust_weights = pd.Series(result.weights, index=prepared.index, name="robust_weight")
    else:
        result = MSTL(prepared, periods=periods, stl_kwargs={"robust": True}).fit()
        seasonal_values = result.seasonal
        if isinstance(seasonal_values, pd.DataFrame):
            seasonal = seasonal_values.copy()
            seasonal.index = prepared.index
        else:
            seasonal = pd.Series(seasonal_values, index=prepared.index, name=f"seasonal_{periods[0]}")
        trend = pd.Series(result.trend, index=prepared.index, name="trend")
        residual = pd.Series(result.resid, index=prepared.index, name="residual")
        robust_weights = None
    return DecompositionResult(observed, prepared, seasonal, trend, residual, robust_weights, mode, periods)


def _seasonal_total(decomposition: DecompositionResult) -> pd.Series:
    seasonal = decomposition.seasonal
    if seasonal is None:
        return pd.Series(0.0, index=decomposition.prepared.index)
    if isinstance(seasonal, pd.DataFrame):
        return seasonal.sum(axis=1)
    return seasonal


def _is_exactly_affine(values: np.ndarray, times: np.ndarray, observed: np.ndarray) -> bool:
    if observed.sum() < 3:
        return False
    x, y = times[observed], values[observed]
    coeff = np.polyfit(x, y, 1)
    residual = y - np.polyval(coeff, x)
    return bool(np.allclose(residual, 0.0, rtol=1e-10, atol=1e-12 * max(1.0, np.ptp(y))))


def _autocorr_penalty_inflation(
    scale_source: pd.Series,
    observed_mask: pd.Series,
    notices: list[str],
) -> float:
    """Long-run variance correction for positively autocorrelated noise.

    MODIFIED: the second-difference MAD targets Var(D2)/6, which equals the
    marginal noise variance only under independence. For AR(1) noise with
    coefficient rho, Var(D2) = sigma^2 * (6 - 8*rho + 2*rho^2), while the
    residual-sum-of-squares reduction a spurious affine knot can harvest
    scales with the LONG-RUN variance sigma^2 * (1 + rho) / (1 - rho). The
    calibrated inflation on the sigma_hat^2 estimate is therefore

        F(rho) = [(1 + rho) / (1 - rho)] * [6 / (6 - 8*rho + 2*rho^2)]
               = 3 * (1 + rho) / ((1 - rho)^2 * (3 - rho)),      F(0) = 1.

    rho is estimated from the lag-1 autocorrelation of fully observed
    SECOND differences. Second differences annihilate every affine segment
    exactly, so genuine piecewise-affine structure contributes only sparse
    knot spikes, which are removed by a 5-sigma MAD trim before the
    correlation is computed — the estimator therefore cannot mistake true
    level/slope regimes for serial correlation. Under AR(1) noise,

        r1(rho) = Corr(D2_t, D2_{t+1})
                = (rho^3 - 4*rho^2 + 7*rho - 4) / (6 - 8*rho + 2*rho^2),

    which is monotone increasing from -2/3 at rho=0 to -1/2 as rho -> 1;
    rho_hat inverts it on a grid, clipped to [0, 0.95]. Negative rho_hat is
    never used to shrink the penalty (independence is the floor).
    """

    values = scale_source.to_numpy(dtype=float)
    observed = observed_mask.to_numpy(dtype=bool)
    if len(values) < 15:
        return 1.0
    d2 = values[2:] - 2.0 * values[1:-1] + values[:-2]
    valid = observed[2:] & observed[1:-1] & observed[:-2]
    if valid.sum() < 12:
        return 1.0
    med = float(np.median(d2[valid]))
    mad = float(np.median(np.abs(d2[valid] - med)))
    if mad <= 0.0:
        return 1.0
    inlier = valid & (np.abs(d2 - med) <= 5.0 * 1.4826 * mad)
    pair = inlier[1:] & inlier[:-1]
    lead, lag = d2[1:][pair], d2[:-1][pair]
    if lead.size < 10:
        return 1.0
    lead = lead - lead.mean()
    lag = lag - lag.mean()
    denominator = float(np.sqrt(float((lead * lead).sum()) * float((lag * lag).sum())))
    if denominator <= 0.0:
        return 1.0
    r1 = float((lead * lag).sum() / denominator)
    if r1 <= -2.0 / 3.0:
        return 1.0
    grid = np.linspace(0.0, 0.95, 96)
    theoretical = (grid**3 - 4.0 * grid**2 + 7.0 * grid - 4.0) / (
        6.0 - 8.0 * grid + 2.0 * grid**2
    )
    rho = float(grid[int(np.argmin(np.abs(theoretical - r1)))])
    if rho <= 0.0:
        return 1.0
    factor = 3.0 * (1.0 + rho) / (((1.0 - rho) ** 2) * (3.0 - rho))
    if factor > 1.0:
        notices.append(
            f"positive serial correlation in the detector noise "
            f"(rho_hat={rho:.2f}); penalty inflated by long-run variance "
            f"factor {factor:.1f} to prevent over-segmentation"
        )
    return factor


def _scalar_penalty(
    scale_source: pd.Series,
    observed_mask: pd.Series,
    weights: pd.Series,
    penalty_multiplier: float,
    notices: list[str],
) -> tuple[float | None, float | None]:
    values = scale_source.to_numpy(dtype=float)
    observed = observed_mask.to_numpy(dtype=bool)
    n_eff = int((weights.to_numpy(dtype=float) > 0).sum())
    if n_eff < 6:
        notices.append("fewer than six positive-weight observations: no internal affine breakpoints considered")
        return None, None
    valid = observed[2:] & observed[1:-1] & observed[:-2]
    d2 = values[2:] - 2.0 * values[1:-1] + values[:-2]
    sample = d2[valid]
    if sample.size == 0:
        notices.append("no fully observed second differences: no internal affine breakpoints considered")
        return None, None
    mad = float(np.median(np.abs(sample - np.median(sample))))
    sigma = 1.4826 * mad / sqrt(6.0)
    time = np.arange(len(values), dtype=float)
    if sigma <= np.finfo(float).eps * max(1.0, np.ptp(values)):
        if _is_exactly_affine(values, time, observed):
            notices.append("second-difference scale is zero and observed signal is affine: no internal breakpoints")
            return None, 0.0
        nonzero = np.abs(sample[np.abs(sample) > np.finfo(float).eps])
        if nonzero.size:
            sigma = float(np.median(nonzero) / sqrt(6.0))
            notices.append("second-difference MAD was degenerate; used median non-zero second difference scale")
        else:
            sigma = float(np.finfo(float).eps * max(1.0, np.ptp(values)))
            notices.append("second-difference scale was degenerate; used finite machine-scale fallback")
    # MODIFIED: correct the iid-calibrated scale for serial correlation
    # before forming the penalty (see _autocorr_penalty_inflation).
    inflation = _autocorr_penalty_inflation(scale_source, observed_mask, notices)
    penalty = float(penalty_multiplier * sigma * sigma * log(n_eff) * inflation)
    if not np.isfinite(penalty) or penalty <= 0:
        raise TemporalRecommendationError("scalar affine penalty must be finite and strictly positive")
    return penalty, sigma


def _snap_to_observed(timestamp: pd.Timestamp, observed_index: pd.DatetimeIndex) -> pd.Timestamp:
    if timestamp in observed_index:
        return timestamp
    position = int(observed_index.searchsorted(timestamp))
    candidates: list[pd.Timestamp] = []
    if position > 0:
        candidates.append(pd.Timestamp(observed_index[position - 1]))
    if position < len(observed_index):
        candidates.append(pd.Timestamp(observed_index[position]))
    # tuple key deliberately resolves an equal-distance tie to the later date.
    return min(candidates, key=lambda value: (abs(value - timestamp), -value.value))


def _detect_scalar_breakpoints(
    detector: pd.Series,
    weights: pd.Series,
    observed_mask: pd.Series,
    scale_source: pd.Series,
    min_width: pd.Timedelta,
    penalty_multiplier: float,
    notices: list[str],
) -> tuple[pd.DatetimeIndex, WeightedAffineCost, float | None, float | None]:
    _require_ruptures()
    time = ((detector.index - detector.index[0]) / pd.Timedelta(days=1)).to_numpy(dtype=float)
    cost = WeightedAffineCost(time, weights.to_numpy(dtype=float)).fit(detector.to_numpy(dtype=float))
    penalty, sigma = _scalar_penalty(scale_source, observed_mask, weights, penalty_multiplier, notices)
    n = len(detector)
    min_size = max(cost.min_size, int(ceil(min_width / pd.Timedelta(days=1))))
    if penalty is None or n < 2 * min_size:
        return pd.DatetimeIndex([]), cost, penalty, sigma
    rpt = _require_ruptures()
    predicted = rpt.Pelt(custom_cost=cost, min_size=min_size, jump=1).fit(detector.to_numpy(dtype=float)).predict(pen=penalty)
    raw_indexes = [int(point) for point in predicted if int(point) < n]
    observed_index = pd.DatetimeIndex(detector.index[observed_mask.to_numpy(dtype=bool)])
    snapped = [_snap_to_observed(pd.Timestamp(detector.index[index]), observed_index) for index in raw_indexes]
    return pd.DatetimeIndex(sorted(set(snapped))), cost, penalty, sigma


def _l2_cost(values: np.ndarray, start: int, end: int) -> float:
    segment = values[start:end]
    if len(segment) == 0:
        return float("inf")
    centre = segment.mean(axis=0)
    return float(np.square(segment - centre).sum())


def _transition_penalty(values: np.ndarray, notices: list[str]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    diffs = np.diff(values, axis=0)
    scales = []
    for column in diffs.T:
        if np.allclose(column, column[0]):
            scales.append(0.0)
            continue
        mad = float(np.median(np.abs(column - np.median(column))))
        sigma = 1.4826 * mad / sqrt(2.0)
        if sigma == 0:
            nonzero = np.abs(column[np.abs(column) > np.finfo(float).eps])
            sigma = float(np.median(nonzero) / sqrt(2.0)) if nonzero.size else 0.0
            if sigma:
                notices.append("transition robust scale used median non-zero first difference fallback")
        scales.append(sigma)
    variance_budget = float(np.square(scales).sum())
    if variance_budget <= 0:
        notices.append("all aligned transition coordinates are constant: no internal breakpoints")
        return None, 0.0
    penalty = float(2.0 * variance_budget * log(len(values)))
    if not np.isfinite(penalty) or penalty <= 0:
        raise TemporalRecommendationError("transition penalty must be finite and strictly positive")
    return penalty, variance_budget


def _detector_index(times: pd.DatetimeIndex, boundary: pd.Timestamp) -> int:
    return int(times.searchsorted(boundary, side="left"))


def _boundary_gain(
    boundaries: Sequence[pd.Timestamp],
    index: int,
    detector_times: pd.DatetimeIndex,
    cost: WeightedAffineCost | None,
    transition_values: np.ndarray | None,
) -> float:
    left = _detector_index(detector_times, boundaries[index - 1])
    middle = _detector_index(detector_times, boundaries[index])
    right = _detector_index(detector_times, boundaries[index + 1])
    if middle <= left or right <= middle:
        return float("inf")
    if cost is not None:
        gain = float(cost.error(left, right) - cost.error(left, middle) - cost.error(middle, right))
    else:
        assert transition_values is not None
        gain = _l2_cost(transition_values, left, right) - _l2_cost(transition_values, left, middle) - _l2_cost(transition_values, middle, right)
    # MODIFIED: inf - inf = NaN when both the merged span and a sub-span are
    # degenerate (< 3 positive-weight observations, e.g. imputation-heavy
    # regions). NaN poisons min()-based merge selection because every NaN
    # comparison is False. A boundary with no measurable structure is the
    # weakest candidate: map NaN to -inf deterministically.
    return gain if not isnan(gain) else float("-inf")


def _epoch_counts(boundaries: Sequence[pd.Timestamp], anchors: pd.DatetimeIndex) -> list[int]:
    # Avoid comparing raw integer units: pandas 3 may use datetime64[us]
    # internally while Timestamp.value remains nanoseconds.
    return [int(((anchors >= left) & (anchors < right)).sum()) for left, right in zip(boundaries[:-1], boundaries[1:])]


def _select_removal(
    boundaries: list[pd.Timestamp],
    epoch_index: int,
    immutable: set[pd.Timestamp],
    detector_times: pd.DatetimeIndex,
    cost: WeightedAffineCost | None,
    transition_values: np.ndarray | None,
) -> tuple[int, tuple[tuple[pd.Timestamp, float], ...]]:
    candidate_indexes = []
    # The only removable boundaries adjacent to the target epoch.
    if epoch_index > 0 and boundaries[epoch_index] not in immutable:
        candidate_indexes.append(epoch_index)
    if epoch_index + 1 < len(boundaries) - 1 and boundaries[epoch_index + 1] not in immutable:
        candidate_indexes.append(epoch_index + 1)
    if not candidate_indexes:
        raise EpochWidthError("no removable boundary is adjacent to the violating immutable epoch")
    scores: list[tuple[int, pd.Timestamp, float]] = []
    for boundary_index in candidate_indexes:
        gain = _boundary_gain(boundaries, boundary_index, detector_times, cost, transition_values)
        scores.append((boundary_index, boundaries[boundary_index], gain))
    # Smaller objective gain is weaker.  Ties remove the later boundary.
    chosen = min(scores, key=lambda item: (item[2], -item[1].value))
    diagnostics = tuple((timestamp, float(gain)) for _, timestamp, gain in scores)
    return chosen[0], diagnostics


def _apply_pins(
    raw_breakpoints: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pinned: Mapping[pd.Interval, str | pd.DatetimeIndex] | None,
) -> tuple[list[pd.Timestamp], set[pd.Timestamp], list[PinAction]]:
    if not pinned:
        return list(raw_breakpoints), {start, end}, []
    resolved: list[tuple[pd.Interval, str | pd.DatetimeIndex]] = []
    for interval, value in pinned.items():
        if not isinstance(interval, pd.Interval):
            raise PinOverlapError("pinned keys must be pandas.Interval values")
        if interval.closed != "left":
            raise PinOverlapError("pinned intervals must be left-closed, right-open")
        left, right = pd.Timestamp(interval.left).normalize(), pd.Timestamp(interval.right).normalize()
        if left < start or right > end or left >= right:
            raise PinOverlapError(
                f"pin interval [{left}, {right}) must lie inside recommendation window [{start}, {end})"
            )
        resolved.append((pd.Interval(left, right, closed="left"), value))
    resolved.sort(key=lambda item: item[0].left)
    for (previous, _), (current, _) in zip(resolved, resolved[1:]):
        if current.left < previous.right:
            raise PinOverlapError(f"pinned intervals {previous} and {current} overlap")

    immutable: set[pd.Timestamp] = {start, end}
    pin_actions: list[PinAction] = []
    candidates = list(raw_breakpoints)
    for interval, specification in resolved:
        left, right = pd.Timestamp(interval.left), pd.Timestamp(interval.right)
        immutable.update({left, right})
        pin_actions.append(PinAction(interval, "pin_edge", left))
        pin_actions.append(PinAction(interval, "pin_edge", right))
        if isinstance(specification, str):
            try:
                internal = pd.date_range(left, right, freq=specification)
            except (TypeError, ValueError) as exc:
                raise PinOverlapError(f"invalid pin frequency {specification!r} for {interval}") from exc
        elif isinstance(specification, pd.DatetimeIndex):
            internal = specification
        else:
            raise PinOverlapError("pin values must be a calendar frequency alias or pd.DatetimeIndex")
        for boundary in pd.DatetimeIndex(pd.to_datetime(internal)).normalize():
            if boundary < left or boundary > right:
                raise PinOverlapError(f"pinned boundary {boundary} lies outside pin interval {interval}")
            immutable.add(pd.Timestamp(boundary))
            if boundary not in (left, right):
                pin_actions.append(PinAction(interval, "pin_internal", pd.Timestamp(boundary)))
        kept: list[pd.Timestamp] = []
        for boundary in candidates:
            if left < boundary < right:
                pin_actions.append(PinAction(interval, "remove_recommender_cut", boundary, "strictly inside pinned interval"))
            else:
                kept.append(boundary)
        candidates = kept
    return candidates, immutable, pin_actions


def _merge_boundaries(
    boundaries: list[pd.Timestamp],
    *,
    immutable: set[pd.Timestamp],
    anchors: pd.DatetimeIndex,
    min_width: pd.Timedelta,
    min_samples: int,
    detector_times: pd.DatetimeIndex,
    cost: WeightedAffineCost | None,
    transition_values: np.ndarray | None,
) -> tuple[list[pd.Timestamp], list[MergeAction]]:
    merge_log: list[MergeAction] = []
    while True:
        widths = [right - left for left, right in zip(boundaries[:-1], boundaries[1:])]
        violating = [i for i, width in enumerate(widths) if width < min_width]
        if not violating:
            break
        epoch_index = min(violating, key=lambda i: (widths[i], boundaries[i].value))
        interval = pd.Interval(boundaries[epoch_index], boundaries[epoch_index + 1], closed="left")
        try:
            remove_index, scores = _select_removal(boundaries, epoch_index, immutable, detector_times, cost, transition_values)
        except EpochWidthError as exc:
            raise EpochWidthError(
                f"epoch {interval} has width {widths[epoch_index]}, below min_epoch_width={min_width}; "
                "its adjacent boundaries are pinned"
            ) from exc
        removed = boundaries.pop(remove_index)
        merge_log.append(MergeAction(interval, "min_epoch_width", removed, scores))

    while True:
        counts = _epoch_counts(boundaries, anchors)
        widths = [right - left for left, right in zip(boundaries[:-1], boundaries[1:])]
        violating = [i for i, count in enumerate(counts) if count < min_samples]
        if not violating:
            break
        epoch_index = min(violating, key=lambda i: (widths[i], boundaries[i].value))
        interval = pd.Interval(boundaries[epoch_index], boundaries[epoch_index + 1], closed="left")
        try:
            remove_index, scores = _select_removal(boundaries, epoch_index, immutable, detector_times, cost, transition_values)
        except EpochWidthError as exc:
            raise EpochSufficiencyError(
                f"epoch {interval} has {counts[epoch_index]} paths, below min_samples={min_samples}; "
                "its adjacent boundaries are pinned"
            ) from exc
        removed = boundaries.pop(remove_index)
        merge_log.append(MergeAction(interval, "min_samples", removed, scores))
    return boundaries, merge_log


def _segment_fits(
    boundaries: Sequence[pd.Timestamp],
    detector_times: pd.DatetimeIndex,
    cost: WeightedAffineCost,
    anchors: pd.DatetimeIndex,
) -> tuple[SegmentFit, ...]:
    fits: list[SegmentFit] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        start, end = _detector_index(detector_times, left), _detector_index(detector_times, right)
        # The final boundary is exclusive and one day after the last daily
        # detector observation, so it naturally maps to len(detector_times).
        intercept, slope, rss, n_observed = cost.segment_parameters(start, end)
        midpoint_time = (cost.times[start] + cost.times[end - 1]) / 2.0 if end > start else float("nan")
        midpoint_value = intercept + slope * midpoint_time if np.isfinite(intercept) else float("nan")
        n_paths = int(((anchors >= left) & (anchors < right)).sum())
        fits.append(SegmentFit(left, right, float(midpoint_value), slope, rss, n_observed, n_paths))
    return tuple(fits)


def _boundary_scores(
    all_candidates: Iterable[pd.Timestamp],
    final_boundaries: Sequence[pd.Timestamp],
    raw: set[pd.Timestamp],
    immutable: set[pd.Timestamp],
    detector_times: pd.DatetimeIndex,
    cost: WeightedAffineCost | None,
    transition_values: np.ndarray | None,
    merge_log: Sequence[MergeAction] = (),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final = list(final_boundaries)
    removal_reasons = {action.removed_boundary: action.reason for action in merge_log}
    for boundary in sorted(set(all_candidates)):
        retained = boundary in final
        gain = np.nan
        level_change = np.nan
        slope_change = np.nan
        if retained and boundary not in (final[0], final[-1]):
            index = final.index(boundary)
            gain = _boundary_gain(final, index, detector_times, cost, transition_values)
            if cost is not None:
                left_start = _detector_index(detector_times, final[index - 1])
                middle = _detector_index(detector_times, boundary)
                right_end = _detector_index(detector_times, final[index + 1])
                left_i, left_slope, _, _ = cost.segment_parameters(left_start, middle)
                right_i, right_slope, _, _ = cost.segment_parameters(middle, right_end)
                at = (boundary - detector_times[0]) / pd.Timedelta(days=1)
                level_change = (right_i + right_slope * at) - (left_i + left_slope * at)
                slope_change = right_slope - left_slope
        classification = "transition" if cost is None else (
            "slope-dominant" if abs(slope_change) > abs(level_change) else "level-dominant"
        )
        rows.append({
            "boundary": boundary,
            "gain": gain,
            "level_change": level_change,
            "slope_change": slope_change,
            "classification": classification,
            "retained": retained,
            "source": "pin" if boundary in immutable and boundary not in (final[0], final[-1]) else ("raw" if boundary in raw else "endpoint"),
            "removal_reason": removal_reasons.get(boundary),
        })
    if not rows:
        return pd.DataFrame(columns=["gain", "level_change", "slope_change", "classification", "retained", "source", "removal_reason"])
    return pd.DataFrame(rows).set_index("boundary").sort_index()


def _epoch_reporting_metrics(
    boundaries: Sequence[pd.Timestamp],
    anchors: pd.DatetimeIndex,
    conversions: np.ndarray,
    min_samples: int,
) -> pd.DataFrame:
    """Return the fixed recommendation reporting schema for final epochs."""

    bounds = pd.DatetimeIndex(boundaries)
    labels = [f"E{index}" for index in range(1, len(bounds))]
    path_counts: list[int] = []
    conversion_counts: list[int] = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        mask = (anchors >= left) & (anchors < right)
        path_counts.append(int(mask.sum()))
        conversion_counts.append(int(conversions[mask].sum()))
    return build_epoch_metrics(
        epoch_labels=labels,
        starts=list(bounds[:-1]),
        ends=list(bounds[1:]),
        path_counts=path_counts,
        conversion_counts=conversion_counts,
        sample_floors=min_samples,
    )


def _callback_signal(
    callback: SignalBuilder,
    paths: np.ndarray,
    conversions: np.ndarray,
    anchors: pd.DatetimeIndex,
) -> pd.Series:
    result = callback(list(paths), list(conversions), list(anchors))
    if not isinstance(result, pd.Series):
        raise SignalShapeError("custom signal callback must return a one-dimensional pandas.Series")
    if result.empty or result.ndim != 1:
        raise SignalShapeError("custom signal callback must return a non-empty one-dimensional Series")
    if isinstance(result.index, pd.PeriodIndex):
        index = result.index.to_timestamp()
    elif isinstance(result.index, pd.DatetimeIndex):
        index = result.index
    else:
        raise SignalShapeError("custom signal callback index must be DatetimeIndex or PeriodIndex")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise SignalShapeError("custom signal callback index must be sorted and unique")
    if len(index) > 2:
        steps = np.diff(index.asi8)
        if not np.all(steps == steps[0]):
            raise SignalShapeError("custom signal callback index must have regular frequency")
    if pd.api.types.is_object_dtype(result.dtype) or np.iscomplexobj(result.to_numpy()):
        raise SignalShapeError("custom signal callback values must use a real, non-object numeric dtype")
    try:
        numeric = pd.to_numeric(result, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise SignalShapeError("custom signal callback values must be real numeric values") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise SignalShapeError("custom signal callback values must be finite")
    return pd.Series(numeric.to_numpy(), index=pd.DatetimeIndex(index), name=result.name or "custom_signal")


def _recommend_scalar(
    paths: np.ndarray,
    conversions: np.ndarray,
    anchors: pd.DatetimeIndex,
    *,
    signal: Literal["conversion_rate"] | SignalBuilder,
    min_width: pd.Timedelta,
    min_samples: int,
    stl_period: int | tuple[int, ...] | Literal["auto"] | None,
    max_gap: pd.Timedelta,
    penalty_multiplier: float,
    pinned: Mapping[pd.Interval, str | pd.DatetimeIndex] | None,
) -> EpochRecommendation:
    notices: list[str] = []
    if signal == "conversion_rate":
        raw, path_counts, _ = _daily_conversion_rate(conversions, anchors)
        prepared, imputed = _prepare_missing_rate(raw, max_gap)
        median_count = float(path_counts[path_counts > 0].median())
        weights = path_counts.astype(float).where(path_counts > 0, 0.0) / median_count
        decomposition = _decompose(raw, prepared, stl_period=stl_period, n_observed=int((path_counts > 0).sum()), notices=notices)
        detector = decomposition.trend.rename("detector_signal")
        scale_source = (prepared - _seasonal_total(decomposition)).rename("scale_source")
        observed_mask = (~imputed).rename("observed")
        signal_name: str | Callable[..., Any] = "conversion_rate"
    else:
        raw = _callback_signal(signal, paths, conversions, anchors)
        path_counts = pd.Series(1, index=raw.index, name="path_count")
        prepared = raw.copy()
        imputed = pd.Series(False, index=raw.index)
        weights = pd.Series(1.0, index=raw.index, name="detector_weight")
        decomposition = _decompose(raw, prepared, stl_period=stl_period, n_observed=len(raw), notices=notices)
        detector = decomposition.trend.rename("detector_signal")
        scale_source = (prepared - _seasonal_total(decomposition)).rename("scale_source")
        observed_mask = (~imputed).rename("observed")
        signal_name = signal
    weights.name = "detector_weight"
    raw_breakpoints, cost, penalty, sigma = _detect_scalar_breakpoints(
        detector, weights, observed_mask, scale_source, min_width, penalty_multiplier, notices
    )
    start, end = anchors.min(), anchors.max() + pd.Timedelta(days=1)
    candidates, immutable, pin_log = _apply_pins(raw_breakpoints, start, end, pinned)
    boundaries = sorted(set([start, end, *candidates, *immutable]))
    all_candidates = set(raw_breakpoints) | immutable
    boundaries, merge_log = _merge_boundaries(
        boundaries,
        immutable=immutable,
        anchors=anchors,
        min_width=min_width,
        min_samples=min_samples,
        detector_times=pd.DatetimeIndex(detector.index),
        cost=cost,
        transition_values=None,
    )
    scores = _boundary_scores(all_candidates, boundaries, set(raw_breakpoints), immutable, pd.DatetimeIndex(detector.index), cost, None, merge_log)
    metadata = {
        "imputed_days": pd.DatetimeIndex(detector.index[imputed.to_numpy()]),
        "penalty": penalty,
        "robust_scale": sigma,
        "min_size": max(cost.min_size, int(ceil(min_width / pd.Timedelta(days=1)))),
    }
    final_boundaries = pd.DatetimeIndex(boundaries)
    return EpochRecommendation(
        boundaries=final_boundaries,
        raw_breakpoints=raw_breakpoints,
        signal_name=signal_name,
        signal=raw,
        detector_signal=detector,
        detector_weights=weights,
        decomposition=decomposition,
        segment_fits=_segment_fits(
            final_boundaries, pd.DatetimeIndex(detector.index), cost, anchors
        ),
        boundary_scores=scores,
        path_counts=path_counts,
        epoch_metrics=_epoch_reporting_metrics(
            final_boundaries, anchors, conversions, min_samples
        ),
        min_samples=min_samples,
        merge_log=tuple(merge_log),
        pin_log=tuple(pin_log),
        warnings=tuple(notices),
        streaming=False,
        metadata=metadata,
    )


def _transition_buckets(
    paths: np.ndarray,
    conversions: np.ndarray,
    anchors: pd.DatetimeIndex,
    separator: str,
    conversion_labels: Sequence[str] | None,
    bucket: str,
    min_samples: int,
) -> tuple[pd.DatetimeIndex, pd.Series, list[pd.DataFrame]]:
    try:
        periods = anchors.to_period(bucket)
    except (TypeError, ValueError) as exc:
        raise TemporalRecommendationError(f"transition_bucket {bucket!r} is not a valid calendar frequency") from exc
    # Edge periods can be partial because the analysis window rarely starts
    # and ends exactly on a calendar bucket boundary.  The transition signal
    # is defined on complete buckets only; partial edges are excluded rather
    # than misclassified as underpowered zero-like regimes.
    window_start, window_end = anchors.min(), anchors.max() + pd.Timedelta(days=1)
    all_periods = [
        period for period in pd.period_range(periods.min(), periods.max(), freq=bucket)
        if pd.Timestamp(period.start_time).normalize() >= window_start
        and pd.Timestamp(period.end_time).normalize() + pd.Timedelta(days=1) <= window_end
    ]
    if not all_periods:
        raise SignalSufficiencyError(
            f"analysis window [{window_start}, {window_end}) contains no complete {bucket!r} transition buckets"
        )
    starts: list[pd.Timestamp] = []
    counts: list[int] = []
    matrices: list[pd.DataFrame] = []
    for period in all_periods:
        mask = periods == period
        count = int(mask.sum())
        if count < min_samples:
            raise SignalSufficiencyError(
                f"transition bucket {period} has {count} paths; requires transition_min_samples={min_samples}. "
                "Increase transition_bucket, widen the window, or lower the threshold explicitly."
            )
        model = MarkovAttribution(order=1, separator=separator, conversion_labels=conversion_labels, return_transition_matrix=True)
        result = model.fit(paths[mask], conversions[mask])
        starts.append(pd.Timestamp(period.start_time).normalize())
        counts.append(count)
        matrices.append(result["transition_matrix"])
    return pd.DatetimeIndex(starts), pd.Series(counts, index=pd.DatetimeIndex(starts), name="path_count"), matrices


def _recommend_transition(
    paths: np.ndarray,
    conversions: np.ndarray,
    anchors: pd.DatetimeIndex,
    *,
    min_width: pd.Timedelta,
    min_samples: int,
    pinned: Mapping[pd.Interval, str | pd.DatetimeIndex] | None,
    separator: str,
    conversion_labels: Sequence[str] | None,
    transition_bucket: str,
    transition_min_samples: int | None,
) -> EpochRecommendation:
    _require_ruptures()
    notices: list[str] = []
    floor = min_samples if transition_min_samples is None else transition_min_samples
    if floor <= 0:
        raise TemporalRecommendationError("transition_min_samples must be positive")
    times, path_counts, matrices = _transition_buckets(paths, conversions, anchors, separator, conversion_labels, transition_bucket, floor)
    basis = sorted(set().union(*(set(matrix.index) | set(matrix.columns) for matrix in matrices)), key=str)
    vectors = np.vstack([matrix.reindex(index=basis, columns=basis, fill_value=0.0).to_numpy().reshape(-1) for matrix in matrices])
    columns = [f"{left}->{right}" for left in basis for right in basis]
    detector = pd.DataFrame(vectors, index=times, columns=columns)
    frobenius = np.empty(len(vectors), dtype=float)
    frobenius[0] = np.nan
    if len(vectors) > 1:
        frobenius[1:] = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
    diagnostic = pd.Series(frobenius, index=times, name="adjacent_frobenius_distance")
    penalty, variance_budget = _transition_penalty(vectors, notices)
    if len(times) > 1:
        # Do not reinterpret DatetimeIndex.asi8 as nanoseconds: pandas 3 can
        # store an index at microsecond resolution.  Timedelta arithmetic
        # retains the correct calendar duration on every supported version.
        widths_in_days = [(times[index + 1] - times[index]) / pd.Timedelta(days=1) for index in range(len(times) - 1)]
        typical_width = pd.Timedelta(days=float(np.median(widths_in_days)))
    else:
        typical_width = pd.Timedelta(days=7)
    min_size = max(2, int(ceil(min_width / typical_width)))
    raw_breakpoints = pd.DatetimeIndex([])
    if penalty is not None and len(vectors) >= 2 * min_size:
        rpt = _require_ruptures()
        points = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(vectors).predict(pen=penalty)
        raw_breakpoints = pd.DatetimeIndex([times[int(point)] for point in points if int(point) < len(times)])
    start, end = anchors.min(), anchors.max() + pd.Timedelta(days=1)
    candidates, immutable, pin_log = _apply_pins(raw_breakpoints, start, end, pinned)
    boundaries = sorted(set([start, end, *candidates, *immutable]))
    all_candidates = set(raw_breakpoints) | immutable
    boundaries, merge_log = _merge_boundaries(
        boundaries,
        immutable=immutable,
        anchors=anchors,
        min_width=min_width,
        min_samples=min_samples,
        detector_times=times,
        cost=None,
        transition_values=vectors,
    )
    scores = _boundary_scores(all_candidates, boundaries, set(raw_breakpoints), immutable, times, None, vectors, merge_log)
    final_boundaries = pd.DatetimeIndex(boundaries)
    return EpochRecommendation(
        boundaries=final_boundaries,
        raw_breakpoints=raw_breakpoints,
        signal_name="transition_drift",
        signal=diagnostic,
        detector_signal=detector,
        detector_weights=None,
        decomposition=None,
        segment_fits=(),
        boundary_scores=scores,
        path_counts=path_counts,
        epoch_metrics=_epoch_reporting_metrics(
            final_boundaries, anchors, conversions, min_samples
        ),
        min_samples=min_samples,
        merge_log=tuple(merge_log),
        pin_log=tuple(pin_log),
        warnings=tuple(notices),
        streaming=False,
        metadata={
            "state_basis": tuple(basis),
            "transition_penalty": penalty,
            "variance_budget": variance_budget,
            "transition_bucket": transition_bucket,
            "detector_input": "aligned order-1 transition-matrix vectors",
        },
    )


def recommend_epochs(
    paths: Sequence[str],
    conversions: Sequence[int],
    anchor_timestamps: Sequence[pd.Timestamp],
    *,
    signal: Literal["conversion_rate", "transition_drift"] | SignalBuilder = "conversion_rate",
    min_epoch_width: str | pd.Timedelta = "14D",
    min_samples: int = 50,
    stl_period: int | tuple[int, ...] | Literal["auto"] | None = "auto",
    max_imputation_gap: str | pd.Timedelta = "7D",
    penalty_multiplier: float = 2.0,
    pinned: Mapping[pd.Interval, str | pd.DatetimeIndex] | None = None,
    separator: str = DEFAULT_SEPARATOR,
    conversion_labels: Sequence[str] | None = None,
    transition_bucket: str = "W-MON",
    transition_min_samples: int | None = None,
) -> EpochRecommendation:
    """Recommend auditable global epoch boundaries for independent static fits."""

    paths_arr, conversion_arr, anchors = _validate_inputs(paths, conversions, anchor_timestamps)
    if not isinstance(min_samples, int) or isinstance(min_samples, bool) or min_samples <= 0:
        raise TemporalRecommendationError(f"min_samples must be a positive integer, got {min_samples!r}")
    try:
        valid_penalty = bool(np.isfinite(penalty_multiplier)) and float(penalty_multiplier) > 0
    except (TypeError, ValueError):
        valid_penalty = False
    if not valid_penalty:
        raise TemporalRecommendationError("penalty_multiplier must be finite and strictly positive")
    min_width = _positive_timedelta(min_epoch_width, "min_epoch_width")
    max_gap = _positive_timedelta(max_imputation_gap, "max_imputation_gap")
    if signal == "transition_drift":
        return _recommend_transition(
            paths_arr, conversion_arr, anchors, min_width=min_width, min_samples=min_samples,
            pinned=pinned, separator=separator, conversion_labels=conversion_labels,
            transition_bucket=transition_bucket, transition_min_samples=transition_min_samples,
        )
    if signal == "conversion_rate" or callable(signal):
        return _recommend_scalar(
            paths_arr, conversion_arr, anchors, signal=signal, min_width=min_width, min_samples=min_samples,
            stl_period=stl_period, max_gap=max_gap, penalty_multiplier=float(penalty_multiplier), pinned=pinned,
        )
    raise TemporalRecommendationError(
        "signal must be 'conversion_rate', 'transition_drift', or a scalar signal callback"
    )


def assign_paths_to_epochs(
    anchor_timestamps: Sequence[Any], boundaries: Sequence[pd.Timestamp]
) -> list[pd.Interval]:
    """Assign each complete path anchor to exactly one left-closed epoch."""

    anchors = _normalize_anchors(anchor_timestamps)
    bounds = pd.DatetimeIndex(pd.to_datetime(boundaries)).normalize()
    if len(bounds) < 2 or not bounds.is_monotonic_increasing or not bounds.is_unique:
        raise TemporalRecommendationError("boundaries must be a strictly increasing sequence with at least two entries")
    positions = bounds.searchsorted(anchors, side="right") - 1
    if (positions < 0).any() or (positions >= len(bounds) - 1).any():
        bad = int(np.flatnonzero((positions < 0) | (positions >= len(bounds) - 1))[0])
        raise TemporalRecommendationError(
            f"anchor timestamp {anchors[bad]} at index {bad} falls outside epoch boundaries [{bounds[0]}, {bounds[-1]})"
        )
    return [pd.Interval(bounds[i], bounds[i + 1], closed="left") for i in positions]


@dataclass(frozen=True)
class DaySummary:
    """One strictly ordered calendar-day update for the streaming recommender."""

    date: pd.Timestamp
    paths: Sequence[str]
    conversions: Sequence[int]


@dataclass(frozen=True)
class BoundaryEvent:
    """A confirmed streaming conversion-rate regime event."""

    timestamp: pd.Timestamp
    confirmed_at: pd.Timestamp
    posterior_probability: float


class StreamingEpochRecommender:
    """Conversion-rate-only, slope-aware streaming recommendation helper.

    The online state is a bounded BOCPD run-length posterior.  Each hypothesis
    carries Normal-Inverse-Gamma weighted affine-regression sufficient
    statistics, so its Student-t predictive distribution can distinguish a
    slope change from an ordinary level fluctuation.  It retains only JSON
    primitives in ``snapshot``; transition-matrix streaming remains out of
    scope.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        expected_run_length: int = 60,
        confirmation_window: int = 7,
        posterior_threshold: float = 0.5,
        max_run_length: int = 365,
    ) -> None:
        if not 0 < alpha <= 1 or expected_run_length <= 0 or confirmation_window <= 0 or not 0 < posterior_threshold < 1 or max_run_length <= 0:
            raise TemporalRecommendationError("streaming parameters must be positive and probabilities must lie in (0, 1)")
        self.alpha = float(alpha)
        self.expected_run_length = int(expected_run_length)
        self.confirmation_window = int(confirmation_window)
        self.posterior_threshold = float(posterior_threshold)
        self.max_run_length = int(max_run_length)
        self._dates: list[pd.Timestamp] = []
        self._rates: list[float | None] = []
        self._counts: list[int] = []
        self._seasonal = [0.0] * 7
        self._warmup_counts: list[int] = []
        self._weight_reference: float | None = None
        self._bocpd_states: list[dict[str, Any]] = []
        self._events: list[BoundaryEvent] = []
        self._confirmation: list[pd.Timestamp] = []

    @property
    def events(self) -> tuple[BoundaryEvent, ...]:
        return tuple(self._events)

    def update(self, summary: DaySummary) -> BoundaryEvent | None:
        date = pd.Timestamp(summary.date).normalize()
        if self._dates and date != self._dates[-1] + pd.Timedelta(days=1):
            raise TemporalRecommendationError("DaySummary dates must arrive in strict contiguous calendar order")
        if len(summary.paths) != len(summary.conversions):
            raise TemporalRecommendationError(
                f"DaySummary conversions length ({len(summary.conversions)}) does not match paths length ({len(summary.paths)})"
            )
        paths, conversions, anchors = _validate_inputs(summary.paths, summary.conversions, [date] * len(summary.paths)) if len(summary.paths) else (np.array([], dtype=object), np.array([], dtype=int), pd.DatetimeIndex([]))
        del paths, anchors
        count = len(summary.conversions)
        converted = int(conversions.sum()) if count else 0
        rate = (converted / count) if count else None
        self._dates.append(date)
        self._rates.append(rate)
        self._counts.append(count)
        if rate is None:
            return None
        self._warmup_counts.append(count)
        established_reference = False
        if self._weight_reference is None and len(self._warmup_counts) >= 28:
            self._weight_reference = float(np.median([value for value in self._warmup_counts if value > 0]))
            established_reference = True
        median = float(np.median([value for value in self._rates[-28:] if value is not None]))
        dow = date.dayofweek
        self._seasonal[dow] = (1.0 - self.alpha) * self._seasonal[dow] + self.alpha * (rate - median)
        if self._weight_reference is None or established_reference:
            return None
        probability = self._bocpd_update(date, rate - self._seasonal[dow], count)
        if probability > self.posterior_threshold:
            self._confirmation.append(date)
        else:
            self._confirmation.clear()
        if len(self._confirmation) >= self.confirmation_window:
            timestamp = self._confirmation[0]
            if not self._events or self._events[-1].timestamp != timestamp:
                event = BoundaryEvent(timestamp, date, probability)
                self._events.append(event)
                self._confirmation.clear()
                return event
        return None

    @staticmethod
    def _empty_bocpd_state() -> dict[str, Any]:
        return {
            "run_length": 0, "start": None, "probability": 1.0,
            "n": 0, "sw": 0.0, "sx": 0.0, "sx2": 0.0,
            "sy": 0.0, "sxy": 0.0, "sy2": 0.0,
        }

    @staticmethod
    def _posterior_parameters(state: Mapping[str, Any]) -> tuple[float, float, float, float, float, float, float]:
        """Return ``m0, m1, covariance entries, alpha, beta`` via NIG update."""
        prior_precision = 1e-3
        a = prior_precision + float(state["sw"])
        b = float(state["sx"])
        d = prior_precision + float(state["sx2"])
        determinant = a * d - b * b
        if determinant <= np.finfo(float).eps:
            determinant = np.finfo(float).eps
        inv00, inv01, inv11 = d / determinant, -b / determinant, a / determinant
        rhs0, rhs1 = float(state["sy"]), float(state["sxy"])
        mean0 = inv00 * rhs0 + inv01 * rhs1
        mean1 = inv01 * rhs0 + inv11 * rhs1
        alpha = 2.0 + 0.5 * int(state["n"])
        beta = 0.01 + 0.5 * max(0.0, float(state["sy2"]) - (rhs0 * mean0 + rhs1 * mean1))
        return mean0, mean1, inv00, inv01, inv11, alpha, beta

    def _predictive_logpdf(self, state: Mapping[str, Any], y: float, q: float, weight: float) -> float:
        mean0, mean1, inv00, inv01, inv11, alpha, beta = self._posterior_parameters(state)
        mean = mean0 + mean1 * q
        leverage = inv00 + 2.0 * q * inv01 + q * q * inv11
        scale2 = beta / alpha * max(np.finfo(float).eps, 1.0 / weight + leverage)
        scale = sqrt(scale2)
        degrees = 2.0 * alpha
        z = (y - mean) / scale
        return (
            lgamma((degrees + 1.0) / 2.0) - lgamma(degrees / 2.0)
            - 0.5 * log(degrees * pi) - log(scale)
            - (degrees + 1.0) / 2.0 * log(1.0 + z * z / degrees)
        )

    @staticmethod
    def _updated_state(state: Mapping[str, Any], *, date: pd.Timestamp, y: float, q: float, weight: float, probability: float, is_change: bool) -> dict[str, Any]:
        base = StreamingEpochRecommender._empty_bocpd_state() if is_change else dict(state)
        return {
            "run_length": 1 if is_change else int(base["run_length"]) + 1,
            "start": date.isoformat() if is_change else base["start"],
            "probability": float(probability),
            "n": int(base["n"]) + 1,
            "sw": float(base["sw"]) + weight,
            "sx": float(base["sx"]) + weight * q,
            "sx2": float(base["sx2"]) + weight * q * q,
            "sy": float(base["sy"]) + weight * y,
            "sxy": float(base["sxy"]) + weight * q * y,
            "sy2": float(base["sy2"]) + weight * y * y,
        }

    def _bocpd_update(self, date: pd.Timestamp, y: float, count: int) -> float:
        """Update weighted affine BOCPD and return ``P(r < confirmation_window)``."""
        assert self._weight_reference is not None
        weight = max(np.finfo(float).eps, float(count) / self._weight_reference)
        previous = self._bocpd_states or [self._empty_bocpd_state()]
        hazard = 1.0 / self.expected_run_length
        proposals: list[tuple[dict[str, Any], float]] = []
        for state in previous:
            start = pd.Timestamp(state["start"]) if state["start"] is not None else date
            q = float((date - start) / pd.Timedelta(days=1))
            growth_likelihood = self._predictive_logpdf(state, y, q, weight)
            change_likelihood = self._predictive_logpdf(self._empty_bocpd_state(), y, 0.0, weight)
            probability = max(np.finfo(float).tiny, float(state["probability"]))
            proposals.append((self._updated_state(state, date=date, y=y, q=0.0, weight=weight, probability=0.0, is_change=True), log(probability) + change_likelihood + log(hazard)))
            if int(state["run_length"]) < self.max_run_length and hazard < 1.0:
                proposals.append((self._updated_state(state, date=date, y=y, q=q, weight=weight, probability=0.0, is_change=False), log(probability) + growth_likelihood + log(1.0 - hazard)))
        # Multiple paths can lead to the same run length only for the change
        # state.  Preserve every hypothesis until normalization, then merge
        # equal (run_length, start) states deterministically.
        maximum = max(score for _, score in proposals)
        normalizer = maximum + log(sum(np.exp(score - maximum) for _, score in proposals))
        merged: dict[tuple[int, str | None], dict[str, Any]] = {}
        for state, score in proposals:
            probability = float(np.exp(score - normalizer))
            key = (int(state["run_length"]), state["start"])
            if key in merged:
                merged[key]["probability"] += probability
            else:
                state["probability"] = probability
                merged[key] = state
        self._bocpd_states = sorted(merged.values(), key=lambda state: (int(state["run_length"]), str(state["start"])))
        return float(sum(state["probability"] for state in self._bocpd_states if int(state["run_length"]) < self.confirmation_window))

    def get_current_recommendation(self) -> EpochRecommendation:
        if not self._dates:
            raise TemporalRecommendationError("stream has no observations")
        start = self._dates[0]
        end = self._dates[-1] + pd.Timedelta(days=1)
        boundaries = pd.DatetimeIndex([start, *(event.timestamp for event in self._events), end])
        signal = pd.Series(self._rates, index=pd.DatetimeIndex(self._dates), name="conversion_rate")
        counts = pd.Series(self._counts, index=pd.DatetimeIndex(self._dates), name="path_count")
        anchors = pd.DatetimeIndex(self._dates)
        # MODIFIED: a zero-path day stores rate=None; None * 0 raised
        # TypeError and permanently broke get_current_recommendation().
        conversions = np.asarray(
            [
                int(round(rate * count)) if rate is not None else 0
                for rate, count in zip(self._rates, self._counts)
            ],
            dtype=int,
        )
        # MODIFIED: per-epoch metrics must aggregate PATH counts, not count
        # calendar days. _epoch_reporting_metrics counts index entries (one
        # per day here), which reported paths=n_days and conversion rates
        # far above 100%.
        counts_arr = np.asarray(self._counts, dtype=int)
        labels = [f"E{index}" for index in range(1, len(boundaries))]
        epoch_paths: list[int] = []
        epoch_conversions: list[int] = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            mask = (anchors >= left) & (anchors < right)
            epoch_paths.append(int(counts_arr[mask].sum()))
            epoch_conversions.append(int(conversions[mask].sum()))
        metrics = build_epoch_metrics(
            epoch_labels=labels,
            starts=list(boundaries[:-1]),
            ends=list(boundaries[1:]),
            path_counts=epoch_paths,
            conversion_counts=epoch_conversions,
            sample_floors=1,
        )
        return EpochRecommendation(
            boundaries=boundaries,
            raw_breakpoints=pd.DatetimeIndex([]),
            signal_name="conversion_rate",
            signal=signal,
            detector_signal=signal,
            detector_weights=None,
            decomposition=None,
            segment_fits=(),
            boundary_scores=pd.DataFrame(),
            path_counts=counts,
            epoch_metrics=metrics,
            min_samples=1,
            merge_log=(),
            pin_log=(),
            warnings=(),
            streaming=True,
            metadata={"provisional_end": end},
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha, "expected_run_length": self.expected_run_length,
            "confirmation_window": self.confirmation_window, "posterior_threshold": self.posterior_threshold,
            "max_run_length": self.max_run_length, "dates": [date.isoformat() for date in self._dates],
            "rates": self._rates, "counts": self._counts, "seasonal": self._seasonal,
            "warmup_counts": self._warmup_counts, "weight_reference": self._weight_reference,
            "bocpd_states": self._bocpd_states,
            "events": [{"timestamp": event.timestamp.isoformat(), "confirmed_at": event.confirmed_at.isoformat(), "posterior_probability": event.posterior_probability} for event in self._events],
            "confirmation": [date.isoformat() for date in self._confirmation],
        }

    @classmethod
    def restore(cls, state: Mapping[str, Any]) -> "StreamingEpochRecommender":
        instance = cls(alpha=state["alpha"], expected_run_length=state["expected_run_length"], confirmation_window=state["confirmation_window"], posterior_threshold=state["posterior_threshold"], max_run_length=state["max_run_length"])
        instance._dates = [pd.Timestamp(value) for value in state["dates"]]
        instance._rates = list(state["rates"])
        instance._counts = [int(value) for value in state["counts"]]
        instance._seasonal = [float(value) for value in state["seasonal"]]
        instance._warmup_counts = [int(value) for value in state["warmup_counts"]]
        instance._weight_reference = state["weight_reference"]
        instance._bocpd_states = [dict(item) for item in state.get("bocpd_states", [])]
        instance._events = [BoundaryEvent(pd.Timestamp(item["timestamp"]), pd.Timestamp(item["confirmed_at"]), float(item["posterior_probability"])) for item in state["events"]]
        instance._confirmation = [pd.Timestamp(value) for value in state["confirmation"]]
        return instance


__all__ = [
    "BoundaryEvent", "DaySummary", "DecompositionResult", "EpochRecommendation", "EpochSufficiencyError",
    "EpochWidthError", "MergeAction", "NonCalendarBoundaryError", "PinAction", "PinOverlapError",
    "SegmentFit", "SignalBuilder", "SignalContinuityError", "SignalShapeError", "SignalSufficiencyError",
    "StreamingEpochRecommender", "TemporalRecommendationError", "WeightedAffineCost", "assign_paths_to_epochs",
    "recommend_epochs",
]
