"""Dependency-free plain-text reporting for attribution results.

The renderer in this module only consumes stored reporting state.  Keeping it
separate from the temporal estimators makes the output available without any
visualisation dependency and, importantly, prevents rendering from triggering
another fit or a recommendation pass.
"""

from __future__ import annotations

from collections import Counter  # MODIFIED: top-path diagnostics
from dataclasses import dataclass, field  # MODIFIED: + field
import math
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import split_path  # MODIFIED: path-diagnostic distillation

# MODIFIED: one distilled model-specific section = (title, headers, rows).
DetailSection = tuple[str, tuple[str, ...], tuple[tuple[str, ...], ...]]


_FIT_REQUIRED = "N/A - fit required"
_UNAVAILABLE = "N/A"
_SHIFT_THRESHOLD = 0.05


@dataclass(frozen=True)
class TemporalReportingSnapshot:
    """Model-agnostic, already-computed input for the text renderer."""

    model_name: str | None
    epoch_source: str
    epoch_labels: tuple[str, ...]
    boundaries: pd.DatetimeIndex | None
    epoch_metrics: pd.DataFrame
    attribution_matrix: pd.DataFrame | None
    sample_floor: int
    # MODIFIED: distilled model-specific sections (raw signed removal
    # effects, top transitions, top coalitions, path diagnostics, ...).
    # Bounded rows only — never full matrices or raw path dumps.
    model_details: tuple = ()
    # MODIFIED (get-well Unit 1): why ``attribution_matrix`` is absent.  Set
    # when a model fitted successfully but could not produce valid
    # normalized shares, so the renderer does not claim the fit is missing.
    attribution_note: str | None = None


def _signed(value: float) -> str:
    return f"{float(value):+.4f}"


def build_markov_detail_sections(
    *,
    removal_effects: Mapping[str, float],
    attribution: Mapping[str, float] | None,
    transition_matrix: pd.DataFrame | None,
    max_channels: int = 10,
    max_transitions: int = 5,
) -> tuple:
    """MODIFIED: distilled static-Markov mechanics for the text renderer.

    Raw signed removal effects are shown next to normalized shares when
    normalization is available.  When it is unavailable, the raw effects
    are rendered alone and are never percentage-formatted.  The full
    transition matrix is summarised as the top probability-mass edges plus
    the top drop-off edges into NULL.
    """

    sections: list = []
    ranked = sorted(
        removal_effects.items(), key=lambda item: (-abs(item[1]), item[0])
    )
    if attribution is None:
        rows = [
            (str(channel), _signed(effect))
            for channel, effect in ranked[:max_channels]
        ]
        if len(ranked) > max_channels:
            rows.append((f"({len(ranked) - max_channels} more channels)", ""))
        if rows:
            sections.append(
                (
                    "Removal effects (raw, signed); normalized attribution unavailable",
                    ("Channel", "Removal effect"),
                    tuple(rows),
                )
            )
    else:
        rows = [
            (
                str(channel),
                _signed(effect),
                _percent(attribution.get(channel, 0.0)),
            )
            for channel, effect in ranked[:max_channels]
        ]
        if len(ranked) > max_channels:
            rows.append((f"({len(ranked) - max_channels} more channels)", "", ""))
        if rows:
            sections.append(
                (
                    "Removal effects (raw, signed) vs normalized shares",
                    ("Channel", "Removal effect", "Share"),
                    tuple(rows),
                )
            )
    if transition_matrix is not None:
        absorbing = {"CONVERSION", "NULL"}
        edges: list[tuple[str, str, float]] = []
        # MODIFIED: iterate positionally — .loc[state] on tuple-valued
        # order>=2 states is interpreted by pandas as multi-axis indexing
        # and raises KeyError.
        matrix_values = transition_matrix.to_numpy()
        columns = list(transition_matrix.columns)
        for row_index, state in enumerate(transition_matrix.index):
            if str(state) in absorbing:
                continue
            for column_index, target in enumerate(columns):
                probability = float(matrix_values[row_index, column_index])
                if probability > 0:
                    edges.append((str(state), str(target), probability))
        edges.sort(key=lambda item: (-item[2], item[0], item[1]))
        top = [
            (f"{left} -> {right}", _percent(probability))
            for left, right, probability in edges[:max_transitions]
        ]
        if top:
            sections.append(
                ("Top transitions", ("Transition", "Probability"), tuple(top))
            )
        drop = [
            (f"{left} -> NULL", _percent(probability))
            for left, right, probability in edges
            if right == "NULL"
        ][:3]
        if drop:
            sections.append(
                (
                    "Top drop-off transitions (bottlenecks)",
                    ("Transition", "Probability"),
                    tuple(drop),
                )
            )
    return tuple(sections)


def _weighted_total(value: object) -> str:
    """Render a weighted total without truncating its fractional part.

    An integer-valued total prints in full, exactly as an unweighted count
    does.  Any other value prints to six significant digits, so a fractional
    total keeps its fraction and a small nonzero total is never shown as
    zero.
    """
    number = float(value)
    if number.is_integer() and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.6g}"


def _signed_weighted(value: object) -> str:
    """Render a signed raw value from a weighted fit.

    Uses the standard four-decimal rendering unless that would print a
    nonzero value as zero, in which case six significant digits are shown.
    """
    number = float(value)
    text = _signed(number)
    if number != 0.0 and float(text) == 0.0:
        return f"{number:+.6g}"
    return text


def build_observation_weight_section(
    *,
    total_weight: float,
    weighted_conversions: float,
    zero_weight_count: int,
) -> tuple:
    """Weighted totals of an explicitly weighted fit, as their own section.

    Kept apart from the observed journey and conversion counts, which stay
    ordinary unweighted counts of the supplied rows.  ``weighted_conversions``
    is the sum of ``weight * outcome`` over every supplied journey, so it
    includes conversions on paths with no channel and need not equal the
    conversions a model credits to channels.
    """

    return (
        "Observation weights",
        ("Metric", "Value"),
        (
            ("Total observation weight", _weighted_total(total_weight)),
            ("Weighted conversions", _weighted_total(weighted_conversions)),
            ("Zero-weight journeys excluded", str(int(zero_weight_count))),
        ),
    )


def build_shapley_detail_sections(
    *,
    shapley_values: Mapping[str, float],
    value_function: Mapping[tuple, float],
    max_coalitions: int = 5,
    max_channels: int = 10,
    weighted: bool = False,
) -> tuple:
    """MODIFIED: distilled static-Shapley mechanics for the text renderer.

    The 2^C coalition space is summarised as the top coalitions by value
    plus the single-channel baselines; raw Shapley values are shown signed.

    ``weighted=True`` marks a fit with explicit observation weights, whose
    coalition totals are weighted conversions and may be fractional.  They
    are then labelled as weighted and rendered without integer truncation,
    and a small nonzero raw value is never printed as zero.  The default
    renders an unweighted fit exactly as before.
    """

    signed = _signed_weighted if weighted else _signed
    total = _weighted_total if weighted else (lambda value: str(int(value)))
    conversions_label = "Weighted conversions" if weighted else "Conversions"

    sections: list = []
    ranked_values = sorted(
        shapley_values.items(), key=lambda item: (-abs(item[1]), item[0])
    )
    value_rows = [
        (str(channel), signed(value)) for channel, value in ranked_values[:max_channels]
    ]
    if len(ranked_values) > max_channels:
        value_rows.append((f"({len(ranked_values) - max_channels} more channels)", ""))
    if value_rows:
        sections.append(
            ("Raw Shapley values (signed)", ("Channel", "Value"), tuple(value_rows))
        )
    coalitions = sorted(
        value_function.items(),
        key=lambda item: (-item[1], " + ".join(map(str, item[0]))),
    )
    coalition_rows = [
        (" + ".join(map(str, coalition)), total(value))
        for coalition, value in coalitions[:max_coalitions]
        if value > 0
    ]
    if coalition_rows:
        sections.append(
            (
                (
                    "Top coalitions by weighted conversions"
                    if weighted
                    else "Top coalitions by conversions"
                ),
                ("Coalition", conversions_label),
                tuple(coalition_rows),
            )
        )
    singles = sorted(
        (
            (coalition[0], value)
            for coalition, value in value_function.items()
            if len(coalition) == 1
        ),
        key=lambda item: (-item[1], item[0]),
    )
    single_rows = [
        (str(channel), total(value)) for channel, value in singles[:max_channels]
    ]
    if single_rows:
        sections.append(
            (
                "Single-channel baselines v({c})",
                ("Channel", conversions_label),
                tuple(single_rows),
            )
        )
    return tuple(sections)


def build_path_diagnostics_section(
    paths: Sequence[str],
    separator: str,
    max_paths: int = 3,
) -> tuple:
    """MODIFIED: scalable path diagnostics — never dumps raw path lists."""

    token_lists = [split_path(str(path), separator) for path in paths if str(path)]
    if not token_lists:
        return (
            "Path diagnostics",
            ("Metric", "Value"),
            (("Non-empty paths", "0"),),
        )
    lengths = [len(tokens) for tokens in token_lists]
    unique_channels = len(set().union(*token_lists))
    counts = Counter(separator.join(tokens) for tokens in token_lists)
    rows: list[tuple[str, str]] = [
        ("Non-empty paths", str(len(token_lists))),
        ("Unique channels", str(unique_channels)),
        ("Average path length", f"{sum(lengths) / len(lengths):.2f}"),
    ]
    for path, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_paths]:
        rows.append((f"Top path: {path}", str(count)))
    return ("Path diagnostics", ("Metric", "Value"), tuple(rows))


def build_markov_epoch_effect_section(
    epoch_labels: Sequence[str],
    epoch_results: Sequence[Mapping[str, Any]],
    top_k: int = 3,
) -> tuple:
    """MODIFIED: temporal Markov distillation — top signed removal effects
    per epoch (bounded at top_k rows per epoch, never the full C x M grid)."""

    rows: list[tuple[str, str, str]] = []
    for label, result in zip(epoch_labels, epoch_results):
        effects = result.get("removal_effects", {}) if result else {}
        ranked = sorted(effects.items(), key=lambda item: (-abs(item[1]), item[0]))
        for channel, effect in ranked[:top_k]:
            rows.append((str(label), str(channel), _signed(effect)))
    if not rows:
        return ()
    return (
        (
            "Top removal effects per epoch (raw, signed)",
            ("Epoch", "Channel", "Removal effect"),
            tuple(rows),
        ),
    )


def build_normalization_unavailable_section(
    epoch_labels: Sequence[str],
    epoch_results: Sequence[Mapping[str, Any]],
) -> tuple:
    """List fitted epochs whose raw values could not form valid shares."""

    rows = []
    for label, result in zip(epoch_labels, epoch_results):
        if result.get("normalization_status") != "unavailable":
            continue
        rows.append(
            (
                str(label),
                str(result.get("normalization_reason") or "unspecified"),
            )
        )
    if not rows:
        return ()
    return (
        (
            "Normalized attribution unavailable by epoch",
            ("Epoch", "Reason"),
            tuple(rows),
        ),
    )


def build_shapley_epoch_coalition_section(
    epoch_labels: Sequence[str],
    epoch_results: Sequence[Mapping[str, Any]],
    top_k: int = 2,
) -> tuple:
    """MODIFIED: temporal Shapley distillation — top coalitions per epoch."""

    rows: list[tuple[str, str, str]] = []
    for label, result in zip(epoch_labels, epoch_results):
        value_function = result.get("value_function", {}) if result else {}
        ranked = sorted(
            value_function.items(),
            key=lambda item: (-item[1], " + ".join(map(str, item[0]))),
        )
        for coalition, value in ranked[:top_k]:
            if value > 0:
                rows.append((str(label), " + ".join(map(str, coalition)), str(int(value))))
    if not rows:
        return ()
    return (
        (
            "Top coalitions per epoch",
            ("Epoch", "Coalition", "Conversions"),
            tuple(rows),
        ),
    )


def build_epoch_metrics(
    *,
    epoch_labels: Sequence[str],
    starts: Sequence[object],
    ends: Sequence[object],
    path_counts: Sequence[int],
    conversion_counts: Sequence[int],
    sample_floors: Sequence[int] | int,
) -> pd.DataFrame:
    """Create the fixed reporting-metric schema in supplied epoch order."""

    labels = list(epoch_labels)
    if isinstance(sample_floors, int):
        floors = [sample_floors] * len(labels)
    else:
        floors = list(sample_floors)
    if not all(
        len(values) == len(labels)
        for values in (starts, ends, path_counts, conversion_counts, floors)
    ):
        raise ValueError("temporal reporting metrics must have equal lengths")

    rows: list[dict[str, object]] = []
    for label, start, end, paths, conversions, floor in zip(
        labels, starts, ends, path_counts, conversion_counts, floors
    ):
        count = int(paths)
        converted = int(conversions)
        required = int(floor)
        rows.append(
            {
                "epoch_label": str(label),
                "start": start,
                "end": end,
                "path_count": count,
                "conversion_count": converted,
                "conversion_rate": (converted / count) if count else np.nan,
                "sample_floor": required,
                "passed_sample_floor": count >= required,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "epoch_label",
            "start",
            "end",
            "path_count",
            "conversion_count",
            "conversion_rate",
            "sample_floor",
            "passed_sample_floor",
        ],
    )


def build_fitted_snapshot(
    *,
    model_name: str,
    epoch_source: str,
    epochs: Sequence[Any],
    epoch_paths: Mapping[Any, Sequence[object]],
    epoch_conversions: Mapping[Any, Sequence[object]],
    epoch_attributions: Mapping[Any, Mapping[str, float]],
    sample_floors: Sequence[int] | int,
    sample_floor: int,
    epoch_labels: Sequence[str] | None = None,
    boundaries: pd.DatetimeIndex | None = None,
    model_details: tuple = (),  # MODIFIED: distilled model sections
) -> TemporalReportingSnapshot:
    """Build a stored snapshot from independently fitted temporal epochs."""

    ordered_epochs = list(epochs)
    labels = list(epoch_labels) if epoch_labels is not None else [str(epoch) for epoch in ordered_epochs]
    if len(labels) != len(ordered_epochs):
        raise ValueError("epoch_labels must align with fitted epochs")
    starts, ends = zip(*(_epoch_dates(epoch) for epoch in ordered_epochs)) if ordered_epochs else ((), ())
    metrics = build_epoch_metrics(
        epoch_labels=labels,
        starts=starts,
        ends=ends,
        path_counts=[len(epoch_paths.get(epoch, ())) for epoch in ordered_epochs],
        conversion_counts=[
            int(np.asarray(epoch_conversions.get(epoch, ()), dtype=int).sum())
            for epoch in ordered_epochs
        ],
        sample_floors=sample_floors,
    )
    if boundaries is None:
        boundaries = _infer_boundaries(starts, ends)
    attribution = pd.DataFrame(
        [dict(epoch_attributions.get(epoch, {})) for epoch in ordered_epochs],
        index=labels,
    )
    attribution = attribution.reindex(sorted(attribution.columns), axis=1)
    return TemporalReportingSnapshot(
        model_name=model_name,
        epoch_source=epoch_source,
        epoch_labels=tuple(str(label) for label in labels),
        boundaries=boundaries,
        epoch_metrics=metrics,
        attribution_matrix=attribution,
        sample_floor=int(sample_floor),
        model_details=tuple(model_details),  # MODIFIED
    )


def render_temporal_report(
    snapshot: TemporalReportingSnapshot,
    output_format: Literal["summary", "report"] = "summary",
) -> str:
    """Render the specified deterministic plain-text temporal output."""

    if output_format != "summary" and output_format != "report":
        raise ValueError("output_format must be 'summary' or 'report'")
    state = _RenderState(snapshot)
    if output_format == "summary":
        return state.summary()
    return state.report()


class _RenderState:
    """Normalized data and small formatting helpers for one render call."""

    def __init__(self, snapshot: TemporalReportingSnapshot) -> None:
        self.snapshot = snapshot
        self.metrics = snapshot.epoch_metrics.copy(deep=True).reset_index(drop=True)
        required = {
            "epoch_label",
            "start",
            "end",
            "path_count",
            "conversion_count",
            "conversion_rate",
            "sample_floor",
            "passed_sample_floor",
        }
        missing = required.difference(self.metrics.columns)
        if missing:
            raise ValueError(
                "temporal reporting snapshot is missing metric columns: "
                + ", ".join(sorted(missing))
            )
        self.labels = tuple(_display(label) for label in snapshot.epoch_labels)
        if len(self.metrics) != len(self.labels):
            raise ValueError("epoch_labels must align with epoch_metrics")
        self.metrics["epoch_label"] = self.labels
        self.attribution = _normalise_attribution(snapshot.attribution_matrix, self.labels)

    @property
    def has_attribution(self) -> bool:
        return self.attribution is not None

    @property
    def is_static(self) -> bool:
        """MODIFIED: static fits render without temporal layout elements."""
        return self.snapshot.epoch_source == "static"

    def summary(self) -> str:
        # MODIFIED: conditional static vs temporal layout. Static fits omit
        # the temporal header lines (Timeline, Boundaries, Sample floor),
        # the Start / End / Health columns, and the adjacent-epoch shift
        # table entirely instead of printing rows of N/A placeholders.
        lines = [
            "Attribution Summary" if self.is_static else "Temporal Attribution Summary",
            f"Model: {self._model_header()}",
            f"Epoch source: {_display(self.snapshot.epoch_source)}",
        ]
        if not self.is_static:
            lines.extend(
                [
                    f"Timeline: {self._timeline()}",
                    f"Boundaries: {self._boundaries()}",
                    f"Sample floor: {int(self.snapshot.sample_floor)} paths",
                ]
            )
        lines.extend([self._overall_line(), "", "Data" if self.is_static else "Epochs"])

        epoch_rows = []
        for index, row in self.metrics.iterrows():
            if self.is_static:
                epoch_rows.append(
                    [
                        _display(row["epoch_label"]),
                        _count(row["path_count"]),
                        _count(row["conversion_count"]),
                        _percent(row["conversion_rate"]),
                        self._top_channels(index),
                    ]
                )
            else:
                epoch_rows.append(
                    [
                        _display(row["epoch_label"]),
                        _date(row["start"]),
                        _date(row["end"]),
                        _count(row["path_count"]),
                        _count(row["conversion_count"]),
                        _percent(row["conversion_rate"]),
                        "PASS" if bool(row["passed_sample_floor"]) else "FAIL",
                        self._top_channels(index),
                    ]
                )
        headers = (
            ["Cohort", "Paths", "Conversions", "Conv. %", "Top channels"]
            if self.is_static
            else [
                "Epoch",
                "Start",
                "End",
                "Paths",
                "Conversions",
                "Conv. %",
                "Health",
                "Top channels",
            ]
        )
        lines.extend(_table(headers, epoch_rows))
        if not self.is_static:
            lines.extend(["", "Adjacent epoch shifts"])
            lines.extend(
                _table(
                    ["Transition", "Channel", "From %", "To %", "Delta"],
                    self._summary_shift_rows(),
                )
            )
        # MODIFIED: bounded model-specific detail sections.
        for title, section_headers, section_rows in self.snapshot.model_details:
            lines.extend(["", _display(title)])
            lines.extend(
                _table(list(section_headers), [list(row) for row in section_rows])
            )
        lines.extend(
            ["", "Interpretation: Descriptive observational changes; not causal effects."]
        )
        return "\n".join(lines).rstrip() + "\n"

    def report(self) -> str:
        # MODIFIED: static fits render a static overview without the
        # temporal timeline / boundary / sample-floor sentences.
        if self.is_static:
            paragraphs = [
                (
                    f"Attribution report for {self._report_identity()}. The model "
                    f"was fitted on {_count(self._total_paths())} paths with "
                    f"{_count(self._total_conversions())} conversions and an overall "
                    f"observed conversion rate of {_percent(self._overall_rate())}."
                )
            ]
            for index, row in self.metrics.iterrows():
                paragraphs.append(
                    f"Cohort {_display(row['epoch_label'])} contains "
                    f"{_count(row['path_count'])} paths and "
                    f"{_count(row['conversion_count'])} conversions, with an observed "
                    f"conversion rate of {_percent(row['conversion_rate'])}. "
                    f"Leading attributed channels: {self._top_channels(index)}."
                )
        else:
            paragraphs = [
                (
                    f"Temporal attribution report for {self._report_identity()}. "
                    f"The {_display(self.snapshot.epoch_source)} timeline runs from "
                    f"{self._timeline_start()} to {self._timeline_end()} across "
                    f"{len(self.metrics)} epochs, with {_count(self._total_paths())} paths, "
                    f"{_count(self._total_conversions())} conversions, and an overall observed "
                    f"conversion rate of {_percent(self._overall_rate())}. "
                    f"Boundaries: {self._boundaries()}."
                )
            ]
            for index, row in self.metrics.iterrows():
                health_word = "passed" if bool(row["passed_sample_floor"]) else "failed"
                paragraphs.append(
                    f"Epoch {_display(row['epoch_label'])} runs from {_date(row['start'])} "
                    f"to {_date(row['end'])}, contains {_count(row['path_count'])} paths "
                    f"and {_count(row['conversion_count'])} conversions, and has an observed "
                    f"conversion rate of {_percent(row['conversion_rate'])}. It {health_word} "
                    f"the {_count(row['sample_floor'])}-path sample floor with "
                    f"{_count(row['path_count'])} paths. Leading attributed channels: "
                    f"{self._top_channels(index)}."
                )
            for left, right in zip(range(len(self.metrics) - 1), range(1, len(self.metrics))):
                paragraphs.append(self._comparison_paragraph(left, right))
        # MODIFIED: bounded model-specific detail sections as prose.
        for title, _headers, section_rows in self.snapshot.model_details:
            entries = "; ".join(", ".join(row) for row in section_rows)
            paragraphs.append(f"{_display(title)}: {entries}.")
        if not self.has_attribution:
            paragraphs.append(
                "Channel attribution requires a fitted temporal model; no channel "
                "interpretation was inferred from path occurrence."
            )
        paragraphs.append(
            "Interpretation: The timeline breaks, conversion-rate movements, and "
            "attribution shifts are descriptive observational results, not causal effects."
        )
        return "\n\n".join(paragraphs).rstrip() + "\n"

    def _model_header(self) -> str:
        return _display(self.snapshot.model_name) if self.snapshot.model_name else "N/A - recommendation only"

    def _report_identity(self) -> str:
        return _display(self.snapshot.model_name) if self.snapshot.model_name else "recommendation only"

    def _timeline(self) -> str:
        if self.snapshot.boundaries is None or len(self.snapshot.boundaries) < 2:
            return "N/A"
        return f"{_date(self.snapshot.boundaries[0])} to {_date(self.snapshot.boundaries[-1])}"

    def _timeline_start(self) -> str:
        if self.snapshot.boundaries is None or len(self.snapshot.boundaries) < 2:
            return "N/A"
        return _date(self.snapshot.boundaries[0])

    def _timeline_end(self) -> str:
        if self.snapshot.boundaries is None or len(self.snapshot.boundaries) < 2:
            return "N/A"
        return _date(self.snapshot.boundaries[-1])

    def _boundaries(self) -> str:
        if self.snapshot.boundaries is None or len(self.snapshot.boundaries) < 2:
            return "N/A"
        return " | ".join(_date(boundary) for boundary in self.snapshot.boundaries)

    def _total_paths(self) -> int:
        return int(pd.to_numeric(self.metrics["path_count"], errors="coerce").fillna(0).sum())

    def _total_conversions(self) -> int:
        return int(pd.to_numeric(self.metrics["conversion_count"], errors="coerce").fillna(0).sum())

    def _overall_rate(self) -> float:
        paths = self._total_paths()
        return self._total_conversions() / paths if paths else float("nan")

    def _overall_line(self) -> str:
        return (
            f"Overall: {_count(self._total_paths())} paths, "
            f"{_count(self._total_conversions())} conversions, "
            f"{_percent(self._overall_rate())} conversion rate"
        )

    def _top_channels(self, epoch_index: int) -> str:
        if self.attribution is None:
            # MODIFIED (get-well Unit 1): a fitted model whose normalization
            # is unavailable is reported as such, not as an unfitted model.
            return self.snapshot.attribution_note or _FIT_REQUIRED
        row = self.attribution.iloc[epoch_index]
        ranked = [
            (_display(channel), float(value))
            for channel, value in row.items()
            if _finite(value) and float(value) > 0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        if not ranked:
            return "N/A"
        return ", ".join(f"{channel} {_percent(value)}" for channel, value in ranked[:3])

    def _shifts(self, left: int, right: int) -> list[tuple[str, float, float, float]] | None:
        if self.attribution is None:
            return None
        before = self.attribution.iloc[left]
        after = self.attribution.iloc[right]
        changes: list[tuple[str, float, float, float]] = []
        for channel in self.attribution.columns:
            from_value = float(before[channel]) if _finite(before[channel]) else 0.0
            to_value = float(after[channel]) if _finite(after[channel]) else 0.0
            delta = to_value - from_value
            if abs(delta) >= _SHIFT_THRESHOLD:
                changes.append((_display(channel), from_value, to_value, delta))
        changes.sort(key=lambda item: (-abs(item[3]), item[0]))
        return changes[:3]

    def _summary_shift_rows(self) -> list[list[str]]:
        if len(self.metrics) < 2:
            return [["N/A", "N/A", "N/A", "N/A", "N/A"]]
        rows: list[list[str]] = []
        for left, right in zip(range(len(self.metrics) - 1), range(1, len(self.metrics))):
            transition = f"{self.labels[left]} -> {self.labels[right]}"
            shifts = self._shifts(left, right)
            if shifts is None:
                rows.append([transition, _FIT_REQUIRED, "N/A", "N/A", "N/A"])
            elif not shifts:
                rows.append([transition, "none >= 5.00 pp", "N/A", "N/A", "N/A"])
            else:
                rows.extend(
                    [
                        [
                            transition,
                            channel,
                            _percent(from_value),
                            _percent(to_value),
                            _pp(delta),
                        ]
                        for channel, from_value, to_value, delta in shifts
                    ]
                )
        return rows

    def _comparison_paragraph(self, left: int, right: int) -> str:
        left_row = self.metrics.iloc[left]
        right_row = self.metrics.iloc[right]
        from_rate = _display_rate(left_row["conversion_rate"])
        to_rate = _display_rate(right_row["conversion_rate"])
        delta = to_rate - from_rate
        if delta > 0:
            direction = "increased"
        elif delta < 0:
            direction = "decreased"
        else:
            direction = "was unchanged"
        magnitude = _display_pp(delta) if delta else "0.00 pp"
        shifts = self._shifts(left, right)
        if shifts is None:
            shift_text = _FIT_REQUIRED
        elif not shifts:
            shift_text = "none >= 5.00 pp"
        else:
            descriptions = []
            for channel, from_value, to_value, change in shifts:
                movement = "increased" if change > 0 else "decreased"
                descriptions.append(
                    f"{channel} {movement} by {_pp(change)} "
                    f"({_percent(from_value)} to {_percent(to_value)})"
                )
            shift_text = ", ".join(descriptions)
        return (
            f"From {self.labels[left]} to {self.labels[right]}, observed conversion "
            f"rate {direction} by {magnitude} percentage points, from "
            f"{_percent(left_row['conversion_rate'])} to {_percent(right_row['conversion_rate'])}. "
            f"Major attribution shifts: {shift_text}."
        )


def _normalise_attribution(
    attribution: pd.DataFrame | None, labels: Sequence[str]
) -> pd.DataFrame | None:
    if attribution is None:
        return None
    frame = attribution.copy(deep=True)
    if len(frame) != len(labels):
        raise ValueError("attribution_matrix must align with epoch_labels")
    frame.index = list(labels)
    frame.columns = [_display(column) for column in frame.columns]
    # Duplicate display labels are intentionally combined so display sanitising
    # cannot make table ordering depend on source column order.
    frame = frame.T.groupby(level=0, sort=True).sum(min_count=1).T
    return frame.reindex(sorted(frame.columns), axis=1).apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _epoch_dates(epoch: Any) -> tuple[object, object]:
    if isinstance(epoch, pd.Interval):
        return epoch.left, epoch.right
    if isinstance(epoch, pd.Period):
        return (
            pd.Timestamp(epoch.start_time).normalize(),
            pd.Timestamp(epoch.end_time).normalize() + pd.Timedelta(days=1),
        )
    return pd.NaT, pd.NaT


def _infer_boundaries(
    starts: Sequence[object], ends: Sequence[object]
) -> pd.DatetimeIndex | None:
    if not starts or any(pd.isna(value) for value in (*starts, *ends)):
        return None
    values = pd.DatetimeIndex(pd.to_datetime([*starts, *ends])).unique().sort_values()
    return values if len(values) >= 2 else None


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    rendered_headers = [_display(header) for header in headers]
    rendered_rows = [[_display(cell) for cell in row] for row in rows]
    widths = [len(header) for header in rendered_headers]
    for row in rendered_rows:
        if len(row) != len(widths):
            raise ValueError("all temporal-report table rows must match the header width")
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_row(row: Sequence[str]) -> str:
        return "|" + "|".join(f" {cell.ljust(width)} " for cell, width in zip(row, widths)) + "|"

    lines = [border, render_row(rendered_headers), border]
    lines.extend(render_row(row) for row in rendered_rows)
    lines.append(border)
    return lines


def _display(value: object) -> str:
    if value is None:
        return _UNAVAILABLE
    return re.sub(r"[\r\n\t]+", " ", str(value)).strip() or _UNAVAILABLE


def _date(value: object) -> str:
    if value is None or pd.isna(value):
        return _UNAVAILABLE
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return _UNAVAILABLE


def _finite(value: object) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _count(value: object) -> str:
    if not _finite(value):
        return _UNAVAILABLE
    return str(int(float(value)))


def _percent(value: object) -> str:
    if not _finite(value):
        return _UNAVAILABLE
    return f"{float(value) * 100:.2f}%"


def _display_rate(value: object) -> float:
    if not _finite(value):
        return 0.0
    return float(f"{float(value) * 100:.2f}")


def _pp(value: object) -> str:
    if not _finite(value):
        return _UNAVAILABLE
    return f"{float(value) * 100:+.2f} pp"


def _display_pp(value: object) -> str:
    """Format a percentage-point value that is already on the 0--100 scale."""

    if not _finite(value):
        return _UNAVAILABLE
    return f"{float(value):+.2f} pp"


__all__ = [
    "TemporalReportingSnapshot",
    "build_epoch_metrics",
    "build_fitted_snapshot",
    "render_temporal_report",
    # MODIFIED: distilled model-specific section builders.
    "build_markov_detail_sections",
    "build_shapley_detail_sections",
    "build_observation_weight_section",
    "build_path_diagnostics_section",
    "build_markov_epoch_effect_section",
    "build_shapley_epoch_coalition_section",
]
