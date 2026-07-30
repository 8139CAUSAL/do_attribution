"""Acceptance coverage for the dependency-free temporal text renderer."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from do_attribution import (
    MarkovAttribution,
    ShapleyAttribution,
    TemporalMarkovAttribution,
    TemporalReportingSnapshot,
    TemporalShapleyAttribution,
    recommend_epochs,
)
from do_attribution.reporting import build_epoch_metrics, render_temporal_report


def _paths_by_day() -> tuple[list[str], list[int], list[pd.Timestamp]]:
    paths: list[str] = []
    conversions: list[int] = []
    anchors: list[pd.Timestamp] = []
    for day in range(20):
        for sample in range(10):
            paths.append("A > B" if day < 10 else "B > C")
            conversions.append(int(sample < (3 if day < 10 else 7)))
            anchors.append(pd.Timestamp("2025-01-01") + pd.Timedelta(days=day))
    return paths, conversions, anchors


def _recommendation():
    paths, conversions, anchors = _paths_by_day()
    return recommend_epochs(
        paths,
        conversions,
        anchors,
        min_samples=20,
        min_epoch_width="4D",
        stl_period=None,
    )


def test_standalone_recommendation_summary_golden_string():
    actual = _recommendation().summary()
    expected = """Temporal Attribution Summary
Model: N/A - recommendation only
Epoch source: recommended:conversion_rate
Timeline: 2025-01-01 to 2025-01-21
Boundaries: 2025-01-01 | 2025-01-11 | 2025-01-21
Sample floor: 20 paths
Overall: 200 paths, 100 conversions, 50.00% conversion rate

Epochs
+-------+------------+------------+-------+-------------+---------+--------+--------------------+
| Epoch | Start      | End        | Paths | Conversions | Conv. % | Health | Top channels       |
+-------+------------+------------+-------+-------------+---------+--------+--------------------+
| E1    | 2025-01-01 | 2025-01-11 | 100   | 30          | 30.00%  | PASS   | N/A - fit required |
| E2    | 2025-01-11 | 2025-01-21 | 100   | 70          | 70.00%  | PASS   | N/A - fit required |
+-------+------------+------------+-------+-------------+---------+--------+--------------------+

Adjacent epoch shifts
+------------+--------------------+--------+------+-------+
| Transition | Channel            | From % | To % | Delta |
+------------+--------------------+--------+------+-------+
| E1 -> E2   | N/A - fit required | N/A    | N/A  | N/A   |
+------------+--------------------+--------+------+-------+

Interpretation: Descriptive observational changes; not causal effects.
"""
    assert actual == expected


@pytest.mark.parametrize("model_type", [TemporalShapleyAttribution, TemporalMarkovAttribution])
def test_fitted_models_render_summary_and_report_without_refitting(model_type):
    paths, conversions, anchors = _paths_by_day()
    model = model_type(freq="Q", min_samples_per_epoch=20)
    model.fit(paths, anchors, conversions)

    summary = model.summary()
    report = model.summary(output_format="report")

    assert type(summary) is str
    assert type(report) is str
    assert "Timeline: 2025-01-01 to 2025-04-01" in summary
    assert "Overall: 200 paths, 100 conversions, 50.00% conversion rate" in summary
    assert "Leading attributed channels:" in report
    assert "descriptive observational results, not causal effects" in report
    assert "P(CONVERSION | START)" not in summary
    assert all(not line.endswith(" ") for line in (summary + report).splitlines())
    assert summary.endswith("\n") and not summary.endswith("\n\n")
    assert report.endswith("\n") and not report.endswith("\n\n")
    assert model.summary() == summary


def test_recommendation_attached_to_fit_includes_channel_sections():
    paths, conversions, anchors = _paths_by_day()
    model = TemporalShapleyAttribution(min_samples_per_epoch=20).fit_recommended(
        paths,
        conversions,
        anchors,
        recommender_kwargs={"min_epoch_width": "4D", "stl_period": None},
    )

    summary = model.summary()
    attached = model.epoch_recommendation.summary(output_format="report")

    assert "Model: TemporalShapleyAttribution" in summary
    assert "A 50.00%, B 50.00%" in summary
    assert "N/A - fit required" not in summary
    assert "From E1 to E2, observed conversion rate increased by +40.00 pp" in attached


def test_errors_and_utf8_file_round_trip(tmp_path):
    with pytest.raises(RuntimeError, match=re.escape("summary() requires a fitted temporal model")):
        TemporalShapleyAttribution().summary()
    paths, conversions, anchors = _paths_by_day()
    model = TemporalMarkovAttribution(freq="Q", min_samples_per_epoch=20)
    model.fit(paths, anchors, conversions)
    with pytest.raises(ValueError, match=re.escape("output_format must be 'summary' or 'report'")):
        model.summary(output_format="html")

    report = model.summary(output_format="report")
    for suffix in (".txt", ".md"):
        destination = tmp_path / f"temporal_report{suffix}"
        destination.write_text(report, encoding="utf-8")
        assert destination.read_text(encoding="utf-8") == report


def test_ties_and_embedded_whitespace_are_deterministic():
    metrics = build_epoch_metrics(
        epoch_labels=["E\n1", "E\t2"],
        starts=[pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")],
        ends=[pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-01")],
        path_counts=[10, 10],
        conversion_counts=[2, 3],
        sample_floors=5,
    )
    first = TemporalReportingSnapshot(
        model_name="TemporalShapleyAttribution",
        epoch_source="explicit_labels",
        epoch_labels=("E\n1", "E\t2"),
        boundaries=None,
        epoch_metrics=metrics,
        attribution_matrix=pd.DataFrame(
            [[0.5, 0.5], [0.5, 0.5]], columns=["B", "A\r"], index=["x", "y"]
        ),
        sample_floor=5,
    )
    second = TemporalReportingSnapshot(
        model_name=first.model_name,
        epoch_source=first.epoch_source,
        epoch_labels=first.epoch_labels,
        boundaries=first.boundaries,
        epoch_metrics=first.epoch_metrics.copy(deep=True),
        attribution_matrix=first.attribution_matrix[["A\r", "B"]],
        sample_floor=first.sample_floor,
    )

    rendered = render_temporal_report(first)
    assert rendered == render_temporal_report(second)
    assert "E 1" in rendered and "E 2" in rendered
    assert "A 50.00%, B 50.00%" in rendered
    assert "\r" not in rendered and "\t" not in rendered


def test_explicit_labels_preserve_first_seen_order_and_render_na_dates():
    model = TemporalShapleyAttribution(min_samples_per_epoch=1)
    model.fit_by_epoch(
        ["A", "B", "A", "B"],
        ["Later\nlabel", "Earlier\tlabel", "Later\nlabel", "Earlier\tlabel"],
        [1, 0, 1, 0],
    )

    summary = model.summary()
    assert summary.index("Later label") < summary.index("Earlier label")
    assert "Timeline: N/A" in summary
    assert "Boundaries: N/A" in summary


@pytest.mark.parametrize("model_type", [ShapleyAttribution, MarkovAttribution])
def test_static_models_use_the_same_text_engine(model_type):
    model = model_type()
    with pytest.raises(RuntimeError, match=re.escape("summary() requires a fitted attribution model")):
        model.summary()

    model.fit(["A > B", "A > B", "B"], [1, 0, 1])
    summary = model.summary()
    report = model.summary(output_format="report")

    assert "Epoch source: static" in summary
    assert "All paths" in summary
    assert "Overall: 3 paths, 2 conversions, 66.67% conversion rate" in summary
    assert type(report) is str and report.endswith("\n")
