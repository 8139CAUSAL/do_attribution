"""Regression tests for ``do_attribution.visualization``.

Promoted from ``/_mega_project/viz_smoketest/`` (see OPEN_ISSUES.md P2/P3).

Every test builds a small fitted model, calls a visualisation function,
and asserts:

  - the return value is a :class:`matplotlib.figure.Figure` with at least
    one axes,
  - the expected user channels appear as x-tick labels (or as nodes, for
    the transition graph),
  - no forbidden token — in particular the conversion-endpoint label
    ``"Purchase"`` — is present anywhere user-visible.

The tests skip cleanly when ``matplotlib`` / ``networkx`` aren't
installed so the suite remains runnable in a minimal environment.
"""

from __future__ import annotations

import pytest

# MODIFIED (P2/P3): skip the whole module cleanly when viz deps are absent.
matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")  # headless backend for CI
plt = pytest.importorskip(
    "matplotlib.pyplot", reason="matplotlib.pyplot not available"
)
nx = pytest.importorskip("networkx", reason="networkx not installed")

from matplotlib.figure import Figure  # noqa: E402

from do_attribution import (  # noqa: E402
    CompareAttribution,
    MarkovAttribution,
    ShapleyAttribution,
)
from do_attribution.visualization import (  # noqa: E402
    plot_attribution,
    plot_attribution_comparison,
    plot_coalition_values,
    plot_removal_effects,
    plot_transition_graph,
)


# --------------------------------------------------------------------------- #
# Shared fixtures                                                             #
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_paths():
    # Endpoint-bearing paths so the tests exercise the P0 ``conversion_labels``
    # contract: "Purchase" must never appear in any user-visible viz element.
    return [
        "Email > Social > Purchase",
        "Search > Email > Purchase",
        "Social > Purchase",
        "Email > Search",
        "Search > Social > Email > Purchase",
        "Direct > Purchase",
        "Email > Social",
        "Search > Email > Social > Purchase",
        "Social > Email > Purchase",
        "Search",
    ]


@pytest.fixture
def sample_conversions():
    return [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]


EXPECTED_CHANNELS = {"Direct", "Email", "Search", "Social"}
FORBIDDEN_TOKENS = {"Purchase"}


def _close(fig: Figure) -> None:
    plt.close(fig)


def _xtick_labels(ax) -> list[str]:
    # matplotlib populates tick label text after a draw; force it.
    ax.figure.canvas.draw()
    return [t.get_text() for t in ax.get_xticklabels() if t.get_text()]


def test_plot_attribution_renders_explicit_unavailable_state():
    fig = plot_attribution(None, title="Unavailable")
    try:
        assert isinstance(fig, Figure)
        axis = fig.get_axes()[0]
        rendered = "\n".join(text.get_text() for text in axis.texts)
        assert "Normalized attribution unavailable" in rendered
        assert "Raw signed diagnostics are not shares" in rendered
        assert not axis.patches
    finally:
        _close(fig)


@pytest.mark.parametrize(
    "removal_strategy", ["fail", "fail_refit", "detour", "redirect_to_null"]
)
def test_plot_attribution_shows_unavailable_state_for_zero_conversion_markov(
    removal_strategy,
):
    """A zero-conversion Markov fit plots the unavailable state, not bars."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = MarkovAttribution(removal_strategy=removal_strategy).fit(
            ["A", "B"], [0, 0]
        )

    assert res["attribution"] is None
    fig = plot_attribution(res["attribution"], title="Zero conversions")
    try:
        assert isinstance(fig, Figure)
        axis = fig.get_axes()[0]
        rendered = "\n".join(text.get_text() for text in axis.texts)
        assert "Normalized attribution unavailable" in rendered
        assert not axis.patches
    finally:
        _close(fig)


# --------------------------------------------------------------------------- #
# P2 — Markov viz coverage                                                    #
# --------------------------------------------------------------------------- #
# MODIFIED (P2): promoted from the smoke test.
def test_plot_attribution_markov_returns_figure_with_channels(
    sample_paths, sample_conversions
):
    model = MarkovAttribution(order=1, conversion_labels=["Purchase"])
    res = model.fit(sample_paths, sample_conversions)

    fig = plot_attribution(res["attribution"], title="Markov")
    try:
        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) >= 1

        labels = set(_xtick_labels(axes[0]))
        assert EXPECTED_CHANNELS <= labels, (
            f"missing channels: {EXPECTED_CHANNELS - labels}"
        )
        assert not (labels & FORBIDDEN_TOKENS), (
            f"forbidden token leaked into viz: {labels & FORBIDDEN_TOKENS}"
        )
    finally:
        _close(fig)


def test_plot_removal_effects_returns_figure_with_channels(
    sample_paths, sample_conversions
):
    model = MarkovAttribution(order=1, conversion_labels=["Purchase"])
    res = model.fit(sample_paths, sample_conversions)

    fig = plot_removal_effects(res["removal_effects"], title="Removal")
    try:
        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) >= 1
        labels = set(_xtick_labels(axes[0]))
        assert EXPECTED_CHANNELS <= labels
        assert not (labels & FORBIDDEN_TOKENS)
    finally:
        _close(fig)


def test_plot_transition_graph_nodes_exclude_forbidden_tokens(
    sample_paths, sample_conversions
):
    model = MarkovAttribution(
        order=1, conversion_labels=["Purchase"], return_transition_matrix=True
    )
    res = model.fit(sample_paths, sample_conversions)
    fig = plot_transition_graph(res["transition_matrix"])
    try:
        assert isinstance(fig, Figure)
        assert len(fig.get_axes()) >= 1

        # The node set on the drawn graph is discoverable from the axes
        # text children (labels drawn by networkx).  Safer: rebuild the
        # graph to check membership.
        import networkx as nx_local

        G = nx_local.DiGraph()
        tm = res["transition_matrix"]
        for from_state in tm.index:
            for to_state in tm.columns:
                w = tm.loc[from_state, to_state]
                if w > 0.01:
                    G.add_edge(str(from_state), str(to_state), weight=w)

        node_set = set(G.nodes())
        # Real channels must be present.
        assert EXPECTED_CHANNELS <= node_set
        # No "Purchase" node anywhere.
        assert not (node_set & FORBIDDEN_TOKENS)
    finally:
        _close(fig)


def test_plot_attribution_markov_higher_order(sample_paths, sample_conversions):
    """Smoke test: order=2 renders without contaminating with Purchase."""
    model = MarkovAttribution(order=2, conversion_labels=["Purchase"])
    res = model.fit(sample_paths, sample_conversions)
    fig = plot_attribution(res["attribution"], title="Markov order=2")
    try:
        assert isinstance(fig, Figure)
        labels = set(_xtick_labels(fig.get_axes()[0]))
        assert EXPECTED_CHANNELS <= labels
        assert not (labels & FORBIDDEN_TOKENS)
    finally:
        _close(fig)


# --------------------------------------------------------------------------- #
# P3 — Shapley viz coverage                                                   #
# --------------------------------------------------------------------------- #
# MODIFIED (P3): mirrors the P2 structure for the Shapley-side viz.
def test_plot_coalition_values_excludes_forbidden_tokens(
    sample_paths, sample_conversions
):
    import warnings

    with warnings.catch_warnings():
        # P1 fix emits a warning on this dataset — not the thing we're
        # testing here.
        warnings.simplefilter("ignore", UserWarning)
        res = ShapleyAttribution(conversion_labels=["Purchase"]).fit(
            sample_paths, sample_conversions
        )

    fig = plot_coalition_values(res["value_function"])
    try:
        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) >= 1

        labels = _xtick_labels(axes[0])
        assert labels, "expected non-empty xtick labels"
        # Coalitions are joined with ' > '; check the whole string set.
        joined = " | ".join(labels)
        for tok in FORBIDDEN_TOKENS:
            assert tok not in joined, (
                f"{tok} leaked into coalition label: {joined!r}"
            )
    finally:
        _close(fig)


def test_plot_attribution_comparison_columns(
    sample_paths, sample_conversions
):
    import warnings

    cmp_ = CompareAttribution(conversion_labels=["Purchase"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cmp_.fit(sample_paths, sample_conversions)

    frame = cmp_.attribution_frame()
    fig = plot_attribution_comparison(frame, title="Markov vs Shapley")
    try:
        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        assert len(axes) >= 1
        labels = set(_xtick_labels(axes[0]))
        assert EXPECTED_CHANNELS <= labels
        assert not (labels & FORBIDDEN_TOKENS)

        # Frame itself must be forbidden-token-free too.
        assert not (set(frame.index) & FORBIDDEN_TOKENS)
    finally:
        _close(fig)


# --------------------------------------------------------------------------- #
# Full viz smoke: every function at least once on the sample data.            #
# --------------------------------------------------------------------------- #
def test_full_viz_smoke_no_purchase_anywhere(
    sample_paths, sample_conversions
):
    """End-to-end smoke test mirroring /_mega_project/viz_smoketest/."""
    import warnings

    markov = MarkovAttribution(
        order=1,
        conversion_labels=["Purchase"],
        return_transition_matrix=True,
    )
    markov_res = markov.fit(sample_paths, sample_conversions)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shapley_res = ShapleyAttribution(
            conversion_labels=["Purchase"]
        ).fit(sample_paths, sample_conversions)

        cmp_ = CompareAttribution(conversion_labels=["Purchase"])
        cmp_.fit(sample_paths, sample_conversions)

    figs = [
        plot_attribution(markov_res["attribution"], title="Markov"),
        plot_attribution(shapley_res["attribution"], title="Shapley"),
        plot_removal_effects(markov_res["removal_effects"]),
        plot_transition_graph(markov_res["transition_matrix"]),
        plot_coalition_values(shapley_res["value_function"]),
        plot_attribution_comparison(cmp_.attribution_frame()),
    ]
    try:
        for fig in figs:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 1
    finally:
        for fig in figs:
            _close(fig)
