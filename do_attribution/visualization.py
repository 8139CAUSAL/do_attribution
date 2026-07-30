"""
do_attribution.visualization
---------------------------------

Visualisation helpers for both attribution models plus a side-by-side
plot for comparing their outputs.  ``matplotlib`` is imported lazily so
the library remains usable in environments where it is not installed.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd


# --------------------------------------------------------------------------- #
# Attribution bar charts                                                      #
# --------------------------------------------------------------------------- #
def plot_attribution(
    attribution: Optional[Mapping[str, float]],
    title: str = "Channel Attribution",
    figsize: Tuple[int, int] = (10, 6),
):
    """Bar chart of normalized shares, or an explicit unavailable state."""
    import matplotlib.pyplot as plt  # lazy import

    fig, ax = plt.subplots(figsize=figsize)
    if attribution is None:
        ax.text(
            0.5,
            0.5,
            "Normalized attribution unavailable\n"
            "Raw signed diagnostics are not shares.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title(title)
        ax.set_axis_off()
        plt.tight_layout()
        return fig

    sorted_attr = dict(
        sorted(attribution.items(), key=lambda x: x[1], reverse=True)
    )

    channels = list(sorted_attr.keys())
    values = list(sorted_attr.values())
    bars = ax.bar(channels, values)

    if values:
        v_min, v_max = min(values), max(values)
        colors = plt.cm.Blues(
            [(v - v_min) / (v_max - v_min + 0.001) for v in values]
        )
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    ax.set_xlabel("Channel")
    ax.set_ylabel("Attribution")
    ax.set_title(title)

    for i, (channel, value) in enumerate(sorted_attr.items()):
        ax.text(i, value + 0.01, f"{value:.1%}", ha="center", va="bottom")

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: "{:.0%}".format(y))
    )
    plt.tight_layout()
    return fig


def plot_removal_effects(
    removal_effects: Mapping[str, float],
    title: str = "Channel Removal Effects",
    figsize: Tuple[int, int] = (10, 6),
):
    """Bar chart of raw Markov removal effects (can be negative)."""
    import matplotlib.pyplot as plt  # lazy import

    sorted_effects = dict(
        sorted(removal_effects.items(), key=lambda x: x[1], reverse=True)
    )
    fig, ax = plt.subplots(figsize=figsize)

    channels = list(sorted_effects.keys())
    values = list(sorted_effects.values())
    bars = ax.bar(channels, values)

    if values:
        v_min, v_max = min(values), max(values)
        colors = plt.cm.Reds(
            [(v - v_min) / (v_max - v_min + 0.001) for v in values]
        )
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    ax.set_xlabel("Channel")
    ax.set_ylabel("Removal Effect")
    ax.set_title(title)
    for i, (channel, value) in enumerate(sorted_effects.items()):
        ax.text(i, value + 0.005, f"{value:.3f}", ha="center", va="bottom")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Markov transition graph                                                     #
# --------------------------------------------------------------------------- #
def plot_transition_graph(
    transition_matrix: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    min_edge_weight: float = 0.01,
):
    """Draw the Markov-chain transition graph using NetworkX."""
    import matplotlib.pyplot as plt  # lazy import
    import networkx as nx  # lazy import

    G = nx.DiGraph()
    for from_state in transition_matrix.index:
        for to_state in transition_matrix.columns:
            weight = transition_matrix.loc[from_state, to_state]
            if weight > min_edge_weight:
                G.add_edge(str(from_state), str(to_state), weight=weight)

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    node_colors = []
    for node in G.nodes():
        if node == "START":
            node_colors.append("lightgreen")
        elif node == "CONVERSION":
            node_colors.append("lightcoral")
        elif node == "NULL":
            node_colors.append("lightgray")
        else:
            node_colors.append("lightblue")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    edges = list(G.edges())
    weights = [G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(
        G,
        pos,
        width=[w * 5 for w in weights],
        alpha=0.6,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        ax=ax,
    )
    edge_labels = {
        (u, v): f"{G[u][v]['weight']:.2f}"
        for u, v in edges
        if G[u][v]["weight"] > 0.05
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    ax.set_title("Markov Chain Transition Graph")
    ax.axis("off")
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Shapley coalition bar chart                                                 #
# --------------------------------------------------------------------------- #
def plot_coalition_values(
    value_function: Mapping[Tuple[str, ...], int],
    title: str = "Value of Channel Coalitions",
    figsize: Tuple[int, int] = (12, 8),
):
    """Bar chart of raw conversion totals per Shapley coalition."""
    import matplotlib.pyplot as plt  # lazy import

    records = [
        {"Coalition": " > ".join(coal), "Value": val, "Size": len(coal)}
        for coal, val in value_function.items()
    ]
    df = pd.DataFrame(records).sort_values(["Size", "Value"])

    fig, ax = plt.subplots(figsize=figsize)
    df.plot.bar(x="Coalition", y="Value", ax=ax, legend=False)
    ax.set_ylabel("Conversions")
    ax.set_title(title)
    ax.set_xticklabels(df["Coalition"], rotation=90, ha="right")
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Side-by-side comparison                                                     #
# --------------------------------------------------------------------------- #
def plot_attribution_comparison(
    frame: pd.DataFrame,
    title: str = "Attribution: Markov vs Shapley",
    figsize: Tuple[int, int] = (12, 6),
):
    """Grouped bar chart comparing the columns of an attribution frame."""
    import matplotlib.pyplot as plt  # lazy import

    fig, ax = plt.subplots(figsize=figsize)
    frame.plot.bar(ax=ax)
    ax.set_ylabel("Attribution share")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: "{:.0%}".format(y))
    )
    ax.legend(title="Model")
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Temporal epoch recommendation diagnostic                                    #
# --------------------------------------------------------------------------- #
def plot_epoch_recommendation(
    recommendation,
    *,
    show_raw_signal: bool = True,
    show_prepared_signal: bool = False,
    show_detector_signal: bool = True,
    show_raw_breakpoints: bool = True,
    show_segment_fits: bool = True,
    show_path_counts: bool = True,
    annotate_actions: bool = True,
    title: str = "Epoch Boundary Diagnostic",
    figsize: Tuple[int, int] = (12, 8),
):
    """Plot an :class:`EpochRecommendation` without joining affine pieces.

    Matplotlib is intentionally imported here, rather than at module import,
    so this diagnostic does not turn the visualization extra into a core
    dependency.  Transition plots show adjacent Frobenius distance only as a
    readable diagnostic; their PELT input is the aligned matrix-vector table.
    """
    import matplotlib.pyplot as plt  # lazy import

    is_transition = recommendation.signal_name == "transition_drift"
    if show_path_counts:
        fig, (ax, count_ax) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        count_ax = None

    if is_transition:
        ax.plot(
            recommendation.signal.index, recommendation.signal.values,
            color="tab:purple", marker="o", label="Adjacent Frobenius distance (diagnostic)",
        )
        ax.set_ylabel("Frobenius distance")
        ax.set_title(f"{title}\nPELT used aligned order-1 transition-matrix vectors, not this distance series")
    else:
        if show_raw_signal:
            ax.plot(
                recommendation.signal.index, recommendation.signal.values,
                color="0.45", marker="o", markersize=3, linewidth=1, label="Observed conversion rate",
            )
        decomposition = recommendation.decomposition
        if show_prepared_signal and decomposition is not None:
            ax.plot(
                decomposition.prepared.index, decomposition.prepared.values,
                color="tab:orange", alpha=0.75, linewidth=1.2, label="Prepared / imputed signal",
            )
        if show_detector_signal:
            ax.plot(
                recommendation.detector_signal.index, recommendation.detector_signal.values,
                color="tab:blue", linewidth=1.7, label="Affine detector input",
            )
        if show_segment_fits:
            for fit in recommendation.segment_fits:
                # SegmentFit stores the mid-point level and slope.  Draw each
                # segment separately so a level gap is never bridged at a knot.
                span = (fit.end - fit.start) / pd.Timedelta(days=1)
                end = fit.end - pd.Timedelta(days=1)
                if end < fit.start:
                    end = fit.end
                y0 = fit.midpoint_value - fit.slope * span / 2.0
                y1 = fit.midpoint_value + fit.slope * span / 2.0
                ax.plot([fit.start, end], [y0, y1], color="black", linewidth=2.0, label="Fitted affine epoch" if fit is recommendation.segment_fits[0] else None)
        ax.set_ylabel("Signal")
        ax.set_title(title)

    boundaries = pd.DatetimeIndex(recommendation.boundaries)
    raw = set(pd.DatetimeIndex(recommendation.raw_breakpoints))
    final_internal = set(boundaries[1:-1])
    score_frame = recommendation.boundary_scores
    pin_boundaries = set()
    if isinstance(score_frame, pd.DataFrame) and "source" in score_frame.columns:
        pin_boundaries = set(score_frame.index[score_frame["source"] == "pin"])

    for timestamp in sorted(final_internal):
        ax.axvline(timestamp, color="tab:red", linewidth=1.5, label="Final boundary" if timestamp == min(final_internal) else None)
    if show_raw_breakpoints:
        for timestamp in sorted(raw - final_internal):
            ax.axvline(timestamp, color="tab:gray", linestyle="--", linewidth=1.1, label="Rejected raw candidate" if timestamp == min(raw - final_internal) else None)
    for timestamp in sorted(pin_boundaries & final_internal):
        ax.axvline(timestamp, color="tab:green", linestyle=":", linewidth=2.0, label="Pinned boundary" if timestamp == min(pin_boundaries & final_internal) else None)

    if annotate_actions:
        ymax = ax.get_ylim()[1]
        for action in recommendation.merge_log:
            ax.annotate(action.reason, xy=(action.removed_boundary, ymax), xytext=(0, -8), textcoords="offset points", rotation=90, va="top", fontsize=7, color="tab:gray")
        for action in recommendation.pin_log:
            if action.boundary is not None and action.action.startswith("pin"):
                ax.annotate("pin", xy=(action.boundary, ymax), xytext=(0, -22), textcoords="offset points", rotation=90, va="top", fontsize=7, color="tab:green")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    ax.grid(alpha=0.2)

    if count_ax is not None:
        counts = []
        starts = []
        widths = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            series = recommendation.path_counts
            count = int(series.loc[(series.index >= left) & (series.index < right)].sum())
            counts.append(count)
            starts.append(left)
            widths.append((right - left) / pd.Timedelta(days=1))
        count_ax.bar(starts, counts, width=widths, align="edge", color="tab:blue", alpha=0.55, label="Paths per final epoch")
        count_ax.axhline(recommendation.min_samples, color="tab:red", linestyle="--", linewidth=1.2, label=f"Statistical floor ({recommendation.min_samples})")
        count_ax.set_ylabel("Paths")
        count_ax.legend(loc="best")
        count_ax.grid(axis="y", alpha=0.2)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


__all__ = [
    "plot_attribution",
    "plot_removal_effects",
    "plot_transition_graph",
    "plot_coalition_values",
    "plot_attribution_comparison",
    "plot_epoch_recommendation",
]
