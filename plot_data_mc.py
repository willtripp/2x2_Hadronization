"""
plot_data_mc.py

Utility for making particle-physics-style Data vs. MC comparison plots
from pandas DataFrames

Usage:
    from plot_data_mc import plot_data_mc

    mc_df   = pd.read_csv("MC_Particles_Test.csv")
    data_df = pd.read_csv("Data_Particles_Test.csv")

    plot_data_mc(
        data_df, mc_df,
        column="thickness",
        bins=np.linspace(0, 5, 51),
        xlabel="Track thickness [size / length]",
        title="Track thickness: Data vs MC",
        outfile="thickness_data_mc.png",
    )
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_data_mc(
    data_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    column: str,
    bins,
    xlabel: str = None,
    ylabel: str = "Entries",
    title: str = None,
    mc_label: str = "MC",
    data_label: str = "Data",
    mc_weight_col: str = None,
    normalize_mc_to_data: bool = False,
    log_y: bool = False,
    outfile: str = None,
    category_col: str = None,
):
    """
    Make a Data vs MC comparison histogram with a ratio panel.

    Parameters
    ----------
    data_df, mc_df : pd.DataFrame
        DataFrames containing at least `column` (and `category_col` if used).
    column : str
        Name of the column to histogram (e.g. "thickness").
    bins : int or array-like
        Passed straight to np.histogram / plt.hist.
    mc_weight_col : str, optional
        Column in mc_df to use as per-entry weights (e.g. for cross-section
        normalization). If None, unweighted counts are used.
    normalize_mc_to_data : bool
        If True, scale total MC yield to match total Data yield (shape
        comparison only — useful before you have real POT/livetime norm).
    category_col : str, optional
        If given, MC is stacked by unique values in mc_df[category_col]
        (e.g. particle species / pid) instead of drawn as one histogram.
    outfile : str, optional
        If given, save the figure to this path.

    Returns
    -------
    fig, (ax_main, ax_ratio)
    """

    data_vals = data_df[column].to_numpy()
    mc_vals = mc_df[column].to_numpy()
    mc_weights = mc_df[mc_weight_col].to_numpy() if mc_weight_col else np.ones(len(mc_vals))

    if normalize_mc_to_data and mc_weights.sum() > 0:
        mc_weights = mc_weights * (len(data_vals) / mc_weights.sum())

    bin_edges = np.histogram_bin_edges(np.concatenate([data_vals, mc_vals]), bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    # --- Data histogram ---
    data_counts, _ = np.histogram(data_vals, bins=bin_edges)
    data_errs = np.sqrt(data_counts)

    # --- Figure layout: main panel + ratio panel ---
    fig, (ax_main, ax_ratio) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(7, 6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # --- MC histogram ---
    if category_col is not None:
        #dropna() removed non-alues
        categories = sorted(mc_df[category_col].dropna().unique())
        stacked_vals = [mc_df.loc[mc_df[category_col] == c, column].to_numpy() for c in categories]
        stacked_weights = [
            (mc_df.loc[mc_df[category_col] == c, mc_weight_col].to_numpy()
             if mc_weight_col else np.ones((mc_df[category_col] == c).sum()))
            for c in categories
        ]
        mc_counts_stack, _, _ = ax_main.hist(
            stacked_vals, bins=bin_edges, weights=stacked_weights,
            stacked=True, histtype="stepfilled", alpha=0.7,
            label=[f"{mc_label}: {c}" for c in categories],
        )
        mc_counts = np.sum(mc_counts_stack, axis=0) if len(categories) > 1 else np.asarray(mc_counts_stack)
    else:
        mc_counts, _, _ = ax_main.hist(
            mc_vals, bins=bin_edges, weights=mc_weights,
            histtype="stepfilled", color="steelblue", alpha=0.6, label=mc_label,
        )

    # --- Data points ---
    ax_main.errorbar(
        bin_centers, data_counts, yerr=data_errs, xerr=bin_widths / 2,
        fmt="ko", markersize=4, capsize=0, label=data_label,
    )

    ax_main.set_ylabel(ylabel)
    if title:
        ax_main.set_title(title)
    if log_y:
        ax_main.set_yscale("log")
    ax_main.legend(frameon=False)

    # --- Ratio panel: Data / MC ---
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mc_counts > 0, data_counts / mc_counts, np.nan)
        ratio_err = np.where(mc_counts > 0, data_errs / mc_counts, np.nan)

    ax_ratio.errorbar(
        bin_centers, ratio, yerr=ratio_err, xerr=bin_widths / 2,
        fmt="ko", markersize=4, capsize=0,
    )
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Data / MC")
    ax_ratio.set_xlabel(xlabel if xlabel else column)

    fig.align_ylabels([ax_main, ax_ratio])

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {outfile}")

    return fig, (ax_main, ax_ratio)


if __name__ == "__main__":
    # Example usage with your Track_Thickness.py outputs
    mc_df = pd.read_csv("MC_Particles_Test.csv")
    data_df = pd.read_csv("Data_Particles_Test.csv")

    plot_data_mc(
        data_df, mc_df,
        column="thickness",
        bins=np.linspace(0, 5, 51),
        xlabel="Track thickness [size / length]",
        title="Track thickness: Data vs MC",
        normalize_mc_to_data=True,
        outfile="thickness_data_mc.png",
    )

    # Example: stack MC by particle ID
    plot_data_mc(
        data_df, mc_df,
        column="length",
        bins=np.linspace(0, 200, 41),
        xlabel="Track length [cm]",
        title="Track length: Data vs MC ",
        category_col="pid",
        normalize_mc_to_data=True,
        outfile="length_data_mc.png",
    )

    # plt.show()
