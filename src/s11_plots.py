"""
Visualization — all plotting functions for the analysis.

Functions for creating figures:
- plot_results: Multi-panel summary plot
- plot_event_study: Dynamic effects event study
- plot_multiverse: Specification curve analysis
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import ALBUMS, RELEASE_DATES
from src.s04_estimate import local_estimate


def plot_results(
    df, df_global=None, ri_results=None, local_df=None, corr_results=None, show=True
):
    """Multi-panel plot: time series, event study, decomposition, RI null, LOO, dose-response."""
    n_panels = 2
    if df_global is not None:
        n_panels += 1
    if ri_results is not None:
        n_panels += 1
    if local_df is not None:
        n_panels += 1
    if corr_results is not None:
        n_panels += 1

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    panel = 0

    # Panel 1: full time series of z-scores
    ax = axes[panel]
    panel += 1
    ax.plot(df["date"], df["z_score"], linewidth=0.3, color="steelblue", alpha=0.6)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axhline(2, color="red", linewidth=0.5, linestyle="--", label="z = 2")

    for artist, album, date_str, _ in ALBUMS:
        dt = pd.to_datetime(date_str)
        row = df[df["date"] == dt]
        if len(row) > 0:
            z = row["z_score"].values[0]
            ax.scatter(dt, z, color="red", s=50, zorder=5)
            ax.annotate(
                album[:15], (dt, z), fontsize=6, rotation=30, ha="left", va="bottom"
            )

    ax.set_ylabel("z-score (residual fatalities)")
    ax.set_title(
        "Daily US Traffic Fatalities: Residuals after DOW/Month/Year/Holiday FEs"
    )
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2: Individual event curves overlaid (shows heterogeneity)
    ax = axes[panel]
    panel += 1
    window = 10
    event_curves = []
    album_labels = []
    for artist, album, date_str, _ in ALBUMS:
        dt = pd.to_datetime(date_str)
        mask = (df["date"] >= dt - pd.Timedelta(days=window)) & (
            df["date"] <= dt + pd.Timedelta(days=window)
        )
        sub = df[mask].copy()
        if len(sub) == 0:
            continue
        sub["tau"] = (sub["date"] - dt).dt.days
        curve = sub.set_index("tau")["z_score"]
        event_curves.append(curve)
        album_labels.append(album[:12])

    if event_curves:
        combined = pd.concat(event_curves, axis=1)
        combined.columns = album_labels
        mean_curve = combined.mean(axis=1)

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i / len(event_curves)) for i in range(len(event_curves))]
        for col, color in zip(combined.columns, colors):
            ax.plot(
                combined.index,
                combined[col],
                alpha=0.4,
                linewidth=1.5,
                color=color,
                label=col,
            )

        ax.plot(
            mean_curve.index,
            mean_curve,
            color="black",
            linewidth=2.5,
            linestyle="--",
            label="Mean",
        )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="red", linewidth=1, linestyle=":", alpha=0.7)
        ax.set_xlabel("Days relative to album release")
        ax.set_ylabel("z-score")
        ax.set_title("Event Study: Individual Album Curves (Heterogeneity Visible)")
        ax.legend(fontsize=7, loc="upper left", ncol=2)

    # Panel 3: Decomposition bar chart
    if df_global is not None:
        ax = axes[panel]
        panel += 1
        local_res = local_estimate(df, window=window)

        albums_short = []
        deltas_local = []
        deltas_global = []
        gaps = []
        for _, r in local_res.iterrows():
            dt = pd.to_datetime(r["date"])
            row_g = df_global[df_global["date"] == dt]
            if len(row_g) == 0:
                continue
            y_hat = row_g["fitted_global"].values[0]
            d_g = r["y_release"] - y_hat
            gap = y_hat - r["y_control"]

            albums_short.append(r["album"][:12])
            deltas_local.append(r["delta_local"])
            deltas_global.append(d_g)
            gaps.append(gap)

        x = np.arange(len(albums_short))
        w = 0.35
        ax.bar(
            x - w / 2,
            deltas_local,
            w,
            label="δ local (paper)",
            color="steelblue",
            alpha=0.8,
        )
        ax.bar(
            x + w / 2,
            deltas_global,
            w,
            label="δ global (donut)",
            color="coral",
            alpha=0.8,
        )
        ax.scatter(
            x + w / 2,
            gaps,
            color="black",
            marker="d",
            s=40,
            zorder=5,
            label="Ŷ_rel − Ȳ_ctrl (gap)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(albums_short, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Excess fatalities")
        ax.set_title("Decomposition: Local vs Global Counterfactual per Album")
        ax.legend(fontsize=8)

    # Panel 4: RI null distribution
    if ri_results is not None and df_global is not None:
        ax = axes[panel]
        panel += 1

        release_mask = df_global["date"].dt.date.isin(RELEASE_DATES)
        actual = df_global.loc[release_mask, "resid_global"].mean()

        ax.hist(
            ri_results["null_fri"],
            bins=80,
            density=True,
            alpha=0.6,
            color="steelblue",
            label="Null: 9 rand Fri + 1 rand Sun",
        )
        ax.axvline(actual, color="red", linewidth=2, label=f"Actual = {actual:+.1f}")
        pct95 = np.percentile(ri_results["null_fri"], 95)
        ax.axvline(
            pct95,
            color="orange",
            linewidth=1,
            linestyle="--",
            label=f"95th pctile = {pct95:+.1f}",
        )
        ax.set_xlabel("Average residual fatalities (10-day set)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Randomization Inference: p = {ri_results['p_fri_sun']:.4f} "
            f"(9 Fri + 1 Sun, {len(ri_results['null_fri']):,} draws)"
        )
        ax.legend(fontsize=8)

    # Panel 5: Leave-One-Out analysis
    if local_df is not None:
        ax = axes[panel]
        panel += 1

        n = len(local_df)
        avg_all = local_df["delta_local"].mean()
        loo_avgs = []
        album_labels_loo = []

        for _, r in local_df.iterrows():
            loo_avg = (n * avg_all - r["delta_local"]) / (n - 1)
            loo_avgs.append(loo_avg)
            album_labels_loo.append(f"{r['artist'][:10]}-{r['album'][:8]}")

        loo_avgs = np.array(loo_avgs)
        influences = avg_all - loo_avgs

        colors = [
            (
                "red"
                if inf > np.std(influences)
                else ("blue" if inf < -np.std(influences) else "gray")
            )
            for inf in influences
        ]

        ax.barh(range(n), loo_avgs, color=colors, alpha=0.7)
        ax.axvline(
            avg_all,
            color="black",
            linewidth=2,
            linestyle="--",
            label=f"Full sample: {avg_all:+.1f}",
        )
        ax.set_yticks(range(n))
        ax.set_yticklabels(album_labels_loo, fontsize=8)
        ax.set_xlabel("Pooled δ when album dropped")
        ax.set_title(
            "Leave-One-Out Sensitivity: Red = inflates estimate, Blue = deflates"
        )
        ax.legend(fontsize=8)

        for i, (loo, inf) in enumerate(zip(loo_avgs, influences)):
            ax.annotate(f"({inf:+.1f})", (loo + 0.5, i), fontsize=7, va="center")

    # Panel 6: Dose-response scatter
    if corr_results is not None:
        ax = axes[panel]
        panel += 1

        streams = corr_results["streams"]
        deltas = corr_results["deltas"]
        album_names = corr_results["album_names"]

        tier1_mask = np.array([s >= 75 for s in streams])
        ax.scatter(
            streams[tier1_mask],
            deltas[tier1_mask],
            s=80,
            color="red",
            alpha=0.7,
            label="Tier 1 (top 10)",
            zorder=5,
        )
        ax.scatter(
            streams[~tier1_mask],
            deltas[~tier1_mask],
            s=60,
            color="blue",
            alpha=0.7,
            label="Tier 2 (11-20)",
            zorder=5,
        )

        beta = corr_results["beta_dose"]
        s_range = np.linspace(min(streams), max(streams), 100)
        fitted_line = beta[0] + beta[1] * np.log(s_range)
        ax.plot(
            s_range,
            fitted_line,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"β₁={beta[1]:+.1f}",
        )

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("First-day streams (millions)")
        ax.set_ylabel("Excess fatalities (δ)")
        r = corr_results["r_pearson"]
        ax.set_title(f"Dose-Response: Streams vs Effect (r = {r:+.2f})")
        ax.legend(fontsize=8)

        for i in range(len(streams)):
            if abs(deltas[i]) > 30 or streams[i] > 150:
                ax.annotate(
                    album_names[i].split("-")[1][:10],
                    (streams[i], deltas[i]),
                    fontsize=6,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    plt.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    plt.savefig("figs/analysis.png", dpi=150)
    print("\nPlot saved to figs/analysis.png")
    if show:
        plt.show()
    else:
        plt.close()


def plot_event_study(dynamic_results, show=True):
    """
    Create event study plot showing day-by-day effects.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    days = dynamic_results["day"].values
    effects = dynamic_results["effect"].values
    ci_lower = dynamic_results["ci_lower"].values
    ci_upper = dynamic_results["ci_upper"].values

    ax.fill_between(
        days, ci_lower, ci_upper, alpha=0.3, color="steelblue", label="95% CI"
    )
    ax.plot(
        days,
        effects,
        marker="o",
        markersize=8,
        color="steelblue",
        linewidth=2,
        label="Effect",
    )

    ax.axhline(0, color="gray", linewidth=1, linestyle="-")
    ax.axvline(
        0, color="red", linewidth=2, linestyle="--", alpha=0.7, label="Release day"
    )

    ax.set_xlabel("Days relative to album release", fontsize=12)
    ax.set_ylabel("Excess fatalities (vs. expected)", fontsize=12)
    ax.set_title(
        "Dynamic Effects: Traffic Fatalities Around Album Release", fontsize=14
    )
    ax.legend(fontsize=10)

    ax.set_xticks(days)

    plt.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    plt.savefig("figs/event_study.png", dpi=150)
    print("\nPlot saved to figs/event_study.png")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_multiverse(multiverse_df, show=True):
    """
    Create multiverse specification plot.

    Shows distribution of effects across all specifications.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    effects = multiverse_df["effect"].values
    ax1.hist(effects, bins=15, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(0, color="red", linestyle="--", linewidth=2, label="Null (0)")
    ax1.axvline(
        np.median(effects),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Median ({np.median(effects):+.1f})",
    )
    ax1.set_xlabel("Effect size (deaths)", fontsize=12)
    ax1.set_ylabel("Number of specifications", fontsize=12)
    ax1.set_title("Distribution of Effects Across Specifications", fontsize=14)
    ax1.legend()

    ax2 = axes[1]
    sorted_df = multiverse_df.sort_values("effect").reset_index(drop=True)
    colors = ["red" if p < 0.05 else "gray" for p in sorted_df["p_value"]]

    ax2.barh(range(len(sorted_df)), sorted_df["effect"], color=colors, alpha=0.7)
    ax2.axvline(0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Effect size (deaths)", fontsize=12)
    ax2.set_ylabel("Specification (sorted by effect)", fontsize=12)
    ax2.set_title("Specification Curve", fontsize=14)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="p < 0.05"),
        Patch(facecolor="gray", alpha=0.7, label="p >= 0.05"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    Path("figs").mkdir(exist_ok=True)
    plt.savefig("figs/multiverse.png", dpi=150)
    print("\nPlot saved to figs/multiverse.png")

    if show:
        plt.show()
    else:
        plt.close()

    return fig
