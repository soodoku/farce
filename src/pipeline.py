"""
Pipeline for FARS album release analysis.

Replication and critique of Patel et al. (2026), "Smartphones, Online Music
Streaming, and Traffic Fatalities," NBER WP 34866.

Usage:
    make run            # Standard analysis
    make run-placebos   # With placebo tests
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.s01_load import load_local_fars
from src.s02_preprocess import build_daily_series, residualize
from src.s03_core import (
    decomposition_analysis,
    dose_response_analysis,
    evaluate,
    leave_one_out,
    randomization_inference,
    stream_effect_correlation,
)
from src.s05_visualize import plot_results
from src.s04_placebo import run_all_placebos


def save_tables(df_global, local_df, corr_results, ri_results, dr_results, window, placebo_results=None):
    """Save all analysis results as CSV tables."""

    # T01: Per-album local estimates
    local_df.to_csv("tabs/t01_local_estimates.csv", index=False)

    # T02: Per-album global estimates
    global_df = []
    for _, row in local_df.iterrows():
        dt = pd.to_datetime(row["date"])
        g_row = df_global[df_global["date"] == dt]
        if len(g_row) > 0:
            global_df.append({
                "artist": row["artist"],
                "album": row["album"],
                "date": row["date"],
                "y_release": row["y_release"],
                "y_fitted_global": g_row["fitted_global"].values[0],
                "delta_global": row["y_release"] - g_row["fitted_global"].values[0],
            })
    pd.DataFrame(global_df).to_csv("tabs/t02_global_estimates.csv", index=False)

    # T03: Dose-response (all 20 albums)
    dose_df = pd.DataFrame({
        "album": corr_results["album_names"],
        "streams_millions": corr_results["streams"],
        "delta_deaths": corr_results["deltas"],
    })
    dose_df.to_csv("tabs/t03_dose_response.csv", index=False)

    # T04: Tier comparison
    tier_df = pd.DataFrame([
        {"tier": 1, "avg_delta": dr_results["avg_t1"], "description": "Top 10 albums"},
        {"tier": 2, "avg_delta": dr_results["avg_t2"], "description": "Albums 11-20"},
    ])
    tier_df.to_csv("tabs/t04_tier_comparison.csv", index=False)

    # T05: Randomization inference results
    ri_df = pd.DataFrame([
        {"test": "iid_10_random_days", "p_value": ri_results["p_all"]},
        {"test": "9_fridays_1_sunday", "p_value": ri_results["p_fri_sun"]},
        {"test": "9_fridays_only", "p_value": ri_results["p_fri_only"]},
        {"test": "block_bootstrap_7day", "p_value": ri_results["p_block"]},
    ])
    ri_df.to_csv("tabs/t05_randomization_inference.csv", index=False)

    # T06: Leave-one-out
    n = len(local_df)
    avg_all = local_df["delta_local"].mean()
    loo_df = []
    for _, r in local_df.iterrows():
        loo_avg = (n * avg_all - r["delta_local"]) / (n - 1)
        influence = avg_all - loo_avg
        loo_df.append({
            "album": f"{r['artist']} - {r['album']}",
            "delta_i": r["delta_local"],
            "loo_avg": loo_avg,
            "influence": influence,
        })
    pd.DataFrame(loo_df).to_csv("tabs/t06_leave_one_out.csv", index=False)

    # T07: Summary statistics
    loo_avgs = np.array([row["loo_avg"] for row in loo_df])
    jackknife_var = ((n - 1) / n) * np.sum((loo_avgs - loo_avgs.mean()) ** 2)
    jackknife_se = np.sqrt(jackknife_var)

    summary = pd.DataFrame([
        {"metric": "n_albums_tier1", "value": 10},
        {"metric": "n_albums_tier2", "value": 10},
        {"metric": "window_days", "value": window},
        {"metric": "local_delta_mean", "value": local_df["delta_local"].mean()},
        {"metric": "local_delta_se", "value": local_df["delta_local"].std() / np.sqrt(n)},
        {"metric": "jackknife_se", "value": jackknife_se},
        {"metric": "pearson_r", "value": corr_results["r_pearson"]},
        {"metric": "spearman_r", "value": corr_results["r_spearman"]},
        {"metric": "tier1_avg", "value": dr_results["avg_t1"]},
        {"metric": "tier2_avg", "value": dr_results["avg_t2"]},
        {"metric": "tier2_tier1_ratio", "value": dr_results["avg_t2"] / dr_results["avg_t1"] if dr_results["avg_t1"] != 0 else None},
    ])
    summary.to_csv("tabs/t07_summary.csv", index=False)

    # T08: Placebo tests (if run)
    if placebo_results:
        placebo_df = pd.DataFrame([
            {"test": "pretrends_avg", "value": placebo_results["pretrends"]["avg_pretrend"]},
            {"test": "pretrends_day0", "value": placebo_results["pretrends"]["avg_day0"]},
            {"test": "year_perm_p_value", "value": placebo_results["year_permutation"]["p_value"]},
            {"test": "year_perm_null_mean", "value": placebo_results["year_permutation"]["permuted_mean"]},
        ])
        placebo_df.to_csv("tabs/t08_placebo_tests.csv", index=False)

        # T09: Window sensitivity
        placebo_results["window_sensitivity"].to_csv("tabs/t09_window_sensitivity.csv", index=False)

    print(f"\nTables saved to tabs/")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Replicate and critique Patel et al. (2026) FARS analysis"
    )
    parser.add_argument(
        "--local",
        type=str,
        required=True,
        help="Directory containing FARS accident CSVs",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Control window size in days (default: 10)",
    )
    parser.add_argument(
        "--run-placebos",
        action="store_true",
        help="Run placebo tests",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FARS Album Release Analysis")
    print("Replication of Patel et al. (2026), NBER WP 34866")
    print("=" * 70)

    # Step 1: Load
    accidents = load_local_fars(args.local)
    print(f"\nLoaded {len(accidents)} crash records")

    # Step 2: Preprocess
    daily = build_daily_series(accidents)
    date_min = daily["date"].min()
    date_max = daily["date"].max()
    n_days = len(daily)
    print(f"Data period: {date_min.date()} to {date_max.date()} ({n_days} days)")
    print(f"Average daily fatalities: {daily['fatalities'].mean():.1f}")
    df = residualize(daily)

    # Step 3: Core analysis
    evaluate(df)
    df_global, local_df = decomposition_analysis(df, window=args.window)
    leave_one_out(local_df)
    ri_results = randomization_inference(df_global, block_size=7)
    corr_results = stream_effect_correlation(df, window=args.window)
    dr_results = dose_response_analysis(df, window=args.window)

    # Step 4: Placebo tests (optional)
    placebo_results = None
    if args.run_placebos:
        placebo_results = run_all_placebos(df, df_global, window=args.window)

    # Save tables
    save_tables(df_global, local_df, corr_results, ri_results, dr_results, args.window, placebo_results)

    # Step 5: Visualize
    import matplotlib
    matplotlib.use("Agg")
    plot_results(df, df_global, ri_results, local_df, corr_results, show=False)


if __name__ == "__main__":
    main()
