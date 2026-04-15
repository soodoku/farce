"""
Pipeline for FARS album release analysis.

Replication and critique of Patel et al. (2026), "Smartphones, Online Music
Streaming, and Traffic Fatalities," NBER WP 34866.

Usage:
    make run
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
    paper_regression_estimate,
    randomization_inference,
    stream_effect_correlation,
)
from src.constants import ALBUMS_TIER1, ALBUMS_TIER3, ALBUMS_EXTENDED
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

    forecast_df = pd.read_csv("tabs/t10_forecast_estimates.csv")
    tier3_effects = forecast_df[forecast_df["tier"] == 3]["effect"]
    tier3_avg = tier3_effects.mean() if len(tier3_effects) > 0 else None
    tier3_se = tier3_effects.std() / np.sqrt(len(tier3_effects)) if len(tier3_effects) > 0 else None

    summary = pd.DataFrame([
        {"metric": "n_albums_tier1", "value": 10},
        {"metric": "n_albums_tier2", "value": 10},
        {"metric": "n_albums_tier3", "value": 7},
        {"metric": "window_days", "value": window},
        {"metric": "local_delta_mean", "value": local_df["delta_local"].mean()},
        {"metric": "local_delta_se", "value": local_df["delta_local"].std() / np.sqrt(n)},
        {"metric": "jackknife_se", "value": jackknife_se},
        {"metric": "pearson_r", "value": corr_results["r_pearson"]},
        {"metric": "spearman_r", "value": corr_results["r_spearman"]},
        {"metric": "tier1_avg", "value": dr_results["avg_t1"]},
        {"metric": "tier2_avg", "value": dr_results["avg_t2"]},
        {"metric": "tier3_avg_forecast", "value": tier3_avg},
        {"metric": "tier3_se_forecast", "value": tier3_se},
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


def save_paper_replication_table(df, window):
    """
    Generate t12_paper_replication.csv comparing paper's results to replication.
    """
    print(f"\n{'='*70}")
    print("PAPER REPLICATION COMPARISON (t12)")
    print(f"{'='*70}")

    result_t1_paper = paper_regression_estimate(
        df, albums=ALBUMS_TIER1, window=window, sample_period=(2017, 2022)
    )
    result_t1_full = paper_regression_estimate(
        df, albums=ALBUMS_TIER1, window=window
    )
    result_t3 = paper_regression_estimate(
        df, albums=ALBUMS_TIER3, window=window
    )
    result_all = paper_regression_estimate(
        df, albums=ALBUMS_EXTENDED, window=window
    )

    rows = [
        {
            "estimator": "Paper (reported Figure 2B)",
            "sample": "Tier 1 (2017-2022)",
            "effect": 18.2,
            "se": 5.5,
            "pct_effect": 15.1,
            "n_albums": 10,
            "source": "Patel et al. (2026)",
        },
        {
            "estimator": "Paper spec (our replication)",
            "sample": "Tier 1 (2017-2022)",
            "effect": result_t1_paper["treatment_effect"],
            "se": result_t1_paper["treatment_se"],
            "pct_effect": result_t1_paper["pct_effect"],
            "n_albums": result_t1_paper["n_albums"],
            "source": "This analysis",
        },
        {
            "estimator": "Paper spec (full data)",
            "sample": "Tier 1 (all years)",
            "effect": result_t1_full["treatment_effect"],
            "se": result_t1_full["treatment_se"],
            "pct_effect": result_t1_full["pct_effect"],
            "n_albums": result_t1_full["n_albums"],
            "source": "This analysis",
        },
        {
            "estimator": "Paper spec (out-of-sample)",
            "sample": "Tier 3 (2022-2024)",
            "effect": result_t3["treatment_effect"],
            "se": result_t3["treatment_se"],
            "pct_effect": result_t3["pct_effect"],
            "n_albums": result_t3["n_albums"],
            "source": "This analysis",
        },
        {
            "estimator": "Paper spec (all tiers)",
            "sample": "All 27 albums",
            "effect": result_all["treatment_effect"],
            "se": result_all["treatment_se"],
            "pct_effect": result_all["pct_effect"],
            "n_albums": result_all["n_albums"],
            "source": "This analysis",
        },
    ]

    replication_df = pd.DataFrame(rows)
    replication_df.to_csv("tabs/t12_paper_replication.csv", index=False)

    print("\nComparison of estimators:")
    print(f"{'Estimator':<35} {'Sample':<20} {'Effect':>8} {'SE':>6} {'%':>8}")
    print("-" * 80)
    for _, row in replication_df.iterrows():
        print(
            f"{row['estimator']:<35} {row['sample']:<20} "
            f"{row['effect']:>+8.1f} {row['se']:>6.1f} {row['pct_effect']:>+7.1f}%"
        )

    print("\nKey findings:")
    print(f"  - Paper reports: +18.2 deaths (+15.1%)")
    print(f"  - Our replication: +{result_t1_paper['treatment_effect']:.1f} deaths (+{result_t1_paper['pct_effect']:.1f}%)")
    diff = abs(18.2 - result_t1_paper['treatment_effect'])
    print(f"  - Difference: {diff:.1f} deaths (within 1 SE)")
    print(f"  - Out-of-sample (Tier 3): {result_t3['treatment_effect']:+.1f} deaths ({result_t3['pct_effect']:+.1f}%)")
    if result_t3['treatment_effect'] < 0:
        print("  - OUT-OF-SAMPLE FAILURE: Effect is NEGATIVE for 2023-2024 albums")

    t3_per_album = result_t3['per_album_df']
    print("\n  Tier 3 per-album effects:")
    for _, row in t3_per_album.iterrows():
        streams = [a[4] for a in ALBUMS_TIER3 if a[0] == row['artist'] and a[1] == row['album']]
        streams_str = f"{streams[0]:.0f}M" if streams else ""
        print(f"    {row['artist']:<18} {row['album'][:22]:<22} δ={row['delta_raw']:+6.1f} ({streams_str} streams)")

    return replication_df


DATA_DIR = "data/fars/"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Replicate and critique Patel et al. (2026) FARS analysis"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Control window size in days (default: 10)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FARS Album Release Analysis")
    print("Replication of Patel et al. (2026), NBER WP 34866")
    print("=" * 70)

    # Step 1: Load
    accidents = load_local_fars(DATA_DIR)
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

    # Step 4: Placebo tests
    placebo_results = run_all_placebos(df, df_global, window=args.window)

    # Save tables
    save_tables(df_global, local_df, corr_results, ri_results, dr_results, args.window, placebo_results)

    # Step 5: Paper replication comparison
    save_paper_replication_table(df, args.window)

    # Step 6: Visualize
    import matplotlib
    matplotlib.use("Agg")
    plot_results(df, df_global, ri_results, local_df, corr_results, show=False)


if __name__ == "__main__":
    main()
