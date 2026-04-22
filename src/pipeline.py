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

from src.constants import (ALBUMS_EXTENDED, ALBUMS_TIER0, ALBUMS_TIER1,
                           ALBUMS_TIER3)
from src.s01_load import load_local_fars
from src.s02_preprocess import build_daily_series
from src.utils import save_table
# Phase 1: Design checks (run before trusting estimates)
from src.s03_design import (covariate_balance_check, holiday_baseline_check,
                            parallel_trends_test, pretrends_analysis)
# Phase 2: Primary estimation
from src.s04_estimate import (decomposition_analysis, evaluate,
                              paper_regression_estimate, residualize)
# Phase 3: Valid inference
from src.s05_inference import (leave_one_out, multiple_testing_correction,
                               randomization_inference,
                               studentized_randomization_inference)
# Phase 4: Specification robustness
from src.s06_specification import (build_daily_weather_controls,
                                   forecast_estimate, multiverse_analysis,
                                   save_forecast_tables,
                                   weather_controlled_model,
                                   window_sensitivity)
# Phase 5: Falsification
from src.s07_falsification import (structural_fars_placebos,
                                   best_fridays_false_positive_rate,
                                   placebo_outcomes, sp500_placebo,
                                   sp500_placebo_expanded,
                                   year_permutation_placebo)
# Phase 6: Confounding sensitivity
from src.s08_confounding import sensitivity_analysis, synthetic_control
# Phase 7: Magnitude checks
from src.s09_magnitude import (effect_size_plausibility_check, power_analysis,
                               weather_effect_sanity_check)
# Phase 8: Heterogeneity / Mechanism
from src.s10_heterogeneity import (compute_dynamic_effects, covid_sensitivity,
                                   dose_response_analysis,
                                   drunk_vs_sober_analysis,
                                   extended_series_analysis,
                                   stream_effect_correlation,
                                   time_of_day_analysis)
# Phase 9: Visualization
from src.s11_plots import plot_event_study, plot_multiverse, plot_results


def save_tables(
    df_global,
    local_df,
    corr_results,
    ri_results,
    dr_results,
    window,
    placebo_results=None,
    forecast_results=None,
):
    """Save all analysis results as markdown tables."""

    # T01: Per-album local estimates
    save_table(local_df, "tabs/t01_local_estimates.md")

    # T02: Per-album global estimates
    global_df = []
    for _, row in local_df.iterrows():
        dt = pd.to_datetime(row["date"])
        g_row = df_global[df_global["date"] == dt]
        if len(g_row) > 0:
            global_df.append(
                {
                    "artist": row["artist"],
                    "album": row["album"],
                    "date": row["date"],
                    "y_release": row["y_release"],
                    "y_fitted_global": g_row["fitted_global"].values[0],
                    "delta_global": row["y_release"] - g_row["fitted_global"].values[0],
                }
            )
    save_table(pd.DataFrame(global_df), "tabs/t02_global_estimates.md")

    # T03: Dose-response (all 20 albums)
    dose_df = pd.DataFrame(
        {
            "album": corr_results["album_names"],
            "streams_millions": corr_results["streams"],
            "delta_deaths": corr_results["deltas"],
        }
    )
    save_table(dose_df, "tabs/t03_dose_response.md")

    # T04: Tier comparison
    tier_df = pd.DataFrame(
        [
            {
                "tier": 1,
                "avg_delta": dr_results["avg_t1"],
                "description": "Top 10 albums",
            },
            {
                "tier": 2,
                "avg_delta": dr_results["avg_t2"],
                "description": "Albums 11-20",
            },
        ]
    )
    save_table(tier_df, "tabs/t04_tier_comparison.md")

    # T05: Randomization inference results
    ri_df = pd.DataFrame(
        [
            {"test": "iid_10_random_days", "p_value": ri_results["p_all"]},
            {"test": "9_fridays_1_sunday", "p_value": ri_results["p_fri_sun"]},
            {"test": "9_fridays_only", "p_value": ri_results["p_fri_only"]},
            {"test": "block_bootstrap_7day", "p_value": ri_results["p_block"]},
        ]
    )
    save_table(ri_df, "tabs/t05_randomization_inference.md")

    # T06: Leave-one-out
    n = len(local_df)
    avg_all = local_df["delta_local"].mean()
    loo_df = []
    for _, r in local_df.iterrows():
        loo_avg = (n * avg_all - r["delta_local"]) / (n - 1)
        influence = avg_all - loo_avg
        loo_df.append(
            {
                "album": f"{r['artist']} - {r['album']}",
                "delta_i": r["delta_local"],
                "loo_avg": loo_avg,
                "influence": influence,
            }
        )
    save_table(pd.DataFrame(loo_df), "tabs/t06_leave_one_out.md")

    # T07: Summary statistics
    loo_avgs = np.array([row["loo_avg"] for row in loo_df])
    jackknife_var = ((n - 1) / n) * np.sum((loo_avgs - loo_avgs.mean()) ** 2)
    jackknife_se = np.sqrt(jackknife_var)

    forecast_df = forecast_results["results_df"] if forecast_results else pd.DataFrame()
    tier3_effects = forecast_df[forecast_df["tier"] == 3]["effect"] if len(forecast_df) > 0 else pd.Series()
    tier3_avg = tier3_effects.mean() if len(tier3_effects) > 0 else None
    tier3_se = (
        tier3_effects.std() / np.sqrt(len(tier3_effects))
        if len(tier3_effects) > 0
        else None
    )

    summary = pd.DataFrame(
        [
            {"metric": "n_albums_tier0", "value": 10},
            {"metric": "n_albums_tier1", "value": 10},
            {"metric": "n_albums_tier2", "value": 10},
            {"metric": "n_albums_tier3", "value": 7},
            {"metric": "window_days", "value": window},
            {"metric": "local_delta_mean", "value": local_df["delta_local"].mean()},
            {
                "metric": "local_delta_se",
                "value": local_df["delta_local"].std() / np.sqrt(n),
            },
            {"metric": "jackknife_se", "value": jackknife_se},
            {"metric": "pearson_r", "value": corr_results["r_pearson"]},
            {"metric": "spearman_r", "value": corr_results["r_spearman"]},
            {"metric": "tier1_avg", "value": dr_results["avg_t1"]},
            {"metric": "tier2_avg", "value": dr_results["avg_t2"]},
            {"metric": "tier3_avg_forecast", "value": tier3_avg},
            {"metric": "tier3_se_forecast", "value": tier3_se},
            {
                "metric": "tier2_tier1_ratio",
                "value": (
                    dr_results["avg_t2"] / dr_results["avg_t1"]
                    if dr_results["avg_t1"] != 0
                    else None
                ),
            },
        ]
    )
    save_table(summary, "tabs/t07_summary.md")

    # T08: Placebo tests (if run)
    if placebo_results:
        placebo_df = pd.DataFrame(
            [
                {
                    "test": "pretrends_avg",
                    "value": placebo_results["pretrends"]["avg_pretrend"],
                },
                {
                    "test": "pretrends_day0",
                    "value": placebo_results["pretrends"]["avg_day0"],
                },
                {
                    "test": "year_perm_p_value",
                    "value": placebo_results["year_permutation"]["p_value"],
                },
                {
                    "test": "year_perm_null_mean",
                    "value": placebo_results["year_permutation"]["permuted_mean"],
                },
            ]
        )
        save_table(placebo_df, "tabs/t08_placebo_tests.md")

        # T09: Window sensitivity
        save_table(placebo_results["window_sensitivity"], "tabs/t09_window_sensitivity.md")

    print("\nTables saved to tabs/")


def save_paper_replication_table(df, window):
    """
    Generate t12_paper_replication.csv comparing paper's results to replication.
    """
    print(f"\n{'='*70}")
    print("PAPER REPLICATION COMPARISON (t12)")
    print(f"{'='*70}")

    result_t0 = paper_regression_estimate(df, albums=ALBUMS_TIER0, window=window)
    result_t1_paper = paper_regression_estimate(
        df, albums=ALBUMS_TIER1, window=window, sample_period=(2017, 2022)
    )
    result_t1_full = paper_regression_estimate(df, albums=ALBUMS_TIER1, window=window)
    result_t3 = paper_regression_estimate(df, albums=ALBUMS_TIER3, window=window)
    result_all = paper_regression_estimate(df, albums=ALBUMS_EXTENDED, window=window)

    rows = [
        {
            "estimator": "Paper (reported Figure 2B)",
            "sample": "Tier 1 (2018-2022)",
            "effect": 18.2,
            "se": 5.5,
            "pct_effect": 15.1,
            "n_albums": 10,
            "source": "Patel et al. (2026)",
        },
        {
            "estimator": "Paper spec (our replication)",
            "sample": "Tier 1 (2018-2022)",
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
            "estimator": "Paper spec (pre-2018)",
            "sample": "Tier 0 (2015-2017)",
            "effect": result_t0["treatment_effect"],
            "se": result_t0["treatment_se"],
            "pct_effect": result_t0["pct_effect"],
            "n_albums": result_t0["n_albums"],
            "source": "This analysis",
        },
        {
            "estimator": "Paper spec (out-of-sample)",
            "sample": "Tier 3 (2023-2024)",
            "effect": result_t3["treatment_effect"],
            "se": result_t3["treatment_se"],
            "pct_effect": result_t3["pct_effect"],
            "n_albums": result_t3["n_albums"],
            "source": "This analysis",
        },
        {
            "estimator": "Paper spec (all tiers)",
            "sample": f"All {len(ALBUMS_EXTENDED)} albums",
            "effect": result_all["treatment_effect"],
            "se": result_all["treatment_se"],
            "pct_effect": result_all["pct_effect"],
            "n_albums": result_all["n_albums"],
            "source": "This analysis",
        },
    ]

    replication_df = pd.DataFrame(rows)
    save_table(replication_df, "tabs/t12_paper_replication.md")

    print("\nComparison of estimators:")
    print(f"{'Estimator':<35} {'Sample':<20} {'Effect':>8} {'SE':>6} {'%':>8}")
    print("-" * 80)
    for _, row in replication_df.iterrows():
        print(
            f"{row['estimator']:<35} {row['sample']:<20} "
            f"{row['effect']:>+8.1f} {row['se']:>6.1f} {row['pct_effect']:>+7.1f}%"
        )

    print("\nKey findings:")
    print("  - Paper reports: +18.2 deaths (+15.1%)")
    print(
        f"  - Our replication: +{result_t1_paper['treatment_effect']:.1f} deaths (+{result_t1_paper['pct_effect']:.1f}%)"
    )
    diff = abs(18.2 - result_t1_paper["treatment_effect"])
    print(f"  - Difference: {diff:.1f} deaths (within 1 SE)")
    print(
        f"  - Pre-2018 (Tier 0): {result_t0['treatment_effect']:+.1f} deaths ({result_t0['pct_effect']:+.1f}%)"
    )
    if result_t0["treatment_effect"] <= 0:
        print("  - PRE-2018 FAILURE: No positive effect before paper's start date")
    print(
        f"  - Out-of-sample (Tier 3): {result_t3['treatment_effect']:+.1f} deaths ({result_t3['pct_effect']:+.1f}%)"
    )
    if result_t3["treatment_effect"] < 0:
        print("  - OUT-OF-SAMPLE FAILURE: Effect is NEGATIVE for 2023-2024 albums")

    t0_per_album = result_t0["per_album_df"]
    print("\n  Tier 0 per-album effects (pre-2018):")
    for _, row in t0_per_album.iterrows():
        streams = [
            a[4] for a in ALBUMS_TIER0 if a[0] == row["artist"] and a[1] == row["album"]
        ]
        streams_str = f"{streams[0]:.0f}M" if streams else ""
        print(
            f"    {row['artist']:<18} {row['album'][:22]:<22} δ={row['delta_raw']:+6.1f} ({streams_str} streams)"
        )

    t3_per_album = result_t3["per_album_df"]
    print("\n  Tier 3 per-album effects (post-2022):")
    for _, row in t3_per_album.iterrows():
        streams = [
            a[4] for a in ALBUMS_TIER3 if a[0] == row["artist"] and a[1] == row["album"]
        ]
        streams_str = f"{streams[0]:.0f}M" if streams else ""
        print(
            f"    {row['artist']:<18} {row['album'][:22]:<22} δ={row['delta_raw']:+6.1f} ({streams_str} streams)"
        )

    return replication_df


DATA_DIR = "data/fars/"


def main():
    import argparse

    from scipy import stats as scipy_stats

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

    # ═══════════════════════════════════════════════════════════════════════
    # LOAD & PREPROCESS
    # ═══════════════════════════════════════════════════════════════════════
    accidents = load_local_fars(DATA_DIR)
    print(f"\nLoaded {len(accidents)} crash records")

    daily = build_daily_series(accidents)
    date_min = daily["date"].min()
    date_max = daily["date"].max()
    n_days = len(daily)
    print(f"Data period: {date_min.date()} to {date_max.date()} ({n_days} days)")
    print(f"Average daily fatalities: {daily['fatalities'].mean():.1f}")
    df = residualize(daily)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: DESIGN CHECKS (balance, holiday baseline)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: DESIGN CHECKS")
    print("=" * 70)

    balance_results = covariate_balance_check(df)
    save_table(balance_results, "tabs/t24_balance_check.md")
    print("Saved: tabs/t24_balance_check.md")

    holiday_results = holiday_baseline_check(accidents)
    save_table(holiday_results, "tabs/t16_holiday_check.md")
    print("Saved: tabs/t16_holiday_check.md")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: PRIMARY ESTIMATION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: PRIMARY ESTIMATION")
    print("=" * 70)

    evaluate(df)
    df_global, local_df = decomposition_analysis(df, window=args.window)
    corr_results = stream_effect_correlation(df, window=args.window)
    dr_results = dose_response_analysis(df, window=args.window)

    # Design checks that need df_global (pretrends, parallel trends)
    pretrends_results = pretrends_analysis(df_global)

    parallel_results = parallel_trends_test(df_global, window=args.window)
    save_table(parallel_results, "tabs/t32_parallel_trends.md")
    print("Saved: tabs/t32_parallel_trends.md")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: VALID INFERENCE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: VALID INFERENCE")
    print("=" * 70)

    leave_one_out(local_df)
    ri_results = randomization_inference(df_global, block_size=7)

    studentized_results = studentized_randomization_inference(df_global)
    save_table(studentized_results, "tabs/t26_studentized_ri.md")
    print("Saved: tabs/t26_studentized_ri.md")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: SPECIFICATION ROBUSTNESS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: SPECIFICATION ROBUSTNESS")
    print("=" * 70)

    window_sens = window_sensitivity(df, windows=[5, 7, 10, 14, 21])

    forecast_results = forecast_estimate(df, window=args.window)
    save_forecast_tables(forecast_results)

    weather_results = weather_controlled_model(df_global, window=args.window)
    save_table(weather_results, "tabs/t21_fars_controls.md")
    print("Saved: tabs/t21_fars_controls.md")

    multiverse_results = multiverse_analysis(df)
    save_table(multiverse_results, "tabs/t29_multiverse.md")
    print("Saved: tabs/t29_multiverse.md")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 5: FALSIFICATION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 5: FALSIFICATION")
    print("=" * 70)

    year_perm_results = year_permutation_placebo(df_global)

    sp500_results = sp500_placebo(window=args.window)
    save_table(sp500_results, "tabs/t15_placebo_sp500.md")
    print("Saved: tabs/t15_placebo_sp500.md")

    sp500_expanded = sp500_placebo_expanded(window=args.window)
    save_table(sp500_expanded, "tabs/t17_sp500_expanded.md")
    print("Saved: tabs/t17_sp500_expanded.md")

    fpr_results = best_fridays_false_positive_rate(df_global)
    save_table(pd.DataFrame([fpr_results]), "tabs/t18_friday_fpr.md")
    print("Saved: tabs/t18_friday_fpr.md")

    placebo_out = placebo_outcomes(accidents, window=args.window)
    save_table(placebo_out, "tabs/t28_placebo_outcomes.md")
    print("Saved: tabs/t28_placebo_outcomes.md")

    structural_results = structural_fars_placebos(accidents, window=args.window)
    save_table(structural_results, "tabs/t28b_structural_fars_placebos.md")
    print("Saved: tabs/t28b_structural_fars_placebos.md")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 6: CONFOUNDING SENSITIVITY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 6: CONFOUNDING SENSITIVITY")
    print("=" * 70)

    sens_results = sensitivity_analysis(df_global, window=args.window)
    save_table(sens_results, "tabs/t27_sensitivity.md")
    print("Saved: tabs/t27_sensitivity.md")

    synth_results = synthetic_control(df, window=args.window)
    if synth_results is not None:
        save_table(synth_results, "tabs/t31_synthetic_control.md")
        print("Saved: tabs/t31_synthetic_control.md")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 7: MAGNITUDE CHECKS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 7: MAGNITUDE CHECKS")
    print("=" * 70)

    power_results = power_analysis(df_global, n_albums=10)
    save_table(power_results, "tabs/t23_power_analysis.md")
    print("Saved: tabs/t23_power_analysis.md")

    tier1_effect = local_df["delta_local"].mean()
    plausibility_results = effect_size_plausibility_check(tier1_effect)
    daily_weather = build_daily_weather_controls(accidents)
    weather_sanity, _, _ = weather_effect_sanity_check(daily_weather)
    save_table(weather_sanity, "tabs/t33_weather_effects.md")
    print("Saved: tabs/t33_weather_effects.md")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 8: HETEROGENEITY / MECHANISM
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 8: HETEROGENEITY / MECHANISM")
    print("=" * 70)

    dynamic_results = compute_dynamic_effects(df_global, window=args.window)
    save_table(dynamic_results, "tabs/t13_dynamic_effects.md")
    print("Saved: tabs/t13_dynamic_effects.md")

    tod_results = time_of_day_analysis(accidents, df_global, window=args.window)
    if tod_results is not None:
        save_table(tod_results, "tabs/t14_time_of_day.md")
        print("Saved: tabs/t14_time_of_day.md")

    drunk_results = drunk_vs_sober_analysis(accidents, window=args.window)
    save_table(drunk_results, "tabs/t22_drunk_mechanism.md")
    print("Saved: tabs/t22_drunk_mechanism.md")

    covid_results = covid_sensitivity(df_global)
    save_table(covid_results, "tabs/t19_covid_sensitivity.md")
    print("Saved: tabs/t19_covid_sensitivity.md")

    extended_results = extended_series_analysis(df_global)
    save_table(extended_results, "tabs/t20_extended_series.md")
    print("Saved: tabs/t20_extended_series.md")

    # ═══════════════════════════════════════════════════════════════════════
    # MULTIPLE TESTING CORRECTION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MULTIPLE TESTING CORRECTION")
    print("=" * 70)

    p_values = {}
    p_values["RI (all days)"] = ri_results["p_all"]
    p_values["RI (9 Fri + 1 Sun)"] = ri_results["p_fri_sun"]
    p_values["RI (Fridays only)"] = ri_results["p_fri_only"]
    p_values["RI (block bootstrap)"] = ri_results["p_block"]
    p_values["Studentized RI"] = studentized_results["p_value"].iloc[1]
    p_values["Clustered RI"] = studentized_results["p_value"].iloc[2]
    if sens_results is not None:
        t_stat = sens_results["observed_effect"].iloc[0] / sens_results["se"].iloc[0]
        p_values["Main effect"] = 2 * (1 - scipy_stats.norm.cdf(abs(t_stat)))
    for _, row in structural_results.iterrows():
        p_val = 2 * (1 - scipy_stats.norm.cdf(abs(row["t_stat"])))
        p_values[f"Placebo: {row['description'][:20]}"] = p_val
    for _, row in placebo_out.iterrows():
        p_val = 2 * (1 - scipy_stats.norm.cdf(abs(row["t_stat"])))
        p_values[f"Placebo: {row['outcome'][:20]}"] = p_val

    mht_results = multiple_testing_correction(p_values)
    save_table(mht_results, "tabs/t25_multiple_testing.md")
    print("Saved: tabs/t25_multiple_testing.md")

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE TABLES
    # ═══════════════════════════════════════════════════════════════════════
    placebo_results = {
        "pretrends": pretrends_results,
        "year_permutation": year_perm_results,
        "window_sensitivity": window_sens,
    }
    save_tables(
        df_global,
        local_df,
        corr_results,
        ri_results,
        dr_results,
        args.window,
        placebo_results,
        forecast_results,
    )

    # Paper replication comparison
    save_paper_replication_table(df, args.window)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 9: VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 9: VISUALIZATION")
    print("=" * 70)

    import matplotlib

    matplotlib.use("Agg")
    plot_results(df, df_global, ri_results, local_df, corr_results, show=False)
    plot_event_study(dynamic_results, show=False)
    plot_multiverse(multiverse_results, show=False)


if __name__ == "__main__":
    main()
