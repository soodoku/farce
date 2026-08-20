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
from scipy import stats as scipy_stats

from src import gates
from src.constants import (
    ALBUMS_EXTENDED,
    ALBUMS_TIER0,
    ALBUMS_TIER1,
    ALBUMS_TIER3,
)
from src.numbers import record
from src.numbers import write as write_numbers
from src.s01_load import load_local_fars, load_local_miacc
from src.s02_preprocess import build_daily_series

# Phase 1: Design checks (run before trusting estimates)
from src.s03_design import (
    covariate_balance_check,
    holiday_baseline_check,
    parallel_trends_test,
    pretrends_analysis,
)

# Phase 2: Primary estimation
from src.s04_estimate import (
    decomposition_analysis,
    evaluate,
    paper_regression_estimate,
    residualize,
)

# Phase 3: Valid inference
from src.s05_inference import (
    leave_one_out,
    multiple_testing_correction,
    randomization_inference,
    robustness_to_one_day,
    studentized_randomization_inference,
)

# Phase 4: Specification robustness
from src.s06_specification import (
    build_daily_weather_controls,
    dow_interaction_models,
    forecast_estimate,
    multiverse_analysis,
    release_day_weather,
    save_forecast_tables,
    weather_controlled_model,
    window_sensitivity,
)

# Phase 5: Falsification
from src.s07_falsification import (
    adjacent_friday_contrast,
    calendar_position_placebo,
    cherry_pick_benchmark,
    composition_covariation,
    friday_placebo_test,
    placebo_outcomes,
    sp500_placebo,
    sp500_placebo_expanded,
    structural_fars_composition,
    weekday_dummy_invariance,
    year_permutation_placebo,
)

# Phase 6: Confounding sensitivity
from src.s08_confounding import sensitivity_analysis, synthetic_control

# Phase 7: Magnitude checks
from src.s09_magnitude import (
    effect_size_plausibility_check,
    holiday_adjacency_robustness,
    holiday_benchmark,
    power_analysis,
    publication_filter,
    weather_effect_sanity_check,
)

# Phase 8: Heterogeneity / Mechanism
from src.s10_heterogeneity import (
    alcohol_decomposition,
    compute_dynamic_effects,
    covid_sensitivity,
    dose_response_analysis,
    extended_series_analysis,
    stream_effect_correlation,
    time_of_day_analysis,
)
from src.s11_plots import (
    plot_dose_response,
    plot_event_study,
    plot_multiverse,
    plot_results,
)

# Phase 9: Visualization
from src.s12_weather import build_national_weather, exogenous_weather_model
from src.s13_streaming import (
    benchmark_against_paper,
    dose_matched_contrast,
    dose_response_all,
    load_streaming,
    measured_first_day,
    paper_album_keys,
    reproducible_frame,
    shared_shock_test,
)
from src.utils import album_stats, save_table


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
    save_table(
        local_df,
        "tabs/t01_local_estimates.md",
        caption=(
            "Local estimator. Release-day fatalities minus the mean of the "
            "surrounding +/-10 days, per album, Tier 1. No fixed effects."
        ),
    )

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
    save_table(
        pd.DataFrame(global_df),
        "tabs/t02_global_estimates.md",
        caption=(
            "Residual estimator. Release-day fatalities minus the prediction of a "
            "day-of-week, month, year and holiday model fitted with release "
            "windows excluded (donut)."
        ),
    )

    # T03: Dose-response (all 20 albums)
    dose_df = pd.DataFrame(
        {
            "album": corr_results["album_names"],
            "streams_millions": corr_results["streams"],
            "delta_deaths": corr_results["deltas"],
        }
    )
    save_table(
        dose_df,
        "tabs/t03_dose_response.md",
        caption=(
            "Residual estimator, Tier 1 and Tier 2. Tier 1 first-day streams are "
            "from Patel et al. Table 1; eight of the ten Tier 2 counts are "
            "estimated from chart position, not measured."
        ),
    )

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
    save_table(
        tier_df,
        "tabs/t04_tier_comparison.md",
        caption=("Local estimator. Mean per-album effect by tier."),
    )

    # T05: Randomization inference results
    ri_df = pd.DataFrame(
        [
            {"test": "iid_10_random_days", "p_value": ri_results["p_all"]},
            {"test": "9_fridays_1_sunday", "p_value": ri_results["p_fri_sun"]},
            {"test": "9_fridays_only", "p_value": ri_results["p_fri_only"]},
            {"test": "block_bootstrap_7day", "p_value": ri_results["p_block"]},
        ]
    )
    save_table(
        ri_df,
        "tabs/t05_randomization_inference.md",
        caption=(
            "One-sided p-values for the Tier 1 residual estimate (+16.1) against "
            "10,000 draws under each assignment scheme."
        ),
    )

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
    save_table(
        pd.DataFrame(loo_df),
        "tabs/t06_leave_one_out.md",
        caption=(
            "Local estimator. delta_i is one album's effect; loo_avg is the "
            "pooled mean with that album dropped. Full-sample pooled local mean "
            "is +23.0."
        ),
    )

    # T07: Summary statistics
    loo_avgs = np.array([row["loo_avg"] for row in loo_df])
    jackknife_var = ((n - 1) / n) * np.sum((loo_avgs - loo_avgs.mean()) ** 2)
    jackknife_se = np.sqrt(jackknife_var)

    forecast_df = forecast_results["results_df"] if forecast_results else pd.DataFrame()
    tier3_effects = (
        forecast_df[forecast_df["tier"] == 3]["effect"]
        if len(forecast_df) > 0
        else pd.Series()
    )
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
        save_table(
            placebo_results["window_sensitivity"], "tabs/t09_window_sensitivity.md"
        )

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
    record(
        "EffectRegression",
        result_t1_paper["treatment_effect"],
        "{:.1f}",
        "Paper spec on stacked album-days, CR1 clustered by album",
    )
    record("SEClustered", result_t1_paper["treatment_se"], "{:.2f}")
    _g = result_t1_paper["n_albums"]
    _crit = scipy_stats.t.ppf(0.975, _g - 1)
    _eff = result_t1_paper["treatment_effect"]
    _se = result_t1_paper["treatment_se"]
    record(
        "CIRegressionLo",
        _eff - _crit * _se,
        "{:.1f}",
        f"t with {_g - 1} df, since treatment varies across {_g} clusters",
    )
    record("CIRegressionHi", _eff + _crit * _se, "{:.1f}")
    record("PRegression", 2 * (1 - scipy_stats.t.cdf(_eff / _se, _g - 1)), "{:.3f}")
    record("PctRegression", result_t1_paper["pct_effect"], "{:.1f}")
    record(
        "ControlMean",
        result_t1_paper["control_mean"],
        "{:.1f}",
        "Mean of the control days, the analogue of the paper's 120.9",
    )
    record(
        "EffectOutOfSample",
        result_t3["treatment_effect"],
        "{:.2f}",
        "Tier 3, paper spec",
    )
    record("SEOutOfSample", result_t3["treatment_se"], "{:.2f}")
    record(
        "EffectOutOfSampleLocal",
        result_t3["per_album_df"]["delta_raw"].mean(),
        "{:.1f}",
        "Tier 3 by the raw release-day-minus-window comparison, the "
        "unadjusted analogue of the paper's 139.1 against 120.9",
    )
    record(
        "DiffFromPaper",
        18.2 - result_t1_paper["treatment_effect"],
        "{:.1f}",
        "Gap between the paper's 18.2 and our replication",
    )
    save_table(
        replication_df,
        "tabs/t12_paper_replication.md",
        caption=(
            "Regression estimator. Paper's specification on stacked album-days: "
            "day-of-week, week-of-year, year and holiday fixed effects, standard "
            "errors clustered by album (G = 10, so use t with 9 df)."
        ),
    )

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
        f"  - Our replication: +{result_t1_paper['treatment_effect']:.1f} "
        f"deaths (+{result_t1_paper['pct_effect']:.1f}%)"
    )
    diff = abs(18.2 - result_t1_paper["treatment_effect"])
    print(f"  - Difference: {diff:.1f} deaths (within 1 SE)")
    print(
        f"  - Pre-2018 (Tier 0): {result_t0['treatment_effect']:+.1f} "
        f"deaths ({result_t0['pct_effect']:+.1f}%)"
    )
    if result_t0["treatment_effect"] <= 0:
        print("  - PRE-2018 FAILURE: No positive effect before paper's start date")
    print(
        f"  - Out-of-sample (Tier 3): {result_t3['treatment_effect']:+.1f} "
        f"deaths ({result_t3['pct_effect']:+.1f}%)"
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
            f"    {row['artist']:<18} {row['album'][:22]:<22} "
            f"δ={row['delta_raw']:+6.1f} ({streams_str} streams)"
        )

    t3_per_album = result_t3["per_album_df"]
    print("\n  Tier 3 per-album effects (post-2022):")
    for _, row in t3_per_album.iterrows():
        streams = [
            a[4] for a in ALBUMS_TIER3 if a[0] == row["artist"] and a[1] == row["album"]
        ]
        streams_str = f"{streams[0]:.0f}M" if streams else ""
        print(
            f"    {row['artist']:<18} {row['album'][:22]:<22} "
            f"δ={row['delta_raw']:+6.1f} ({streams_str} streams)"
        )

    return replication_df


DATA_DIR = "data/fars/"


def main():
    warnings.filterwarnings("ignore")
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

    # ═══════════════════════════════════════════════════════════════════════
    # LOAD & PREPROCESS
    # ═══════════════════════════════════════════════════════════════════════
    accidents = load_local_fars(DATA_DIR)
    print(f"\nLoaded {len(accidents)} crash records")

    daily = build_daily_series(accidents)
    gates.run_all(
        daily,
        window=args.window,
        accidents=accidents,
        miacc=load_local_miacc(DATA_DIR),
    )
    _pw = daily[(daily["date"].dt.year >= 2017) & (daily["date"].dt.year <= 2022)]
    record(
        "FridayMean",
        _pw[_pw["date"].dt.dayofweek == 4]["fatalities"].mean(),
        "{:.1f}",
        "Mean fatalities on Fridays, 2017-2022",
    )
    record(
        "OverallMean",
        _pw["fatalities"].mean(),
        "{:.1f}",
        "Mean fatalities on all days, 2017-2022",
    )
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
    save_table(
        balance_results,
        "tabs/t24_balance_check.md",
        caption=(
            "Release days against the days in their own +/-10 windows. Covariates "
            "not recorded for every treated day are skipped rather than compared "
            "across different coverage."
        ),
    )
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
    record(
        "RDose",
        corr_results["r_pearson"],
        "{:.2f}",
        "Pearson correlation of per-album effect with first-day streams, 20 albums",
    )
    record("RDoseP", corr_results["r_pearson_p"], "{:.2f}")
    record("RDoseLo", corr_results["r_pearson_ci_lower"], "{:.2f}")
    record("RDoseHi", corr_results["r_pearson_ci_upper"], "{:.2f}")
    _n = len(corr_results["streams"])
    _tc = scipy_stats.t.ppf(0.975, _n - 2)
    record(
        "RDoseMinDetectable",
        _tc / np.sqrt(_tc**2 + _n - 2),
        "{:.2f}",
        "Smallest correlation this many albums could call significant",
    )
    _t1 = scipy_stats.pearsonr(
        corr_results["streams"][:10], corr_results["deltas"][:10]
    )
    record(
        "RDoseTierOne",
        _t1.statistic,
        "{:.2f}",
        "Tier 1 only, the ten albums whose stream counts are measured "
        "rather than estimated from chart position",
    )
    record("RDoseTierOneP", _t1.pvalue, "{:.2f}")
    _tc10 = scipy_stats.t.ppf(0.975, 8)
    record(
        "RDoseMinDetectableTierOne",
        _tc10 / np.sqrt(_tc10**2 + 8),
        "{:.2f}",
        "Smallest correlation ten albums could call significant",
    )
    dr_results = dose_response_analysis(df, window=args.window)

    # Design checks that need df_global (pretrends, parallel trends)
    pretrends_results = pretrends_analysis(df_global)

    parallel_results = parallel_trends_test(df_global, window=args.window)
    record(
        "PPreTrendDropWorst",
        parallel_results["joint_p_drop_worst_day"].iloc[0],
        "{:.2f}",
        "Joint pre-trend test with the largest single day removed",
    )
    save_table(
        parallel_results,
        "tabs/t32_parallel_trends.md",
        caption=(
            "Residual estimator by event time with the modal weekday of each "
            "offset named. Day -6 is a Saturday; the previous Friday is day -7. "
            "Joint test uses album-level sign flips."
        ),
    )
    print("Saved: tabs/t32_parallel_trends.md")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: VALID INFERENCE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: VALID INFERENCE")
    print("=" * 70)

    leave_one_out(local_df)
    ri_results = randomization_inference(df_global, block_size=7)

    one_day = robustness_to_one_day(df_global)
    record(
        "MedianEffect",
        one_day.iloc[1]["value"],
        "{:.1f}",
        "Median of the ten release-day residuals",
    )
    record("PMedian", one_day.iloc[1]["p_value"], "{:.4f}")
    record("TrimmedEffect", one_day.iloc[2]["value"], "{:.1f}")
    record("NPositiveTierOne", one_day.iloc[3]["value"], "{:.0f}")
    record("PSignTest", one_day.iloc[3]["p_value"], "{:.3f}")
    record("PWilcoxon", one_day.iloc[4]["p_value"], "{:.4f}")
    record(
        "EffectDropMostInfluential",
        one_day.iloc[5]["value"],
        "{:.1f}",
        "Estimate with the single most influential album removed",
    )
    record("PDropMostInfluential", one_day.iloc[5]["p_value"], "{:.3f}")
    save_table(
        one_day,
        "tabs/t39_one_day_robustness.md",
        decimals=4,
        caption=(
            "Statistics a single large day cannot move: rank tests, a trimmed "
            "mean, and the estimate with the most influential album dropped."
        ),
    )
    print("Saved: tabs/t39_one_day_robustness.md")

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

    # Exogenous weather: NOAA station precipitation, nothing from the crash
    # record. Runs only if the cached pull is present; the pipeline stays
    # offline-reproducible without it.
    try:
        import json as _json

        from src.s12_weather import CACHE as _WCACHE
        from src.s12_weather import WEIGHTS as _WWEIGHTS

        if _WCACHE.exists() and _WWEIGHTS.exists():
            _state_daily = pd.read_csv(_WCACHE, parse_dates=["date"])
            _weights = _json.loads(_WWEIGHTS.read_text())
            _national = build_national_weather(_state_daily, _weights)
            exo = exogenous_weather_model(daily, _national, window=args.window)
            record(
                "EffectNoWeather",
                exo.iloc[0]["effect"],
                "{:.2f}",
                "Release-day effect on the weather sample, no weather control",
            )
            record(
                "EffectWeatherControlled",
                exo.iloc[1]["effect"],
                "{:.2f}",
                "The same with fatality-weighted station precipitation added",
            )
            record("WeatherSpread", exo["effect"].max() - exo["effect"].min(), "{:.2f}")
            save_table(
                exo,
                "tabs/t43_exogenous_weather.md",
                decimals=4,
                caption=(
                    "Release-day effect with daily precipitation from NOAA "
                    "stations in all 51 states, weighted by each state's share "
                    "of US fatalities. Unlike the crash-composition shares, "
                    "nothing here is a function of the crash record."
                ),
            )
            print("Saved: tabs/t43_exogenous_weather.md")
        else:
            print("\n  (station weather not cached; skipping exogenous model)")
    except Exception as _exc:  # never let the optional pull break the pipeline
        print(f"\n  (exogenous weather skipped: {type(_exc).__name__}: {_exc})")

    # Spotify Charts: the alternative the paper could not address, a measured
    # dose, and a frame anyone can rebuild. Optional, like the station weather.
    try:
        from src.s13_streaming import US_DAILY as _USD

        if _USD.exists():
            _rows, _sdaily = load_streaming()
            shock = shared_shock_test(daily, _sdaily, window=args.window)
            record(
                "EffectNoStream",
                shock.iloc[0]["effect"],
                "{:.2f}",
                "Release-day effect on the streaming sample, no control",
            )
            record(
                "EffectBackgroundStream",
                shock.iloc[1]["effect"],
                "{:.2f}",
                "The same with ambient listening controlled",
            )
            record(
                "BgStreamCoef",
                shock.iloc[0]["bg_coef"],
                "{:.4f}",
                "Deaths per million ambient streams, release windows excluded",
            )
            record("BgStreamSE", shock.iloc[0]["bg_se"], "{:.4f}")
            record(
                "BgStreamSD",
                shock.iloc[0]["bg_sd"],
                "{:.1f}",
                "SD of daily ambient streaming, millions",
            )
            record("BgStreamSDMove", shock.iloc[0]["bg_sd_move"], "{:.2f}")
            _bm = benchmark_against_paper(_sdaily, window=args.window)
            record(
                "StreamControlMean",
                _bm["control_mean"],
                "{:.1f}",
                "Mean US top-200 streams on the days surrounding the ten "
                "releases, the paper's own estimand, millions",
            )
            record("StreamControlLo", _bm["control_lo"], "{:.1f}")
            record("StreamControlHi", _bm["control_hi"], "{:.1f}")
            record(
                "PaperStreamControlMean",
                86.1,
                "{:.1f}",
                "The same quantity as reported in the original paper",
            )
            record(
                "StreamReleaseMean",
                _bm["release_mean"],
                "{:.1f}",
                "Mean US top-200 streams on the ten release days, millions",
            )
            record(
                "PaperStreamReleaseMean",
                123.3,
                "{:.1f}",
                "The same quantity as reported in the original paper",
            )
            save_table(
                shock.drop(
                    columns=[
                        "bg_coef",
                        "bg_se",
                        "bg_sd",
                        "bg_sd_move",
                        "control_day_streams",
                    ]
                ),
                "tabs/t44_shared_shock.md",
                decimals=3,
                caption=(
                    "Background streaming is the daily US top-200 total with "
                    "every track released that day removed, so it is ambient "
                    "listening the release did not cause. The total-streaming "
                    "row is a bad control shown for contrast, not a robustness "
                    "check."
                ),
            )
            print("Saved: tabs/t44_shared_shock.md")

            _frame, frame_res = reproducible_frame(
                _rows, df_global.set_index("date")["resid_global"]
            )
            record(
                "EffectFrameTen",
                frame_res.iloc[0]["effect"],
                "{:.2f}",
                "Top ten albums by measured first-day US streams",
            )
            record(
                "EffectFrameFifty",
                frame_res.iloc[-1]["effect"],
                "{:.2f}",
                "Top fifty on the same frame",
            )
            save_table(
                frame_res,
                "tabs/t45_reproducible_frame.md",
                decimals=3,
                caption=(
                    "Albums 2018-2022 ranked by measured first-day US streams "
                    "from Spotify Charts, so the sample is generated by a rule "
                    "rather than assembled by hand. The estimate is present at "
                    "every frame size and falls as the frame widens."
                ),
            )
            print("Saved: tabs/t45_reproducible_frame.md")

            _fd = measured_first_day(_rows)
            _fd.to_csv("data/spotify/album_first_day.csv", index=False)
            for _artist, _title, _rdate, _, _global_m in ALBUMS_TIER1:
                if _title not in ("Midnights", "Un Verano Sin Ti"):
                    continue
                _key = "Midnights" if _title == "Midnights" else "UnVerano"
                _hit = _fd[
                    (_fd["lead_artist"] == _artist)
                    & (_fd["date"] == pd.Timestamp(_rdate))
                ]
                if len(_hit):
                    record(
                        f"Measured{_key}",
                        _hit.iloc[0]["first_day_millions"],
                        "{:.1f}",
                        f"Measured US first-day streams for {_title}, millions",
                    )
                record(
                    f"Paper{_key}",
                    _global_m,
                    "{:.1f}",
                    f"First-day streams for {_title} as reported in Table 1 of "
                    f"the original paper, millions",
                )
            _resid = df_global.set_index("date")["resid_global"]
            per_album, binned, dsum = dose_response_all(_rows, _resid)
            per_album.to_csv("data/spotify/dose_per_album.csv", index=False)
            record(
                "NDoseAll",
                dsum["n"],
                "{:.0f}",
                "Albums with a measured dose and a fatality residual",
            )
            record(
                "RDoseAll",
                dsum["pearson"],
                "{:.3f}",
                "Correlation of per-album effect with measured first-day "
                "US streams, every charting album",
            )
            record("RDoseAllP", dsum["pearson_p"], "{:.3f}")
            record("RDoseAllMin", dsum["r_min_detectable"], "{:.3f}")
            record(
                "DoseSlope",
                dsum["slope_per_million"],
                "{:.4f}",
                "Deaths per million first-day streams",
            )
            record("DoseTopDecile", dsum["top_decile_effect"], "{:.2f}")
            save_table(
                binned,
                "tabs/t46_dose_response_all.md",
                decimals=3,
                caption=(
                    "Release-day excess by decile of measured first-day US "
                    "streams, across every album on the chart. The ten-album "
                    "version of this test could not detect a correlation below "
                    "0.63; this one detects above 0.086."
                ),
            )
            print("Saved: tabs/t46_dose_response_all.md")

            _dm, _cut = dose_matched_contrast(per_album, _fd)
            record(
                "DoseCut",
                _cut,
                "{:.1f}",
                "Measured first-day US streams of the smallest of the paper's "
                "ten albums, millions",
            )
            record(
                "NDoseMatched",
                _dm.iloc[1]["n"],
                "{:.0f}",
                "Albums other than the paper's that reach that dose",
            )
            record(
                "EffectDoseMatched",
                _dm.iloc[1]["effect"],
                "{:.2f}",
                "Release-day excess for those albums",
            )
            record("EffectDoseMatchedSE", _dm.iloc[1]["se"], "{:.2f}")
            record("EffectDoseMatchedLo", _dm.iloc[1]["ci_lower"], "{:.2f}")
            record("EffectDoseMatchedHi", _dm.iloc[1]["ci_upper"], "{:.2f}")
            record(
                "EffectDoseMatchedWindow",
                _dm.iloc[2]["effect"],
                "{:.2f}",
                "The same, restricted to the paper's own years",
            )
            record("EffectDoseMatchedWindowSE", _dm.iloc[2]["se"], "{:.2f}")
            record("NDoseMatchedWindow", _dm.iloc[2]["n"], "{:.0f}")
            record(
                "EffectBelowDoseCut",
                _dm.iloc[3]["effect"],
                "{:.2f}",
                "Release-day excess for albums below that dose",
            )
            record("EffectBelowDoseCutSE", _dm.iloc[3]["se"], "{:.2f}")
            record("NBelowDoseCut", _dm.iloc[3]["n"], "{:.0f}")
            save_table(
                _dm,
                "tabs/t47_dose_matched.md",
                decimals=2,
                caption=(
                    "Holding measured dose fixed. Every album on the US daily "
                    "chart whose first-day streams reach the smallest of the "
                    "paper's ten is a comparison unit, which makes the "
                    "comparison set a rule rather than a judgement about which "
                    "releases were major. Effects are means of donut residuals "
                    "with standard errors across albums."
                ),
            )
            print("Saved: tabs/t47_dose_matched.md")
            import matplotlib as _mpl

            _mpl.use("Agg")
            plot_dose_response(
                per_album,
                binned,
                dsum,
                show=False,
                paper_keys=paper_album_keys(_fd),
            )
        else:
            print("\n  (Spotify series not cached; skipping streaming analyses)")
    except Exception as _exc:
        print(f"\n  (streaming analyses skipped: {type(_exc).__name__}: {_exc})")

    rel_weather = release_day_weather(accidents, window=args.window)
    save_table(
        rel_weather,
        "tabs/t40_release_day_weather.md",
        decimals=4,
        caption=(
            "Snow and rain shares of fatal crashes on each release day against "
            "the same album's window. Shares come from the crashes themselves, "
            "not from weather stations, so they describe composition rather "
            "than exposure."
        ),
    )
    print("Saved: tabs/t40_release_day_weather.md")

    weather_results = weather_controlled_model(
        build_daily_weather_controls(accidents), window=args.window
    )
    save_table(
        weather_results,
        "tabs/t21_fars_controls.md",
        caption=(
            "Full-series regression on a release-day indicator with day-of-week, "
            "month, year and holiday fixed effects, plus the listed weather "
            "shares. Not the stacked album-day design, so the level differs from "
            "t12."
        ),
    )
    print("Saved: tabs/t21_fars_controls.md")

    dow_results = dow_interaction_models(daily, window=args.window)
    record(
        "EffectDowInteractMin",
        dow_results["effect"].min(),
        "{:.2f}",
        "Smallest estimate across day-of-week interaction models",
    )
    record("EffectDowInteractMax", dow_results["effect"].max(), "{:.1f}")
    record("PDowInteractMin", dow_results["p_value"].min(), "{:.3f}")
    record("PDowInteractMax", dow_results["p_value"].max(), "{:.3f}")
    save_table(
        dow_results,
        "tabs/t30_dow_interactions.md",
        caption=(
            "Residual estimator with the additive day-of-week term replaced by "
            "the listed interaction. mean_friday_resid shows how little additive "
            "Friday signal the baseline leaves."
        ),
    )
    print("Saved: tabs/t30_dow_interactions.md")

    multiverse_results = multiverse_analysis(df)
    save_table(
        multiverse_results,
        "tabs/t29_multiverse.md",
        caption=(
            "Residual estimator across 7 album-set and sample-period "
            "configurations at 5 window widths. The 5 nulls are one configuration "
            "repeated, so read 6 of 7 configurations rather than 30 of 35 tests."
        ),
    )
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

    friday_results = friday_placebo_test(df_global)
    _fri = friday_results.iloc[0]
    record(
        "EffectResidual",
        _fri["actual_effect"],
        "{:.1f}",
        "Tier 1 mean donut residual, the estimate the placebos are compared with",
    )
    record("FridayPlaceboMean", _fri["null_mean"], "{:.2f}")
    record(
        "FridayPlaceboDraws",
        _fri["n_draws_ge_actual"],
        "{:.0f}",
        "Draws of ten random Fridays reaching the observed effect, of 10,000",
    )
    save_table(
        friday_results,
        "tabs/t18_friday_placebo.md",
        caption=(
            "Placebo distribution for the Tier 1 residual estimate (+16.1). "
            "Fridays drawn at random from days that are not release days."
        ),
    )
    print("Saved: tabs/t18_friday_placebo.md")

    cal_placebo = calendar_position_placebo(df_global, window=args.window)
    _cp = cal_placebo[cal_placebo["album"] == "POOLED"].iloc[0]
    _cp_stats = album_stats(
        cal_placebo[cal_placebo["album"] != "POOLED"]["mean_effect"].values
    )
    record(
        "EffectCalendarPlacebo",
        _cp["mean_effect"],
        "{:.2f}",
        "Same calendar positions in other years, clustered by album",
    )
    record("SECalendarPlacebo", _cp_stats["se"], "{:.2f}")
    record("NCalendarPlacebo", _cp["n_placebo_dates"], "{:.0f}")
    save_table(
        cal_placebo,
        "tabs/t41_calendar_position_placebo.md",
        caption=(
            "Each album's calendar position applied to every other year, "
            "matched to the nearest same weekday. Dates inside a real release "
            "window are dropped; the pooled standard error clusters by album."
        ),
    )
    print("Saved: tabs/t41_calendar_position_placebo.md")

    dow_inv = weekday_dummy_invariance(daily, window=args.window)
    record(
        "ContrastWithDow",
        dow_inv.iloc[0]["contrast"],
        "{:.1f}",
        "Release days minus the Friday mean, weekday dummies in the model",
    )
    record(
        "ContrastNoDow",
        dow_inv.iloc[1]["contrast"],
        "{:.1f}",
        "The same contrast with the weekday dummies removed",
    )
    record("FridayResidNoDow", dow_inv.iloc[1]["mean_friday_resid"], "{:.2f}")
    record("ObservedNoDow", dow_inv.iloc[1]["observed_effect"], "{:.1f}")
    record(
        "PPlaceboNoDow",
        dow_inv.iloc[1]["placebo_share_ge_observed"],
        "{:.4f}",
        "Friday placebo p-value with the weekday dummies removed",
    )
    save_table(
        dow_inv,
        "tabs/t38_weekday_dummy_invariance.md",
        caption=(
            "The Friday placebo with and without weekday dummies. Removing "
            "them shifts both the release days and the Friday pool by the same "
            "amount, leaving the contrast and the p-value unchanged."
        ),
    )
    print("Saved: tabs/t38_weekday_dummy_invariance.md")

    adj_fri = adjacent_friday_contrast(daily)
    _pool = adj_fri[adj_fri["album"] == "POOLED"].iloc[0]
    record(
        "EffectAdjacentFriday",
        _pool["delta"],
        "{:.1f}",
        "Release Friday against the same album's neighbouring Fridays",
    )
    record("RatioAdjacentFriday", _pool["ratio"], "{:.2f}")
    _adj_stats = album_stats(adj_fri[adj_fri["album"] != "POOLED"]["delta"].values)
    record("CIAdjacentFridayLo", _adj_stats["ci_lower"], "{:.1f}")
    record("CIAdjacentFridayHi", _adj_stats["ci_upper"], "{:.1f}")
    record("PAdjacentFriday", _adj_stats["p_value"], "{:.3f}")
    record("SEAdjacentFriday", _adj_stats["se"], "{:.2f}")
    save_table(
        adj_fri,
        "tabs/t37_adjacent_friday.md",
        caption=(
            "Within-album, same-weekday contrast: each Friday release against "
            "the Fridays 7 and 14 days either side. No fixed effects and no "
            "residuals. Donda is the one Sunday release and is excluded."
        ),
    )
    print("Saved: tabs/t37_adjacent_friday.md")

    cherry_results = cherry_pick_benchmark(df_global)
    save_table(
        cherry_results,
        "tabs/t18b_cherry_pick_benchmark.md",
        caption=(
            "NOT a placebo test. Top 10 residuals from 100 drawn days, which "
            "measures what date selection can manufacture. The Tuesday row shows "
            "it is an order statistic, not a day-of-week result."
        ),
    )
    print("Saved: tabs/t18b_cherry_pick_benchmark.md")

    placebo_out = placebo_outcomes(accidents, window=args.window)
    save_table(
        placebo_out,
        "tabs/t28_placebo_outcomes.md",
        caption=(
            "Residual estimator applied to fatality subsets. Rows with fewer than "
            "5 contributing albums carry a point estimate and no standard error."
        ),
    )
    print("Saved: tabs/t28_placebo_outcomes.md")

    covar = composition_covariation(accidents, window=args.window)
    _hl = covar[covar["album"] == "Her Loss"]
    if len(_hl):
        record(
            "LatShiftHerLoss",
            _hl.iloc[0]["latitude_shift"],
            "{:.3f}",
            "Latitude shift on the album with the largest excess",
        )
        record("ExcessHerLoss", _hl.iloc[0]["excess_deaths"], "{:.1f}")
    _c = covar[covar["album"].str.startswith("CORRELATION")].iloc[0]
    record(
        "RLatitudeExcess",
        _c["excess_deaths"],
        "{:.2f}",
        "Correlation between an album's excess deaths and its latitude shift",
    )
    record("PLatitudeExcess", _c["latitude_shift"], "{:.2f}")
    save_table(
        covar,
        "tabs/t42_composition_covariation.md",
        decimals=3,
        caption=(
            "If extra crashes dragged mean latitude, albums with the largest "
            "excess would show the largest shift. They do not, so the "
            "mechanical reading of the latitude result is an assertion rather "
            "than a finding."
        ),
    )
    print("Saved: tabs/t42_composition_covariation.md")

    structural_results = structural_fars_composition(accidents, window=args.window)
    _lat = structural_results[structural_results["variable"] == "LATITUDE"]
    if len(_lat):
        record(
            "EffectLatitude",
            _lat.iloc[0]["effect"],
            "{:.2f}",
            "Mean crash latitude on release days, degrees",
        )
        record("TLatitude", _lat.iloc[0]["t_stat"], "{:.2f}")
    save_table(
        structural_results,
        "tabs/t28b_structural_fars_composition.md",
        caption=(
            "Residual estimator applied to crash composition. These are post- "
            "treatment summaries of the same crashes, not placebo outcomes."
        ),
    )
    print("Saved: tabs/t28b_structural_fars_composition.md")

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
    _null = power_results[power_results["se_source"].str.startswith("null")]
    _p = _null.iloc[0]
    record(
        "ResidSD",
        _p["resid_sd"],
        "{:.1f}",
        "SD of daily fatalities after the fixed effects",
    )
    record("SENull", _p["se"], "{:.2f}", "Null SE of a ten-day mean")
    record(
        "SERealized",
        power_results[power_results["se_source"].str.startswith("realized")]["se"].iloc[
            0
        ],
        "{:.2f}",
        "SE across the ten release-day residuals",
    )
    record(
        "MDEEightyClustered",
        power_results[power_results["se_source"].str.contains("clustered")][
            "mde_80"
        ].iloc[0],
        "{:.1f}",
        "MDE using the paper's own clustered standard error",
    )
    record(
        "MDEEighty",
        _p["mde_80"],
        "{:.1f}",
        "Smallest effect detected 80% of the time, t(9) critical value",
    )
    record("MDEFifty", _p["mde_50"], "{:.1f}")
    _five = _null[_null["true_effect"] == 5].iloc[0]
    record("PowerAtFive", _five["power"], "{:.2f}")
    record("ExaggerationAtFive", _five["exaggeration"], "{:.1f}")
    save_table(power_results, "tabs/t23_power_analysis.md")
    print("Saved: tabs/t23_power_analysis.md")

    tier1_effect = local_df["delta_local"].mean()
    effect_size_plausibility_check(tier1_effect)
    holiday_bench = holiday_benchmark(daily)
    _hb = holiday_bench[
        (holiday_bench["baseline"] == "dow+month+year")
        & (holiday_bench["day"] == "Album release days")
    ].iloc[0]
    record(
        "EffectBenchmark",
        _hb["excess"],
        "{:.1f}",
        "Release-day excess on the holiday-comparable baseline",
    )
    _mem = holiday_bench[
        (holiday_bench["baseline"] == "dow+month+year")
        & (holiday_bench["day"] == "Memorial Day")
    ].iloc[0]
    record("ExcessMemorialDay", _mem["excess"], "{:.1f}")
    hol_adj = holiday_adjacency_robustness(daily, window=args.window)
    record(
        "EffectHolAdjMin",
        hol_adj["effect"].min(),
        "{:.1f}",
        "Smallest estimate across holiday-window variants",
    )
    record("EffectHolAdjMax", hol_adj["effect"].max(), "{:.1f}")
    _drop = hol_adj.iloc[-1]
    record(
        "EffectDropHolAdj",
        _drop["effect"],
        "{:.1f}",
        "Dropping the two holiday-adjacent releases",
    )
    record("NDropHolAdj", _drop["n_albums"], "{:.0f}")
    save_table(
        hol_adj,
        "tabs/t35_holiday_adjacency.md",
        caption=(
            "Residual estimator. Two Tier 1 releases sit within three days of "
            "a federal holiday. Widening the holiday control and dropping both "
            "albums leave the estimate essentially unchanged."
        ),
    )
    print("Saved: tabs/t35_holiday_adjacency.md")
    save_table(
        holiday_bench,
        "tabs/t34_holiday_benchmark.md",
        caption=(
            "Excess over a day-of-week, seasonal and year baseline with holiday "
            "terms removed, so holidays are not absorbed by their own dummies. "
            "All nine federal holidays, two seasonal baselines."
        ),
    )
    print("Saved: tabs/t34_holiday_benchmark.md")
    weather_sanity, _, _ = weather_effect_sanity_check(
        build_daily_weather_controls(accidents)
    )
    save_table(weather_sanity, "tabs/t33_weather_effects.md")
    print("Saved: tabs/t33_weather_effects.md")

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 8: HETEROGENEITY / MECHANISM
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 8: HETEROGENEITY / MECHANISM")
    print("=" * 70)

    dynamic_results = compute_dynamic_effects(df_global, window=args.window)
    _dyn = dynamic_results.set_index("day")
    record(
        "EffectDayPlusOne",
        _dyn.loc[1, "effect"],
        "{:.1f}",
        "Event-study coefficient on the day after release",
    )
    record("TDayPlusOne", _dyn.loc[1, "t_stat"], "{:.2f}")
    record(
        "EffectDayMinusSix",
        _dyn.loc[-6, "effect"],
        "{:.1f}",
        "The Saturday before release for nine of the ten albums",
    )
    record(
        "EffectDayMinusSeven",
        _dyn.loc[-7, "effect"],
        "{:.1f}",
        "The previous Friday, the same weekday as release",
    )
    save_table(
        dynamic_results,
        "tabs/t13_dynamic_effects.md",
        caption=(
            "Residual estimator by event time, Tier 1. Standard errors are across "
            "the 10 albums (ddof = 1); intervals use t with 9 df. Day 0 is the "
            "release day."
        ),
    )
    print("Saved: tabs/t13_dynamic_effects.md")

    tod_results = time_of_day_analysis(accidents, df_global, window=args.window)
    if tod_results is not None:
        save_table(tod_results, "tabs/t14_time_of_day.md")
        print("Saved: tabs/t14_time_of_day.md")

    alcohol_results = alcohol_decomposition(accidents, load_local_miacc(DATA_DIR))
    _no = alcohol_results.iloc[0]
    _yes = alcohol_results.iloc[1]
    record(
        "EffectNoAlcohol",
        _no["effect"],
        "{:.2f}",
        "Release-day excess in crashes with no driver at BAC 0.08 or above",
    )
    record("SENoAlcohol", _no["se"], "{:.2f}")
    record("PNoAlcohol", _no["p_value"], "{:.3f}")
    record(
        "EffectAlcohol",
        _yes["effect"],
        "{:.2f}",
        "The same for crashes with a driver at or above the limit",
    )
    record("SEAlcohol", _yes["se"], "{:.2f}")
    record("PAlcohol", _yes["p_value"], "{:.3f}")
    save_table(
        alcohol_results,
        "tabs/t22_alcohol_decomposition.md",
        decimals=3,
        caption=(
            "Decomposition, not a mechanism test: the split is post-treatment, "
            "so it describes where the release-day excess sits and cannot "
            "identify why. Ten albums, ten NHTSA imputations of driver BAC "
            "combined by Rubin's rules, threshold 0.08."
        ),
    )
    print("Saved: tabs/t22_alcohol_decomposition.md")

    covid_results = covid_sensitivity(df_global)
    save_table(covid_results, "tabs/t19_covid_sensitivity.md")
    print("Saved: tabs/t19_covid_sensitivity.md")

    extended_results = extended_series_analysis(df_global)
    _out = extended_results[extended_results["tier"].str.startswith("Excluding")].iloc[
        0
    ]
    record(
        "EffectOutsidePaper",
        _out["avg_effect"],
        "{:.1f}",
        "Pooled effect over the 27 albums that are not the paper's ten",
    )
    record("CIOutsideLo", _out["ci_lower"], "{:.1f}")
    record("CIOutsideHi", _out["ci_upper"], "{:.1f}")
    record("POutsidePaper", _out["sign_test_p"], "{:.2f}", "Sign test over those 27")
    record("NOutsidePaper", _out["n_albums"], "{:.0f}")
    record("NPositiveOutside", _out["n_positive"], "{:.0f}")
    _t1 = extended_results[extended_results["tier"].str.startswith("Tier 1")].iloc[0]
    _t2 = extended_results[extended_results["tier"].str.startswith("Tier 2")].iloc[0]
    _diff = _t1["avg_effect"] - _t2["avg_effect"]
    _diff_se = float(np.sqrt(_t1["se"] ** 2 + _t2["se"] ** 2))
    record(
        "SelectionDiff",
        _diff,
        "{:.1f}",
        "Top ten by first-day streams minus the next ten. Tests whether "
        "being selected on maximum streaming predicts a larger effect.",
    )
    record("SelectionDiffSE", _diff_se, "{:.1f}")
    _all = extended_results[extended_results["tier"].str.startswith("All")].iloc[0]
    record(
        "EffectAllAlbums",
        _all["avg_effect"],
        "{:.1f}",
        "All 37 albums, residual estimator",
    )
    record("CIAllAlbumsLo", _all["ci_lower"], "{:.1f}")
    record("CIAllAlbumsHi", _all["ci_upper"], "{:.1f}")
    record("NAllAlbums", _all["n_albums"], "{:.0f}")
    pub = publication_filter(
        [
            round(float(_out["avg_effect"]), 2),
            round(float(_all["avg_effect"]), 2),
            16.1,
        ],
        se=7.45,
    )
    record(
        "MeanPublishedAtOutside",
        pub.iloc[0]["mean_published"],
        "{:.1f}",
        "What a ten-album study would report if the truth were the "
        "outside-sample pooled effect and the SE were the paper's",
    )
    record("PowerAtOutside", pub.iloc[0]["power"], "{:.2f}")
    record("PReachReplicated", pub.iloc[0]["p_reaches_replicated"], "{:.2f}")
    save_table(
        pub,
        "tabs/t36_publication_filter.md",
        caption=(
            "What a ten-album study would report, given a true effect and the "
            "paper's album-clustered standard error of 7.45. Reconciles the "
            "pooled estimate outside the paper's sample with its headline."
        ),
    )
    print("Saved: tabs/t36_publication_filter.md")
    save_table(
        extended_results,
        "tabs/t20_extended_series.md",
        caption=(
            "Residual estimator, mean per-album effect by album tier. Comparable "
            "to +16.1 for Tier 1, not to the regression estimate of +17.6."
        ),
    )
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

    # ═══════════════════════════════════════════════════════════════════════
    # ONE SOURCE OF TRUTH: emit every number the prose is allowed to quote
    # ═══════════════════════════════════════════════════════════════════════
    write_numbers()


if __name__ == "__main__":
    main()
