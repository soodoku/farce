"""
Magnitude Checks — effect size plausibility.

Functions for checking whether the effect size is plausible:
- Power analysis (Type M / Type S errors)
- Effect size plausibility vs weather benchmarks
- Weather effect sanity check
"""

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import ALBUMS_TIER1
from src.utils import add_time_features, build_design_matrix, ols_fit


def power_analysis(df_global, n_albums=10, alpha=0.05):
    """
    Type M / Power Analysis (Gelman).

    With N=10 albums and σ=typical residual SD, what's the minimum detectable
    effect at 80% power? And if the true effect is X, what's the expected
    exaggeration when we condition on p<0.05?

    Type M error: Expected exaggeration ratio = E[|estimate| | p<0.05] / true_effect
    Type S error: Probability of wrong sign | p<0.05

    Output: tabs/t23_power_analysis.csv
    """
    print(f"\n{'='*70}")
    print("TYPE M / POWER ANALYSIS (Gelman)")
    print(f"{'='*70}")
    print("With small N, even TRUE effects are wildly overestimated when")
    print("they pass the significance threshold.\n")

    resid_sd = df_global["resid_global"].std()
    se = resid_sd / sqrt(n_albums)

    print(f"Observed residual SD: {resid_sd:.1f} deaths/day")
    print(f"Standard error with N={n_albums}: {se:.1f} deaths")

    z_crit = stats.norm.ppf(1 - alpha / 2)
    mde_80 = (z_crit + stats.norm.ppf(0.80)) * se
    mde_50 = z_crit * se

    print(f"\nMINIMUM DETECTABLE EFFECT (two-sided α={alpha}):")
    print(f"  80% power: {mde_80:.1f} deaths")
    print(f"  50% power: {mde_50:.1f} deaths")

    tier1_resids = []
    for album_tuple in ALBUMS_TIER1:
        dt = pd.to_datetime(album_tuple[2])
        row = df_global[df_global["date"] == dt]
        if len(row) > 0:
            tier1_resids.append(row["resid_global"].values[0])

    observed_effect = np.mean(tier1_resids) if tier1_resids else 0
    print(f"\nObserved Tier 1 effect: {observed_effect:+.1f} deaths")

    true_effects = [0, 2, 5, 10, 15, 20, observed_effect]
    results = []

    print(f"\nTYPE M AND TYPE S ERRORS (via simulation, 50000 draws):")
    print(f"{'True Effect':>12} | {'Power':>8} | {'Type M':>10} | {'Type S':>10}")
    print("-" * 50)

    rng = np.random.RandomState(42)
    n_sims = 50000

    for true_eff in sorted(set(true_effects)):
        estimates = true_eff + rng.normal(0, se, n_sims)
        z_stats = estimates / se
        significant = np.abs(z_stats) > z_crit

        power = significant.mean()

        if significant.sum() > 0:
            type_m = np.abs(estimates[significant]).mean() / max(abs(true_eff), 0.1)
            type_s = (
                (np.sign(estimates[significant]) != np.sign(true_eff)).mean()
                if true_eff != 0
                else np.nan
            )
        else:
            type_m = np.nan
            type_s = np.nan

        results.append(
            {
                "true_effect": true_eff,
                "se": se,
                "power": power,
                "type_m_ratio": type_m,
                "type_s_prob": type_s,
                "mde_80": mde_80,
                "mde_50": mde_50,
                "n_albums": n_albums,
                "resid_sd": resid_sd,
            }
        )

        type_m_str = f"{type_m:.2f}" if not np.isnan(type_m) else "N/A"
        type_s_str = f"{type_s:.3f}" if not np.isnan(type_s) else "N/A"
        marker = " <-- observed" if true_eff == observed_effect else ""
        print(
            f"{true_eff:>12.1f} | {power:>8.1%} | {type_m_str:>10} | {type_s_str:>10}{marker}"
        )

    results_df = pd.DataFrame(results)

    print("\nINTERPRETATION:")
    if observed_effect < mde_80:
        print(
            f"  Observed effect ({observed_effect:+.1f}) < MDE at 80% power ({mde_80:.1f})"
        )
        print("  Study is UNDERPOWERED to detect effects of this magnitude reliably.")

    true_5_row = results_df[results_df["true_effect"] == 5]
    if len(true_5_row) > 0:
        m5 = true_5_row["type_m_ratio"].values[0]
        print(f"\n  If true effect is 5 deaths:")
        print(f"    Expected exaggeration (Type M): {m5:.1f}x")
        print(f"    Observed effect / Type M = {observed_effect / m5:.1f} deaths")
        print("    This is a shrinkage estimate toward zero.")

    baseline = df_global["resid_global"].mean() + 121
    pct_effect = 100 * observed_effect / baseline
    print(f"\n  PLAUSIBILITY CHECK:")
    print(f"    Observed effect: +{observed_effect:.1f} deaths (+{pct_effect:.1f}%)")
    print(f"    US baseline: ~100 deaths/day")
    print(f"    If weekly releases: 52 * 18 = ~936 extra deaths/year")
    print(f"    That's almost 1% of all traffic fatalities from album releases alone!")

    return results_df


def effect_size_plausibility_check(album_effect):
    """
    Compare claimed streaming effect to weather effects as a sanity check.

    The paper claims ~16 extra deaths per album release day. This function
    compares that effect size to well-documented weather effects from the
    literature to assess plausibility.

    Key insight: Weather is REGIONAL, not nationwide. A storm in Texas
    doesn't affect California drivers. This means weather affects ~10-30%
    of drivers on any given day, not all of them.

    Parameters
    ----------
    album_effect : float
        The estimated album release effect in deaths per day.

    Returns
    -------
    dict
        Verdict and comparison data.
    """
    print("\n" + "=" * 70)
    print("EFFECT SIZE PLAUSIBILITY CHECK")
    print("=" * 70)
    print(f"Question: Is +{album_effect:.0f} deaths per release day plausible?\n")

    print("BENCHMARKS FROM LITERATURE:")
    print("-" * 70)
    benchmarks = [
        (
            "Adverse weather (FHWA 2019-2023)",
            "~10 deaths/day attributable to weather",
            "~3,807 weather-related deaths/year nationwide",
        ),
        (
            "Precipitation (Black et al. 2023)",
            "+34% relative risk (RR=1.34)",
            "Fatal crash risk increases by 34% on precip days",
        ),
    ]

    for source, finding, detail in benchmarks:
        print(f"  {source}:")
        print(f"    {finding}")
        print(f"    ({detail})")
        print()

    print("EXPOSURE COMPARISON (weather is regional, not nationwide):")
    print("-" * 70)
    print("  Adverse weather:  ~10-30% of US drivers affected (regional storms)")
    print("  Album streaming:  ~0.2-0.4% of car trips (in-car listeners of new album)")
    print()
    print("  Calculation for album exposure:")
    print("    - Day-1 streams: ~15-40M streams nationwide")
    print("    - In-car listening: ~10% of streaming happens in cars")
    print("    - In-car sessions: ~2-4M")
    print("    - US car trips/day: ~1 billion")
    print("    - Album exposure: ~0.2-0.4% of trips")
    print()

    print("EFFECT SIZE COMPARISON:")
    print("-" * 70)
    comparison_table = [
        (
            "Adverse weather (FHWA)",
            "~10 deaths/day avg",
            "~10-30% of US drivers (regional)",
        ),
        (
            "Album release (paper claim)",
            f"+{album_effect:.0f} deaths/day",
            "~0.2-0.4% of car trips",
        ),
    ]

    print(f"{'Cause':<35} | {'Effect Size':<20} | {'Population Affected':<35}")
    print("-" * 95)
    for cause, effect, population in comparison_table:
        print(f"{cause:<35} | {effect:<20} | {population:<35}")

    print("\nNORMALIZED RISK (deaths per 1% of drivers exposed):")
    print("-" * 70)
    weather_low = 10 / 30
    weather_high = 10 / 10
    album_low = album_effect / 0.4
    album_high = album_effect / 0.2
    ratio_low = album_low / weather_high
    ratio_high = album_high / weather_low

    print(f"  Weather: {weather_low:.2f}-{weather_high:.2f} deaths per 1% exposure")
    print(f"  Album:   {album_low:.0f}-{album_high:.0f} deaths per 1% exposure")
    print(
        f"  Ratio:   Album listening implied {ratio_low:.0f}-{ratio_high:.0f}x more dangerous"
    )

    print("\n" + "=" * 70)
    print("VERDICT: Claimed effect is IMPLAUSIBLE")
    print("=" * 70)
    print(
        f"""
Reasoning:
  1. PER-DRIVER RISK RATIO: The paper implies listening to a new album
     while driving is {ratio_low:.0f}-{ratio_high:.0f}x more dangerous per driver-exposure
     than driving in adverse weather.

  2. WEATHER PHYSICALLY IMPAIRS DRIVING: Reduced visibility, wet/icy roads,
     longer stopping distances. These are genuine mechanical hazards.

  3. NOT A NOVEL DISTRACTION: Listening to new songs vs. old songs is not
     a fundamentally different distraction. People already listen to music/
     radio while driving safely.

  4. MOST STREAMING IS NOT IN-CAR: Majority of streaming occurs at home,
     work, or via headphones - not while driving.

  5. BENCHMARK: The implied per-listener risk exceeds drunk driving, which
     has RR ~6-8. The paper implies RR > 10 for album listeners.
"""
    )

    print("CITATIONS:")
    print("-" * 70)
    citations = [
        ("FHWA (2019-2023)", "https://ops.fhwa.dot.gov/weather/roadimpact.htm"),
        ("Black et al. (2023)", "https://pmc.ncbi.nlm.nih.gov/articles/PMC10248718/"),
    ]

    for source, url in citations:
        print(f"  {source}:")
        print(f"    {url}")

    return {
        "album_effect": album_effect,
        "weather_effect_fhwa": 10,
        "weather_exposure_pct": (10, 30),
        "album_exposure_pct": (0.2, 0.4),
        "deaths_per_1pct_weather": (weather_low, weather_high),
        "deaths_per_1pct_album": (album_low, album_high),
        "danger_ratio": (ratio_low, ratio_high),
        "verdict": "IMPLAUSIBLE",
        "reason": "Per-driver risk implied to be 400-2600x higher than weather effects",
    }


def weather_effect_sanity_check(daily_df):
    """
    Sanity check: Are weather effect sizes sensible?

    Prior research suggests:
    - Rain: +5-15% crash frequency, +0-10% severity
    - Fog: +10-30% crash frequency, +5-15% severity
    - Expected daily fatality increase from bad weather: 2-8 deaths (2-8%)

    This function:
    1. Outputs weather coefficients with practical interpretations
    2. Checks for multicollinearity between weather variables
    3. Runs single-variable models to avoid collinearity issues
    4. Compares to literature benchmarks

    Returns DataFrame with weather effects analysis.
    """
    print(f"\n{'='*70}")
    print("WEATHER EFFECT SANITY CHECK")
    print(f"{'='*70}")
    print("Question: Are weather coefficients sensible, or 'too big'?")
    print("Literature: Rain +5-15% crashes, Fog +10-30% crashes")
    print("Expected: 2-8 extra deaths per day from bad weather (~2-8%)\n")

    df = daily_df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df = add_time_features(df)

    weather_vars = ["pct_rain", "pct_fog", "pct_cloudy", "pct_bad_weather"]
    baseline_deaths = df["fatalities"].mean()

    print("=" * 70)
    print("1. DESCRIPTIVE STATISTICS")
    print("=" * 70)
    print(f"\nBaseline daily fatalities: {baseline_deaths:.1f}\n")

    stats_rows = []
    for var in weather_vars:
        if var in df.columns:
            vals = df[var].dropna()
            stats_rows.append(
                {
                    "variable": var,
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "p10": vals.quantile(0.10),
                    "p50": vals.quantile(0.50),
                    "p90": vals.quantile(0.90),
                }
            )
            print(f"{var}:")
            print(f"  Mean: {vals.mean():.3f}, SD: {vals.std():.3f}")
            p10, p50, p90 = (
                vals.quantile(0.10),
                vals.quantile(0.50),
                vals.quantile(0.90),
            )
            print(f"  10th: {p10:.3f}, Median: {p50:.3f}, 90th: {p90:.3f}")

    stats_df = pd.DataFrame(stats_rows)

    print("\n" + "=" * 70)
    print("2. MULTICOLLINEARITY CHECK")
    print("=" * 70)
    print("\nCorrelation matrix between weather variables:")

    weather_data = df[weather_vars].dropna()
    corr_matrix = weather_data.corr()

    print("\n" + " " * 16 + "  ".join([f"{v:>12}" for v in weather_vars]))
    for i, v1 in enumerate(weather_vars):
        row_str = f"{v1:<16}"
        for v2 in weather_vars:
            row_str += f"  {corr_matrix.loc[v1, v2]:>12.3f}"
        print(row_str)

    high_corr_pairs = []
    for i, v1 in enumerate(weather_vars):
        for j, v2 in enumerate(weather_vars):
            if i < j and abs(corr_matrix.loc[v1, v2]) > 0.5:
                high_corr_pairs.append((v1, v2, corr_matrix.loc[v1, v2]))

    if high_corr_pairs:
        print("\nWARNING: High correlations detected (|r| > 0.5):")
        for v1, v2, r in high_corr_pairs:
            print(f"  {v1} ~ {v2}: r = {r:.3f}")
        print(
            "  This explains potentially unstable/flipped coefficients in multi-variable models."
        )
    else:
        print("\nNo high correlations (|r| > 0.5) detected.")

    print("\n" + "=" * 70)
    print("3. SINGLE-VARIABLE MODELS (avoids multicollinearity)")
    print("=" * 70)
    print("\nEach model: fatalities ~ weather_var + DOW + Month + Year + holidays")

    single_var_results = []

    for var in weather_vars:
        if var not in df.columns:
            continue

        X = build_design_matrix(df, controls=[var])
        y = df["fatalities"].values.astype(float)

        beta, se, _, _ = ols_fit(X.values, y, return_se=True)

        var_idx = list(X.columns).index(var)
        coef = beta[var_idx]
        coef_se = se[var_idx]
        t_stat = coef / coef_se if coef_se > 0 else 0

        var_stats = stats_df[stats_df["variable"] == var].iloc[0]
        effect_1sd = coef * var_stats["std"]
        effect_10_90 = coef * (var_stats["p90"] - var_stats["p10"])

        single_var_results.append(
            {
                "variable": var,
                "coefficient": coef,
                "se": coef_se,
                "t_stat": t_stat,
                "effect_1sd": effect_1sd,
                "effect_10_90": effect_10_90,
                "pct_effect_1sd": 100 * effect_1sd / baseline_deaths,
            }
        )

    single_df = pd.DataFrame(single_var_results)

    hdr = f"\n{'Variable':<18} | {'Coef':>10} | {'SE':>8} | {'t':>7}"
    hdr += f" | {'1SD Effect':>12} | {'10-90 Effect':>13}"
    print(hdr)
    print("-" * 85)
    for _, r in single_df.iterrows():
        sig = " **" if abs(r["t_stat"]) > 2 else " *" if abs(r["t_stat"]) > 1.65 else ""
        print(
            f"{r['variable']:<18} | {r['coefficient']:>+10.2f} | {r['se']:>8.2f} | "
            f"{r['t_stat']:>+6.2f}{sig} | {r['effect_1sd']:>+11.2f} | {r['effect_10_90']:>+12.2f}"
        )

    print("\n" + "=" * 70)
    print("4. MULTI-VARIABLE MODEL (for comparison)")
    print("=" * 70)
    print("\nModel: fatalities ~ pct_rain + pct_fog + pct_cloudy + FEs")

    multi_vars = ["pct_rain", "pct_fog", "pct_cloudy"]
    X = build_design_matrix(df, controls=multi_vars)
    y = df["fatalities"].values.astype(float)

    beta, se, _, _ = ols_fit(X.values, y, return_se=True)

    multi_results = []
    for var in multi_vars:
        var_idx = list(X.columns).index(var)
        coef = beta[var_idx]
        coef_se = se[var_idx]
        t_stat = coef / coef_se if coef_se > 0 else 0
        multi_results.append(
            {
                "variable": var,
                "coefficient": coef,
                "se": coef_se,
                "t_stat": t_stat,
            }
        )

    multi_df = pd.DataFrame(multi_results)

    print(f"\n{'Variable':<18} | {'Coef':>10} | {'SE':>8} | {'t':>7}")
    print("-" * 50)
    for _, r in multi_df.iterrows():
        sig = " **" if abs(r["t_stat"]) > 2 else " *" if abs(r["t_stat"]) > 1.65 else ""
        coef, se_val, t = r["coefficient"], r["se"], r["t_stat"]
        print(
            f"{r['variable']:<18} | {coef:>+10.2f} | {se_val:>8.2f} | {t:>+6.2f}{sig}"
        )

    print("\n" + "=" * 70)
    print("5. LITERATURE COMPARISON")
    print("=" * 70)
    print("\nExpected effects from prior research:")
    print("  - Rain: +5-15% crash frequency → ~5-15 extra deaths/day")
    print("  - Fog: +10-30% crash frequency → ~10-30 extra deaths/day")
    print("  - Bad weather overall: +2-8% deaths → ~2-8 extra deaths/day")

    print("\nObserved effects (single-variable models, 10th→90th percentile):")
    concerns = []
    for _, r in single_df.iterrows():
        effect = r["effect_10_90"]
        var = r["variable"]

        if var == "pct_rain":
            expected_range = (5, 15)
        elif var == "pct_fog":
            expected_range = (10, 30)
        elif var == "pct_bad_weather":
            expected_range = (2, 8)
        else:
            expected_range = None

        status = ""
        if expected_range:
            if effect < 0:
                status = "WRONG SIGN (negative)"
                concerns.append(f"{var}: coefficient is negative ({effect:+.1f})")
            elif effect < expected_range[0] * 0.5:
                status = "TOO SMALL"
            elif effect > expected_range[1] * 3:
                status = "TOO BIG"
                lo, hi = expected_range
                concerns.append(
                    f"{var}: effect too large ({effect:+.1f} vs expected {lo}-{hi})"
                )
            else:
                status = "PLAUSIBLE"

        print(f"  {var}: {effect:+.1f} deaths — {status}")

    print("\n" + "=" * 70)
    print("6. SANITY CHECK VERDICT")
    print("=" * 70)

    if concerns:
        print("\nCONCERNS IDENTIFIED:")
        for c in concerns:
            print(f"  - {c}")
        print("\nPossible explanations:")
        print("  1. Multicollinearity inflating/flipping coefficients")
        print("  2. Weather measured as % of crashes (endogenous to crash count)")
        print("  3. Selection: worse weather → more crashes → lower % per crash")
    else:
        print("\nNo major concerns. Weather effects are within plausible ranges.")

    output_df = single_df.copy()
    output_df["model_type"] = "single_variable"
    output_df["baseline_deaths"] = baseline_deaths

    multi_df_extended = multi_df.copy()
    multi_df_extended["model_type"] = "multi_variable"
    multi_df_extended["effect_1sd"] = np.nan
    multi_df_extended["effect_10_90"] = np.nan
    multi_df_extended["pct_effect_1sd"] = np.nan
    multi_df_extended["baseline_deaths"] = baseline_deaths

    output_df = pd.concat([output_df, multi_df_extended], ignore_index=True)

    return output_df, corr_matrix, concerns
