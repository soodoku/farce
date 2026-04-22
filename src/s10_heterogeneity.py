"""
Heterogeneity / Mechanism / Subgroups — effect variation analysis.

Functions for understanding effect variation:
- Dynamic effects (event study)
- Time-of-day analysis
- Stream-effect correlation (dose-response)
- Tier comparison (dose-response)
- Drunk vs sober mechanism test
- COVID sensitivity
- Extended series analysis
"""

import datetime
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import (ALBUMS, ALBUMS_ALL, ALBUMS_EXTENDED, ALBUMS_TIER0,
                           ALBUMS_TIER1, ALBUMS_TIER2, ALBUMS_TIER3,
                           RELEASE_DATES_ALL)
from src.utils import add_time_features, build_design_matrix, ols_fit


def compute_dynamic_effects(df_global, window=5):
    """
    Compute effects for each day in [-window, +window] relative to release.

    Shows whether effect is concentrated on day 0 or spread across multiple days.
    Pre-release effects suggest anticipation or confounding.
    Post-release persistence suggests using post-days in control is biased.

    Returns DataFrame with day, effect, se, ci_lower, ci_upper.
    """
    print(f"\n{'='*70}")
    print("DYNAMIC EFFECTS ANALYSIS (Event Study)")
    print(f"{'='*70}")
    print(f"Computing effects for days {-window} to +{window} around release.")
    print("If mechanism is causal, effect should be concentrated on day 0.\n")

    results = []

    for day_offset in range(-window, window + 1):
        day_resids = []

        for _, _, date_str, _ in ALBUMS:
            dt = pd.to_datetime(date_str)
            target_dt = dt + pd.Timedelta(days=day_offset)

            row = df_global[df_global["date"] == target_dt]
            if len(row) > 0:
                day_resids.append(row["resid_global"].values[0])

        if day_resids:
            effect = np.mean(day_resids)
            se = np.std(day_resids) / sqrt(len(day_resids))
            ci_lower = effect - 1.96 * se
            ci_upper = effect + 1.96 * se

            results.append(
                {
                    "day": day_offset,
                    "effect": effect,
                    "se": se,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_albums": len(day_resids),
                }
            )

    results_df = pd.DataFrame(results)

    print(f"{'Day':>5} | {'Effect':>10} | {'SE':>8} | {'95% CI':>20} | {'n':>3}")
    print("-" * 55)
    for _, r in results_df.iterrows():
        ci_str = f"[{r['ci_lower']:+.1f}, {r['ci_upper']:+.1f}]"
        marker = " <-- RELEASE" if r["day"] == 0 else ""
        print(
            f"{int(r['day']):>5} | {r['effect']:>+10.1f} | "
            f"{r['se']:>8.1f} | {ci_str:>20} | {int(r['n_albums']):>3}{marker}"
        )

    day0_effect = results_df[results_df["day"] == 0]["effect"].values[0]
    pre_effect = results_df[results_df["day"] < 0]["effect"].mean()
    post_effect = results_df[(results_df["day"] > 0) & (results_df["day"] <= 3)][
        "effect"
    ].mean()

    print("\nSUMMARY:")
    print(f"  Day 0 effect: {day0_effect:+.1f} deaths")
    print(f"  Pre-release avg (days {-window} to -1): {pre_effect:+.1f} deaths")
    print(f"  Post-release avg (days +1 to +3): {post_effect:+.1f} deaths")

    print("\nINTERPRETATION:")
    if pre_effect > 0 and day0_effect > 0:
        pre_pct = 100 * pre_effect / day0_effect
        print(f"  Pre-release effects are {pre_pct:.0f}% of day-0 effect.")
        if pre_pct > 30:
            print(
                "  WARNING: Large pre-release effects suggest confounding, not causation."
            )
    if post_effect > 0.3 * day0_effect:
        print("  WARNING: Effects persist after release day.")
        print("           Using post-release days in control window may be biased.")

    return results_df


def load_fars_by_hour(accidents, hour_range):
    """Load FARS data filtered to specific hours."""
    df = accidents.copy()

    cols = {c.upper(): c for c in df.columns}
    if "HOUR" not in cols:
        raise ValueError("HOUR column not found in FARS data")

    hour_col = cols["HOUR"]
    return df[df[hour_col].isin(hour_range)]


def build_hourly_daily_series(accidents, hour_ranges):
    """
    Build separate daily fatality series for each time window.

    Parameters
    ----------
    accidents : DataFrame
        Raw FARS accident data
    hour_ranges : dict
        Mapping of time window name to list of hours

    Returns
    -------
    dict of DataFrames
        Daily fatality series for each time window
    """
    from src.s02_preprocess import build_daily_series

    results = {}
    for name, hours in hour_ranges.items():
        subset = load_fars_by_hour(accidents, hours)
        if len(subset) > 0:
            daily = build_daily_series(subset)
            results[name] = daily

    return results


def time_of_day_analysis(accidents, df_global, window=10):
    """
    Test if effects concentrate during driving hours.

    If streaming causes distracted driving, effects should be highest during
    rush hour (7-9am, 4-7pm) and low during late night (11pm-5am).

    Returns DataFrame with results for each time window.
    """
    print(f"\n{'='*70}")
    print("TIME-OF-DAY ANALYSIS (Mechanism Test)")
    print(f"{'='*70}")
    print("If streaming causes distracted driving, effects should concentrate")
    print("during rush hour (driving time) and be minimal at night.\n")

    hour_ranges = {
        "rush_hour": list(range(7, 10)) + list(range(16, 20)),
        "midday": list(range(10, 16)),
        "evening": list(range(20, 23)),
        "late_night": list(range(23, 24)) + list(range(0, 6)),
    }

    cols = {c.upper(): c for c in accidents.columns}
    if "HOUR" not in cols:
        print(
            "ERROR: HOUR column not found in FARS data. Cannot perform time-of-day analysis."
        )
        return None

    hour_col = cols["HOUR"]

    for candidate in ["YEAR", "CASEYEAR"]:
        if candidate in cols:
            accidents["_year"] = accidents[cols[candidate]]
            break

    for candidate in ["MONTH"]:
        if candidate in cols:
            accidents["_month"] = accidents[cols[candidate]]

    for candidate in ["DAY", "DAY_OF_CRASH"]:
        if candidate in cols:
            accidents["_day"] = accidents[cols[candidate]]
            break

    if "FATALS" in cols:
        accidents["_fatals"] = accidents[cols["FATALS"]]
    else:
        accidents["_fatals"] = 1

    accidents = accidents.dropna(subset=["_year", "_month", "_day"])

    def safe_date(row):
        try:
            return datetime.date(
                int(row["_year"]), int(row["_month"]), int(row["_day"])
            )
        except ValueError:
            return None

    accidents["_date"] = accidents.apply(safe_date, axis=1)
    accidents = accidents.dropna(subset=["_date"])
    accidents["_date"] = pd.to_datetime(accidents["_date"])

    results = []

    for window_name, hours in hour_ranges.items():
        subset = accidents[accidents[hour_col].isin(hours)].copy()

        daily = subset.groupby("_date")["_fatals"].sum().reset_index()
        daily.columns = ["date", "fatalities"]
        daily["date"] = pd.to_datetime(daily["date"])

        all_dates_exclude = set()
        for a in ALBUMS_TIER1:
            dt = pd.to_datetime(a[2]).date()
            for offset in range(-window, window + 1):
                all_dates_exclude.add(dt + datetime.timedelta(days=offset))

        daily = add_time_features(daily)

        est_mask = ~daily["date"].dt.date.isin(all_dates_exclude)
        if est_mask.sum() < 100:
            continue

        X_est = build_design_matrix(daily[est_mask])
        y_est = daily.loc[est_mask, "fatalities"].values.astype(float)
        beta, _, _ = ols_fit(X_est.values, y_est)

        X_all = build_design_matrix(daily)
        daily["fitted"] = X_all.values @ beta
        daily["resid"] = daily["fatalities"].values - daily["fitted"]

        release_resids = []
        for a in ALBUMS_TIER1:
            dt = pd.to_datetime(a[2])
            row = daily[daily["date"] == dt]
            if len(row) > 0:
                release_resids.append(row["resid"].values[0])

        if release_resids:
            effect = np.mean(release_resids)
            se = np.std(release_resids) / sqrt(len(release_resids))
            baseline = daily["fatalities"].mean()

            results.append(
                {
                    "time_window": window_name,
                    "hours": str(hours),
                    "baseline_deaths": baseline,
                    "effect": effect,
                    "se": se,
                    "n_albums": len(release_resids),
                }
            )

    if not results:
        print("No results computed. Check if HOUR data is available.")
        return None

    results_df = pd.DataFrame(results)

    total_effect = results_df["effect"].sum()
    results_df["pct_of_total"] = (
        100 * results_df["effect"] / total_effect if total_effect != 0 else 0
    )

    print(
        f"{'Time Window':<12} | {'Hours':>20} | {'Baseline':>10} | {'Effect':>10} | {'SE':>8} | {'% Total':>8}"
    )
    print("-" * 80)
    for _, r in results_df.iterrows():
        print(
            f"{r['time_window']:<12} | {r['hours'][:20]:>20} | {r['baseline_deaths']:>10.1f} | "
            f"{r['effect']:>+10.1f} | {r['se']:>8.1f} | {r['pct_of_total']:>7.1f}%"
        )

    print("\nINTERPRETATION:")
    rush = results_df[results_df["time_window"] == "rush_hour"]
    night = results_df[results_df["time_window"] == "late_night"]

    if len(rush) > 0 and len(night) > 0:
        rush_effect = rush["effect"].values[0]
        night_effect = night["effect"].values[0]

        if rush_effect > night_effect:
            print("  Rush hour effect > late night effect.")
            print("  This is CONSISTENT with distracted driving mechanism.")
        else:
            print("  WARNING: Late night effect >= rush hour effect.")
            print("  This is INCONSISTENT with distracted driving mechanism.")
            print("  (People don't drive much late at night)")

    return results_df


def stream_effect_correlation(df, window=10):
    """
    Compute correlation between first-day streams and fatality effect.
    A real mechanism should show positive correlation (more streams -> more effect).
    """
    print(f"\n{'='*70}")
    print("STREAM-EFFECT CORRELATION ANALYSIS")
    print(f"{'='*70}")

    df = df.copy()
    all_dates_exclude = set()
    for a in ALBUMS_ALL:
        dt = pd.to_datetime(a[2]).date()
        for offset in range(-window, window + 1):
            all_dates_exclude.add(dt + datetime.timedelta(days=offset))

    df = add_time_features(df)

    est_mask = ~df["date"].dt.date.isin(all_dates_exclude)
    X_est = build_design_matrix(df[est_mask])
    y_est = df.loc[est_mask, "fatalities"].values.astype(float)
    beta, _, _ = ols_fit(X_est.values, y_est)

    X_all = build_design_matrix(df)
    df["fitted_dr"] = X_all.values @ beta
    df["resid_dr"] = df["fatalities"].values - df["fitted_dr"]

    streams = []
    deltas = []
    album_names = []

    print(f"\n{'Album':<40} {'Streams (M)':>12} {'δ (deaths)':>12} {'z-score':>10}")
    print("-" * 78)

    for a in ALBUMS_ALL:
        dt = pd.to_datetime(a[2])
        row = df[df["date"] == dt]
        if len(row) == 0:
            continue
        delta = row["resid_dr"].values[0]
        streams.append(a[4])
        deltas.append(delta)
        album_names.append(f"{a[0]} - {a[1][:20]}")

        resid_sd = (y_est - X_est.values @ beta).std()
        z = delta / resid_sd
        print(f"  {a[0]:<18} {a[1]:<20} {a[4]:>8.0f} {delta:>12.1f} {z:>10.2f}")

    streams = np.array(streams)
    deltas = np.array(deltas)

    r_pearson, p_pearson = stats.pearsonr(streams, deltas)
    r_spearman, p_spearman = stats.spearmanr(streams, deltas)

    print("\n── Correlation Analysis ──")
    print(f"  Pearson r:  {r_pearson:+.3f}  (p = {p_pearson:.4f})")
    print(f"  Spearman ρ: {r_spearman:+.3f}  (p = {p_spearman:.4f})")

    if r_pearson < 0:
        print("\n  WARNING: Negative correlation between streams and effect!")
        print("           This is OPPOSITE to dose-response prediction.")
        print("           Albums with more streams show SMALLER fatality effects.")

    weighted_avg = np.sum(streams * deltas) / np.sum(streams)
    unweighted_avg = np.mean(deltas)
    print("\n── Weighted vs Unweighted Estimates ──")
    print(f"  Unweighted avg δ: {unweighted_avg:+.1f} deaths")
    print(f"  Stream-weighted avg δ: {weighted_avg:+.1f} deaths")
    if weighted_avg < unweighted_avg:
        print("  Weighted estimate is SMALLER — high-stream albums contribute less.")

    print("\n── Formal Dose-Response Regression ──")
    print("  Model: δᵢ = β₀ + β₁ × log(streamsᵢ) + ε")

    log_streams = np.log(streams)
    X_reg = np.column_stack([np.ones(len(streams)), log_streams])
    beta_reg = np.linalg.lstsq(X_reg, deltas, rcond=None)[0]
    fitted_reg = X_reg @ beta_reg
    residuals_reg = deltas - fitted_reg
    se_reg = np.sqrt(
        np.sum(residuals_reg**2)
        / (len(deltas) - 2)
        * np.diag(np.linalg.inv(X_reg.T @ X_reg))
    )
    t_stat = beta_reg[1] / se_reg[1]

    print(f"  β₀ (intercept): {beta_reg[0]:+.1f} (SE = {se_reg[0]:.1f})")
    print(f"  β₁ (log-streams): {beta_reg[1]:+.1f} (SE = {se_reg[1]:.1f})")
    print(f"  t-stat for β₁: {t_stat:.2f}")
    print("  Interpretation: A 10% increase in streams is associated with")
    print(f"                  {beta_reg[1] * np.log(1.1):+.2f} additional deaths")

    if beta_reg[1] < 0:
        print("\n  CRITICAL: β₁ < 0 means ANTI-dose-response!")
        print("            More streams → FEWER excess deaths")
        print("            This contradicts the distraction mechanism.")

    return {
        "streams": streams,
        "deltas": deltas,
        "album_names": album_names,
        "r_pearson": r_pearson,
        "r_spearman": r_spearman,
        "beta_dose": beta_reg,
        "weighted_avg": weighted_avg,
    }


def dose_response_analysis(df, window=10):
    """
    Compare the fatality effect for Tier 1 (top 10) vs Tier 2 (albums 11-20).
    If the mechanism is streaming-induced distraction, the effect should be
    monotonically decreasing in streaming intensity.
    """
    print(f"\n{'='*70}")
    print("DOSE-RESPONSE: TOP 10 vs ALBUMS 11-20")
    print(f"{'='*70}")

    df = df.copy()
    all_dates_exclude = set()
    for a in ALBUMS_ALL:
        dt = pd.to_datetime(a[2]).date()
        for offset in range(-window, window + 1):
            all_dates_exclude.add(dt + datetime.timedelta(days=offset))

    df = add_time_features(df)

    est_mask = ~df["date"].dt.date.isin(all_dates_exclude)
    X_est = build_design_matrix(df[est_mask])
    y_est = df.loc[est_mask, "fatalities"].values.astype(float)
    beta, _, _ = ols_fit(X_est.values, y_est)

    X_all = build_design_matrix(df)
    df["fitted_dr"] = X_all.values @ beta
    df["resid_dr"] = df["fatalities"].values - df["fitted_dr"]
    resid_sd = (y_est - X_est.values @ beta).std()

    print("\n── Tier 1 (Top 10, paper's sample) ──")
    deltas_t1 = []
    for a in ALBUMS_TIER1:
        dt = pd.to_datetime(a[2])
        row = df[df["date"] == dt]
        if len(row) == 0:
            continue
        delta = row["resid_dr"].values[0]
        deltas_t1.append(delta)
        print(f"  {a[0]:<22} {a[1]:<30} δ={delta:+6.1f}  (~{a[4]:.0f}M streams)")

    avg_t1 = np.mean(deltas_t1)
    se_t1 = np.std(deltas_t1) / np.sqrt(len(deltas_t1))
    print(
        f"  Avg δ (Tier 1): {avg_t1:+.1f} (SE={se_t1:.1f}), " f"z={avg_t1/resid_sd:.2f}"
    )

    print("\n── Tier 2 (Albums 11-20) ──")
    deltas_t2 = []
    for a in ALBUMS_TIER2:
        dt = pd.to_datetime(a[2])
        row = df[df["date"] == dt]
        if len(row) == 0:
            continue
        delta = row["resid_dr"].values[0]
        deltas_t2.append(delta)
        print(f"  {a[0]:<22} {a[1]:<30} δ={delta:+6.1f}  (~{a[4]:.0f}M streams)")

    avg_t2 = np.mean(deltas_t2)
    se_t2 = np.std(deltas_t2) / np.sqrt(len(deltas_t2))
    print(
        f"  Avg δ (Tier 2): {avg_t2:+.1f} (SE={se_t2:.1f}), " f"z={avg_t2/resid_sd:.2f}"
    )

    diff = avg_t1 - avg_t2
    se_diff = np.sqrt(se_t1**2 + se_t2**2)
    print("\n── Dose-Response Summary ──")
    print(f"  Tier 1 avg δ: {avg_t1:+.1f} deaths ({100*avg_t1/121:+.1f}%)")
    print(f"  Tier 2 avg δ: {avg_t2:+.1f} deaths ({100*avg_t2/121:+.1f}%)")
    print(f"  Difference:   {diff:+.1f} (SE={se_diff:.1f})")
    if se_diff > 0:
        print(f"  t-stat (T1 vs T2): {diff/se_diff:.2f}")

    if avg_t1 > 0 and avg_t2 > 0:
        ratio = avg_t2 / avg_t1
        print(f"  Tier2/Tier1 ratio: {ratio:.2f}")
        print("  Expected ratio if proportional to streams: ~0.50")
    elif avg_t2 <= 0:
        print("  Tier 2 effect is zero or negative — no dose-response gradient")

    fri_pool = df[
        (df["date"].dt.dayofweek == 4) & ~df["date"].dt.date.isin(RELEASE_DATES_ALL)
    ].index.values

    rng = np.random.RandomState(123)
    n_sims = 10000
    null_avgs = np.zeros(n_sims)
    for s in range(n_sims):
        idx = rng.choice(fri_pool, size=len(deltas_t2), replace=False)
        null_avgs[s] = df.loc[idx, "resid_dr"].mean()

    p_t2 = (null_avgs >= avg_t2).mean()
    print(f"\n  RI p-value (Tier 2, {len(deltas_t2)} random Fridays): {p_t2:.4f}")

    return {"avg_t1": avg_t1, "avg_t2": avg_t2, "p_t2": p_t2}


def drunk_vs_sober_analysis(accidents, window=10):
    """
    MECHANISM TEST: Split accidents by DRUNK_DR and test effect separately.

    NOTE: FARS changed data format in 2021 - DRUNK_DR column is no longer
    in the Accident table (moved to Person/Driver tables). Analysis limited
    to 2007-2020 data where DRUNK_DR is available.

    This is NOT a control — drunk driving is potentially endogenous.
    This is a MECHANISM TEST: if streaming causes distracted driving,
    the effect should appear in SOBER accidents (where distraction matters),
    not drunk accidents (where alcohol already impairs driving).

    Prediction: Sober effect > Drunk effect if mechanism is distraction.

    Returns DataFrame with drunk vs sober comparison.
    """
    print(f"\n{'='*70}")
    print("DRUNK VS SOBER ANALYSIS (Mechanism Test)")
    print(f"{'='*70}")
    print("MECHANISM TEST: Where does the effect come from?")
    print("  - If streaming → distracted driving:")
    print("    Effect should be in SOBER accidents (distraction matters)")
    print("    NOT in DRUNK accidents (alcohol already impairs)")
    print("  - If both show similar effects: mechanism unclear")
    print("\nNOTE: DRUNK_DR not available in FARS 2021+ (data format change).")
    print("      Analysis limited to 2007-2020 releases.\n")

    df = accidents.copy()
    cols = {c.upper(): c for c in df.columns}

    if "DRUNK_DR" not in cols:
        print("ERROR: DRUNK_DR column not found in FARS data.")
        return None

    drunk_col = cols["DRUNK_DR"]

    for candidate in ["YEAR", "CASEYEAR"]:
        if candidate in cols:
            df["_year"] = df[cols[candidate]]
            break

    if "MONTH" in cols:
        df["_month"] = df[cols["MONTH"]]

    for candidate in ["DAY", "DAY_OF_CRASH"]:
        if candidate in cols:
            df["_day"] = df[cols[candidate]]
            break

    if "FATALS" in cols:
        df["_fatals"] = df[cols["FATALS"]]
    else:
        df["_fatals"] = 1

    df["_drunk"] = df[drunk_col] >= 1
    df = df.dropna(subset=["_year", "_month", "_day"])

    df = df[df["_year"] <= 2020]
    if len(df) == 0:
        print("ERROR: No data with DRUNK_DR available (pre-2021 required).")
        return None

    def safe_date(row):
        try:
            return datetime.date(
                int(row["_year"]), int(row["_month"]), int(row["_day"])
            )
        except ValueError:
            return None

    df["_date"] = df.apply(safe_date, axis=1)
    df = df.dropna(subset=["_date"])
    df["_date"] = pd.to_datetime(df["_date"])

    pre_2021_albums = [a for a in ALBUMS_TIER1 if pd.to_datetime(a[2]).year <= 2020]
    print(f"Pre-2021 Tier 1 albums with DRUNK_DR data: {len(pre_2021_albums)}")
    for a in pre_2021_albums:
        print(f"  - {a[0]}: {a[1]} ({a[2]})")
    print()

    if len(pre_2021_albums) == 0:
        print("ERROR: No Tier 1 albums fall in pre-2021 period.")
        return None

    results = []

    for label, mask in [
        ("Sober (DRUNK_DR=0)", ~df["_drunk"]),
        ("Drunk (DRUNK_DR>=1)", df["_drunk"]),
    ]:
        subset = df[mask].copy()

        daily = subset.groupby("_date")["_fatals"].sum().reset_index()
        daily.columns = ["date", "fatalities"]
        daily["date"] = pd.to_datetime(daily["date"])

        all_dates_exclude = set()
        for a in pre_2021_albums:
            dt = pd.to_datetime(a[2]).date()
            for offset in range(-window, window + 1):
                all_dates_exclude.add(dt + datetime.timedelta(days=offset))

        daily = add_time_features(daily)

        est_mask = ~daily["date"].dt.date.isin(all_dates_exclude)
        if est_mask.sum() < 100:
            continue

        X_est = build_design_matrix(daily[est_mask])
        y_est = daily.loc[est_mask, "fatalities"].values.astype(float)
        beta, _, _ = ols_fit(X_est.values, y_est)

        X_all = build_design_matrix(daily)
        daily["fitted"] = X_all.values @ beta
        daily["resid"] = daily["fatalities"].values - daily["fitted"]

        release_resids = []
        for a in pre_2021_albums:
            dt = pd.to_datetime(a[2])
            row = daily[daily["date"] == dt]
            if len(row) > 0:
                release_resids.append(row["resid"].iloc[0])

        if release_resids:
            effect = np.mean(release_resids)
            se = np.std(release_resids) / sqrt(len(release_resids))
            t_stat = effect / se if se > 0 else 0
            baseline = daily["fatalities"].mean()
            pct_effect = 100 * effect / baseline

            results.append(
                {
                    "sample": label,
                    "baseline_deaths": baseline,
                    "effect": effect,
                    "se": se,
                    "t_stat": t_stat,
                    "pct_effect": pct_effect,
                    "n_albums": len(release_resids),
                }
            )

    if not results:
        print("No results computed.")
        return None

    results_df = pd.DataFrame(results)

    print(
        f"{'Sample':<25} | {'Baseline':>10} | {'Effect':>10} | {'SE':>8} | {'t-stat':>8} | {'%':>8}"
    )
    print("-" * 85)
    for _, r in results_df.iterrows():
        sig = " **" if abs(r["t_stat"]) > 2 else " *" if abs(r["t_stat"]) > 1.65 else ""
        print(
            f"{r['sample']:<25} | {r['baseline_deaths']:>10.1f} | {r['effect']:>+10.1f} | "
            f"{r['se']:>8.1f} | {r['t_stat']:>+7.2f}{sig} | {r['pct_effect']:>+7.1f}%"
        )

    print("\nMECHANISM TEST RESULT:")
    sober = results_df[results_df["sample"].str.contains("Sober")]
    drunk = results_df[results_df["sample"].str.contains("Drunk")]

    if len(sober) > 0 and len(drunk) > 0:
        sober_effect = sober["effect"].iloc[0]
        drunk_effect = drunk["effect"].iloc[0]

        if sober_effect > 0 and drunk_effect <= 0:
            print(f"  Sober effect: {sober_effect:+.1f} (positive)")
            print(f"  Drunk effect: {drunk_effect:+.1f} (null/negative)")
            print("  STRONG SUPPORT for distracted driving mechanism.")
        elif sober_effect > drunk_effect and sober_effect > 0:
            print(
                f"  Sober effect ({sober_effect:+.1f}) > Drunk effect ({drunk_effect:+.1f})"
            )
            print("  CONSISTENT with distracted driving mechanism.")
        elif drunk_effect > sober_effect:
            print(
                f"  Drunk effect ({drunk_effect:+.1f}) > Sober effect ({sober_effect:+.1f})"
            )
            print("  INCONSISTENT with distracted driving mechanism.")
            print("  (Why would streaming affect drunk drivers more than sober?)")
        else:
            print(
                f"  Effects similar: Sober ({sober_effect:+.1f}), Drunk ({drunk_effect:+.1f})"
            )
            print("  Mechanism ambiguous — could be something other than distraction.")

    return results_df


def covid_sensitivity(df_global, window=10):
    """
    Test if the effect is driven by COVID-era albums.

    COVID period: July 2020 - Nov 2021 (abnormal driving patterns)

    4 of 10 Tier 1 albums released during COVID:
    - Folklore (2020-07-24)
    - Donda (2021-08-29)
    - Certified Lover Boy (2021-09-03)
    - Red (Taylor's Version) (2021-11-12)

    Returns DataFrame with COVID vs non-COVID effects.
    """
    print(f"\n{'='*70}")
    print("COVID SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print("Testing whether COVID-era albums drive the effect.")
    print("COVID period: July 2020 - Nov 2021 (abnormal driving patterns)\n")

    covid_start = pd.to_datetime("2020-07-01")
    covid_end = pd.to_datetime("2021-11-30")

    covid_albums = []
    non_covid_albums = []

    for album_tuple in ALBUMS_TIER1:
        artist, album, date_str = album_tuple[0], album_tuple[1], album_tuple[2]
        dt = pd.to_datetime(date_str)

        if covid_start <= dt <= covid_end:
            covid_albums.append((artist, album, date_str, dt))
        else:
            non_covid_albums.append((artist, album, date_str, dt))

    print(f"COVID-era albums (n={len(covid_albums)}):")
    for artist, album, date_str, _ in covid_albums:
        print(f"  - {artist}: {album} ({date_str})")

    print(f"\nNon-COVID albums (n={len(non_covid_albums)}):")
    for artist, album, date_str, _ in non_covid_albums:
        print(f"  - {artist}: {album} ({date_str})")

    def compute_effect(album_list, label):
        """Compute effect for a list of albums."""
        resids = []
        per_album = []
        for artist, album, date_str, dt in album_list:
            row = df_global[df_global["date"] == dt]
            if len(row) > 0:
                resid = row["resid_global"].values[0]
                resids.append(resid)
                per_album.append(
                    {
                        "artist": artist,
                        "album": album,
                        "date": date_str,
                        "effect": resid,
                    }
                )

        if not resids:
            return None, None

        avg_effect = np.mean(resids)
        se = np.std(resids) / sqrt(len(resids))
        t_stat = avg_effect / se if se > 0 else 0

        return {
            "sample": label,
            "n_albums": len(resids),
            "avg_effect": avg_effect,
            "se": se,
            "t_stat": t_stat,
        }, per_album

    results = []

    full_list = [(a[0], a[1], a[2], pd.to_datetime(a[2])) for a in ALBUMS_TIER1]
    full_result, _ = compute_effect(full_list, "Full Tier 1")
    if full_result:
        results.append(full_result)

    covid_result, covid_per_album = compute_effect(covid_albums, "COVID albums")
    if covid_result:
        results.append(covid_result)

    non_covid_result, non_covid_per_album = compute_effect(
        non_covid_albums, "Non-COVID albums"
    )
    if non_covid_result:
        results.append(non_covid_result)

    if not results:
        print("No results computed.")
        return None

    results_df = pd.DataFrame(results)

    print(
        f"\n{'Sample':<20} | {'N':>5} | {'Avg Effect':>12} | {'SE':>10} | {'t-stat':>8}"
    )
    print("-" * 65)
    for _, r in results_df.iterrows():
        sig = " **" if abs(r["t_stat"]) > 2 else " *" if abs(r["t_stat"]) > 1.65 else ""
        print(
            f"{r['sample']:<20} | {r['n_albums']:>5} | {r['avg_effect']:>+12.1f} | "
            f"{r['se']:>10.1f} | {r['t_stat']:>+7.2f}{sig}"
        )

    if covid_per_album:
        print("\nPer-album effects (COVID era):")
        for item in covid_per_album:
            print(
                f"  {item['artist']:<18} {item['album'][:22]:<22} δ={item['effect']:+6.1f}"
            )

    if non_covid_per_album:
        print("\nPer-album effects (non-COVID):")
        for item in non_covid_per_album:
            print(
                f"  {item['artist']:<18} {item['album'][:22]:<22} δ={item['effect']:+6.1f}"
            )

    print("\nINTERPRETATION:")
    if covid_result and non_covid_result:
        covid_eff = covid_result["avg_effect"]
        non_covid_eff = non_covid_result["avg_effect"]

        if covid_eff > non_covid_eff * 1.5:
            print(
                f"  WARNING: COVID albums effect ({covid_eff:+.1f}) >> non-COVID ({non_covid_eff:+.1f})"
            )
            print("  COVID-era albums are driving the overall effect.")
            print("  This is confounded by COVID-era driving pattern anomalies.")
        elif non_covid_eff < 0:
            print(f"  CRITICAL: Non-COVID effect is NEGATIVE ({non_covid_eff:+.1f})")
            print("  Without COVID albums, there is NO positive effect!")
            print("  The finding is entirely driven by COVID anomalies.")
        else:
            print(
                f"  COVID effect: {covid_eff:+.1f}, Non-COVID effect: {non_covid_eff:+.1f}"
            )
            if abs(covid_eff - non_covid_eff) < 10:
                print("  Effects are similar across periods — not COVID-driven.")
            else:
                print(
                    "  Effects differ substantially — COVID period may be confounded."
                )

    return results_df


def extended_series_analysis(df_global):
    """
    Test effects across all tiers: Tier 0 (pre-2018) through Tier 3 (post-2022).

    Key question: Does the effect generalize beyond the paper's specific sample?

    If Tier 0 (pre-2018) shows:
    - Positive effect: Mechanism plausible OR methodology produces false positives
    - Null/negative effect: Paper's 2018+ start date may have been cherry-picked

    Returns DataFrame with tier, n_albums, avg_effect, se, t_stat.
    """
    print(f"\n{'='*70}")
    print("EXTENDED SERIES ANALYSIS (All Tiers)")
    print(f"{'='*70}")
    print("Testing effects across streaming eras:")
    print("  Tier 0: Pre-2018 streaming (2015-2017) — before paper's cutoff")
    print("  Tier 1: Paper's sample (2018-2022)")
    print("  Tier 2: Extended 2018-2022 (albums 11-20)")
    print("  Tier 3: Post-paper (2023-2024) — out-of-sample")
    print()

    tiers = [
        ("Tier 0 (pre-2018)", ALBUMS_TIER0),
        ("Tier 1 (paper)", ALBUMS_TIER1),
        ("Tier 2 (extended)", ALBUMS_TIER2),
        ("Tier 3 (post-paper)", ALBUMS_TIER3),
        ("All combined", ALBUMS_EXTENDED),
    ]

    results = []

    for tier_name, album_list in tiers:
        resids = []
        per_album = []

        for album_tuple in album_list:
            artist, album, date_str = album_tuple[0], album_tuple[1], album_tuple[2]
            dt = pd.to_datetime(date_str)
            row = df_global[df_global["date"] == dt]

            if len(row) > 0:
                resid = row["resid_global"].values[0]
                resids.append(resid)
                per_album.append(
                    {
                        "artist": artist,
                        "album": album,
                        "date": date_str,
                        "effect": resid,
                    }
                )

        if not resids:
            continue

        avg_effect = np.mean(resids)
        se = np.std(resids) / sqrt(len(resids))
        t_stat = avg_effect / se if se > 0 else 0

        results.append(
            {
                "tier": tier_name,
                "n_albums": len(resids),
                "avg_effect": avg_effect,
                "se": se,
                "t_stat": t_stat,
                "per_album": per_album,
            }
        )

    if not results:
        print("No results computed.")
        return None

    results_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "per_album"} for r in results]
    )

    print(f"{'Tier':<25} | {'N':>5} | {'Avg Effect':>12} | {'SE':>10} | {'t-stat':>8}")
    print("-" * 70)
    for r in results:
        sig = " **" if abs(r["t_stat"]) > 2 else " *" if abs(r["t_stat"]) > 1.65 else ""
        print(
            f"{r['tier']:<25} | {r['n_albums']:>5} | {r['avg_effect']:>+12.1f} | "
            f"{r['se']:>10.1f} | {r['t_stat']:>+7.2f}{sig}"
        )

    tier0_result = next((r for r in results if "pre-2018" in r["tier"]), None)
    tier1_result = next((r for r in results if "paper" in r["tier"]), None)
    tier3_result = next((r for r in results if "post-paper" in r["tier"]), None)
    all_result = next((r for r in results if "All" in r["tier"]), None)

    if tier0_result:
        print(f"\nTier 0 (pre-2018) per-album effects:")
        for item in tier0_result["per_album"]:
            print(
                f"  {item['artist']:<18} {item['album'][:22]:<22} δ={item['effect']:+6.1f}"
            )

    print("\nINTERPRETATION:")

    if tier0_result and tier1_result:
        t0_eff = tier0_result["avg_effect"]
        t1_eff = tier1_result["avg_effect"]

        if t0_eff > 0 and t1_eff > 0:
            ratio = t0_eff / t1_eff if t1_eff != 0 else float("inf")
            print(f"  Tier 0 effect: {t0_eff:+.1f}, Tier 1 effect: {t1_eff:+.1f}")
            print(f"  Ratio (Tier 0 / Tier 1): {ratio:.2f}")
            if ratio > 0.5:
                print("  Pre-2018 shows substantial effect — mechanism may be real")
                print("  OR methodology produces false positives across all eras.")
            else:
                print("  Pre-2018 effect is much smaller — suggests something")
                print("  specific to 2018-2022 period (COVID? selection bias?).")
        elif t0_eff <= 0 and t1_eff > 0:
            print(f"  Tier 0 effect: {t0_eff:+.1f} (null/negative)")
            print(f"  Tier 1 effect: {t1_eff:+.1f} (positive)")
            print("  CRITICAL: Pre-2018 shows NO positive effect!")
            print("  This suggests paper's 2018+ start date was cherry-picked.")

    if tier3_result and tier1_result:
        t3_eff = tier3_result["avg_effect"]
        t1_eff = tier1_result["avg_effect"]

        if t3_eff < 0 and t1_eff > 0:
            print(f"\n  Out-of-sample failure confirmed:")
            print(f"    Tier 1: {t1_eff:+.1f} deaths (paper's sample)")
            print(f"    Tier 3: {t3_eff:+.1f} deaths (2023-2024)")

    if all_result and tier1_result:
        all_eff = all_result["avg_effect"]
        t1_eff = tier1_result["avg_effect"]

        print(
            f"\n  Combined effect (all {all_result['n_albums']} albums): {all_eff:+.1f} deaths"
        )
        if abs(all_eff) < abs(t1_eff) / 2:
            print("  Combined effect is much smaller than Tier 1 alone.")
            print("  Tier 1 is an outlier — effect doesn't generalize.")

    return results_df
