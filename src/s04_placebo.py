"""
Placebo tests for FARS album release study.

Tests:
1. Pre-trends analysis: Do effects appear before release?
2. Window sensitivity: Is the effect stable across different window sizes?
3. Year permutation: Would the effect appear on the same calendar dates in other years?
"""

import datetime

import numpy as np
import pandas as pd

from src.constants import ALBUMS, RELEASE_DATES, us_holidays
from src.s02_preprocess import _build_design


def _ols_residuals(X, y):
    """OLS via normal equations, return (beta, fitted, residuals)."""
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    fitted = X @ beta
    return beta, fitted, y - fitted


def pretrends_analysis(df_global, pre_days=None):
    """
    Test for anticipation effects before release.

    If streaming causes deaths, effects should NOT appear before release.
    Compute average residual for each of days -5 to -1 (pre-treatment).

    Returns dict with pre-trend residuals and summary statistics.
    """
    if pre_days is None:
        pre_days = range(-5, 0)

    print(f"\n{'='*70}")
    print("PRE-TRENDS ANALYSIS")
    print(f"{'='*70}")
    print("If streaming causes deaths, effects should NOT appear before release.")
    print("Testing residuals on days -5 to -1 (before album release).\n")

    pre_resids = {d: [] for d in pre_days}
    day0_resids = []

    for _, _, date_str, _ in ALBUMS:
        dt = pd.to_datetime(date_str)

        # Day 0 (release day)
        row_0 = df_global[df_global["date"] == dt]
        if len(row_0) > 0:
            day0_resids.append(row_0["resid_global"].values[0])

        # Pre-release days
        for d in pre_days:
            pre_dt = dt + pd.Timedelta(days=d)
            row = df_global[df_global["date"] == pre_dt]
            if len(row) > 0:
                pre_resids[d].append(row["resid_global"].values[0])

    # Get residual SD from estimation sample for z-scores
    resid_sd = df_global["resid_global"].std()

    print("PRE-TREATMENT (before release):")
    pre_avgs = []
    for d in sorted(pre_days):
        if pre_resids[d]:
            avg = np.mean(pre_resids[d])
            z = avg / (resid_sd / np.sqrt(len(pre_resids[d])))
            pre_avgs.append(avg)
            print(f"  Day {d:>2}:  {avg:+.1f} deaths (z = {z:+.2f}, n={len(pre_resids[d])})")

    avg_pretrend = np.mean(pre_avgs) if pre_avgs else 0
    avg_day0 = np.mean(day0_resids) if day0_resids else 0

    print(f"\n  Avg pre-trend (days -5 to -1): {avg_pretrend:+.1f} deaths")
    print(f"\nTREATMENT:")
    print(f"  Day  0:  {avg_day0:+.1f} deaths (actual release day)")

    # Interpretation
    print("\nINTERPRETATION:")
    if avg_pretrend > 0 and avg_day0 > 0:
        pretrend_pct = 100 * avg_pretrend / avg_day0
        print(f"  Pre-trend is {pretrend_pct:.1f}% of day-0 effect.")
        if pretrend_pct > 30:
            print("  WARNING: Pre-trend > 30% of treatment effect!")
            print("  This suggests confounding, not causal streaming effect.")
        elif pretrend_pct > 15:
            print("  CAUTION: Pre-trend is substantial (15-30% of effect).")
        else:
            print("  Pre-trend appears small relative to treatment effect.")
    elif avg_pretrend <= 0:
        print("  No positive pre-trend detected (good sign for causal identification).")

    return {
        "pre_resids": pre_resids,
        "pre_avgs": pre_avgs,
        "avg_pretrend": avg_pretrend,
        "avg_day0": avg_day0,
        "resid_sd": resid_sd,
    }


def window_sensitivity(df, windows=None):
    """
    Test effect stability across window sizes.

    A real effect should be stable (within SE) across different windows.
    A spurious effect tends to shrink with larger windows as more
    averaging dilutes chance spikes.

    Returns DataFrame with results for each window size.
    """
    if windows is None:
        windows = [5, 7, 10, 15, 20, 30]

    print(f"\n{'='*70}")
    print("WINDOW SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print("Testing effect stability across different control window sizes.")
    print("Real effect: stable across windows (within SE).")
    print("Spurious effect: shrinks dramatically with larger windows.\n")

    # Add required columns if not present
    df = df.copy()
    if "dow" not in df.columns:
        df["dow"] = df["date"].dt.dayofweek
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "holiday" not in df.columns:
        holidays = us_holidays(df["year"].unique())
        df["holiday"] = df["date"].dt.date.isin(holidays).astype(int)
        hol_adj = set()
        for h in holidays:
            hol_adj.add(h - datetime.timedelta(1))
            hol_adj.add(h + datetime.timedelta(1))
        df["holiday_adj"] = df["date"].dt.date.isin(hol_adj).astype(int)

    results = []

    for w in windows:
        # Local estimate
        local_deltas = []
        for _, _, date_str, _ in ALBUMS:
            dt = pd.to_datetime(date_str)
            release_row = df[df["date"] == dt]
            if len(release_row) == 0:
                continue

            y_release = release_row["fatalities"].values[0]

            mask = (
                (df["date"] >= dt - pd.Timedelta(days=w))
                & (df["date"] <= dt + pd.Timedelta(days=w))
                & (df["date"] != dt)
            )
            control = df[mask]
            y_control = control["fatalities"].mean()
            local_deltas.append(y_release - y_control)

        avg_local = np.mean(local_deltas)
        se_local = np.std(local_deltas) / np.sqrt(len(local_deltas))

        # Global estimate (donut)
        exclude = set()
        for _, _, date_str, _ in ALBUMS:
            dt = pd.to_datetime(date_str).date()
            for offset in range(-w, w + 1):
                exclude.add(dt + datetime.timedelta(days=offset))

        est_mask = ~df["date"].dt.date.isin(exclude)
        X_est = _build_design(df[est_mask])
        y_est = df.loc[est_mask, "fatalities"].values.astype(float)
        beta, _, _ = _ols_residuals(X_est.values, y_est)

        X_all = _build_design(df)
        fitted_all = X_all.values @ beta

        global_deltas = []
        for _, _, date_str, _ in ALBUMS:
            dt = pd.to_datetime(date_str)
            release_row = df[df["date"] == dt]
            if len(release_row) == 0:
                continue
            idx = release_row.index[0]
            y_release = release_row["fatalities"].values[0]
            y_hat = fitted_all[df.index.get_loc(idx)]
            global_deltas.append(y_release - y_hat)

        avg_global = np.mean(global_deltas)
        se_global = np.std(global_deltas) / np.sqrt(len(global_deltas))
        t_stat = avg_global / se_global if se_global > 0 else 0

        results.append({
            "window": w,
            "local_delta": avg_local,
            "global_delta": avg_global,
            "se": se_global,
            "t_stat": t_stat,
        })

    results_df = pd.DataFrame(results)

    print(f"{'Window':<8} | {'Local δ':>9} | {'Global δ':>10} | {'SE':>7} | {'t-stat':>7}")
    print("-" * 50)
    for _, r in results_df.iterrows():
        marker = " (current)" if r["window"] == 10 else ""
        print(
            f"±{int(r['window']):<6} | {r['local_delta']:>+9.1f} | "
            f"{r['global_delta']:>+10.1f} | {r['se']:>7.1f} | {r['t_stat']:>7.2f}{marker}"
        )

    # Check for shrinkage
    baseline_idx = results_df[results_df["window"] == 10].index
    if len(baseline_idx) > 0:
        baseline = results_df.loc[baseline_idx[0], "global_delta"]
        large_w = results_df[results_df["window"] == 30]
        if len(large_w) > 0:
            large_effect = large_w.iloc[0]["global_delta"]
            shrinkage = 100 * (1 - large_effect / baseline) if baseline != 0 else 0

            print(f"\nEFFECT SHRINKAGE:")
            print(f"  Window=10 δ: {baseline:+.1f}")
            print(f"  Window=30 δ: {large_effect:+.1f}")
            print(f"  Shrinkage: {shrinkage:.1f}%")

            if shrinkage > 50:
                print("  WARNING: >50% shrinkage suggests effect is window-dependent!")
            elif shrinkage > 25:
                print("  CAUTION: Moderate shrinkage (25-50%) with larger windows.")
            else:
                print("  Effect appears stable across window sizes.")

    return results_df


def year_permutation_placebo(df_global, n_perms=1000, seed=42):
    """
    Test if effect is specific to release years.

    Keep (month, day) of each album but assign to different years.
    If wrong-year dates show similar effects, the finding is likely
    a calendar artifact, not a streaming effect.

    Returns dict with permutation distribution and p-value.
    """
    print(f"\n{'='*70}")
    print(f"YEAR PERMUTATION PLACEBO ({n_perms:,} permutations)")
    print(f"{'='*70}")
    print("Testing: Would the same calendar dates show effects in other years?")
    print("Keep (month, day) of each album, assign to random years.\n")

    rng = np.random.RandomState(seed)

    # Get actual effect
    release_mask = df_global["date"].dt.date.isin(RELEASE_DATES)
    actual_avg = df_global.loc[release_mask, "resid_global"].mean()

    # Get available years
    available_years = sorted(df_global["date"].dt.year.unique())

    # Extract month/day from each album
    album_md = []
    for _, _, date_str, dow in ALBUMS:
        dt = pd.to_datetime(date_str)
        album_md.append({
            "month": dt.month,
            "day": dt.day,
            "dow_original": dt.dayofweek,
            "dow_name": dow,
        })

    permuted_avgs = np.zeros(n_perms)

    for p in range(n_perms):
        permuted_dates = []
        for album in album_md:
            # Find a matching date in a random year
            perm_year = rng.choice(available_years)
            try:
                perm_date = datetime.date(perm_year, album["month"], album["day"])
                permuted_dates.append(perm_date)
            except ValueError:
                pass

        # Get average residual for permuted dates
        permuted_mask = df_global["date"].dt.date.isin(set(permuted_dates))
        if permuted_mask.sum() > 0:
            permuted_avgs[p] = df_global.loc[permuted_mask, "resid_global"].mean()
        else:
            permuted_avgs[p] = 0

    p_value = (permuted_avgs >= actual_avg).mean()

    print(f"Actual release dates avg residual: {actual_avg:+.1f}")
    print(f"\nWrong-year permutation distribution:")
    print(f"  Mean: {permuted_avgs.mean():+.1f}")
    print(f"  SD: {permuted_avgs.std():.1f}")
    print(f"  5th percentile: {np.percentile(permuted_avgs, 5):+.1f}")
    print(f"  95th percentile: {np.percentile(permuted_avgs, 95):+.1f}")
    print(f"\np-value (actual vs permuted): {p_value:.4f}")

    # Interpretation
    print("\nINTERPRETATION:")
    if permuted_avgs.mean() > 0.3 * actual_avg:
        print("  WARNING: Wrong-year dates show substantial positive residuals!")
        print("  This suggests the effect may be a calendar artifact.")
    elif p_value > 0.05:
        print("  Effect is not significantly larger than wrong-year dates.")
        print("  Cannot rule out calendar/seasonal artifacts.")
    else:
        print("  Effect is significantly larger than wrong-year dates.")
        print("  This supports (but doesn't prove) year-specific causation.")

    return {
        "actual_avg": actual_avg,
        "permuted_avgs": permuted_avgs,
        "p_value": p_value,
        "permuted_mean": permuted_avgs.mean(),
        "permuted_sd": permuted_avgs.std(),
    }


def run_all_placebos(df, df_global, window=10):
    """
    Run all three placebo tests and provide summary.

    Returns dict with all placebo results.
    """
    print("\n" + "=" * 70)
    print("RUNNING ALL PLACEBO TESTS")
    print("=" * 70)

    # 1. Pre-trends
    pretrend_results = pretrends_analysis(df_global)

    # 2. Window sensitivity
    window_results = window_sensitivity(df)

    # 3. Year permutation
    year_perm_results = year_permutation_placebo(df_global)

    # Summary
    print("\n" + "=" * 70)
    print("PLACEBO TEST SUMMARY")
    print("=" * 70)

    concerns = []

    # Pre-trends check
    if pretrend_results["avg_day0"] > 0:
        pretrend_pct = 100 * pretrend_results["avg_pretrend"] / pretrend_results["avg_day0"]
        if pretrend_pct > 30:
            concerns.append(f"Pre-trend is {pretrend_pct:.0f}% of treatment effect")

    # Window sensitivity check
    baseline = window_results[window_results["window"] == 10]["global_delta"].values
    large = window_results[window_results["window"] == 30]["global_delta"].values
    if len(baseline) > 0 and len(large) > 0 and baseline[0] != 0:
        shrinkage = 100 * (1 - large[0] / baseline[0])
        if shrinkage > 50:
            concerns.append(f"Effect shrinks {shrinkage:.0f}% at window=30")

    # Year permutation check
    if year_perm_results["p_value"] > 0.05:
        concerns.append(f"Year permutation p={year_perm_results['p_value']:.3f}")

    if concerns:
        print("\nCONCERNS IDENTIFIED:")
        for c in concerns:
            print(f"  - {c}")
        print("\nConclusion: Evidence for causation is WEAK.")
    else:
        print("\nNo major concerns from placebo tests.")
        print("Conclusion: Placebo tests support causal interpretation.")

    return {
        "pretrends": pretrend_results,
        "window_sensitivity": window_results,
        "year_permutation": year_perm_results,
        "concerns": concerns,
    }
