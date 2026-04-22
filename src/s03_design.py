"""
Design / Identification Checks — run before trusting estimates.

Functions to verify identifying assumptions:
- Covariate balance between treatment and control
- Holiday baseline confounding
- Pre-trends analysis
- Parallel trends formal test
"""

import datetime
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import ALBUMS, ALBUMS_TIER1
from src.utils import add_time_features, build_design_matrix, ols_fit


def covariate_balance_check(df_global, window=10):
    """
    Covariate Balance Check (Hainmueller).

    Compare release days vs control days on observable characteristics.
    If imbalanced, the treatment effect estimate may be confounded.

    Output: tabs/t24_balance_check.csv
    """
    print(f"\n{'='*70}")
    print("COVARIATE BALANCE CHECK (Hainmueller)")
    print(f"{'='*70}")
    print("Are release days balanced on observables vs control days?")
    print("If not, simple comparison may be confounded.\n")

    release_dates = set()
    for a in ALBUMS_TIER1:
        release_dates.add(pd.to_datetime(a[2]).date())

    control_window_dates = set()
    for a in ALBUMS_TIER1:
        dt = pd.to_datetime(a[2]).date()
        for offset in range(-window, window + 1):
            if offset != 0:
                control_window_dates.add(dt + datetime.timedelta(days=offset))

    df = df_global.copy()
    df["is_release"] = df["date"].dt.date.isin(release_dates).astype(int)
    df["is_control_window"] = df["date"].dt.date.isin(control_window_dates).astype(int)

    release_df = df[df["is_release"] == 1]
    control_df = df[df["is_control_window"] == 1]

    covariates = [
        ("dow", "Day of week (0=Mon)"),
        ("month", "Month (1-12)"),
        ("holiday", "Holiday (0/1)"),
        ("holiday_adj", "Holiday adjacent (0/1)"),
    ]

    if "pct_bad_weather" in df.columns:
        covariates.append(("pct_bad_weather", "% bad weather crashes"))
    if "pct_dark" in df.columns:
        covariates.append(("pct_dark", "% dark conditions"))
    if "pct_alcohol" in df.columns:
        covariates.append(("pct_alcohol", "% alcohol-involved"))

    covariates.append(("fitted_global", "Expected fatalities (model)"))

    results = []

    print(
        f"{'Covariate':<30} | {'Release':>10} | {'Control':>10} | {'SMD':>8} | {'p-value':>10}"
    )
    print("-" * 80)

    for col, label in covariates:
        if col not in df.columns:
            continue

        rel_mean = release_df[col].mean()
        ctrl_mean = control_df[col].mean()
        pooled_std = np.sqrt((release_df[col].var() + control_df[col].var()) / 2)
        smd = (rel_mean - ctrl_mean) / pooled_std if pooled_std > 0 else 0

        t_stat, p_value = stats.ttest_ind(
            release_df[col].dropna(), control_df[col].dropna()
        )

        results.append(
            {
                "covariate": col,
                "description": label,
                "release_mean": rel_mean,
                "control_mean": ctrl_mean,
                "smd": smd,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )

        sig = " *" if p_value < 0.1 else " **" if p_value < 0.05 else ""
        print(
            f"{label:<30} | {rel_mean:>10.3f} | {ctrl_mean:>10.3f} | {smd:>+8.3f} | {p_value:>10.3f}{sig}"
        )

    results_df = pd.DataFrame(results)

    print("\nINTERPRETATION (SMD thresholds: |0.1| small, |0.25| medium, |0.4| large):")
    large_imbalance = results_df[np.abs(results_df["smd"]) > 0.25]
    if len(large_imbalance) > 0:
        print("  WARNING: Medium/large imbalances detected:")
        for _, r in large_imbalance.iterrows():
            print(f"    - {r['description']}: SMD = {r['smd']:+.3f}")
        print("  Consider propensity weighting or matching.")
    else:
        print("  No large imbalances detected. Good covariate balance.")

    return results_df


def holiday_baseline_check(accidents, window=10):
    """
    Check if release dates landed on systematically high-fatality dates.

    Uses historical baseline fatality rates for each calendar date.
    Certified Lover Boy (2021-09-03) is Labor Day weekend — +10% fatalities.

    Returns DataFrame with baseline check for each release date.
    """
    from src.constants import us_holidays

    print(f"\n{'='*70}")
    print("HOLIDAY / BASELINE CHECK")
    print(f"{'='*70}")
    print("Checking if release dates coincide with high-fatality periods.")
    print("Key: Certified Lover Boy = Labor Day weekend (+10% fatalities).\n")

    cols = {c.upper(): c for c in accidents.columns}

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

    daily = accidents.groupby("_date")["_fatals"].sum().reset_index()
    daily.columns = ["date", "fatalities"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")

    daily["month"] = daily.index.month
    daily["day"] = daily.index.day
    daily["dow"] = daily.index.dayofweek
    daily["year"] = daily.index.year

    years = daily["year"].unique()
    all_holidays = us_holidays(years)

    major_holiday_weekends = []
    for y in years:
        for h in all_holidays:
            if h.year == y:
                for offset in range(-3, 4):
                    major_holiday_weekends.append(h + datetime.timedelta(days=offset))
    major_holiday_weekends = set(major_holiday_weekends)

    def get_labor_day(year):
        d = datetime.date(year, 9, 1)
        while d.weekday() != 0:
            d += datetime.timedelta(1)
        return d

    def get_memorial_day(year):
        d = datetime.date(year, 5, 31)
        while d.weekday() != 0:
            d -= datetime.timedelta(1)
        return d

    labor_day_weekends = set()
    memorial_day_weekends = set()
    july4_weekends = set()
    thanksgiving_weekends = set()

    for y in years:
        ld = get_labor_day(y)
        for offset in range(-3, 2):
            labor_day_weekends.add(ld + datetime.timedelta(days=offset))

        md = get_memorial_day(y)
        for offset in range(-3, 2):
            memorial_day_weekends.add(md + datetime.timedelta(days=offset))

        j4 = datetime.date(y, 7, 4)
        for offset in range(-3, 4):
            july4_weekends.add(j4 + datetime.timedelta(days=offset))

        d = datetime.date(y, 11, 1)
        while d.weekday() != 3:
            d += datetime.timedelta(1)
        tg = d + datetime.timedelta(21)
        for offset in range(-1, 4):
            thanksgiving_weekends.add(tg + datetime.timedelta(days=offset))

    overall_mean = daily["fatalities"].mean()

    friday_only = daily[daily["dow"] == 4]
    friday_mean = friday_only["fatalities"].mean()

    results = []

    for artist, album, date_str, dow, *_ in ALBUMS_TIER1:
        dt = pd.to_datetime(date_str).date()

        same_md = daily[(daily["month"] == dt.month) & (daily["day"] == dt.day)]
        baseline = same_md["fatalities"].mean() if len(same_md) > 0 else np.nan

        is_labor = dt in labor_day_weekends
        is_memorial = dt in memorial_day_weekends
        is_july4 = dt in july4_weekends
        is_thanksgiving = dt in thanksgiving_weekends
        is_holiday_weekend = is_labor or is_memorial or is_july4 or is_thanksgiving

        notes = []
        if is_labor:
            notes.append("Labor Day weekend")
        if is_memorial:
            notes.append("Memorial Day weekend")
        if is_july4:
            notes.append("July 4th weekend")
        if is_thanksgiving:
            notes.append("Thanksgiving weekend")

        pct_above_avg = (
            100 * (baseline - overall_mean) / overall_mean
            if baseline and overall_mean
            else 0
        )
        pct_above_fri = (
            100 * (baseline - friday_mean) / friday_mean
            if baseline and friday_mean
            else 0
        )

        results.append(
            {
                "artist": artist,
                "album": album,
                "date": date_str,
                "baseline_fatalities": baseline,
                "overall_mean": overall_mean,
                "friday_mean": friday_mean,
                "pct_above_avg": pct_above_avg,
                "pct_above_friday": pct_above_fri,
                "is_holiday_weekend": is_holiday_weekend,
                "notes": "; ".join(notes) if notes else "",
            }
        )

    results_df = pd.DataFrame(results)

    print(
        f"{'Album':<35} | {'Date':>12} | {'Baseline':>10} | {'% vs Avg':>10} | {'Holiday?':>10}"
    )
    print("-" * 90)
    for _, r in results_df.iterrows():
        holiday_marker = "YES" if r["is_holiday_weekend"] else ""
        print(
            f"{r['artist'][:15] + ' - ' + r['album'][:17]:<35} | "
            f"{r['date']:>12} | {r['baseline_fatalities']:>10.1f} | "
            f"{r['pct_above_avg']:>+9.1f}% | {holiday_marker:>10}"
        )

    print(f"\nBASELINE STATISTICS:")
    print(f"  Overall daily mean: {overall_mean:.1f} fatalities")
    print(f"  Friday mean: {friday_mean:.1f} fatalities")
    print(
        f"  Release date avg baseline: {results_df['baseline_fatalities'].mean():.1f} fatalities"
    )

    n_holiday = results_df["is_holiday_weekend"].sum()
    print(f"\nHOLIDAY WEEKEND RELEASES: {n_holiday} / {len(results_df)}")

    if n_holiday > 0:
        holiday_rows = results_df[results_df["is_holiday_weekend"]]
        for _, r in holiday_rows.iterrows():
            print(f"  {r['album']}: {r['notes']} ({r['date']})")

        print("\n  WARNING: Holiday weekends have elevated fatality baselines!")
        print("  This confounds the treatment effect estimate.")

    avg_pct_above = results_df["pct_above_avg"].mean()
    if avg_pct_above > 5:
        print(
            f"\nCAUTION: Release dates average {avg_pct_above:.1f}% above overall mean."
        )
        print(
            "  This suggests selection of high-fatality dates (intentionally or not)."
        )

    return results_df


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

        row_0 = df_global[df_global["date"] == dt]
        if len(row_0) > 0:
            day0_resids.append(row_0["resid_global"].values[0])

        for d in pre_days:
            pre_dt = dt + pd.Timedelta(days=d)
            row = df_global[df_global["date"] == pre_dt]
            if len(row) > 0:
                pre_resids[d].append(row["resid_global"].values[0])

    resid_sd = df_global["resid_global"].std()

    print("PRE-TREATMENT (before release):")
    pre_avgs = []
    for d in sorted(pre_days):
        if pre_resids[d]:
            avg = np.mean(pre_resids[d])
            z = avg / (resid_sd / np.sqrt(len(pre_resids[d])))
            pre_avgs.append(avg)
            print(
                f"  Day {d:>2}:  {avg:+.1f} deaths (z = {z:+.2f}, n={len(pre_resids[d])})"
            )

    avg_pretrend = np.mean(pre_avgs) if pre_avgs else 0
    avg_day0 = np.mean(day0_resids) if day0_resids else 0

    print(f"\n  Avg pre-trend (days -5 to -1): {avg_pretrend:+.1f} deaths")
    print(f"\nTREATMENT:")
    print(f"  Day  0:  {avg_day0:+.1f} deaths (actual release day)")

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


def parallel_trends_test(df_global, window=10):
    """
    Formal Test of Parallel Trends (Hainmueller).

    Pre-trends should be zero if the effect is causal. Test whether
    pre-release days show systematic patterns.

    Output: tabs/t32_parallel_trends.csv
    """
    print(f"\n{'='*70}")
    print("PARALLEL TRENDS TEST (Hainmueller)")
    print(f"{'='*70}")
    print("Testing whether pre-release trends are parallel to control trends.\n")

    results = []

    for day_offset in range(-window, window + 1):
        day_resids = []

        for album_tuple in ALBUMS_TIER1:
            dt = pd.to_datetime(album_tuple[2])
            target_dt = dt + pd.Timedelta(days=day_offset)

            row = df_global[df_global["date"] == target_dt]
            if len(row) > 0:
                day_resids.append(row["resid_global"].values[0])

        if day_resids:
            effect = np.mean(day_resids)
            se = np.std(day_resids) / sqrt(len(day_resids))
            t_stat = effect / se if se > 0 else 0

            results.append(
                {
                    "day": day_offset,
                    "effect": effect,
                    "se": se,
                    "t_stat": t_stat,
                    "p_value": 2 * (1 - stats.norm.cdf(abs(t_stat))),
                    "n": len(day_resids),
                }
            )

    results_df = pd.DataFrame(results)

    pre_df = results_df[results_df["day"] < 0]

    if len(pre_df) > 0:
        pre_effects = pre_df["effect"].values
        pre_days = pre_df["day"].values

        slope, _, _, p_value, _ = stats.linregress(pre_days, pre_effects)

        joint_f_stat = (pre_df["effect"] ** 2 / pre_df["se"] ** 2).sum() / len(pre_df)
        joint_p_value = 1 - stats.f.cdf(joint_f_stat, len(pre_df), 1000)

        print("PRE-TREND ANALYSIS:")
        print(f"  Pre-release days (t < 0):")
        print(f"    Mean effect: {pre_df['effect'].mean():+.1f}")
        print(f"    Trend slope: {slope:+.3f} deaths/day")
        print(f"    Slope p-value: {p_value:.4f}")
        print(f"    Joint F-test: F = {joint_f_stat:.2f}, p = {joint_p_value:.4f}")

        if p_value < 0.1 or joint_p_value < 0.1:
            print("\n  WARNING: Significant pre-trends detected!")
            print("  Parallel trends assumption may be violated.")
        else:
            print("\n  No significant pre-trends. Parallel trends assumption holds.")

        results_df["pre_trend_slope"] = slope
        results_df["pre_trend_p"] = p_value
        results_df["joint_f_stat"] = joint_f_stat
        results_df["joint_p_value"] = joint_p_value

    return results_df
