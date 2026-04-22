"""
Falsification / Placebo Tests — tests that should show null effects.

Functions for placebo and falsification tests:
- Year permutation placebo (same dates, different years)
- S&P 500 placebo (unrelated outcome)
- Placebo outcomes (shouldn't be affected by streaming)
- Structural FARS placebos (structural variables)
- Best-Fridays false positive rate
"""

import datetime
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (ALBUMS, ALBUMS_EXTENDED, ALBUMS_TIER0, ALBUMS_TIER1,
                           ALBUMS_TIER2, ALBUMS_TIER3, RELEASE_DATES)
from src.utils import add_time_features, build_design_matrix, ols_fit


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

    release_mask = df_global["date"].dt.date.isin(RELEASE_DATES)
    actual_avg = df_global.loc[release_mask, "resid_global"].mean()

    available_years = sorted(df_global["date"].dt.year.unique())

    album_md = []
    for _, _, date_str, dow in ALBUMS:
        dt = pd.to_datetime(date_str)
        album_md.append(
            {
                "month": dt.month,
                "day": dt.day,
                "dow_original": dt.dayofweek,
                "dow_name": dow,
            }
        )

    permuted_avgs = np.zeros(n_perms)

    for p in range(n_perms):
        permuted_dates = []
        for album in album_md:
            perm_year = rng.choice(available_years)
            try:
                perm_date = datetime.date(perm_year, album["month"], album["day"])
                permuted_dates.append(perm_date)
            except ValueError:
                pass

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


def sp500_placebo(window=10):
    """
    Test effect on S&P 500 returns — an unrelated placebo.

    If album releases "cause" stock returns, the methodology is picking up noise
    from the small N of events, not a real streaming-driving effect.

    Returns DataFrame with per-album results.
    """
    print(f"\n{'='*70}")
    print("S&P 500 PLACEBO TEST (Absurd Placebo)")
    print(f"{'='*70}")
    print("Testing effect on stock market returns — NOTHING to do with driving.")
    print("If we find a 'significant' effect, methodology is picking up noise.\n")

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance package not installed.")
        print("Install with: pip install yfinance")
        return None

    print("Downloading S&P 500 data...")
    sp500 = yf.download("^GSPC", start="2017-01-01", end="2024-12-31", progress=False)

    if len(sp500) == 0:
        print("ERROR: Could not download S&P 500 data.")
        return None

    sp500 = sp500.reset_index()
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)

    sp500["return"] = sp500["Close"].pct_change() * 100
    sp500["Date"] = pd.to_datetime(sp500["Date"])

    results = []

    for artist, album, date_str, dow, *_ in ALBUMS_TIER1:
        dt = pd.to_datetime(date_str)

        release_row = sp500[sp500["Date"] == dt]
        if len(release_row) == 0:
            next_day = dt + pd.Timedelta(days=1)
            release_row = sp500[sp500["Date"] == next_day]

        if len(release_row) == 0:
            continue

        sp500_return = release_row["return"].values[0]

        control_mask = (
            (sp500["Date"] >= dt - pd.Timedelta(days=window))
            & (sp500["Date"] <= dt + pd.Timedelta(days=window))
            & (sp500["Date"] != dt)
        )
        control_returns = sp500[control_mask]["return"]
        control_mean = control_returns.mean()
        control_std = control_returns.std()

        effect = sp500_return - control_mean

        results.append(
            {
                "artist": artist,
                "album": album,
                "date": date_str,
                "sp500_return": sp500_return,
                "control_return": control_mean,
                "effect": effect,
                "control_std": control_std,
            }
        )

    if not results:
        print("No S&P 500 data found for release dates.")
        return None

    results_df = pd.DataFrame(results)

    print(f"{'Album':<35} | {'SP500 %':>10} | {'Control %':>10} | {'Effect':>10}")
    print("-" * 75)
    for _, r in results_df.iterrows():
        print(
            f"{r['artist'][:15] + ' - ' + r['album'][:17]:<35} | "
            f"{r['sp500_return']:>+10.2f} | {r['control_return']:>+10.2f} | {r['effect']:>+10.2f}"
        )

    avg_effect = results_df["effect"].mean()
    se_effect = results_df["effect"].std() / sqrt(len(results_df))
    t_stat = avg_effect / se_effect if se_effect > 0 else 0

    print(f"\nPOOLED RESULTS:")
    print(f"  Average effect: {avg_effect:+.3f}%")
    print(f"  SE: {se_effect:.3f}%")
    print(f"  t-stat: {t_stat:.2f}")

    print("\nINTERPRETATION:")
    if abs(t_stat) > 2:
        print(f"  NOTE: t-stat = {t_stat:.2f} is 'significant' (|t| > 2).")
        print("  Album releases appear to 'cause' stock market movements.")
        print("  This is likely spurious — suggests methodology may be sensitive to noise.")
    else:
        print(f"  t-stat = {t_stat:.2f} is not significant.")
        print("  Good: No spurious 'effect' on unrelated outcome.")

    return results_df


def sp500_placebo_expanded(window=10):
    """
    S&P 500 placebo for all tiers separately and combined.

    Shows how t-stat changes with sample size. If adding more albums
    increases the spurious t-stat, there's a systematic confound in the
    methodology.

    Returns DataFrame with tier, n_albums, avg_effect, se, t_stat.
    """
    print(f"\n{'='*70}")
    print("EXPANDED S&P 500 PLACEBO (All Tiers)")
    print(f"{'='*70}")
    print("Testing spurious 'effect' on stock returns by tier.")
    print("With N=10, even random noise can produce t≈1.5.\n")

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance package not installed.")
        return None

    print("Downloading S&P 500 data...")
    sp500 = yf.download("^GSPC", start="2017-01-01", end="2024-12-31", progress=False)

    if len(sp500) == 0:
        print("ERROR: Could not download S&P 500 data.")
        return None

    sp500 = sp500.reset_index()
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)

    sp500["return"] = sp500["Close"].pct_change() * 100
    sp500["Date"] = pd.to_datetime(sp500["Date"])

    def compute_sp500_effect(albums_list, tier_name):
        """Compute S&P 500 effect for a list of albums."""
        effects = []
        for album_tuple in albums_list:
            artist, album, date_str = album_tuple[0], album_tuple[1], album_tuple[2]
            dt = pd.to_datetime(date_str)

            release_row = sp500[sp500["Date"] == dt]
            if len(release_row) == 0:
                next_day = dt + pd.Timedelta(days=1)
                release_row = sp500[sp500["Date"] == next_day]

            if len(release_row) == 0:
                continue

            sp500_return = release_row["return"].values[0]

            control_mask = (
                (sp500["Date"] >= dt - pd.Timedelta(days=window))
                & (sp500["Date"] <= dt + pd.Timedelta(days=window))
                & (sp500["Date"] != dt)
            )
            control_returns = sp500[control_mask]["return"]
            control_mean = control_returns.mean()

            effects.append(sp500_return - control_mean)

        if not effects:
            return None

        avg_effect = np.mean(effects)
        se = np.std(effects) / sqrt(len(effects))
        t_stat = avg_effect / se if se > 0 else 0

        return {
            "tier": tier_name,
            "n_albums": len(effects),
            "avg_effect": avg_effect,
            "se": se,
            "t_stat": t_stat,
        }

    results = []

    tier0_result = compute_sp500_effect(ALBUMS_TIER0, "Tier 0")
    if tier0_result:
        results.append(tier0_result)

    tier1_result = compute_sp500_effect(ALBUMS_TIER1, "Tier 1")
    if tier1_result:
        results.append(tier1_result)

    tier2_result = compute_sp500_effect(ALBUMS_TIER2, "Tier 2")
    if tier2_result:
        results.append(tier2_result)

    tier3_result = compute_sp500_effect(ALBUMS_TIER3, "Tier 3")
    if tier3_result:
        results.append(tier3_result)

    all_result = compute_sp500_effect(ALBUMS_EXTENDED, f"All {len(ALBUMS_EXTENDED)}")
    if all_result:
        results.append(all_result)

    if not results:
        print("No results computed.")
        return None

    results_df = pd.DataFrame(results)

    print(f"{'Tier':<10} | {'N':>5} | {'Avg Effect':>12} | {'SE':>10} | {'t-stat':>8}")
    print("-" * 55)
    for _, r in results_df.iterrows():
        sig = " **" if abs(r["t_stat"]) > 2 else " *" if abs(r["t_stat"]) > 1.65 else ""
        print(
            f"{r['tier']:<10} | {r['n_albums']:>5} | {r['avg_effect']:>+12.3f}% | "
            f"{r['se']:>10.3f}% | {r['t_stat']:>+7.2f}{sig}"
        )

    print("\nINTERPRETATION:")
    if len(results_df) > 1:
        t1_row = results_df[results_df["tier"] == "Tier 1"]
        all_row = results_df[results_df["tier"].str.startswith("All")]

        if len(t1_row) > 0 and len(all_row) > 0:
            t1_t = t1_row["t_stat"].values[0]
            all_t = all_row["t_stat"].values[0]
            all_n = all_row["n_albums"].values[0]

            if abs(all_t) > abs(t1_t):
                print(
                    f"  WARNING: t-stat INCREASES with more albums ({t1_t:.2f} → {all_t:.2f})"
                )
                print("  This suggests systematic confound, not random noise.")
            else:
                print(f"  t-stat decreases with more albums ({t1_t:.2f} → {all_t:.2f})")
                print("  Consistent with small-N noise in Tier 1.")

            if abs(all_t) > 2:
                print(
                    f"  NOTE: All {all_n} albums show 'significant' S&P effect (t={all_t:.2f})."
                )
                print(
                    "  The placebo is significant — methodology may be sensitive to noise."
                )

    return results_df


def best_fridays_false_positive_rate(df_global, n_sims=10000, n_pick=10):
    """
    Simulation: How often does cherry-picking 10 'best' Fridays produce
    an effect >= the actual album release effect?

    This tests researcher degrees of freedom. If we can cherry-pick ANY
    10 Fridays from 2017-2024 and frequently find large effects, the
    methodology is vulnerable to selection bias.

    Returns dict with FPR and simulation details.
    """
    print(f"\n{'='*70}")
    print("BEST-10-FRIDAYS FALSE POSITIVE RATE SIMULATION")
    print(f"{'='*70}")
    print(f"Testing: If we cherry-pick the 'best' {n_pick} Fridays from a sample,")
    print("how often do we find an effect >= the actual album effect?\n")

    all_fridays = df_global[df_global["date"].dt.dayofweek == 4].copy()
    print(f"Total Fridays in data: {len(all_fridays)}")

    tier1_resids = []
    for album_tuple in ALBUMS_TIER1:
        dt = pd.to_datetime(album_tuple[2])
        row = df_global[df_global["date"] == dt]
        if len(row) > 0:
            tier1_resids.append(row["resid_global"].values[0])

    actual_effect = np.mean(tier1_resids) if tier1_resids else 0
    print(f"Actual Tier 1 effect (mean residual): {actual_effect:+.1f} deaths")
    print(f"\nRunning {n_sims:,} simulations...")

    exceeds_count = 0
    effects_distribution = []

    np.random.seed(42)
    n_sample = min(100, len(all_fridays))

    for _ in range(n_sims):
        sample_idx = np.random.choice(len(all_fridays), size=n_sample, replace=True)
        sample = all_fridays.iloc[sample_idx]

        top_n = sample.nlargest(n_pick, "resid_global")
        effect = top_n["resid_global"].mean()
        effects_distribution.append(effect)

        if effect >= actual_effect:
            exceeds_count += 1

    fpr = exceeds_count / n_sims
    effects_distribution = np.array(effects_distribution)

    print(f"\nRESULTS:")
    print(f"  Simulated effect distribution:")
    print(f"    Mean: {effects_distribution.mean():+.1f} deaths")
    print(f"    Median: {np.median(effects_distribution):+.1f} deaths")
    print(f"    5th percentile: {np.percentile(effects_distribution, 5):+.1f} deaths")
    print(f"    95th percentile: {np.percentile(effects_distribution, 95):+.1f} deaths")
    print(f"\n  False Positive Rate: {fpr:.1%}")
    print(
        f"  (Times cherry-picked effect >= {actual_effect:.1f}: {exceeds_count:,} / {n_sims:,})"
    )

    print("\nINTERPRETATION:")
    if fpr > 0.05:
        print(f"  WARNING: FPR = {fpr:.1%} > 5%")
        print(
            "  Cherry-picking ANY 10 Fridays can easily produce 'significant' effects."
        )
        print("  The methodology is vulnerable to researcher degrees of freedom.")
    else:
        print(f"  FPR = {fpr:.1%} is within acceptable range.")
        print("  Finding an effect of {actual_effect:.1f} by chance is unlikely.")

    return {
        "actual_effect": actual_effect,
        "fpr": fpr,
        "n_sims": n_sims,
        "n_pick": n_pick,
        "n_fridays": len(all_fridays),
        "mean_simulated": effects_distribution.mean(),
        "p95_simulated": np.percentile(effects_distribution, 95),
    }


def placebo_outcomes(accidents, window=10):
    """
    Placebo Outcomes Test (Green).

    Test on outcomes that SHOULDN'T be affected by album releases:
    - Weather-related crashes only (rain/snow/fog)
    - Work zone crashes
    - School bus involved crashes

    If we find effects on these, methodology is suspect.

    Output: tabs/t28_placebo_outcomes.csv
    """
    print(f"\n{'='*70}")
    print("PLACEBO OUTCOMES TEST (Green)")
    print(f"{'='*70}")
    print("Testing on outcomes that shouldn't be affected by streaming.")
    print("If we find effects here, methodology is picking up noise.\n")

    df = accidents.copy()
    cols = {c.upper(): c for c in df.columns}

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

    df = df.dropna(subset=["_year", "_month", "_day"])

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

    placebo_outcomes_list = []

    if "WEATHER" in cols:
        weather_col = cols["WEATHER"]
        df["_weather_related"] = df[weather_col].isin([2, 3, 4, 5, 11, 12]).astype(int)
        placebo_outcomes_list.append(
            ("Weather-related only", df[df["_weather_related"] == 1])
        )

    if "WRK_ZONE" in cols:
        wrk_col = cols["WRK_ZONE"]
        df["_work_zone"] = (df[wrk_col] >= 1).astype(int)
        placebo_outcomes_list.append(("Work zone crashes", df[df["_work_zone"] == 1]))

    if "SCH_BUS" in cols:
        bus_col = cols["SCH_BUS"]
        df["_school_bus"] = (df[bus_col] >= 1).astype(int)
        placebo_outcomes_list.append(
            ("School bus involved", df[df["_school_bus"] == 1])
        )

    if "NHS" in cols:
        nhs_col = cols["NHS"]
        df["_nhs"] = (df[nhs_col] == 1).astype(int)
        placebo_outcomes_list.append(("National Highway System", df[df["_nhs"] == 1]))

    all_dates_exclude = set()
    for a in ALBUMS_TIER1:
        dt = pd.to_datetime(a[2]).date()
        for offset in range(-window, window + 1):
            all_dates_exclude.add(dt + datetime.timedelta(days=offset))

    results = []

    for outcome_name, subset in placebo_outcomes_list:
        if len(subset) == 0:
            continue

        daily = subset.groupby("_date")["_fatals"].sum().reset_index()
        daily.columns = ["date", "fatalities"]
        daily["date"] = pd.to_datetime(daily["date"])
        daily = add_time_features(daily)

        est_mask = ~daily["date"].dt.date.isin(all_dates_exclude)
        if est_mask.sum() < 50:
            continue

        try:
            X_est = build_design_matrix(daily[est_mask])
            y_est = daily.loc[est_mask, "fatalities"].values.astype(float)
            beta, _, _ = ols_fit(X_est.values, y_est)

            X_all = build_design_matrix(daily)
            daily["fitted"] = X_all.values @ beta
            daily["resid"] = daily["fatalities"].values - daily["fitted"]
        except Exception:
            continue

        release_resids = []
        for a in ALBUMS_TIER1:
            dt = pd.to_datetime(a[2])
            row = daily[daily["date"] == dt]
            if len(row) > 0:
                release_resids.append(row["resid"].values[0])

        if len(release_resids) < 3:
            continue

        effect = np.mean(release_resids)
        se = np.std(release_resids) / sqrt(len(release_resids))
        t_stat = effect / se if se > 0 else 0
        baseline = daily["fatalities"].mean()
        pct_effect = 100 * effect / baseline if baseline > 0 else 0

        results.append(
            {
                "outcome": outcome_name,
                "baseline_deaths": baseline,
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "pct_effect": pct_effect,
                "n_albums": len(release_resids),
            }
        )

    if not results:
        print("No placebo outcomes could be computed.")
        return None

    results_df = pd.DataFrame(results)

    print(
        f"{'Outcome':<25} | {'Baseline':>10} | {'Effect':>10} | {'SE':>8} | {'t':>8} | {'%':>8}"
    )
    print("-" * 85)
    for _, r in results_df.iterrows():
        sig = " **" if abs(r["t_stat"]) > 2 else " *" if abs(r["t_stat"]) > 1.65 else ""
        print(
            f"{r['outcome']:<25} | {r['baseline_deaths']:>10.1f} | {r['effect']:>+10.1f} | "
            f"{r['se']:>8.1f} | {r['t_stat']:>+7.2f}{sig} | {r['pct_effect']:>+7.1f}%"
        )

    print("\nINTERPRETATION:")
    significant_placebos = results_df[np.abs(results_df["t_stat"]) > 2]
    if len(significant_placebos) > 0:
        print(
            f"  WARNING: {len(significant_placebos)} placebo outcomes show 'significant' effects!"
        )
        for _, r in significant_placebos.iterrows():
            print(
                f"    - {r['outcome']}: effect = {r['effect']:+.1f}, t = {r['t_stat']:.2f}"
            )
        print("  Methodology is detecting spurious patterns.")
    else:
        print("  No significant effects on placebo outcomes. Good.")

    return results_df


def structural_fars_placebos(accidents, window=10):
    """
    Structural FARS Placebos (Green/Gelman).

    Test if album releases "predict" structural FARS variables that
    should not be causally affected by streaming:
    - Mean crash latitude (geography)
    - Mean crash longitude
    - Mean vehicles per crash
    - Mean persons per crash

    If methodology finds effects here, it may be sensitive to noise.

    Output: tabs/t28b_structural_fars_placebos.csv
    """
    print(f"\n{'='*70}")
    print("STRUCTURAL FARS PLACEBOS (Green/Gelman)")
    print(f"{'='*70}")
    print("Testing 'effects' on structural variables:")
    print("  - Mean crash latitude/longitude (geography)")
    print("  - Mean vehicles/persons per crash (crash structure)")
    print("These variables should not be causally affected by streaming.\n")

    df = accidents.copy()
    cols = {c.upper(): c for c in df.columns}

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

    df = df.dropna(subset=["_year", "_month", "_day"])

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

    structural_vars = []

    if "LATITUDE" in cols:
        lat_col = cols["LATITUDE"]
        df["_lat"] = df[lat_col]
        df.loc[df["_lat"] > 90, "_lat"] = np.nan
        df.loc[df["_lat"] < -90, "_lat"] = np.nan
        structural_vars.append(("LATITUDE", "_lat", "Mean crash latitude"))

    if "LONGITUD" in cols:
        lon_col = cols["LONGITUD"]
        df["_lon"] = df[lon_col]
        df.loc[df["_lon"] > 180, "_lon"] = np.nan
        df.loc[df["_lon"] < -180, "_lon"] = np.nan
        structural_vars.append(("LONGITUD", "_lon", "Mean crash longitude"))

    if "VE_TOTAL" in cols:
        ve_col = cols["VE_TOTAL"]
        df["_ve"] = df[ve_col]
        structural_vars.append(("VE_TOTAL", "_ve", "Mean vehicles per crash"))

    if "PERSONS" in cols:
        per_col = cols["PERSONS"]
        df["_persons"] = df[per_col]
        structural_vars.append(("PERSONS", "_persons", "Mean persons per crash"))

    if "RAIL" in cols:
        rail_col = cols["RAIL"]
        try:
            df["_rail"] = pd.to_numeric(df[rail_col], errors="coerce")
            df["_rail"] = (df["_rail"] >= 1).astype(float)
            structural_vars.append(("RAIL", "_rail", "% railroad crossing"))
        except Exception:
            pass

    if "SCH_BUS" in cols:
        bus_col = cols["SCH_BUS"]
        try:
            df["_bus"] = pd.to_numeric(df[bus_col], errors="coerce")
            df["_bus"] = (df["_bus"] >= 1).astype(float)
            structural_vars.append(("SCH_BUS", "_bus", "% school bus involved"))
        except Exception:
            pass

    if "WRK_ZONE" in cols:
        wrk_col = cols["WRK_ZONE"]
        try:
            df["_wrk"] = pd.to_numeric(df[wrk_col], errors="coerce")
            df["_wrk"] = (df["_wrk"] >= 1).astype(float)
            structural_vars.append(("WRK_ZONE", "_wrk", "% work zone"))
        except Exception:
            pass

    all_dates_exclude = set()
    for a in ALBUMS_TIER1:
        dt = pd.to_datetime(a[2]).date()
        for offset in range(-window, window + 1):
            all_dates_exclude.add(dt + datetime.timedelta(days=offset))

    results = []

    for var_name, internal_col, description in structural_vars:
        daily = (
            df.groupby("_date")
            .agg(
                mean_var=(internal_col, "mean"),
                n_obs=(internal_col, "count"),
            )
            .reset_index()
        )
        daily.columns = ["date", "outcome", "n_obs"]
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.dropna(subset=["outcome"])

        if len(daily) < 100:
            continue

        daily = add_time_features(daily)

        est_mask = ~daily["date"].dt.date.isin(all_dates_exclude)
        if est_mask.sum() < 50:
            continue

        X_est = build_design_matrix(daily[est_mask])
        y_est = daily.loc[est_mask, "outcome"].values.astype(float)

        try:
            beta, _, _ = ols_fit(X_est.values, y_est)
        except Exception:
            continue

        X_all = build_design_matrix(daily)

        for col in X_est.columns:
            if col not in X_all.columns:
                X_all[col] = 0
        X_all = X_all[X_est.columns]

        daily["fitted"] = X_all.values @ beta
        daily["resid"] = daily["outcome"].values - daily["fitted"]

        release_resids = []
        for a in ALBUMS_TIER1:
            dt = pd.to_datetime(a[2])
            row = daily[daily["date"] == dt]
            if len(row) > 0:
                release_resids.append(row["resid"].values[0])

        if len(release_resids) < 3:
            continue

        effect = np.mean(release_resids)
        se = np.std(release_resids) / sqrt(len(release_resids))
        t_stat = effect / se if se > 0 else 0
        baseline = daily["outcome"].mean()

        results.append(
            {
                "variable": var_name,
                "description": description,
                "baseline": baseline,
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "n_albums": len(release_resids),
            }
        )

    if not results:
        print("No structural placebos could be computed (missing columns).")
        return None

    results_df = pd.DataFrame(results)

    print(f"{'Variable':<25} | {'Baseline':>12} | {'Effect':>12} | {'t-stat':>8}")
    print("-" * 65)
    for _, r in results_df.iterrows():
        sig = " **" if abs(r["t_stat"]) > 2 else " *" if abs(r["t_stat"]) > 1.65 else ""
        print(
            f"{r['description']:<25} | {r['baseline']:>12.3f} | {r['effect']:>+12.4f} | {r['t_stat']:>+7.2f}{sig}"
        )

    print("\nINTERPRETATION:")
    significant_structural = results_df[np.abs(results_df["t_stat"]) > 2]
    if len(significant_structural) > 0:
        print(
            f"  NOTE: {len(significant_structural)} structural placebo(s) show 'significant' effects."
        )
        for _, r in significant_structural.iterrows():
            print(f"    - {r['description']}: t = {r['t_stat']:.2f}")
        print("  These variables should not be causally affected by streaming.")
        print("  This suggests the methodology may be sensitive to noise.")
    else:
        print(
            "  No significant effects on structural placebos. Methodology passes this check."
        )

    return results_df
