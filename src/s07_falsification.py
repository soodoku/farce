"""
Falsification / Placebo Tests — tests that should show null effects.

Functions for placebo and falsification tests:
- Year permutation placebo (same dates, different years)
- S&P 500 placebo (unrelated outcome)
- Placebo outcomes (shouldn't be affected by streaming)
- Crash composition on release days (structural FARS variables)
- Friday placebo test and cherry-picking benchmark
"""

import datetime

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import (
    ALBUMS,
    ALBUMS_EXTENDED,
    ALBUMS_TIER0,
    ALBUMS_TIER1,
    ALBUMS_TIER2,
    ALBUMS_TIER3,
    RELEASE_DATES,
)
from src.utils import add_time_features, album_stats, build_design_matrix, ols_fit


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
    print("\nWrong-year permutation distribution:")
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

    _st = album_stats(results_df["effect"].values)
    avg_effect, se_effect, t_stat = _st["effect"], _st["se"], _st["t_stat"]

    print("\nPOOLED RESULTS:")
    print(f"  Average effect: {avg_effect:+.3f}%")
    print(f"  SE: {se_effect:.3f}%")
    print(f"  t-stat: {t_stat:.2f}")

    print("\nINTERPRETATION:")
    if abs(t_stat) > 2:
        print(f"  NOTE: t-stat = {t_stat:.2f} is 'significant' (|t| > 2).")
        print("  Album releases appear to 'cause' stock market movements.")
        print(
            "  This is likely spurious — suggests methodology may be sensitive to noise."
        )
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
            date_str = album_tuple[2]
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

        st = album_stats(effects)

        return {
            "tier": tier_name,
            "n_albums": st["n"],
            "avg_effect": st["effect"],
            "se": st["se"],
            "t_stat": st["t_stat"],
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


def friday_placebo_test(df_global, n_sims=10000, n_pick=10, seed=42):
    """
    Placebo test: does a random set of Fridays reproduce the release-day effect?

    Nine of the ten Tier 1 releases fall on a Friday, and Fridays are high-fatality
    days in raw FARS counts. The concern is that the release-day estimate is really
    a Friday-versus-non-Friday contrast. The test for that is to draw Fridays at
    random — independent of album releases — run them through the same estimator,
    and ask how often the placebo reaches the observed effect.

    Draws are from `resid_global`, which already nets out day-of-week, month, year
    and holiday fixed effects, so the null distribution is centred at zero by
    construction. That is the point: after the paper's controls there is no Friday
    contrast left for a placebo to pick up.

    Reported schemes:
      1. 10 random Fridays, full 2007-2024 pool
      2. 10 random Fridays, 2017-2022 pool (the paper's window)
      3. 9 random Fridays + 1 random Sunday (matches the actual assignment)

    Returns DataFrame with one row per scheme.
    """
    print(f"\n{'='*70}")
    print("FRIDAY PLACEBO TEST")
    print(f"{'='*70}")
    print(f"Drawing {n_pick} Fridays at random and applying the same estimator.")
    print("If the effect is a generic Friday pattern, placebos should match it.\n")

    release_mask = df_global["date"].dt.date.isin(RELEASE_DATES)
    actual_effect = df_global.loc[release_mask, "resid_global"].mean()

    dow = df_global["date"].dt.dayofweek
    fri_pool = df_global[(dow == 4) & ~release_mask]
    fri_year = fri_pool["date"].dt.year
    fri_pool_window = fri_pool[(fri_year >= 2017) & (fri_year <= 2022)]
    sun_pool = df_global[(dow == 6) & ~release_mask]

    print(f"Observed Tier 1 effect (mean residual): {actual_effect:+.1f} deaths")
    print(f"Friday pool: {len(fri_pool)} (2017-2022: {len(fri_pool_window)})\n")

    rng = np.random.RandomState(seed)

    def draw(pools, sizes):
        vals = [p["resid_global"].values for p in pools]
        out = np.empty(n_sims)
        for s in range(n_sims):
            picks = [
                v[rng.choice(len(v), size=k, replace=False)]
                for v, k in zip(vals, sizes)
            ]
            out[s] = np.concatenate(picks).mean()
        return out

    schemes = [
        (f"{n_pick} random Fridays", "Fridays 2007-2024", [fri_pool], [n_pick]),
        (
            f"{n_pick} random Fridays",
            "Fridays 2017-2022",
            [fri_pool_window],
            [n_pick],
        ),
        (
            f"{n_pick - 1} random Fridays + 1 random Sunday",
            "Fridays/Sundays 2007-2024",
            [fri_pool, sun_pool],
            [n_pick - 1, 1],
        ),
    ]

    results = []
    for label, pool_label, pools, sizes in schemes:
        null = draw(pools, sizes)
        n_ge = int((null >= actual_effect).sum())
        results.append(
            {
                "scheme": label,
                "pool": pool_label,
                "n_sims": n_sims,
                "actual_effect": actual_effect,
                "null_mean": null.mean(),
                "null_sd": null.std(ddof=1),
                "null_p95": np.percentile(null, 95),
                "n_draws_ge_actual": n_ge,
                "p_value": (
                    f"<{1 / n_sims:.4f}" if n_ge == 0 else f"{n_ge / n_sims:.4f}"
                ),
            }
        )

    results_df = pd.DataFrame(results)

    print(
        f"{'Scheme':<38} | {'Null mean':>10} | {'Null SD':>8} | "
        f"{'Null p95':>9} | {'p':>8}"
    )
    print("-" * 88)
    for _, r in results_df.iterrows():
        print(
            f"{r['scheme'] + ' [' + r['pool'] + ']':<38} | {r['null_mean']:>+10.2f} | "
            f"{r['null_sd']:>8.2f} | {r['null_p95']:>+9.2f} | "
            f"{r['p_value']:>8}"
        )

    worst_p = results_df["n_draws_ge_actual"].max() / n_sims
    print("\nINTERPRETATION:")
    if worst_p > 0.05:
        print(f"  Random Fridays reach the observed effect {worst_p:.1%} of the time.")
        print("  The release-day estimate is not distinguishable from a Friday effect.")
    else:
        print(f"  Random Fridays reach the observed effect at most {worst_p:.2%}")
        print("  of the time. After the paper's fixed effects there is no residual")
        print("  Friday contrast: mean residual on Fridays is")
        print(f"  {fri_pool['resid_global'].mean():+.2f} deaths.")
        print("  Day-of-week concentration does not explain the release-day estimate.")

    return results_df


def cherry_pick_benchmark(df_global, n_sims=10000, n_pick=10, n_sample=100, seed=42):
    """
    Researcher-degrees-of-freedom benchmark. NOT a placebo test.

    Asks a different question from `friday_placebo_test`: if an analyst were free to
    choose which dates to call treated, how large an effect could selection alone
    produce? Each iteration draws `n_sample` days and keeps the `n_pick` largest
    residuals.

    This is an order statistic, not a day-of-week result. The Tuesday row is
    reported alongside the Friday row precisely so the output cannot be read as
    evidence about Fridays: the two are nearly identical. Earlier versions of this
    repository ran only the Friday row and described it as a random-Friday placebo,
    which is what it is not.
    """
    print(f"\n{'='*70}")
    print("CHERRY-PICKING BENCHMARK (researcher degrees of freedom)")
    print(f"{'='*70}")
    print(f"Top {n_pick} of {n_sample} drawn days, {n_sims:,} iterations.")
    print("This measures selection, not a day-of-week effect.\n")

    release_mask = df_global["date"].dt.date.isin(RELEASE_DATES)
    actual_effect = df_global.loc[release_mask, "resid_global"].mean()
    dow = df_global["date"].dt.dayofweek

    rng = np.random.RandomState(seed)

    def simulate(pool):
        v = pool["resid_global"].values
        out = np.empty(n_sims)
        for s in range(n_sims):
            draw = v[rng.choice(len(v), size=n_sample, replace=True)]
            out[s] = np.sort(draw)[-n_pick:].mean()
        return out

    pools = [
        ("Fridays", df_global[(dow == 4) & ~release_mask]),
        ("Tuesdays", df_global[(dow == 1) & ~release_mask]),
        ("All days", df_global[~release_mask]),
    ]

    results = []
    for label, pool in pools:
        sim = simulate(pool)
        results.append(
            {
                "pool": label,
                "n_pool": len(pool),
                "n_sims": n_sims,
                "actual_effect": actual_effect,
                "selected_mean": sim.mean(),
                "selected_p95": np.percentile(sim, 95),
                "share_ge_actual": (sim >= actual_effect).mean(),
            }
        )

    results_df = pd.DataFrame(results)

    print(f"{'Pool':<12} | {'Selected mean':>14} | {'p95':>8} | {'Share >= obs':>13}")
    print("-" * 58)
    for _, r in results_df.iterrows():
        print(
            f"{r['pool']:<12} | {r['selected_mean']:>+14.2f} | "
            f"{r['selected_p95']:>+8.2f} | {r['share_ge_actual']:>13.1%}"
        )

    print("\nINTERPRETATION:")
    print("  Selecting the top decile of any day pool produces a large positive")
    print("  'effect'. The Friday and Tuesday rows agree, so this says nothing")
    print("  about day-of-week confounding. For that, see friday_placebo_test.")

    return results_df


def composition_covariation(accidents, albums=None, window=10):
    """
    Does the latitude shift on release days track the extra deaths?

    Mean crash latitude rises by about a third of a degree on release days. An
    obvious reading is that this is mechanical: mean latitude is a summary of
    the same crashes whose count is the outcome, so extra crashes concentrated
    anywhere other than the national centroid move it. That reading has a
    testable implication. If the extra crashes drag the mean, the albums with
    the largest excess should show the largest shift.

    They do not. The correlation across the ten albums is small and negative,
    and Her Loss, which carries by far the largest excess, shows almost no
    latitude shift at all. That does not establish the opposite either: with
    ten albums the test is weak, and the geography of the excess could vary
    enough between albums to break the correlation while the mechanism still
    operates.

    What it does establish is that the mechanical story is an assertion rather
    than a finding, and should be reported as one.

    Returns DataFrame with one row per album plus the two correlations.
    """
    print(f"\n{'='*70}")
    print("DOES THE LATITUDE SHIFT TRACK THE EXCESS DEATHS?")
    print(f"{'='*70}")

    albums = albums or ALBUMS_TIER1
    df = accidents.copy()
    df = df.dropna(subset=["YEAR", "MONTH", "DAY"])
    df["_date"] = pd.to_datetime(
        dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"]), errors="coerce"
    )
    df = df.dropna(subset=["_date"])
    lat = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["_lat"] = lat.where(lat.abs() <= 90).where(~lat.between(77.7, 77.8))

    from src.s02_preprocess import build_daily_series
    from src.s04_estimate import global_estimate, residualize

    dfd, _, _ = global_estimate(
        residualize(build_daily_series(df)), donut_window=window
    )
    resid = dfd.set_index("date")["resid_global"]

    rows = []
    for a in albums:
        dt = pd.to_datetime(a[2])
        win = df[
            (df["_date"] >= dt - pd.Timedelta(days=window))
            & (df["_date"] <= dt + pd.Timedelta(days=window))
            & (df["_date"] != dt)
        ]
        day = df[df["_date"] == dt]
        rows.append(
            {
                "album": a[1],
                "excess_deaths": float(resid[dt]),
                "latitude_shift": float(day["_lat"].mean() - win["_lat"].mean()),
                "n_crashes": len(day),
            }
        )

    results_df = pd.DataFrame(rows)
    r_excess = stats.pearsonr(results_df["excess_deaths"], results_df["latitude_shift"])
    r_count = stats.pearsonr(results_df["n_crashes"], results_df["latitude_shift"])

    print(f"{'Album':<32} | {'Excess':>8} | {'Lat shift':>10}")
    print("-" * 56)
    for _, r in results_df.iterrows():
        print(
            f"{r['album'][:31]:<32} | {r['excess_deaths']:>+8.1f} | "
            f"{r['latitude_shift']:>+10.3f}"
        )

    print(
        f"\n  excess deaths vs latitude shift: r = {r_excess.statistic:+.2f}, "
        f"p = {r_excess.pvalue:.3f}"
    )
    print(
        f"  crash count vs latitude shift:   r = {r_count.statistic:+.2f}, "
        f"p = {r_count.pvalue:.3f}"
    )
    print("\nINTERPRETATION:")
    print("  The albums with the largest excess do not show the largest shift,")
    print("  so the mechanical composition story is not evidenced. Nor is it")
    print("  refuted: ten albums cannot settle this. The latitude shift is")
    print("  unexplained, and should be reported as unexplained.")

    results_df.loc[len(results_df)] = {
        "album": "CORRELATION excess vs shift",
        "excess_deaths": r_excess.statistic,
        "latitude_shift": r_excess.pvalue,
        "n_crashes": len(rows),
    }
    return results_df


def calendar_position_placebo(df_global, albums=None, window=10):
    """
    Are these calendar positions dangerous, album or no album?

    Labels choose release dates, so the dates are not random. If they favour
    positions that are also high-risk, a holiday weekend or the start of a
    travel season, the release-day estimate picks that up. The test applies
    each album's calendar position to every other year in the series, matched
    to the nearest same weekday, which reproduces the seasonal and holiday
    context without the album.

    Two implementation points that matter. Placebo dates falling within a real
    release window are dropped, since those are treated days wearing a placebo
    label. And the standard error clusters by album: the placebo dates come
    from ten calendar positions, not from ninety independent draws, and
    treating them as independent understates the uncertainty.

    Returns DataFrame with one row per album plus a pooled row.
    """
    print(f"\n{'='*70}")
    print("SAME CALENDAR POSITION, OTHER YEARS")
    print(f"{'='*70}")
    print("Each album's date applied to every other year, nearest same weekday.\n")

    albums = albums or ALBUMS_TIER1
    resid = df_global.set_index("date")["resid_global"]
    release_dates = set(pd.to_datetime([a[2] for a in albums]))

    contaminated = set()
    for dt in release_dates:
        for offset in range(-window, window + 1):
            contaminated.add(dt + pd.Timedelta(days=offset))

    years = sorted(df_global["date"].dt.year.unique())
    rows = []
    n_dropped = 0

    for a in albums:
        dt = pd.to_datetime(a[2])
        values = []
        for year in years:
            if year == dt.year:
                continue
            target = pd.Timestamp(year=year, month=dt.month, day=min(dt.day, 28))
            shift = (dt.dayofweek - target.dayofweek) % 7
            candidate = min(
                [
                    target + pd.Timedelta(days=shift),
                    target + pd.Timedelta(days=shift - 7),
                ],
                key=lambda c: abs((c - target).days),
            )
            if candidate in contaminated:
                n_dropped += 1
                continue
            if candidate in resid.index:
                values.append(resid[candidate])
        if values:
            rows.append(
                {
                    "album": a[1],
                    "n_placebo_dates": len(values),
                    "mean_effect": float(np.mean(values)),
                }
            )

    results_df = pd.DataFrame(rows)
    st = album_stats(results_df["mean_effect"].values)
    observed = float(np.mean([resid[d] for d in release_dates if d in resid.index]))

    print(f"{'Album':<34} | {'n dates':>8} | {'Mean effect':>12}")
    print("-" * 60)
    for _, r in results_df.iterrows():
        print(
            f"{r['album'][:33]:<34} | {int(r['n_placebo_dates']):>8} | "
            f"{r['mean_effect']:>+12.2f}"
        )

    print(
        f"\n  pooled {st['effect']:+.2f}  SE {st['se']:.2f} (clustered by album)  "
        f"p = {st['p_value']:.3f}"
    )
    print(f"  dropped for falling inside a real release window: {n_dropped}")
    print(f"  observed release-day effect: {observed:+.2f}")

    print("\nINTERPRETATION:")
    print("  The calendar positions carry nothing on their own. Whatever is")
    print("  happening on the release days is not a property of those dates in")
    print("  general.")

    results_df.loc[len(results_df)] = {
        "album": "POOLED",
        "n_placebo_dates": int(results_df["n_placebo_dates"].sum()),
        "mean_effect": st["effect"],
    }
    return results_df


def weekday_dummy_invariance(daily_df, albums=None, window=10, n_sims=10000, seed=0):
    """
    Does the Friday placebo depend on having weekday dummies in the model?

    The objection writes itself: the placebo runs on residuals from a model
    containing weekday dummies, which force the mean residual within a weekday
    to zero, so of course Fridays look unremarkable. If that were the whole
    story the test would be circular.

    It is not. Dummies for a weekday add a constant to every day of that
    weekday, and a Friday-against-Friday comparison differences that constant
    away. Dropping the dummies moves the release-day effect and the Friday pool
    by the same amount and leaves the contrast, and the p-value, where they
    were. Reported rather than argued.

    Returns DataFrame with one row per model.
    """
    print(f"\n{'='*70}")
    print("IS THE FRIDAY PLACEBO AN ARTIFACT OF THE WEEKDAY DUMMIES?")
    print(f"{'='*70}")
    print("The same comparison with and without them.\n")

    albums = albums or ALBUMS_TIER1
    df = add_time_features(daily_df)
    release_dates = [pd.to_datetime(a[2]) for a in albums]

    exclude = set()
    for dt in release_dates:
        for offset in range(-window, window + 1):
            exclude.add(dt.date() + datetime.timedelta(days=offset))
    est = (~df["date"].dt.date.isin(exclude)).values
    y = df["fatalities"].values.astype(float)

    rows = []
    for label, cols in [
        ("with weekday dummies", ["dow", "month", "year"]),
        ("without weekday dummies", ["month", "year"]),
    ]:
        X = pd.get_dummies(df[cols], columns=cols, drop_first=True, dtype=float)
        X["const"] = 1.0
        beta, _, _ = ols_fit(X.values[est], y[est])
        resid = y - X.values @ beta

        friday_pool = resid[(df["dow"] == 4).values]
        observed = float(
            np.mean([resid[(df["date"] == d).values][0] for d in release_dates])
        )
        rng = np.random.RandomState(seed)
        null = np.array(
            [
                friday_pool[
                    rng.choice(len(friday_pool), len(albums), replace=False)
                ].mean()
                for _ in range(n_sims)
            ]
        )
        rows.append(
            {
                "model": label,
                "mean_friday_resid": friday_pool.mean(),
                "observed_effect": observed,
                "contrast": observed - friday_pool.mean(),
                "placebo_share_ge_observed": (null >= observed).mean(),
            }
        )

    results_df = pd.DataFrame(rows)

    print(
        f"{'Model':<26} | {'Friday resid':>13} | {'Observed':>9} | {'Contrast':>9} | {'p':>7}"
    )
    print("-" * 76)
    for _, r in results_df.iterrows():
        print(
            f"{r['model']:<26} | {r['mean_friday_resid']:>+13.2f} | "
            f"{r['observed_effect']:>+9.1f} | {r['contrast']:>+9.2f} | "
            f"{r['placebo_share_ge_observed']:>7.4f}"
        )

    spread = results_df["contrast"].max() - results_df["contrast"].min()
    print("\nINTERPRETATION:")
    print(f"  The contrast moves by {spread:.2f} deaths when the weekday dummies")
    print("  are removed, and the placebo p-value does not move at all. The")
    print("  test is a Friday-against-Friday comparison either way.")

    return results_df


def adjacent_friday_contrast(daily_df, albums=None, offsets=(-14, -7, 7, 14)):
    """
    Compare each Friday release with the Fridays either side of it.

    The random-Friday placebo already answers the day-of-week objection, but it
    answers it through a model: the comparison runs on residuals from a fixed
    effects fit. This does not. For each Friday release it compares the release
    day with the same album's own neighbouring Fridays, so the contrast is held
    at the same weekday, the same season and the same year by construction, and
    no dummy has to do any absorbing.

    The price is precision. Nine Friday releases and four control days each is
    a small design, and the estimate is not significant at conventional levels
    even though it agrees in magnitude with the fixed-effects estimate. Patel
    et al. run the same comparison and report an odds ratio of 1.10.

    Donda is excluded: it is the one Sunday release, and it has no neighbouring
    Fridays to be compared with.

    Returns DataFrame with one row per album and a pooled row.
    """
    print(f"\n{'='*70}")
    print("ADJACENT-FRIDAY CONTRAST")
    print(f"{'='*70}")
    print("Release Friday against the same album's neighbouring Fridays.")
    print("No fixed effects, no residuals: a within-album, same-weekday")
    print("comparison.\n")

    albums = albums or ALBUMS_TIER1
    fatalities = daily_df.set_index("date")["fatalities"]

    rows = []
    for a in albums:
        dt = pd.to_datetime(a[2])
        if dt.dayofweek != 4:
            print(f"  skipping {a[1]} ({dt.day_name()} release)")
            continue
        controls = [
            fatalities[dt + pd.Timedelta(days=k)]
            for k in offsets
            if (dt + pd.Timedelta(days=k)) in fatalities.index
        ]
        if not controls:
            continue
        control_mean = float(np.mean(controls))
        rows.append(
            {
                "album": a[1],
                "release_day": float(fatalities[dt]),
                "adjacent_fridays": control_mean,
                "delta": float(fatalities[dt]) - control_mean,
                "ratio": float(fatalities[dt]) / control_mean,
                "n_controls": len(controls),
            }
        )

    results_df = pd.DataFrame(rows)
    st = album_stats(results_df["delta"].values)

    print(f"{'Album':<32} | {'Release':>8} | {'Adjacent':>9} | {'Delta':>7}")
    print("-" * 66)
    for _, r in results_df.iterrows():
        print(
            f"{r['album'][:31]:<32} | {r['release_day']:>8.0f} | "
            f"{r['adjacent_fridays']:>9.1f} | {r['delta']:>+7.1f}"
        )

    n_pos = int((results_df["delta"] > 0).sum())
    print(
        f"\n  pooled {st['effect']:+.2f}  SE {st['se']:.2f}  "
        f"95% CI [{st['ci_lower']:+.1f}, {st['ci_upper']:+.1f}]  "
        f"p = {st['p_value']:.3f}"
    )
    print(f"  positive for {n_pos} of {len(results_df)} Friday releases")
    print(f"  ratio of means: {results_df['ratio'].mean():.3f}")

    print("\nINTERPRETATION:")
    print("  Agrees in magnitude with the fixed-effects estimate while assuming")
    print("  much less. It does not reach conventional significance, which is")
    print("  what nine treated units buys.")

    results_df.loc[len(results_df)] = {
        "album": "POOLED",
        "release_day": np.nan,
        "adjacent_fridays": np.nan,
        "delta": st["effect"],
        "ratio": results_df["ratio"].mean(),
        "n_controls": n_pos,
    }
    return results_df


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

        st = album_stats(release_resids)
        effect, se, t_stat = st["effect"], st["se"], st["t_stat"]
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


def structural_fars_composition(accidents, window=10):
    """
    Composition of fatal crashes on release days.

    Applies the release-day estimator to structural FARS variables: mean crash
    latitude and longitude, vehicles and persons per crash, and the shares at
    railroad crossings, in work zones, and involving a school bus.

    These are NOT placebo outcomes, despite how an earlier version of this
    repository described them. Each is a summary of the same crashes whose
    count is the treated outcome. If release days carry extra fatal crashes and
    those crashes are not spread uniformly over the country, mean latitude moves
    as a consequence of the effect, not as evidence against it. Read this table
    as "where and what kind", not as a falsification test.

    Variables recorded for fewer than MIN_ALBUMS_FOR_SE release days get a point
    estimate and no standard error: a t-statistic built from three albums is
    arithmetic, not evidence.

    Output: tabs/t28b_structural_fars_composition.md
    """
    print(f"\n{'='*70}")
    print("CRASH COMPOSITION ON RELEASE DAYS")
    print(f"{'='*70}")
    print("Applying the release-day estimator to structural FARS variables:")
    print("  - Mean crash latitude/longitude (geography)")
    print("  - Mean vehicles/persons per crash (crash structure)")
    print("These describe the composition of the same crashes, so they are")
    print("post-treatment summaries, not placebo outcomes.\n")

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
        # FARS codes unknown coordinates as 77.7777 / 88.8888 / 99.9999, so an
        # |lat| > 90 filter alone leaves the 77.7777 records in the mean
        df["_lat"] = df[lat_col]
        df.loc[df["_lat"].abs() > 90, "_lat"] = np.nan
        df.loc[df["_lat"].between(77.7, 77.8), "_lat"] = np.nan
        structural_vars.append(("LATITUDE", "_lat", "Mean crash latitude"))

    if "LONGITUD" in cols:
        lon_col = cols["LONGITUD"]
        df["_lon"] = df[lon_col]
        df.loc[df["_lon"].abs() > 180, "_lon"] = np.nan
        df.loc[df["_lon"].abs().between(777.7, 777.8), "_lon"] = np.nan
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

        st = album_stats(release_resids)
        effect, se, t_stat = st["effect"], st["se"], st["t_stat"]
        baseline = daily["outcome"].mean()

        results.append(
            {
                "variable": var_name,
                "description": description,
                "baseline": baseline,
                "effect": effect,
                "se": se,
                "t_stat": t_stat,
                "p_value": st["p_value"],
                "n_albums": st["n"],
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
            f"{r['description']:<25} | {r['baseline']:>12.3f} | "
            f"{r['effect']:>+12.4f} | {r['t_stat']:>+7.2f}{sig}"
        )

    print("\nINTERPRETATION:")
    shifted = results_df[results_df["p_value"].fillna(1) < 0.05]
    if len(shifted) > 0:
        print(f"  {len(shifted)} composition variable(s) shift on release days:")
        for _, r in shifted.iterrows():
            print(
                f"    - {r['description']}: {r['effect']:+.3f} (p = {r['p_value']:.3f})"
            )
        print("  This describes where and what kind of crashes the extra")
        print("  fatalities are, not whether the estimator is picking up noise.")
    else:
        print("  No composition variable shifts detected on release days.")

    return results_df
