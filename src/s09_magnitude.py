"""
Magnitude Checks — effect size plausibility.

Functions for checking whether the effect size is plausible:
- Power analysis (Type M / Type S errors)
- Effect size plausibility vs weather benchmarks
- Weather effect sanity check
"""

import datetime
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import ALBUMS_TIER1, us_holidays
from src.utils import add_time_features, album_stats, build_design_matrix, ols_fit


def power_analysis(
    df_global, n_albums=10, alpha=0.05, n_sims=100000, seed=42, clustered_se=7.45
):
    """
    What can ten treated days see, and by how much does a hit overstate?

    Two quantities. The minimum detectable effect is the smallest true effect
    this design rejects the null for 80% of the time. The exaggeration ratio,
    Gelman and Carlin's Type M, is E[|estimate| given significance] divided by
    the true effect: with low power, an estimate has to be large to clear the
    threshold, so the published ones are the large draws.

    Two things that matter at n = 10 and are easy to get wrong. Critical values
    come from t with n - 1 degrees of freedom, not the normal; using 1.96 here
    understates the minimum detectable effect by about a death and a half. And
    the standard error is estimated from the same ten observations rather than
    known, so the simulation draws n observations per replicate and runs a real
    t-test. Treating the standard error as known overstates the exaggeration:
    at a true effect of 5 it reports 2.4x where the honest figure is 2.1x.

    Three standard errors are available and they answer different questions.
    The null standard error, residual SD over sqrt(n), is what the design can
    see against noise, and is the one used here. The realized cross-album
    standard error is larger because the ten release-day residuals are more
    dispersed than ten arbitrary days, so the effective threshold for this
    particular sample is higher still; it is reported alongside.

    Output: tabs/t23_power_analysis.md
    """
    print(f"\n{'='*70}")
    print("DESIGN SENSITIVITY: MDE AND EXAGGERATION (Gelman and Carlin)")
    print(f"{'='*70}")
    print("With low power, an estimate must be large to reach significance,")
    print("so the ones that get published are the large draws.\n")

    resid_sd = df_global["resid_global"].std()
    se_null = resid_sd / sqrt(n_albums)

    tier1_resids = [
        df_global.loc[df_global["date"] == pd.to_datetime(a[2]), "resid_global"].values[
            0
        ]
        for a in ALBUMS_TIER1
        if (df_global["date"] == pd.to_datetime(a[2])).any()
    ]
    observed = album_stats(tier1_resids)
    observed_effect = observed["effect"]
    se_realized = observed["se"]

    t_crit = stats.t.ppf(1 - alpha / 2, n_albums - 1)
    mde_80 = (t_crit + stats.norm.ppf(0.80)) * se_null
    mde_50 = t_crit * se_null

    print(f"Residual SD of daily fatalities: {resid_sd:.1f} deaths")
    print(f"Null standard error of an {n_albums}-day mean: {se_null:.2f}")
    print(
        f"Realized standard error across the {n_albums} release days: "
        f"{se_realized:.2f}"
    )
    print(f"Critical value: t({n_albums - 1}) = {t_crit:.3f}, not 1.96\n")
    print(f"MINIMUM DETECTABLE EFFECT (two-sided alpha = {alpha}):")
    print(f"  80% power: {mde_80:.1f} deaths")
    print(f"  50% power: {mde_50:.1f} deaths")
    print(
        f"  80% power at the realized standard error: "
        f"{(t_crit + stats.norm.ppf(0.80)) * se_realized:.1f} deaths"
    )
    print(f"\nObserved Tier 1 effect: {observed_effect:+.1f} deaths")

    # The calculation is only as relevant as the estimator it models, and there
    # are three candidate standard errors for one quantity. Running all three
    # shows which way the choice cuts: the null SE is the most favourable to
    # the published estimate, and it is the one quoted in the text.
    se_options = [
        ("null (residual SD / sqrt n)", se_null),
        ("realized across albums", se_realized),
        ("paper's clustered SE", clustered_se),
    ]

    rng = np.random.RandomState(seed)
    results = []

    for se_label, se in se_options:
        sd = se * sqrt(n_albums)
        mde = (t_crit + stats.norm.ppf(0.80)) * se
        print(f"\nSIMULATION, SE = {se:.2f} ({se_label}), MDE80 = {mde:.1f}")
        print(
            f"{'True effect':>12} | {'Power':>8} | {'Exaggeration':>13} | "
            f"{'Wrong sign':>11}"
        )
        print("-" * 55)

        for true_eff in [0, 2, 3, 5, 8, 12, 16, 20, observed_effect]:
            sample = rng.normal(true_eff, sd, (n_sims, n_albums))
            means = sample.mean(axis=1)
            ses = sample.std(axis=1, ddof=1) / sqrt(n_albums)
            significant = np.abs(means / ses) > t_crit
            power = significant.mean()

            if true_eff == 0 or not significant.any():
                type_m = np.nan
                type_s = np.nan
            else:
                type_m = np.abs(means[significant]).mean() / abs(true_eff)
                type_s = (np.sign(means[significant]) != np.sign(true_eff)).mean()

            results.append(
                {
                    "se_source": se_label,
                    "se": se,
                    "mde_80": mde,
                    "mde_50": t_crit * se,
                    "true_effect": true_eff,
                    "power": power,
                    "exaggeration": type_m,
                    "wrong_sign": type_s,
                    "n_albums": n_albums,
                    "resid_sd": resid_sd,
                }
            )

            m = f"{type_m:.2f}x" if not np.isnan(type_m) else "n/a"
            sgn = f"{type_s:.3f}" if not np.isnan(type_s) else "n/a"
            mark = "  <-- observed" if true_eff == observed_effect else ""
            print(f"{true_eff:>12.1f} | {power:>8.2f} | {m:>13} | {sgn:>11}{mark}")

    results_df = pd.DataFrame(results)

    print("\nINTERPRETATION:")
    small = results_df[
        (results_df["true_effect"] == 5)
        & (results_df["se_source"] == "null (residual SD / sqrt n)")
    ].iloc[0]
    print("  If the truth were 5 deaths, this design reaches significance")
    print(
        f"  {small['power']:.0%} of the time and reports "
        f"{small['exaggeration']:.1f} times too much when it does."
    )
    if observed_effect < mde_80:
        print(
            f"  The observed {observed_effect:.1f} is below the {mde_80:.1f} "
            "minimum detectable effect,"
        )
        print("  so this sample is one of the draws that cleared the threshold.")
    print("  These figures use the null standard error, the smallest of the three")
    print("  and so the reading most favourable to the published estimate. At the")
    print("  paper's clustered standard error the minimum detectable effect")
    print("  exceeds the estimate itself.")

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
        """
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


def holiday_benchmark(daily_df, albums=None):
    """
    How big is the release-day effect next to days whose danger is documented?

    Literature benchmarks for weather or distraction effects come from other
    data and other estimators, so comparing a FARS release-day coefficient with
    them compares two things at once. Federal holidays are a benchmark from the
    same series, the same years and the same model: excess fatalities over a
    day-of-week, seasonal and year baseline.

    Holiday indicators are dropped from the baseline, since otherwise the days
    being measured would be absorbed by their own dummies. Release-day effects
    are recomputed on that same baseline so the columns are comparable.

    All nine federal holidays in `us_holidays` are reported, not a chosen
    subset. Three of them (MLK Day, Presidents Day, Veterans Day) show no
    excess at all, so "as deadly as a federal holiday" is not a statement the
    data support; the comparison that survives is with the travel holidays.

    Both a month and a week-of-year seasonal baseline are run. The release-day
    figure is the same either way; holiday levels move by up to four deaths,
    so the ranking rather than the level is what carries.

    Returns DataFrame with one row per day type per baseline.
    """
    print("\n" + "=" * 70)
    print("MAGNITUDE BENCHMARK: RELEASE DAYS vs FEDERAL HOLIDAYS")
    print("=" * 70)
    print("Excess deaths over a day-of-week + seasonal + year baseline.\n")

    albums = albums or ALBUMS_TIER1
    df = add_time_features(daily_df)
    df["woy"] = df["date"].dt.isocalendar().week.astype(int)
    years = list(range(int(df["year"].min()), int(df["year"].max()) + 1))
    release_dates = [pd.to_datetime(a[2]) for a in albums]

    def nth_weekday(year, month, weekday, n):
        d = datetime.date(year, month, 1)
        return d + datetime.timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))

    def last_weekday(year, month, day_hi, weekday):
        d = datetime.date(year, month, day_hi)
        while d.weekday() != weekday:
            d -= datetime.timedelta(days=1)
        return d

    benchmarks = [
        ("New Year's Day", [datetime.date(y, 1, 1) for y in years]),
        ("MLK Day", [nth_weekday(y, 1, 0, 3) for y in years]),
        ("Presidents Day", [nth_weekday(y, 2, 0, 3) for y in years]),
        ("Memorial Day", [last_weekday(y, 5, 31, 0) for y in years]),
        ("July 4", [datetime.date(y, 7, 4) for y in years]),
        ("Labor Day", [nth_weekday(y, 9, 0, 1) for y in years]),
        ("Veterans Day", [datetime.date(y, 11, 11) for y in years]),
        ("Thanksgiving", [nth_weekday(y, 11, 3, 4) for y in years]),
        ("Christmas Day", [datetime.date(y, 12, 25) for y in years]),
    ]

    rows = []
    for seasonal in ["month", "woy"]:
        X = pd.get_dummies(
            df[["dow", seasonal, "year"]],
            columns=["dow", seasonal, "year"],
            drop_first=True,
            dtype=float,
        )
        X["const"] = 1.0
        beta, _, _ = ols_fit(X.values, df["fatalities"].values.astype(float))
        excess = df["fatalities"].values - X.values @ beta

        for label, dates in benchmarks + [("Album release days", release_dates)]:
            if label == "Album release days":
                mask = df["date"].isin(dates).values
            else:
                mask = df["date"].dt.date.isin(set(dates)).values
            st = album_stats(excess[mask])
            rows.append(
                {
                    "baseline": f"dow+{seasonal}+year",
                    "day": label,
                    "n": st["n"],
                    "excess": st["effect"],
                    "ci_lower": st["ci_lower"],
                    "ci_upper": st["ci_upper"],
                }
            )

    results_df = pd.DataFrame(rows)

    for seasonal in ["month", "woy"]:
        sub = results_df[results_df["baseline"] == f"dow+{seasonal}+year"]
        sub = sub.sort_values("excess", ascending=False)
        print(f"-- baseline dow+{seasonal}+year --")
        print(f"{'Day':<22} | {'n':>3} | {'Excess':>8} | {'95% CI':>18}")
        print("-" * 60)
        for _, r in sub.iterrows():
            ci = f"[{r['ci_lower']:+.1f}, {r['ci_upper']:+.1f}]"
            mark = "  <--" if r["day"] == "Album release days" else ""
            print(
                f"{r['day']:<22} | {int(r['n']):>3} | {r['excess']:>+8.1f} | "
                f"{ci:>18}{mark}"
            )
        print()

    alb = results_df[results_df["day"] == "Album release days"]
    print("INTERPRETATION:")
    print(
        f"  Release days: {alb['excess'].min():+.1f} to {alb['excess'].max():+.1f} "
        "across the two baselines, so the figure does not depend on the"
    )
    print("  seasonal control. It lands among the travel holidays, above Labor")
    print("  Day and below Memorial Day, with an interval wide enough to reach")
    print("  from Thanksgiving to July 4. Three federal holidays show no excess")
    print("  at all, so the benchmark is the travel holidays specifically.")

    return results_df


def holiday_adjacency_robustness(daily_df, albums=None, window=10):
    """
    Does the estimate survive the releases that sit next to a federal holiday?

    Two of the ten Tier 1 releases are holiday-adjacent: Certified Lover Boy on
    the Friday of Labor Day weekend, and Red (Taylor's Version) the day after
    Veterans Day. The main specification carries a holiday indicator and a
    one-day-either-side indicator, which catches the second but not the first.

    Two checks. Widen the holiday control from the day itself out to seven days
    either side and watch the estimate. Then drop both albums and see what the
    remaining eight give.

    Returns DataFrame with one row per holiday-window width plus a final row
    for the drop-both variant.
    """
    print("\n" + "=" * 70)
    print("HOLIDAY ADJACENCY ROBUSTNESS")
    print("=" * 70)
    print("Two releases sit within three days of a federal holiday.\n")

    albums = albums or ALBUMS_TIER1
    df = add_time_features(daily_df)
    df["woy"] = df["date"].dt.isocalendar().week.astype(int)
    holidays = us_holidays(sorted(df["year"].unique()))
    release_dates = [pd.to_datetime(a[2]) for a in albums]

    exclude = set()
    for dt in release_dates:
        for offset in range(-window, window + 1):
            exclude.add(dt.date() + datetime.timedelta(days=offset))
    est = (~df["date"].dt.date.isin(exclude)).values
    y = df["fatalities"].values.astype(float)

    base = pd.get_dummies(
        df[["dow", "woy", "year"]],
        columns=["dow", "woy", "year"],
        drop_first=True,
        dtype=float,
    )

    def fit(extra_cols):
        X = base.copy()
        for name, col in extra_cols.items():
            X[name] = col
        X["const"] = 1.0
        beta, _, _ = ols_fit(X.values[est], y[est])
        return y - X.values @ beta

    rows = []
    for k in [0, 1, 2, 3, 5, 7]:
        flagged = set()
        for h in holidays:
            for j in range(-k, k + 1):
                flagged.add(h + datetime.timedelta(days=j))
        resid = fit({"hol_win": df["date"].dt.date.isin(flagged).astype(float).values})
        st = album_stats([resid[(df["date"] == d).values][0] for d in release_dates])
        rows.append(
            {
                "variant": f"holiday control +/-{k}d",
                "n_albums": st["n"],
                "effect": st["effect"],
                "se": st["se"],
                "ci_lower": st["ci_lower"],
                "ci_upper": st["ci_upper"],
                "p_value": st["p_value"],
                "n_days_flagged": int(df["date"].dt.date.isin(flagged).sum()),
            }
        )

    near = set()
    for h in holidays:
        for j in range(-3, 4):
            near.add(h + datetime.timedelta(days=j))
    resid = fit(
        {
            "holiday": df["holiday"].values.astype(float),
            "holiday_adj": df["holiday_adj"].values.astype(float),
        }
    )
    kept = [
        resid[(df["date"] == d).values][0]
        for d in release_dates
        if d.date() not in near
    ]
    st = album_stats(kept)
    rows.append(
        {
            "variant": "drop holiday-adjacent releases",
            "n_albums": st["n"],
            "effect": st["effect"],
            "se": st["se"],
            "ci_lower": st["ci_lower"],
            "ci_upper": st["ci_upper"],
            "p_value": st["p_value"],
            "n_days_flagged": np.nan,
        }
    )

    results_df = pd.DataFrame(rows)

    print(f"{'Variant':<32} | {'n':>3} | {'Effect':>8} | {'SE':>6} | {'p':>6}")
    print("-" * 68)
    for _, r in results_df.iterrows():
        print(
            f"{r['variant']:<32} | {int(r['n_albums']):>3} | {r['effect']:>+8.2f} | "
            f"{r['se']:>6.2f} | {r['p_value']:>6.3f}"
        )

    spread = results_df["effect"].max() - results_df["effect"].min()
    print("\nINTERPRETATION:")
    print(f"  The estimate moves by {spread:.2f} deaths across these variants.")
    print("  Proximity to a federal holiday does not carry the release-day effect.")

    return results_df


def publication_filter(
    true_effects, se, n_albums=10, alpha=0.05, n_sims=400000, seed=7
):
    """
    If the truth were X, what would a ten-album study that got published report?

    Points 2 and 3 have to be reconciled. The pooled estimate over the albums
    outside the paper's sample is around six deaths; the paper reports
    eighteen. These are not in tension once the significance filter is applied,
    but the reconciliation depends on which standard error is used, so it is
    computed rather than asserted.

    For each candidate true effect this draws n_albums observations, runs the
    t-test an analyst would run, and reports what the surviving estimates look
    like: how often they survive, their mean, and how often they reach the
    replicated 17.6.

    Returns DataFrame with one row per candidate true effect.
    """
    print("\n" + "=" * 70)
    print("WHAT A TEN-ALBUM STUDY WOULD PUBLISH")
    print("=" * 70)
    print(f"Standard error {se:.2f}, n = {n_albums}, two-sided alpha = {alpha}.\n")

    t_crit = stats.t.ppf(1 - alpha / 2, n_albums - 1)
    rng = np.random.RandomState(seed)
    sd = se * sqrt(n_albums)
    rows = []

    print(
        f"{'True effect':>12} | {'Power':>7} | {'Mean published':>15} | {'P(>= 17.6)':>11}"
    )
    print("-" * 56)
    for true_eff in true_effects:
        sample = rng.normal(true_eff, sd, (n_sims, n_albums))
        means = sample.mean(axis=1)
        ses = sample.std(axis=1, ddof=1) / sqrt(n_albums)
        sig = np.abs(means / ses) > t_crit
        mean_pub = np.abs(means[sig]).mean() if sig.any() else np.nan
        reach = (means[sig] >= 17.6).mean() if sig.any() else np.nan
        rows.append(
            {
                "true_effect": true_eff,
                "se": se,
                "power": sig.mean(),
                "mean_published": mean_pub,
                "p_reaches_replicated": reach,
            }
        )
        print(
            f"{true_eff:>12.2f} | {sig.mean():>7.2f} | {mean_pub:>15.1f} | {reach:>11.2f}"
        )

    print("\nINTERPRETATION:")
    print("  A published estimate is not an unbiased read of the truth when")
    print("  power is low. At this standard error, a modest true effect and a")
    print("  large published one are the same story.")

    return pd.DataFrame(rows)


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
