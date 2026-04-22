"""
Primary Estimation — core treatment effect estimates.

Functions for estimating the causal effect of album releases on fatalities:
- Residualization (removing time fixed effects)
- Local counterfactual (paper's ±window approach)
- Global counterfactual (donut regression)
- Paper's exact regression specification
- Decomposition analysis
"""

import datetime

import numpy as np
import pandas as pd

from src.constants import (ALBUMS, ALBUMS_ALL, ALBUMS_TIER1, ALBUMS_TIER2,
                           RELEASE_DATES, RELEASE_DATES_ALL)
from src.utils import add_time_features, build_design_matrix, ols_fit, cluster_robust_se


def residualize(daily):
    """
    Regress daily fatalities on day-of-week, month, year, holiday FEs,
    and crash-level predictors (dark, rural, bad weather, night, alcohol).
    Return the DataFrame with residuals attached.
    """
    df = add_time_features(daily)
    X = build_design_matrix(df)
    y = df["fatalities"].values.astype(float)
    beta, fitted, residuals = ols_fit(X.values, y)
    df["fitted"] = fitted
    df["residual"] = residuals
    df["z_score"] = (residuals - residuals.mean()) / residuals.std()
    return df


def local_estimate(df, window=10, pre_only=False):
    """
    Paper's approach: for each release date, compare fatalities on release day
    to the average of the ±window surrounding days.

    Parameters
    ----------
    df : DataFrame
        Daily fatality data
    window : int
        Number of days before (and optionally after) release to include
    pre_only : bool
        If True, use only pre-treatment days as control (avoids post-treatment
        contamination if spillover effects exist). Default: False (paper's spec).

    Returns
    -------
    DataFrame with per-event deltas and the pooled estimate.
    """
    results = []
    for artist, album, date_str, dow in ALBUMS:
        dt = pd.to_datetime(date_str)
        release_row = df[df["date"] == dt]
        if len(release_row) == 0:
            continue

        y_release = release_row["fatalities"].values[0]

        if pre_only:
            mask = (
                (df["date"] >= dt - pd.Timedelta(days=window))
                & (df["date"] < dt)
            )
        else:
            mask = (
                (df["date"] >= dt - pd.Timedelta(days=window))
                & (df["date"] <= dt + pd.Timedelta(days=window))
                & (df["date"] != dt)
            )
        control = df[mask]
        y_control = control["fatalities"].mean()

        delta = y_release - y_control

        results.append(
            {
                "artist": artist,
                "album": album,
                "date": date_str,
                "dow": dow,
                "y_release": y_release,
                "y_control": y_control,
                "delta_local": delta,
            }
        )

    return pd.DataFrame(results)


def paper_regression_estimate(df, albums=None, window=10, sample_period=None):
    """
    Paper's exact regression specification from page 7:

    "multivariable linear regression at the album-day level to estimate national
    daily counts of traffic fatalities in each of the 10 days surrounding album
    releases (indicator variables for each day relative to album release, defined
    as day zero), adjusting for fixed effects (i.e., indicators) of federal holidays,
    day-of-week (Monday through Sunday), week-of-year (weeks 1 through 52), and
    calendar-year."

    Parameters
    ----------
    df : DataFrame
        Daily fatality data
    albums : list
        Albums to analyze (default: ALBUMS_TIER1)
    window : int
        Days before/after release to include (default: 10)
    sample_period : tuple
        (start_year, end_year) to filter sample (default: None = all years)

    Returns
    -------
    dict
        Regression results including treatment effect and SE
    """
    if albums is None:
        albums = ALBUMS_TIER1

    df = df.copy()

    if sample_period is not None:
        start_year, end_year = sample_period
        df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]

    df = add_time_features(df)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    album_dfs = []
    for a in albums:
        dt = pd.to_datetime(a[2])
        for offset in range(-window, window + 1):
            day = dt + pd.Timedelta(days=offset)
            row = df[df["date"] == day]
            if len(row) == 0:
                continue

            album_dfs.append(
                {
                    "artist": a[0],
                    "album": a[1],
                    "release_date": a[2],
                    "day": day,
                    "day_relative": offset,
                    "fatalities": row["fatalities"].values[0],
                    "dow": row["dow"].values[0],
                    "week_of_year": row["week_of_year"].values[0],
                    "year": row["year"].values[0],
                    "holiday": row["holiday"].values[0],
                }
            )

    reg_df = pd.DataFrame(album_dfs)

    reg_df["release_day"] = (reg_df["day_relative"] == 0).astype(int)

    dow_dummies = pd.get_dummies(
        reg_df["dow"], prefix="dow", drop_first=True, dtype=float
    )
    week_dummies = pd.get_dummies(
        reg_df["week_of_year"], prefix="week", drop_first=True, dtype=float
    )
    year_dummies = pd.get_dummies(
        reg_df["year"], prefix="year", drop_first=True, dtype=float
    )

    X = pd.concat(
        [
            dow_dummies,
            week_dummies,
            year_dummies,
        ],
        axis=1,
    )
    X["holiday"] = reg_df["holiday"].values
    X["release_day"] = reg_df["release_day"].values
    X["const"] = 1.0

    y = reg_df["fatalities"].values.astype(float)

    beta, se_ols, fitted, resid = ols_fit(X.values, y, return_se=True, ridge_lambda=1e-8)

    album_ids = reg_df["release_date"].values
    se_cluster = cluster_robust_se(X.values, resid, album_ids)

    release_day_idx = list(X.columns).index("release_day")
    treatment_effect = beta[release_day_idx]
    treatment_se = se_cluster[release_day_idx]

    control_mean = reg_df[reg_df["release_day"] == 0]["fatalities"].mean()
    pct_effect = 100 * treatment_effect / control_mean

    per_album_effects = []
    for a in albums:
        album_rows = reg_df[
            (reg_df["release_date"] == a[2]) & (reg_df["day_relative"] == 0)
        ]
        if len(album_rows) == 0:
            continue

        y_release = album_rows["fatalities"].values[0]

        control_rows = reg_df[
            (reg_df["release_date"] == a[2]) & (reg_df["day_relative"] != 0)
        ]
        y_control = (
            control_rows["fatalities"].mean() if len(control_rows) > 0 else np.nan
        )

        per_album_effects.append(
            {
                "artist": a[0],
                "album": a[1],
                "date": a[2],
                "y_release": y_release,
                "y_control": y_control,
                "delta_raw": (
                    y_release - y_control if not np.isnan(y_control) else np.nan
                ),
            }
        )

    per_album_df = pd.DataFrame(per_album_effects)

    return {
        "treatment_effect": treatment_effect,
        "treatment_se": treatment_se,
        "t_stat": treatment_effect / treatment_se,
        "pct_effect": pct_effect,
        "control_mean": control_mean,
        "n_obs": len(reg_df),
        "n_albums": len(albums),
        "n_release_days": reg_df["release_day"].sum(),
        "per_album_df": per_album_df,
        "sample_period": sample_period,
    }


def global_estimate(df, donut_window=None):
    """
    Global counterfactual: fit DOW + month + year + holiday FEs on the full
    time series (or on the donut sample excluding ±donut_window days around
    each release), then compute residuals for all days.

    Returns (df with global residuals, beta from the global regression).
    """
    df = df.copy()

    if donut_window is not None:
        exclude = set()
        for _, _, date_str, _ in ALBUMS:
            dt = pd.to_datetime(date_str).date()
            for offset in range(-donut_window, donut_window + 1):
                exclude.add(dt + datetime.timedelta(days=offset))
        estimation_mask = ~df["date"].dt.date.isin(exclude)
        label = f"donut(±{donut_window})"
    else:
        estimation_mask = pd.Series(True, index=df.index)
        label = "full sample"

    X_est = build_design_matrix(df[estimation_mask])
    y_est = df.loc[estimation_mask, "fatalities"].values.astype(float)
    beta, _, _ = ols_fit(X_est.values, y_est)

    X_all = build_design_matrix(df)
    fitted_all = X_all.values @ beta
    df["fitted_global"] = fitted_all
    df["resid_global"] = df["fatalities"].values - fitted_all

    resid_est = y_est - X_est.values @ beta
    resid_sd = resid_est.std()
    df["z_global"] = df["resid_global"] / resid_sd

    return df, beta, label


def decomposition_analysis(df, window=10):
    """
    Core analysis: compute local, global, and donut-global estimates.
    Show the decomposition: δ_local = δ_global + (Ŷ_release - Ȳ_control).
    """
    print(f"\n{'='*70}")
    print("LOCAL vs GLOBAL COUNTERFACTUAL ANALYSIS")
    print(f"{'='*70}")

    local = local_estimate(df, window=window)

    print(f"\n── 1. LOCAL ESTIMATE (paper's ±{window} day window) ──")
    print(f"{'Artist':<22} {'Album':<25} {'Y_rel':>6} {'Ȳ_ctrl':>7} {'δ_loc':>7}")
    print("-" * 72)
    for _, r in local.iterrows():
        print(
            f"{r['artist']:<22} {r['album']:<25} {r['y_release']:>6.0f} "
            f"{r['y_control']:>7.1f} {r['delta_local']:>7.1f}"
        )
    avg_local = local["delta_local"].mean()
    se_local = local["delta_local"].std() / np.sqrt(len(local))
    print(f"\n  Pooled local δ:  {avg_local:+.1f} deaths  (SE = {se_local:.1f})")
    print(f"  Pooled local %:  {100*avg_local/local['y_control'].mean():+.1f}%")

    df_g, _, lbl_g = global_estimate(df, donut_window=None)
    df_d, _, lbl_d = global_estimate(df, donut_window=window)

    for tag, dff, lbl in [
        ("FULL-SAMPLE GLOBAL", df_g, lbl_g),
        (f"DONUT GLOBAL (excl ±{window}d)", df_d, lbl_d),
    ]:
        print(f"\n── 2. {tag} ({lbl}) ──")
        print(f"{'Artist':<22} {'Album':<25} {'Y_rel':>6} {'Ŷ_glob':>7} {'δ_glob':>7}")
        print("-" * 72)

        deltas_g = []
        fitted_releases = []
        for _, _, date_str, _ in ALBUMS:
            dt = pd.to_datetime(date_str)
            row = dff[dff["date"] == dt]
            if len(row) == 0:
                continue
            y_rel = row["fatalities"].values[0]
            y_hat = row["fitted_global"].values[0]
            delta_g = y_rel - y_hat
            deltas_g.append(delta_g)
            fitted_releases.append(y_hat)

            album_info = [a for a in ALBUMS if a[2] == date_str][0]
            print(
                f"{album_info[0]:<22} {album_info[1]:<25} {y_rel:>6.0f} "
                f"{y_hat:>7.1f} {delta_g:>7.1f}"
            )

        avg_global = np.mean(deltas_g)
        se_global = np.std(deltas_g) / np.sqrt(len(deltas_g))
        print(f"\n  Pooled global δ: {avg_global:+.1f} deaths  (SE = {se_global:.1f})")
        print(f"  Pooled global %: {100*avg_global/np.mean(fitted_releases):+.1f}%")

    print(f"\n── 3. DECOMPOSITION: δ_local = δ_global + (Ŷ_release − Ȳ_control) ──")
    print(f"  This last term measures whether the ±{window}d control window is")
    print("  representative of 'normal' days as predicted by the global model.\n")

    print(
        f"{'Artist':<22} {'Album':<20} {'δ_loc':>7} {'δ_glob':>7} {'Ŷ−Ȳ_c':>7} {'Check':>7}"
    )
    print("-" * 72)

    for _, r in local.iterrows():
        dt = pd.to_datetime(r["date"])
        row_d = df_d[df_d["date"] == dt]
        if len(row_d) == 0:
            continue
        y_hat_release = row_d["fitted_global"].values[0]
        delta_g = r["y_release"] - y_hat_release
        gap = y_hat_release - r["y_control"]
        check = delta_g + gap

        print(
            f"{r['artist']:<22} {r['album']:<20} {r['delta_local']:>7.1f} "
            f"{delta_g:>7.1f} {gap:>7.1f} {check:>7.1f}"
        )

    gaps = []
    for _, r in local.iterrows():
        dt = pd.to_datetime(r["date"])
        row_d = df_d[df_d["date"] == dt]
        if len(row_d) == 0:
            continue
        gaps.append(row_d["fitted_global"].values[0] - r["y_control"])

    avg_gap = np.mean(gaps)
    print(f"\n  Avg (Ŷ_release − Ȳ_control): {avg_gap:+.1f}")
    print(
        f"  Interpretation: the ±{window}d control window runs "
        f"{'BELOW' if avg_gap > 0 else 'ABOVE'} the global prediction"
    )
    print(
        f"  by {abs(avg_gap):.1f} deaths/day, "
        f"{'inflating' if avg_gap > 0 else 'deflating'} "
        "the local estimate relative to global."
    )

    return df_d, local


def evaluate(df, k_values=None):
    """
    For each threshold k, flag days with z_score > k as predicted release days.
    Report precision, recall, and the rank of actual release dates.
    """
    if k_values is None:
        k_values = [1.0, 1.5, 2.0, 2.5]

    release_mask = df["date"].dt.date.isin(RELEASE_DATES)
    n_releases = release_mask.sum()

    print(f"\n{'='*70}")
    print("RELEASE DATE PREDICTION FROM FARS RESIDUALS")
    print(f"{'='*70}")
    print(f"Total days in sample: {len(df)}")
    print(f"Release dates found in sample: {n_releases} / {len(RELEASE_DATES)}")

    df_sorted = df.sort_values("z_score", ascending=False).reset_index(drop=True)
    df_sorted["rank"] = range(1, len(df_sorted) + 1)

    print("\n--- Ranks of release dates (by z-score, descending) ---")
    print(
        f"{'Artist':<22} {'Album':<32} {'Date':<12} {'Fatals':>7} {'z':>6} "
        f"{'Rank':>6} {'Pctile':>7}"
    )
    print("-" * 100)

    for artist, album, date_str, _ in ALBUMS:
        dt = pd.to_datetime(date_str).date()
        row = df_sorted[df_sorted["date"].dt.date == dt]
        if len(row) == 0:
            print(
                f"{artist:<22} {album:<32} {date_str:<12} {'N/A':>7} {'N/A':>6} "
                f"{'N/A':>6} {'N/A':>7}"
            )
        else:
            r = row.iloc[0]
            pctile = 100 * (1 - r["rank"] / len(df_sorted))
            print(
                f"{artist:<22} {album:<32} {date_str:<12} {r['fatalities']:>7.0f} "
                f"{r['z_score']:>6.2f} {r['rank']:>6.0f} {pctile:>6.1f}%"
            )

    print("\n--- Anomaly detection (z > k) ---")
    print(f"{'k':>5} {'Flagged':>8} {'TP':>5} {'Precision':>10} {'Recall':>8}")
    print("-" * 42)
    for k in k_values:
        flagged = df["z_score"] > k
        n_flagged = flagged.sum()
        tp = (flagged & release_mask).sum()
        prec = tp / n_flagged if n_flagged > 0 else 0
        rec = tp / n_releases if n_releases > 0 else 0
        print(f"{k:>5.1f} {n_flagged:>8} {tp:>5} {prec:>10.3f} {rec:>8.3f}")

    return df_sorted
