"""
Analysis functions for FARS album release study.

Includes local/global counterfactual estimation, randomization inference,
leave-one-out analysis, and dose-response analysis.
"""

import datetime

import numpy as np
import pandas as pd

from src.constants import (
    ALBUMS,
    ALBUMS_ALL,
    ALBUMS_TIER1,
    ALBUMS_TIER2,
    RELEASE_DATES,
    RELEASE_DATES_ALL,
    us_holidays,
)
from src.s02_preprocess import _build_design


def _ols_with_se(X, y):
    """OLS with standard errors (handles multicollinearity via ridge regularization)."""
    n, k = X.shape
    XtX = X.T @ X
    Xty = X.T @ y

    ridge_lambda = 1e-8
    XtX_reg = XtX + ridge_lambda * np.eye(k)
    beta = np.linalg.solve(XtX_reg, Xty)

    fitted = X @ beta
    resid = y - fitted

    XtX_inv = np.linalg.inv(XtX_reg)
    sigma2 = np.sum(resid**2) / max(n - k, 1)
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0))

    return beta, se, fitted, resid


def _ols_residuals(X, y):
    """OLS via normal equations, return (beta, fitted, residuals)."""
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    fitted = X @ beta
    return beta, fitted, y - fitted


def local_estimate(df, window=10):
    """
    Paper's approach: for each release date, compare fatalities on release day
    to the average of the ±window surrounding days.
    Returns per-event deltas and the pooled estimate.
    """
    results = []
    for artist, album, date_str, dow in ALBUMS:
        dt = pd.to_datetime(date_str)
        release_row = df[df["date"] == dt]
        if len(release_row) == 0:
            continue

        y_release = release_row["fatalities"].values[0]

        # Control window: ±window days, excluding release day itself
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

    df["dow"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year

    holidays = us_holidays(df["year"].unique())
    df["holiday"] = df["date"].dt.date.isin(holidays).astype(int)

    album_dfs = []
    for a in albums:
        dt = pd.to_datetime(a[2])
        for offset in range(-window, window + 1):
            day = dt + pd.Timedelta(days=offset)
            row = df[df["date"] == day]
            if len(row) == 0:
                continue

            album_dfs.append({
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
            })

    reg_df = pd.DataFrame(album_dfs)

    reg_df["release_day"] = (reg_df["day_relative"] == 0).astype(int)

    dow_dummies = pd.get_dummies(reg_df["dow"], prefix="dow", drop_first=True, dtype=float)
    week_dummies = pd.get_dummies(reg_df["week_of_year"], prefix="week", drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(reg_df["year"], prefix="year", drop_first=True, dtype=float)

    X = pd.concat([
        dow_dummies,
        week_dummies,
        year_dummies,
    ], axis=1)
    X["holiday"] = reg_df["holiday"].values
    X["release_day"] = reg_df["release_day"].values
    X["const"] = 1.0

    y = reg_df["fatalities"].values.astype(float)

    beta, se, fitted, resid = _ols_with_se(X.values, y)

    release_day_idx = list(X.columns).index("release_day")
    treatment_effect = beta[release_day_idx]
    treatment_se = se[release_day_idx]

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
        y_control = control_rows["fatalities"].mean() if len(control_rows) > 0 else np.nan

        per_album_effects.append({
            "artist": a[0],
            "album": a[1],
            "date": a[2],
            "y_release": y_release,
            "y_control": y_control,
            "delta_raw": y_release - y_control if not np.isnan(y_control) else np.nan,
        })

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

    # Identify donut exclusion zone
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

    # Fit on estimation sample
    X_est = _build_design(df[estimation_mask])
    y_est = df.loc[estimation_mask, "fatalities"].values.astype(float)
    beta, _, _ = _ols_residuals(X_est.values, y_est)

    # Predict for ALL days (including release days)
    X_all = _build_design(df)
    fitted_all = X_all.values @ beta
    df["fitted_global"] = fitted_all
    df["resid_global"] = df["fatalities"].values - fitted_all

    # Use estimation-sample residual SD for z-scores
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

    # Local estimates (paper's approach)
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

    # Global estimates
    # (a) Full-sample global
    df_g, _, lbl_g = global_estimate(df, donut_window=None)
    # (b) Donut global (exclude ±10 days around releases)
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

    # Decomposition
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

    # Averages
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


def leave_one_out(local_df):
    """Jackknife: drop each album and report the leave-one-out pooled estimate."""
    print(f"\n{'='*70}")
    print("LEAVE-ONE-OUT (JACKKNIFE) ANALYSIS")
    print(f"{'='*70}")

    n = len(local_df)
    avg_all = local_df["delta_local"].mean()

    print(f"\n  Full-sample pooled δ: {avg_all:+.1f}")
    print(f"\n{'Album dropped':<35} {'δ_i':>7} {'LOO avg':>8} {'Influence':>10}")
    print("-" * 65)

    loo_avgs = []
    for _, r in local_df.iterrows():
        loo_avg = (n * avg_all - r["delta_local"]) / (n - 1)
        inf = avg_all - loo_avg
        loo_avgs.append(loo_avg)
        print(
            f"{r['artist'] + ' - ' + r['album'][:18]:<35} "
            f"{r['delta_local']:>7.1f} {loo_avg:>8.1f} {inf:>10.1f}"
        )

    loo_avgs = np.array(loo_avgs)
    jackknife_var = ((n - 1) / n) * np.sum((loo_avgs - loo_avgs.mean()) ** 2)
    jackknife_se = np.sqrt(jackknife_var)

    print(f"\n  Jackknife SE:  {jackknife_se:.1f}")
    print(
        f"  Jackknife 95% CI: [{avg_all - 1.96*jackknife_se:+.1f}, "
        f"{avg_all + 1.96*jackknife_se:+.1f}]"
    )
    print(f"  t-stat (jackknife): {avg_all / jackknife_se:.2f}")

    n_neg = (local_df["delta_local"] < 0).sum()
    print(f"\n  Albums with δ_i < 0: {n_neg} / {n}")

    return jackknife_se


def randomization_inference(df_global, n_sims=10000, seed=42, block_size=7):
    """
    Exact randomization inference using the global residual distribution.
    Draw random sets of 10 days (or 9 Fridays + 1 Sunday) and compare their
    average residual to the actual release dates.

    Now includes block bootstrap option to account for autocorrelation.
    """
    rng = np.random.RandomState(seed)

    print(f"\n{'='*70}")
    print(f"RANDOMIZATION INFERENCE (global residuals, {n_sims:,} draws)")
    print(f"{'='*70}")

    release_mask = df_global["date"].dt.date.isin(RELEASE_DATES)
    actual_avg_resid = df_global.loc[release_mask, "resid_global"].mean()

    # Strategy 1: Draw 10 random days from all days
    all_indices = df_global.index.values
    null_avgs_all = np.zeros(n_sims)
    for s in range(n_sims):
        idx = rng.choice(all_indices, size=10, replace=False)
        null_avgs_all[s] = df_global.loc[idx, "resid_global"].mean()

    p_all = (null_avgs_all >= actual_avg_resid).mean()

    print("\n── Strategy 1: 10 random days from full sample ──")
    print(f"  Actual avg residual on release days: {actual_avg_resid:+.1f}")
    print(
        f"  Null distribution: mean={null_avgs_all.mean():+.1f}, "
        f"SD={null_avgs_all.std():.1f}"
    )
    print(f"  p-value (one-sided): {p_all:.4f}")
    print(f"  Null 95th percentile: {np.percentile(null_avgs_all, 95):+.1f}")
    print(f"  Null 99th percentile: {np.percentile(null_avgs_all, 99):+.1f}")

    # Strategy 2: Draw 9 random Fridays + 1 random Sunday
    fri_mask = df_global["date"].dt.dayofweek == 4
    sun_mask = df_global["date"].dt.dayofweek == 6
    fri_pool = df_global[fri_mask & ~release_mask].index.values
    sun_pool = df_global[sun_mask & ~release_mask].index.values

    null_avgs_fri = np.zeros(n_sims)
    for s in range(n_sims):
        fri_idx = rng.choice(fri_pool, size=9, replace=False)
        sun_idx = rng.choice(sun_pool, size=1, replace=False)
        idx = np.concatenate([fri_idx, sun_idx])
        null_avgs_fri[s] = df_global.loc[idx, "resid_global"].mean()

    p_fri = (null_avgs_fri >= actual_avg_resid).mean()

    print("\n── Strategy 2: 9 random Fridays + 1 random Sunday ──")
    print(f"  Actual avg residual on release days: {actual_avg_resid:+.1f}")
    print(
        f"  Null distribution: mean={null_avgs_fri.mean():+.1f}, "
        f"SD={null_avgs_fri.std():.1f}"
    )
    print(f"  p-value (one-sided): {p_fri:.4f}")
    print(f"  Null 95th percentile: {np.percentile(null_avgs_fri, 95):+.1f}")
    print(f"  Null 99th percentile: {np.percentile(null_avgs_fri, 99):+.1f}")

    # Strategy 3: Friday-only analysis (drop Donda, use 9 releases)
    fri_releases = [pd.to_datetime(a[2]).date() for a in ALBUMS if a[3] == "Friday"]
    fri_release_mask = df_global["date"].dt.date.isin(set(fri_releases))
    actual_avg_fri_only = df_global.loc[fri_release_mask, "resid_global"].mean()

    null_avgs_fri9 = np.zeros(n_sims)
    for s in range(n_sims):
        idx = rng.choice(fri_pool, size=9, replace=False)
        null_avgs_fri9[s] = df_global.loc[idx, "resid_global"].mean()

    p_fri9 = (null_avgs_fri9 >= actual_avg_fri_only).mean()

    print("\n── Strategy 3: 9 random Fridays (Donda excluded) ──")
    print(f"  Actual avg residual (9 Friday releases): {actual_avg_fri_only:+.1f}")
    print(
        f"  Null distribution: mean={null_avgs_fri9.mean():+.1f}, "
        f"SD={null_avgs_fri9.std():.1f}"
    )
    print(f"  p-value (one-sided): {p_fri9:.4f}")

    n_fridays = len(fri_pool) + fri_release_mask.sum()
    print(f"\n  Total Fridays in sample: {n_fridays}")
    print(f"  Effective comparison: 9 treated Fridays vs {n_fridays - 9} control Fridays")

    # Strategy 4: Block bootstrap to account for autocorrelation
    print(f"\n── Strategy 4: Block Bootstrap ({block_size}-day blocks) ──")
    print("  Preserves temporal autocorrelation in fatality data")

    n_days = len(df_global)
    n_blocks = n_days // block_size
    block_starts = np.arange(0, n_blocks * block_size, block_size)

    null_avgs_block = np.zeros(n_sims)
    for s in range(n_sims):
        sampled_blocks = rng.choice(block_starts, size=10 // block_size + 1, replace=True)
        sampled_days = []
        for bs in sampled_blocks:
            sampled_days.extend(range(bs, min(bs + block_size, n_days)))
        sampled_days = sampled_days[:10]
        null_avgs_block[s] = df_global.iloc[sampled_days]["resid_global"].mean()

    p_block = (null_avgs_block >= actual_avg_resid).mean()
    print(f"  Actual avg residual: {actual_avg_resid:+.1f}")
    print(
        f"  Block null distribution: mean={null_avgs_block.mean():+.1f}, "
        f"SD={null_avgs_block.std():.1f}"
    )
    print(f"  Block bootstrap p-value: {p_block:.4f}")
    print(f"  Compare to iid p-value: {p_all:.4f}")
    if p_block > p_all * 1.5:
        print("  WARNING: Block p-value substantially larger than iid p-value,")
        print("           suggesting autocorrelation inflates significance.")

    return {
        "p_all": p_all,
        "p_fri_sun": p_fri,
        "p_fri_only": p_fri9,
        "p_block": p_block,
        "null_all": null_avgs_all,
        "null_fri": null_avgs_fri,
        "null_block": null_avgs_block,
    }


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

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    holidays = us_holidays(df["year"].unique())
    df["holiday"] = df["date"].dt.date.isin(holidays).astype(int)
    hol_adj = set()
    for h in holidays:
        hol_adj.add(h - datetime.timedelta(1))
        hol_adj.add(h + datetime.timedelta(1))
    df["holiday_adj"] = df["date"].dt.date.isin(hol_adj).astype(int)

    est_mask = ~df["date"].dt.date.isin(all_dates_exclude)
    X_est = _build_design(df[est_mask])
    y_est = df.loc[est_mask, "fatalities"].values.astype(float)
    beta, _, _ = _ols_residuals(X_est.values, y_est)

    X_all = _build_design(df)
    df["fitted_dr"] = X_all.values @ beta
    df["resid_dr"] = df["fatalities"].values - df["fitted_dr"]

    streams = []
    deltas = []
    album_names = []

    print(
        f"\n{'Album':<40} {'Streams (M)':>12} {'δ (deaths)':>12} {'z-score':>10}"
    )
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

    from scipy import stats

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

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    holidays = us_holidays(df["year"].unique())
    df["holiday"] = df["date"].dt.date.isin(holidays).astype(int)
    hol_adj = set()
    for h in holidays:
        hol_adj.add(h - datetime.timedelta(1))
        hol_adj.add(h + datetime.timedelta(1))
    df["holiday_adj"] = df["date"].dt.date.isin(hol_adj).astype(int)

    est_mask = ~df["date"].dt.date.isin(all_dates_exclude)
    X_est = _build_design(df[est_mask])
    y_est = df.loc[est_mask, "fatalities"].values.astype(float)
    beta, _, _ = _ols_residuals(X_est.values, y_est)

    X_all = _build_design(df)
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
        print(
            f"  {a[0]:<22} {a[1]:<30} δ={delta:+6.1f}  (~{a[4]:.0f}M streams)"
        )

    avg_t1 = np.mean(deltas_t1)
    se_t1 = np.std(deltas_t1) / np.sqrt(len(deltas_t1))
    print(
        f"  Avg δ (Tier 1): {avg_t1:+.1f} (SE={se_t1:.1f}), "
        f"z={avg_t1/resid_sd:.2f}"
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
        print(
            f"  {a[0]:<22} {a[1]:<30} δ={delta:+6.1f}  (~{a[4]:.0f}M streams)"
        )

    avg_t2 = np.mean(deltas_t2)
    se_t2 = np.std(deltas_t2) / np.sqrt(len(deltas_t2))
    print(
        f"  Avg δ (Tier 2): {avg_t2:+.1f} (SE={se_t2:.1f}), "
        f"z={avg_t2/resid_sd:.2f}"
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
