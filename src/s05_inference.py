"""
Valid Inference — proper p-values, SEs, and shrinkage estimators.

Functions for valid statistical inference:
- Leave-one-out (jackknife) analysis
- Randomization inference (various strategies)
- Studentized randomization inference
- Multiple testing correction
- Hierarchical model with shrinkage
"""

from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import (ALBUMS, ALBUMS_EXTENDED, ALBUMS_TIER1, ALBUMS_TIER2,
                           RELEASE_DATES)


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
    print(
        f"  Effective comparison: 9 treated Fridays vs {n_fridays - 9} control Fridays"
    )

    print(f"\n── Strategy 4: Block Bootstrap ({block_size}-day blocks) ──")
    print("  Preserves temporal autocorrelation in fatality data")

    n_days = len(df_global)
    n_blocks = n_days // block_size
    block_starts = np.arange(0, n_blocks * block_size, block_size)

    null_avgs_block = np.zeros(n_sims)
    for s in range(n_sims):
        sampled_blocks = rng.choice(
            block_starts, size=10 // block_size + 1, replace=True
        )
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


def _cluster_se(resids, clusters, X):
    """
    Compute cluster-robust standard errors.

    Uses the HC1 adjustment for finite sample.
    """
    n = len(resids)
    k = X.shape[1]
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))

    for c in unique_clusters:
        mask = clusters == c
        X_c = X[mask]
        e_c = resids[mask]
        score = X_c.T @ e_c
        meat += np.outer(score, score)

    sandwich = XtX_inv @ meat @ XtX_inv
    dof_adj = (G / (G - 1)) * ((n - 1) / (n - k))
    var_beta = dof_adj * sandwich

    return np.sqrt(np.diag(var_beta))


def studentized_randomization_inference(df_global, n_sims=10000, seed=42):
    """
    Studentized Randomization Inference (Green).

    Use t-statistics instead of raw means. This accounts for varying
    standard errors across permutations and is more powerful.

    Also implements cluster-robust SEs at the week level.

    Output: tabs/t26_studentized_ri.csv
    """
    print(f"\n{'='*70}")
    print("STUDENTIZED RANDOMIZATION INFERENCE (Green)")
    print(f"{'='*70}")
    print("Using t-statistics instead of raw means for sharper inference.")
    print("Also clustering at week level to account for autocorrelation.\n")

    rng = np.random.RandomState(seed)

    release_dates = set()
    for a in ALBUMS_TIER1:
        release_dates.add(pd.to_datetime(a[2]).date())

    release_mask = df_global["date"].dt.date.isin(release_dates)

    release_resids = df_global.loc[release_mask, "resid_global"].values
    n_release = len(release_resids)

    actual_mean = np.mean(release_resids)
    actual_se = np.std(release_resids) / sqrt(n_release)
    actual_t = actual_mean / actual_se if actual_se > 0 else 0

    print(f"Observed statistics (N={n_release} release days):")
    print(f"  Mean residual: {actual_mean:+.1f}")
    print(f"  SE: {actual_se:.1f}")
    print(f"  t-statistic: {actual_t:.2f}")

    all_resids = df_global["resid_global"].values
    n_total = len(all_resids)

    print(f"\nRunning {n_sims:,} permutations...")

    null_means = np.zeros(n_sims)
    null_t_stats = np.zeros(n_sims)

    for s in range(n_sims):
        idx = rng.choice(n_total, size=n_release, replace=False)
        sample = all_resids[idx]
        null_means[s] = np.mean(sample)
        se = np.std(sample) / sqrt(n_release)
        null_t_stats[s] = null_means[s] / se if se > 0 else 0

    p_mean = (null_means >= actual_mean).mean()
    p_t = (null_t_stats >= actual_t).mean()

    print("\n── Non-studentized (raw means) ──")
    print(f"  p-value: {p_mean:.4f}")
    print(f"  Null mean: {null_means.mean():+.1f}")
    print(f"  Null SD: {null_means.std():.1f}")

    print("\n── Studentized (t-statistics) ──")
    print(f"  p-value: {p_t:.4f}")
    print(f"  Null mean t: {null_t_stats.mean():+.2f}")
    print(f"  Null SD t: {null_t_stats.std():.2f}")

    df_global["week_id"] = (df_global["date"] - df_global["date"].min()).dt.days // 7
    weeks = df_global["week_id"].values

    release_weeks = df_global.loc[release_mask, "week_id"].values
    n_weeks_treated = len(np.unique(release_weeks))

    all_weeks = np.unique(weeks)
    n_all_weeks = len(all_weeks)

    null_cluster_t = np.zeros(n_sims)

    for s in range(n_sims):
        sampled_weeks = rng.choice(all_weeks, size=n_weeks_treated, replace=False)
        sample_mask = np.isin(weeks, sampled_weeks)
        sample_resids = all_resids[sample_mask]

        if len(sample_resids) == 0:
            continue

        sample_mean = np.mean(sample_resids)
        sample_se = np.std(sample_resids) / sqrt(len(sample_resids))
        null_cluster_t[s] = sample_mean / sample_se if sample_se > 0 else 0

    p_cluster = (null_cluster_t >= actual_t).mean()

    print("\n── Week-clustered RI ──")
    print(f"  Number of treated weeks: {n_weeks_treated}")
    print(f"  Total weeks: {n_all_weeks}")
    print(f"  Cluster p-value: {p_cluster:.4f}")

    results = pd.DataFrame(
        [
            {
                "method": "Non-studentized (means)",
                "p_value": p_mean,
                "test_stat": actual_mean,
            },
            {"method": "Studentized (t-stat)", "p_value": p_t, "test_stat": actual_t},
            {"method": "Week-clustered", "p_value": p_cluster, "test_stat": actual_t},
        ]
    )

    print("\nINTERPRETATION:")
    if p_cluster > p_t * 1.5:
        print(f"  Cluster p-value ({p_cluster:.4f}) >> studentized p-value ({p_t:.4f})")
        print("  Autocorrelation inflates significance. Be cautious.")
    else:
        print("  Results robust to clustering.")

    return results


def multiple_testing_correction(p_values_dict):
    """
    Multiple Testing Correction (Green).

    Apply Bonferroni and Benjamini-Hochberg corrections to all tests.

    Output: tabs/t25_multiple_testing.csv
    """
    print(f"\n{'='*70}")
    print("MULTIPLE TESTING CORRECTION (Green)")
    print(f"{'='*70}")
    print("Correcting for multiple hypothesis tests.\n")

    tests = list(p_values_dict.items())
    n_tests = len(tests)
    p_values = np.array([t[1] for t in tests])

    bonferroni_adj = np.minimum(p_values * n_tests, 1.0)

    sorted_idx = np.argsort(p_values)
    bh_adj = np.zeros(n_tests)
    for i, idx in enumerate(sorted_idx):
        rank = i + 1
        bh_adj[idx] = min(p_values[idx] * n_tests / rank, 1.0)

    for i in range(n_tests - 2, -1, -1):
        idx = sorted_idx[i]
        next_idx = sorted_idx[i + 1]
        bh_adj[idx] = min(bh_adj[idx], bh_adj[next_idx])

    results = []
    print(f"{'Test':<40} | {'Raw p':>10} | {'Bonferroni':>12} | {'BH':>10}")
    print("-" * 80)

    for idx, (test_name, p_raw) in enumerate(tests):
        results.append(
            {
                "test": test_name,
                "p_raw": p_raw,
                "p_bonferroni": bonferroni_adj[idx],
                "p_bh": bh_adj[idx],
                "significant_raw": p_raw < 0.05,
                "significant_bonf": bonferroni_adj[idx] < 0.05,
                "significant_bh": bh_adj[idx] < 0.05,
            }
        )

        raw_sig = "*" if p_raw < 0.05 else ""
        bonf_sig = "*" if bonferroni_adj[idx] < 0.05 else ""
        bh_sig = "*" if bh_adj[idx] < 0.05 else ""
        print(
            f"{test_name:<40} | {p_raw:>10.4f}{raw_sig} | {bonferroni_adj[idx]:>11.4f}{bonf_sig} | {bh_adj[idx]:>9.4f}{bh_sig}"
        )

    results_df = pd.DataFrame(results)

    print(f"\nSUMMARY:")
    print(f"  Total tests: {n_tests}")
    print(f"  Significant at raw p<0.05: {results_df['significant_raw'].sum()}")
    print(f"  Significant after Bonferroni: {results_df['significant_bonf'].sum()}")
    print(f"  Significant after BH: {results_df['significant_bh'].sum()}")

    return results_df


def hierarchical_model(df_global, albums=None):
    """
    Hierarchical Model with Shrinkage (Gelman).

    Each album is a "unit" with its own effect. Pool toward grand mean
    using empirical Bayes shrinkage.

    Model: effect_i ~ N(μ, τ² + σ²)
    Shrinkage: effect_i_shrunk = μ + (τ²/(τ²+σ²)) * (effect_i - μ)

    Output: tabs/t30_hierarchical.csv
    """
    print(f"\n{'='*70}")
    print("HIERARCHICAL MODEL WITH SHRINKAGE (Gelman)")
    print(f"{'='*70}")
    print("Each album has its own effect. Pool toward grand mean.\n")

    if albums is None:
        albums = ALBUMS_TIER1

    effects = []
    album_info = []

    for album_tuple in albums:
        artist, album, date_str = album_tuple[0], album_tuple[1], album_tuple[2]
        dt = pd.to_datetime(date_str)
        row = df_global[df_global["date"] == dt]

        if len(row) > 0:
            effect = row["resid_global"].values[0]
            effects.append(effect)
            album_info.append(
                {
                    "artist": artist,
                    "album": album,
                    "date": date_str,
                    "effect_raw": effect,
                }
            )

    effects = np.array(effects)
    n = len(effects)

    grand_mean = np.mean(effects)
    total_var = np.var(effects, ddof=1)
    resid_sd = df_global["resid_global"].std()
    within_var = resid_sd**2

    between_var = max(total_var - within_var / n, 0)

    shrinkage_factor = (
        between_var / (between_var + within_var)
        if (between_var + within_var) > 0
        else 0
    )

    for i, info in enumerate(album_info):
        raw = info["effect_raw"]
        shrunk = grand_mean + shrinkage_factor * (raw - grand_mean)
        info["effect_shrunk"] = shrunk
        info["shrinkage_amount"] = raw - shrunk

    results_df = pd.DataFrame(album_info)

    print(f"{'Album':<40} | {'Raw':>10} | {'Shrunk':>10} | {'Shrinkage':>10}")
    print("-" * 80)
    for _, r in results_df.iterrows():
        print(
            f"{r['artist'][:18] + ' - ' + r['album'][:18]:<40} | "
            f"{r['effect_raw']:>+10.1f} | {r['effect_shrunk']:>+10.1f} | {r['shrinkage_amount']:>+10.1f}"
        )

    print(f"\nSHRINKAGE PARAMETERS:")
    print(f"  Grand mean (μ): {grand_mean:+.1f}")
    print(f"  Between-album variance (τ²): {between_var:.1f}")
    print(f"  Within-album variance (σ²): {within_var:.1f}")
    print(f"  Shrinkage factor: {shrinkage_factor:.3f}")

    print(f"\nSUMMARY:")
    print(f"  Raw average effect: {effects.mean():+.1f}")
    print(f"  Shrunk average effect: {results_df['effect_shrunk'].mean():+.1f}")
    print(
        f"  Shrinkage reduced estimate by: {effects.mean() - results_df['effect_shrunk'].mean():.1f}"
    )

    if shrinkage_factor < 0.5:
        print(f"\n  INTERPRETATION: Shrinkage factor ({shrinkage_factor:.2f}) < 0.5")
        print(
            "  More than half the variance is noise. True effects may be much smaller."
        )

    return results_df
