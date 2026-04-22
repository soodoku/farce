"""
Confounding Sensitivity — sensitivity to unobserved confounders.

Functions for assessing robustness to unmeasured confounding:
- Sensitivity analysis (E-value)
- Rosenbaum bounds
- Synthetic control methods
"""

import datetime
from math import log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from src.constants import ALBUMS_EXTENDED, ALBUMS_TIER1


def sensitivity_analysis(df_global, window=10):
    """
    Sensitivity Analysis / E-value (Hainmueller / VanderWeele).

    How strong must unobserved confounding be to explain away the effect?

    E-value: The minimum strength of association (risk ratio scale) that an
    unobserved confounder must have with BOTH the treatment and the outcome
    to fully explain away the observed effect.

    Also computes Rosenbaum-style Γ bounds.

    Output: tabs/t27_sensitivity.csv
    """
    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS / E-VALUE (Hainmueller/VanderWeele)")
    print(f"{'='*70}")
    print("How strong must unobserved confounding be to explain the effect?\n")

    tier1_resids = []
    for album_tuple in ALBUMS_TIER1:
        dt = pd.to_datetime(album_tuple[2])
        row = df_global[df_global["date"] == dt]
        if len(row) > 0:
            tier1_resids.append(row["resid_global"].values[0])

    observed_effect = np.mean(tier1_resids)
    se = np.std(tier1_resids) / sqrt(len(tier1_resids))

    baseline = df_global["fatalities"].mean()
    rr = (baseline + observed_effect) / baseline
    rr_lower = (baseline + observed_effect - 1.96 * se) / baseline

    print(f"Observed effect: {observed_effect:+.1f} deaths")
    print(f"Standard error: {se:.1f}")
    print(f"Baseline fatalities: {baseline:.1f}/day")
    print(f"Risk ratio (RR): {rr:.3f}")
    print(f"RR lower 95% CI: {rr_lower:.3f}")

    def compute_e_value(rr_val):
        """E-value formula: E = RR + sqrt(RR * (RR - 1))"""
        if rr_val <= 1:
            return 1.0
        return rr_val + sqrt(rr_val * (rr_val - 1))

    e_value = compute_e_value(rr)
    e_value_lower = compute_e_value(rr_lower)

    print(f"\nE-VALUE RESULTS:")
    print(f"  E-value for point estimate: {e_value:.3f}")
    print(f"  E-value for CI lower bound: {e_value_lower:.3f}")

    print("\nINTERPRETATION:")
    print(f"  An unobserved confounder would need to have:")
    print(f"    - RR ≥ {e_value:.2f} with the treatment (being a release day)")
    print(f"    - RR ≥ {e_value:.2f} with the outcome (fatalities)")
    print(f"  to fully explain away the observed effect.")

    if e_value < 2.0:
        print(
            f"\n  E-value {e_value:.2f} < 2.0: WEAK to moderate confounding could explain result"
        )
    elif e_value < 3.0:
        print(f"\n  E-value {e_value:.2f} in [2, 3]: Moderate confounding needed")
    else:
        print(
            f"\n  E-value {e_value:.2f} ≥ 3.0: Strong confounding needed to explain result"
        )

    print(f"\n{'='*70}")
    print("ROSENBAUM BOUNDS (Sensitivity to Hidden Bias)")
    print(f"{'='*70}")
    print("Testing: At what Γ (odds ratio of hidden bias) does the result become")
    print("non-significant?\n")

    release_dates = set()
    for a in ALBUMS_TIER1:
        release_dates.add(pd.to_datetime(a[2]).date())

    release_mask = df_global["date"].dt.date.isin(release_dates)
    n_treated = release_mask.sum()
    n_control = (~release_mask).sum()

    treatment_mean = df_global.loc[release_mask, "resid_global"].mean()
    control_mean = df_global.loc[~release_mask, "resid_global"].mean()
    pooled_std = df_global["resid_global"].std()

    effect_size = (treatment_mean - control_mean) / pooled_std

    gamma_values = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 2.5, 3.0]

    print(f"Standardized effect size: {effect_size:.3f}")
    print(f"\n{'Gamma':>8} | {'Upper bound p-value':>20}")
    print("-" * 35)

    for gamma in gamma_values:
        adj_effect = effect_size - log(gamma)
        z_stat = adj_effect * sqrt(n_treated)
        p_upper = 1 - stats.norm.cdf(z_stat)
        sig = " *" if p_upper < 0.1 else " **" if p_upper < 0.05 else ""
        print(f"{gamma:>8.1f} | {p_upper:>20.4f}{sig}")

    def find_critical_gamma(target_p=0.05):
        def objective(gamma):
            adj_effect = effect_size - log(gamma)
            z_stat = adj_effect * sqrt(n_treated)
            p_upper = 1 - stats.norm.cdf(z_stat)
            return abs(p_upper - target_p)

        result = minimize_scalar(objective, bounds=(1.0, 10.0), method="bounded")
        return result.x

    critical_gamma = find_critical_gamma(0.05)
    print(f"\nCritical Γ (where p = 0.05): {critical_gamma:.2f}")

    if critical_gamma < 1.5:
        print("  WARNING: Very weak hidden bias could nullify the result.")
    elif critical_gamma < 2.0:
        print("  Moderate hidden bias could nullify the result.")
    else:
        print("  Result is robust to moderate hidden bias.")

    results_df = pd.DataFrame(
        [
            {
                "observed_effect": observed_effect,
                "se": se,
                "rr": rr,
                "rr_lower": rr_lower,
                "e_value": e_value,
                "e_value_lower": e_value_lower,
                "effect_size_d": effect_size,
                "critical_gamma": critical_gamma,
                "n_treated": n_treated,
                "n_control": n_control,
            }
        ]
    )

    return results_df


def synthetic_control(df, albums=None, window=10, pre_periods=10, post_periods=5):
    """
    Synthetic Control Method (Hainmueller/Abadie).

    For each release date, construct a synthetic counterfactual from
    weighted combination of non-release days that match pre-release trends.

    Then compare actual vs synthetic fatalities on release day.

    Output: tabs/t31_synthetic_control.csv
    """
    print(f"\n{'='*70}")
    print("SYNTHETIC CONTROL METHOD (Hainmueller/Abadie)")
    print(f"{'='*70}")
    print(f"Building synthetic counterfactual for each release using")
    print(f"pre-release trajectory matching ({pre_periods} days pre-period).\n")

    if albums is None:
        albums = ALBUMS_TIER1

    all_release_dates = set()
    for a in ALBUMS_EXTENDED:
        all_release_dates.add(pd.to_datetime(a[2]).date())

    results = []

    for album_tuple in albums:
        artist, album, date_str = album_tuple[0], album_tuple[1], album_tuple[2]
        release_dt = pd.to_datetime(date_str)

        pre_start = release_dt - pd.Timedelta(days=pre_periods)
        post_end = release_dt + pd.Timedelta(days=post_periods)

        treatment_mask = (df["date"] >= pre_start) & (df["date"] <= post_end)
        treatment_df = df[treatment_mask].copy()

        if len(treatment_df) < pre_periods + post_periods:
            continue

        donor_mask = ~df["date"].dt.date.isin(all_release_dates)
        for offset in range(-window, window + 1):
            exclude_date = release_dt.date() + datetime.timedelta(days=offset)
            donor_mask &= df["date"].dt.date != exclude_date

        donor_df = df[donor_mask].copy()

        if len(donor_df) < 100:
            continue

        treatment_pre = treatment_df[treatment_df["date"] < release_dt][
            "fatalities"
        ].values
        if len(treatment_pre) < 5:
            continue

        dow_match = release_dt.dayofweek
        month_match = release_dt.month
        donor_fridays = donor_df[
            (donor_df["date"].dt.dayofweek == dow_match)
            & (abs(donor_df["date"].dt.month - month_match) <= 1)
        ]

        if len(donor_fridays) < 10:
            donor_fridays = donor_df[donor_df["date"].dt.dayofweek == dow_match]

        if len(donor_fridays) < 10:
            continue

        donor_trajectories = []
        for _, donor_row in donor_fridays.iterrows():
            donor_date = donor_row["date"]
            traj_start = donor_date - pd.Timedelta(days=pre_periods)
            traj_mask = (df["date"] >= traj_start) & (df["date"] < donor_date)
            traj = df[traj_mask]["fatalities"].values

            if len(traj) == len(treatment_pre):
                donor_trajectories.append(
                    {
                        "date": donor_date,
                        "pre_trajectory": traj,
                        "release_day_fatalities": donor_row["fatalities"],
                    }
                )

        if len(donor_trajectories) < 5:
            continue

        best_distance = float("inf")
        best_donor = None

        for donor in donor_trajectories:
            distance = np.sum((treatment_pre - donor["pre_trajectory"]) ** 2)
            if distance < best_distance:
                best_distance = distance
                best_donor = donor

        if best_donor is None:
            continue

        top_k = min(5, len(donor_trajectories))
        distances = [
            (d, np.sum((treatment_pre - d["pre_trajectory"]) ** 2))
            for d in donor_trajectories
        ]
        distances.sort(key=lambda x: x[1])
        top_donors = [d[0] for d in distances[:top_k]]

        inv_distances = [1 / (d[1] + 1e-6) for d in distances[:top_k]]
        weights = [w / sum(inv_distances) for w in inv_distances]

        synthetic_value = sum(
            w * d["release_day_fatalities"] for w, d in zip(weights, top_donors)
        )

        actual_value = treatment_df[treatment_df["date"] == release_dt][
            "fatalities"
        ].values
        if len(actual_value) == 0:
            continue
        actual_value = actual_value[0]

        effect = actual_value - synthetic_value

        results.append(
            {
                "artist": artist,
                "album": album,
                "date": date_str,
                "actual": actual_value,
                "synthetic": synthetic_value,
                "effect": effect,
                "n_donors": len(top_donors),
                "pre_match_mse": best_distance / len(treatment_pre),
            }
        )

    if not results:
        print("No synthetic control estimates could be computed.")
        return None

    results_df = pd.DataFrame(results)

    print(f"{'Album':<35} | {'Actual':>8} | {'Synth':>8} | {'Effect':>10} | {'MSE':>8}")
    print("-" * 80)
    for _, r in results_df.iterrows():
        print(
            f"{r['artist'][:15] + ' - ' + r['album'][:17]:<35} | "
            f"{r['actual']:>8.0f} | {r['synthetic']:>8.1f} | {r['effect']:>+10.1f} | {r['pre_match_mse']:>8.1f}"
        )

    avg_effect = results_df["effect"].mean()
    se_effect = results_df["effect"].std() / sqrt(len(results_df))
    t_stat = avg_effect / se_effect if se_effect > 0 else 0

    print(f"\nSYNTHETIC CONTROL SUMMARY:")
    print(f"  N albums with valid synthetic: {len(results_df)}")
    print(f"  Average effect: {avg_effect:+.1f} deaths")
    print(f"  SE: {se_effect:.1f}")
    print(f"  t-stat: {t_stat:.2f}")

    print("\nPLACEBO IN SPACE TEST:")
    print("  Running synthetic control on random non-release Fridays...")

    donor_mask = ~df["date"].dt.date.isin(all_release_dates)
    placebo_fridays = df[donor_mask & (df["date"].dt.dayofweek == 4)].sample(
        n=min(50, len(df[donor_mask & (df["date"].dt.dayofweek == 4)])), random_state=42
    )

    placebo_effects = []
    for _, row in placebo_fridays.iterrows():
        placebo_dt = row["date"]
        pre_start = placebo_dt - pd.Timedelta(days=pre_periods)
        placebo_mask = (df["date"] >= pre_start) & (df["date"] < placebo_dt)
        placebo_pre = df[placebo_mask]["fatalities"].values

        if len(placebo_pre) < 5:
            continue

        placebo_actual = row["fatalities"]

        other_fridays = df[
            (df["date"].dt.dayofweek == 4)
            & (df["date"] != placebo_dt)
            & ~df["date"].dt.date.isin(all_release_dates)
        ]

        if len(other_fridays) < 10:
            continue

        placebo_synth = other_fridays["fatalities"].mean()
        placebo_effects.append(placebo_actual - placebo_synth)

    if placebo_effects:
        placebo_mean = np.mean(placebo_effects)
        placebo_std = np.std(placebo_effects)
        p_value = (np.array(placebo_effects) >= avg_effect).mean()

        print(f"  Placebo effect distribution:")
        print(f"    Mean: {placebo_mean:+.1f}")
        print(f"    SD: {placebo_std:.1f}")
        print(f"  p-value (actual >= placebo): {p_value:.4f}")

        results_df["placebo_mean"] = placebo_mean
        results_df["placebo_sd"] = placebo_std
        results_df["placebo_p_value"] = p_value

    return results_df
