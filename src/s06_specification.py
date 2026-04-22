"""
Specification Robustness — sensitivity to analytical choices.

Functions for testing robustness to specification choices:
- Window sensitivity analysis
- Forecast-based estimation (Gary King style)
- Weather-controlled models
- Multiverse analysis
"""

import datetime
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import (ALBUMS, ALBUMS_ALL, ALBUMS_EXTENDED, ALBUMS_TIER1,
                           ALBUMS_TIER2, ALBUMS_TIER3)
from src.utils import add_time_features, build_design_matrix, ols_fit, save_table


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

    df = df.copy()
    if "dow" not in df.columns or "holiday" not in df.columns:
        df = add_time_features(df)

    results = []

    for w in windows:
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

        exclude = set()
        for _, _, date_str, _ in ALBUMS:
            dt = pd.to_datetime(date_str).date()
            for offset in range(-w, w + 1):
                exclude.add(dt + datetime.timedelta(days=offset))

        est_mask = ~df["date"].dt.date.isin(exclude)
        X_est = build_design_matrix(df[est_mask])
        y_est = df.loc[est_mask, "fatalities"].values.astype(float)
        beta, _, _ = ols_fit(X_est.values, y_est)

        X_all = build_design_matrix(df)
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

        results.append(
            {
                "window": w,
                "local_delta": avg_local,
                "global_delta": avg_global,
                "se": se_global,
                "t_stat": t_stat,
            }
        )

    results_df = pd.DataFrame(results)

    print(
        f"{'Window':<8} | {'Local δ':>9} | {'Global δ':>10} | {'SE':>7} | {'t-stat':>7}"
    )
    print("-" * 50)
    for _, r in results_df.iterrows():
        marker = " (current)" if r["window"] == 10 else ""
        print(
            f"±{int(r['window']):<6} | {r['local_delta']:>+9.1f} | "
            f"{r['global_delta']:>+10.1f} | {r['se']:>7.1f} | {r['t_stat']:>7.2f}{marker}"
        )

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


def build_features(df):
    """Build feature matrix for forecasting model."""
    df = add_time_features(df)

    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(
        df["month"], prefix="month", drop_first=True, dtype=float
    )
    year_dummies = pd.get_dummies(
        df["year"], prefix="year", drop_first=True, dtype=float
    )

    X = pd.concat([dow_dummies, month_dummies, year_dummies], axis=1)
    X["holiday"] = df["holiday"].values
    X["holiday_adj"] = df["holiday_adj"].values

    predictor_cols = [
        "pct_dark",
        "pct_rural",
        "pct_bad_weather",
        "pct_night",
        "pct_alcohol",
    ]
    for col in predictor_cols:
        if col in df.columns:
            X[col] = df[col].fillna(df[col].median()).values

    return X


def get_exclusion_mask(df, albums, window):
    """Create mask for days to exclude (within ±window of any release)."""
    exclude_dates = set()
    for a in albums:
        dt = pd.to_datetime(a[2]).date()
        for offset in range(-window, window + 1):
            exclude_dates.add(dt + datetime.timedelta(days=offset))
    return df["date"].dt.date.isin(exclude_dates)


def forecast_estimate(df, window=10, model_type="ridge", cv_folds=5, albums=None):
    """
    Forecast-based causal effect estimation.

    1. Train model on non-treatment days (excluding ±window around releases)
    2. Predict release-day fatalities
    3. Effect = actual - predicted
    4. Use CV to estimate prediction uncertainty

    Parameters
    ----------
    df : DataFrame
        Daily fatality data with 'date' and 'fatalities' columns
    window : int
        Days to exclude around each release date
    model_type : str
        'ridge' or 'gbm' (gradient boosting)
    cv_folds : int
        Number of cross-validation folds for uncertainty estimation
    albums : list
        List of albums to analyze (default: ALBUMS_ALL)

    Returns
    -------
    results : dict
        Per-album effects, pooled estimates, prediction intervals
    """
    if albums is None:
        albums = ALBUMS_ALL

    df = df.copy()

    X = build_features(df)
    y = df["fatalities"].values.astype(float)

    exclusion_mask = get_exclusion_mask(df, albums, window)
    train_mask = ~exclusion_mask

    X_train = X[train_mask].values
    y_train = y[train_mask]

    if model_type == "ridge":
        from sklearn.linear_model import RidgeCV

        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=cv_folds)
    elif model_type == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)

    y_pred_all = model.predict(X.values)
    df["y_pred"] = y_pred_all
    df["resid_forecast"] = y - y_pred_all

    train_resid = y_train - model.predict(X_train)
    resid_sd = train_resid.std()

    from sklearn.model_selection import cross_val_predict

    y_cv = cross_val_predict(model, X_train, y_train, cv=cv_folds)
    cv_resid = y_train - y_cv
    cv_rmse = np.sqrt(np.mean(cv_resid**2))

    results_list = []
    for a in albums:
        dt = pd.to_datetime(a[2])
        row = df[df["date"] == dt]
        if len(row) == 0:
            continue

        y_actual = row["fatalities"].values[0]
        y_predicted = row["y_pred"].values[0]
        effect = y_actual - y_predicted
        z_score = effect / resid_sd

        if a in ALBUMS_TIER1:
            tier = 1
        elif a in ALBUMS_TIER2:
            tier = 2
        else:
            tier = 3

        results_list.append(
            {
                "artist": a[0],
                "album": a[1],
                "date": a[2],
                "streams_millions": a[4],
                "tier": tier,
                "y_actual": y_actual,
                "y_predicted": y_predicted,
                "effect": effect,
                "z_score": z_score,
                "pred_ci_lower": y_predicted - 1.96 * cv_rmse,
                "pred_ci_upper": y_predicted + 1.96 * cv_rmse,
            }
        )

    results_df = pd.DataFrame(results_list)

    tier1_effects = results_df[results_df["tier"] == 1]["effect"].values
    tier2_effects = results_df[results_df["tier"] == 2]["effect"].values
    all_effects = results_df["effect"].values

    pooled_all = np.mean(all_effects)
    pooled_se_all = np.std(all_effects) / np.sqrt(len(all_effects))

    pooled_t1 = np.mean(tier1_effects)
    pooled_se_t1 = np.std(tier1_effects) / np.sqrt(len(tier1_effects))

    pooled_t2 = np.mean(tier2_effects)
    pooled_se_t2 = np.std(tier2_effects) / np.sqrt(len(tier2_effects))

    streams = results_df["streams_millions"].values
    effects = results_df["effect"].values
    r_pearson, p_pearson = stats.pearsonr(streams, effects)

    return {
        "results_df": results_df,
        "model": model,
        "model_type": model_type,
        "cv_rmse": cv_rmse,
        "resid_sd": resid_sd,
        "pooled_all": pooled_all,
        "pooled_se_all": pooled_se_all,
        "pooled_t1": pooled_t1,
        "pooled_se_t1": pooled_se_t1,
        "pooled_t2": pooled_t2,
        "pooled_se_t2": pooled_se_t2,
        "r_pearson": r_pearson,
        "p_pearson": p_pearson,
        "n_train": train_mask.sum(),
        "n_excluded": exclusion_mask.sum(),
    }


def print_forecast_results(results):
    """Print formatted forecast estimation results."""
    print(f"\n{'='*70}")
    print("FORECAST-BASED CAUSAL EFFECT ESTIMATION")
    print(f"{'='*70}")

    print(f"\nModel: {results['model_type'].upper()}")
    print(f"Training days: {results['n_train']}")
    print(f"Excluded days (±window around releases): {results['n_excluded']}")
    print(f"Cross-validation RMSE: {results['cv_rmse']:.1f} deaths/day")
    print(f"Training residual SD: {results['resid_sd']:.1f} deaths/day")

    print(f"\n{'─'*70}")
    print("PER-ALBUM EFFECTS (actual - predicted)")
    print(f"{'─'*70}")
    print(
        f"{'Artist':<18} {'Album':<22} {'Actual':>6} {'Pred':>6} {'Effect':>7} {'z':>5} {'95% CI':>16}"
    )
    print("-" * 84)

    df = results["results_df"]
    for _, r in df.iterrows():
        ci = f"[{r['pred_ci_lower']:.0f}, {r['pred_ci_upper']:.0f}]"
        tier_marker = "•" if r["tier"] == 1 else " "
        print(
            f"{tier_marker}{r['artist']:<17} {r['album'][:21]:<22} "
            f"{r['y_actual']:>6.0f} {r['y_predicted']:>6.0f} {r['effect']:>+7.1f} "
            f"{r['z_score']:>5.2f} {ci:>16}"
        )

    print(f"\n{'─'*70}")
    print("POOLED ESTIMATES")
    print(f"{'─'*70}")

    print(f"\n  All 20 albums:")
    print(
        f"    Effect: {results['pooled_all']:+.1f} deaths (SE = {results['pooled_se_all']:.1f})"
    )
    t_all = results["pooled_all"] / results["pooled_se_all"]
    print(f"    t-stat: {t_all:.2f}")

    print(f"\n  Tier 1 (top 10):")
    print(
        f"    Effect: {results['pooled_t1']:+.1f} deaths (SE = {results['pooled_se_t1']:.1f})"
    )
    t_t1 = results["pooled_t1"] / results["pooled_se_t1"]
    print(f"    t-stat: {t_t1:.2f}")

    print(f"\n  Tier 2 (albums 11-20):")
    print(
        f"    Effect: {results['pooled_t2']:+.1f} deaths (SE = {results['pooled_se_t2']:.1f})"
    )
    t_t2 = results["pooled_t2"] / results["pooled_se_t2"]
    print(f"    t-stat: {t_t2:.2f}")

    print(f"\n{'─'*70}")
    print("DOSE-RESPONSE CHECK")
    print(f"{'─'*70}")
    print(f"  Pearson r (streams vs effect): {results['r_pearson']:+.3f}")
    print(f"  p-value: {results['p_pearson']:.4f}")
    if results["r_pearson"] < 0:
        print("  → Negative correlation: more streams associated with SMALLER effects")


def compare_estimators(local_df, global_df, forecast_results):
    """Compare local, global, and forecast estimators side by side."""
    print(f"\n{'='*70}")
    print("ESTIMATOR COMPARISON")
    print(f"{'='*70}")

    forecast_df = forecast_results["results_df"]

    print(f"\n{'Album':<35} {'Local':>8} {'Global':>8} {'Forecast':>8}")
    print("-" * 65)

    for _, fr in forecast_df.iterrows():
        local_row = local_df[local_df["album"] == fr["album"]]
        local_delta = (
            local_row["delta_local"].values[0] if len(local_row) > 0 else np.nan
        )

        dt = pd.to_datetime(fr["date"])
        global_row = global_df[global_df["date"] == dt]
        if len(global_row) > 0:
            global_delta = (
                global_row["fatalities"].values[0]
                - global_row["fitted_global"].values[0]
            )
        else:
            global_delta = np.nan

        name = f"{fr['artist'][:12]} - {fr['album'][:18]}"
        print(
            f"{name:<35} {local_delta:>+8.1f} {global_delta:>+8.1f} {fr['effect']:>+8.1f}"
        )

    local_mean = local_df["delta_local"].mean()
    local_se = local_df["delta_local"].std() / np.sqrt(len(local_df))

    release_mask = global_df["date"].isin(pd.to_datetime(local_df["date"]))
    global_deltas = (
        global_df.loc[release_mask, "fatalities"].values
        - global_df.loc[release_mask, "fitted_global"].values
    )
    global_mean = np.mean(global_deltas)
    global_se = np.std(global_deltas) / np.sqrt(len(global_deltas))

    t1_mask = forecast_df["tier"] == 1
    forecast_mean = forecast_df.loc[t1_mask, "effect"].mean()
    forecast_se = forecast_df.loc[t1_mask, "effect"].std() / np.sqrt(t1_mask.sum())

    print(f"\n{'─'*65}")
    print("POOLED ESTIMATES (Tier 1 only)")
    print(f"{'─'*65}")
    print(f"  Local (±10 day):    {local_mean:+.1f} deaths (SE = {local_se:.1f})")
    print(f"  Donut-global:       {global_mean:+.1f} deaths (SE = {global_se:.1f})")
    print(f"  Forecast:           {forecast_mean:+.1f} deaths (SE = {forecast_se:.1f})")

    print(f"\n  Interpretation:")
    print(f"    Local includes post-treatment days in control → likely biased")
    print(f"    Global uses fixed effects, excludes ±10d window")
    print(f"    Forecast trains on all non-release days, predicts counterfactual")


def save_forecast_tables(results, output_dir="tabs"):
    """Save forecast results to markdown."""
    Path(output_dir).mkdir(exist_ok=True)

    save_table(results["results_df"], f"{output_dir}/t10_forecast_estimates.md")

    summary = pd.DataFrame(
        [
            {"metric": "model_type", "value": results["model_type"]},
            {"metric": "n_train_days", "value": results["n_train"]},
            {"metric": "n_excluded_days", "value": results["n_excluded"]},
            {"metric": "cv_rmse", "value": results["cv_rmse"]},
            {"metric": "pooled_all_effect", "value": results["pooled_all"]},
            {"metric": "pooled_all_se", "value": results["pooled_se_all"]},
            {"metric": "pooled_t1_effect", "value": results["pooled_t1"]},
            {"metric": "pooled_t1_se", "value": results["pooled_se_t1"]},
            {"metric": "pooled_t2_effect", "value": results["pooled_t2"]},
            {"metric": "pooled_t2_se", "value": results["pooled_se_t2"]},
            {"metric": "pearson_r_streams_effect", "value": results["r_pearson"]},
        ]
    )
    save_table(summary, f"{output_dir}/t11_forecast_summary.md")

    print(f"\nForecast tables saved to {output_dir}/t10_*, t11_*")


def build_daily_weather_controls(accidents):
    """
    For each day, compute weather condition percentages.

    Weather is EXOGENOUS - not affected by album releases.

    Returns DataFrame with:
    - fatalities
    - pct_rain, pct_fog, pct_cloudy (weather conditions)
    """
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

    if "WEATHER" in cols:
        weather = df[cols["WEATHER"]]
        df["_rain"] = (weather == 2).astype(int)
        df["_fog"] = (weather == 5).astype(int)
        df["_cloudy"] = (weather == 10).astype(int)
        df["_bad_weather"] = weather.isin([2, 3, 4, 5, 11, 12]).astype(int)
    else:
        df["_rain"] = np.nan
        df["_fog"] = np.nan
        df["_cloudy"] = np.nan
        df["_bad_weather"] = np.nan

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

    daily = (
        df.groupby("_date")
        .agg(
            fatalities=("_fatals", "sum"),
            n_crashes=("_fatals", "count"),
            n_rain=("_rain", "sum"),
            n_fog=("_fog", "sum"),
            n_cloudy=("_cloudy", "sum"),
            n_bad_weather=("_bad_weather", "sum"),
        )
        .reset_index()
    )

    daily["pct_rain"] = daily["n_rain"] / daily["n_crashes"]
    daily["pct_fog"] = daily["n_fog"] / daily["n_crashes"]
    daily["pct_cloudy"] = daily["n_cloudy"] / daily["n_crashes"]
    daily["pct_bad_weather"] = daily["n_bad_weather"] / daily["n_crashes"]

    daily = daily.drop(
        columns=["n_crashes", "n_rain", "n_fog", "n_cloudy", "n_bad_weather"]
    )
    daily.columns = ["date"] + list(daily.columns[1:])
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily


def weather_controlled_model(daily_df, window=10):
    """
    Test if effect is robust to weather controls.

    Weather is EXOGENOUS - not caused by album releases. If release days
    happen to have bad weather, that could explain higher fatalities.

    Models:
    1. Base: fatalities ~ release + DOW + Month + Year + holidays
    2. +Weather: + pct_rain + pct_fog + pct_cloudy
    3. +All weather: + pct_bad_weather

    Returns DataFrame with model comparisons.
    """
    print(f"\n{'='*70}")
    print("WEATHER-CONTROLLED MODEL (Exogenous Controls)")
    print(f"{'='*70}")
    print("Weather is exogenous - not affected by album releases.")
    print("If effect shrinks with weather controls, some confounding present.\n")

    df = add_time_features(daily_df)

    release_dates = set()
    for a in ALBUMS_TIER1:
        release_dates.add(pd.to_datetime(a[2]).date())

    df["treatment"] = df["date"].dt.date.isin(release_dates).astype(int)

    model_specs = [
        ("Base (DOW+Month+Year)", None),
        ("+Rain", ["pct_rain"]),
        ("+Rain+Fog", ["pct_rain", "pct_fog"]),
        ("+Rain+Fog+Cloudy", ["pct_rain", "pct_fog", "pct_cloudy"]),
        ("+All bad weather", ["pct_bad_weather"]),
    ]

    results = []

    for model_name, controls in model_specs:
        X = build_design_matrix(df, controls=controls)
        X["treatment"] = df["treatment"].values
        y = df["fatalities"].values.astype(float)

        beta, se, _, _ = ols_fit(X.values, y, return_se=True)

        treatment_idx = list(X.columns).index("treatment")
        treatment_effect = beta[treatment_idx]
        treatment_se = se[treatment_idx]
        t_stat = treatment_effect / treatment_se if treatment_se > 0 else 0

        baseline = df["fatalities"].mean()
        pct_effect = 100 * treatment_effect / baseline

        results.append(
            {
                "model": model_name,
                "effect": treatment_effect,
                "se": treatment_se,
                "t_stat": t_stat,
                "pct_effect": pct_effect,
                "controls": ", ".join(controls) if controls else "None",
            }
        )

    results_df = pd.DataFrame(results)

    print(
        f"{'Model':<25} | {'Effect':>10} | {'SE':>8} | {'t-stat':>8} | {'% Effect':>10}"
    )
    print("-" * 75)
    for _, r in results_df.iterrows():
        sig = " **" if abs(r["t_stat"]) > 2 else " *" if abs(r["t_stat"]) > 1.65 else ""
        print(
            f"{r['model']:<25} | {r['effect']:>+10.1f} | {r['se']:>8.1f} | "
            f"{r['t_stat']:>+7.2f}{sig} | {r['pct_effect']:>+9.1f}%"
        )

    print("\nINTERPRETATION:")
    base_row = results_df[results_df["model"] == "Base (DOW+Month+Year)"]
    weather_row = results_df[results_df["model"] == "+All bad weather"]

    if len(base_row) > 0 and len(weather_row) > 0:
        base_effect = base_row["effect"].iloc[0]
        weather_effect = weather_row["effect"].iloc[0]
        change_pct = (
            100 * (weather_effect - base_effect) / base_effect
            if base_effect != 0
            else 0
        )

        if abs(change_pct) < 10:
            print(f"  Effect changes by {change_pct:+.1f}% with weather controls.")
            print("  Weather does NOT explain the effect.")
        else:
            print(f"  Effect changes by {change_pct:+.1f}% with weather controls.")
            if change_pct < 0:
                print(
                    "  Effect DECREASES with weather — some confounding from bad weather days."
                )
            else:
                print(
                    "  Effect INCREASES with weather — weather suppresses the true effect."
                )

    return results_df


def multiverse_analysis(df, window_sizes=None, album_sets=None):
    """
    Multiverse Analysis (Gelman).

    Run analysis under ALL reasonable specification choices:
    - Window sizes: 5, 7, 10, 14, 21 days
    - Album sets: Tier1, Tier1+2, All tiers
    - Sample periods: pre-2018, 2018-2022, all years

    Output: tabs/t29_multiverse.csv
    """
    print(f"\n{'='*70}")
    print("MULTIVERSE ANALYSIS (Gelman)")
    print(f"{'='*70}")
    print("Running ALL reasonable specifications to show distribution of effects.\n")

    if window_sizes is None:
        window_sizes = [5, 7, 10, 14, 21]

    if album_sets is None:
        album_sets = {
            "Tier1": ALBUMS_TIER1,
            "Tier1+2": ALBUMS_TIER1 + ALBUMS_TIER2,
            "All": ALBUMS_EXTENDED,
        }

    sample_periods = {
        "pre-2018": (2007, 2017),
        "2018-2022": (2018, 2022),
        "all_years": (2007, 2024),
    }

    results = []
    total_specs = len(window_sizes) * len(album_sets) * len(sample_periods)
    print(f"Running {total_specs} specifications...")

    spec_num = 0
    for window in window_sizes:
        for album_name, albums in album_sets.items():
            for period_name, (start_yr, end_yr) in sample_periods.items():
                spec_num += 1
                df_sub = df[
                    (df["date"].dt.year >= start_yr) & (df["date"].dt.year <= end_yr)
                ].copy()

                if len(df_sub) < 100:
                    continue

                albums_in_period = [
                    a for a in albums if start_yr <= pd.to_datetime(a[2]).year <= end_yr
                ]

                if len(albums_in_period) == 0:
                    continue

                all_dates_exclude = set()
                for a in albums_in_period:
                    dt = pd.to_datetime(a[2]).date()
                    for offset in range(-window, window + 1):
                        all_dates_exclude.add(dt + datetime.timedelta(days=offset))

                df_sub = add_time_features(df_sub)

                est_mask = ~df_sub["date"].dt.date.isin(all_dates_exclude)
                if est_mask.sum() < 50:
                    continue

                try:
                    X_est = build_design_matrix(df_sub[est_mask])
                    y_est = df_sub.loc[est_mask, "fatalities"].values.astype(float)
                    beta, _, _ = ols_fit(X_est.values, y_est)

                    X_all = build_design_matrix(df_sub)
                    df_sub["fitted"] = X_all.values @ beta
                    df_sub["resid"] = df_sub["fatalities"].values - df_sub["fitted"]
                except Exception:
                    continue

                release_resids = []
                for a in albums_in_period:
                    dt = pd.to_datetime(a[2])
                    row = df_sub[df_sub["date"] == dt]
                    if len(row) > 0:
                        release_resids.append(row["resid"].values[0])

                if len(release_resids) < 2:
                    continue

                effect = np.mean(release_resids)
                se = np.std(release_resids) / sqrt(len(release_resids))
                t_stat = effect / se if se > 0 else 0

                results.append(
                    {
                        "window": window,
                        "album_set": album_name,
                        "sample_period": period_name,
                        "n_albums": len(release_resids),
                        "effect": effect,
                        "se": se,
                        "t_stat": t_stat,
                        "p_value": 2 * (1 - stats.norm.cdf(abs(t_stat))),
                    }
                )

    results_df = pd.DataFrame(results)

    print(f"\nCompleted {len(results_df)} specifications.\n")

    print(
        f"{'Window':>8} | {'Albums':>10} | {'Period':>12} | {'N':>5} | {'Effect':>10} | {'SE':>8} | {'t':>8}"
    )
    print("-" * 75)
    for _, r in results_df.iterrows():
        sig = " **" if abs(r["t_stat"]) > 2 else " *" if abs(r["t_stat"]) > 1.65 else ""
        print(
            f"{r['window']:>8} | {r['album_set']:>10} | {r['sample_period']:>12} | "
            f"{r['n_albums']:>5} | {r['effect']:>+10.1f} | {r['se']:>8.1f} | {r['t_stat']:>+7.2f}{sig}"
        )

    print("\nMULTIVERSE SUMMARY:")
    print(f"  Total specifications: {len(results_df)}")
    print(
        f"  Effect range: [{results_df['effect'].min():+.1f}, {results_df['effect'].max():+.1f}]"
    )
    print(f"  Median effect: {results_df['effect'].median():+.1f}")
    print(f"  Mean effect: {results_df['effect'].mean():+.1f}")
    print(
        f"  % significant (p<0.05): {100 * (results_df['p_value'] < 0.05).mean():.1f}%"
    )
    print(f"  % positive effect: {100 * (results_df['effect'] > 0).mean():.1f}%")

    tier1_only = results_df[results_df["album_set"] == "Tier1"]
    if len(tier1_only) > 0:
        print(f"\n  Tier 1 only specifications:")
        print(
            f"    Effect range: [{tier1_only['effect'].min():+.1f}, {tier1_only['effect'].max():+.1f}]"
        )
        print(f"    Median: {tier1_only['effect'].median():+.1f}")

    return results_df
