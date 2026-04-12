"""
Forecast-based causal effect estimation (Gary King style).

Train a prediction model on non-treatment days, then compare actual release-day
fatalities to model predictions. The residual = causal effect estimate.

Advantages over ±10 day local estimator:
1. No post-treatment contamination (only uses pre-treatment patterns)
2. Uses all available predictors
3. Quantifies prediction uncertainty via cross-validation
4. Transparent counterfactual
"""

import datetime

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import (
    ALBUMS_ALL,
    ALBUMS_EXTENDED,
    ALBUMS_TIER1,
    ALBUMS_TIER2,
    ALBUMS_TIER3,
    us_holidays,
)


def build_features(df):
    """Build feature matrix for forecasting model."""
    df = df.copy()

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

    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True, dtype=float)
    month_dummies = pd.get_dummies(
        df["month"], prefix="month", drop_first=True, dtype=float
    )
    year_dummies = pd.get_dummies(df["year"], prefix="year", drop_first=True, dtype=float)

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
        local_delta = local_row["delta_local"].values[0] if len(local_row) > 0 else np.nan

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
    """Save forecast results to CSV."""
    from pathlib import Path

    Path(output_dir).mkdir(exist_ok=True)

    results["results_df"].to_csv(f"{output_dir}/t10_forecast_estimates.csv", index=False)

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
    summary.to_csv(f"{output_dir}/t11_forecast_summary.csv", index=False)

    print(f"\nForecast tables saved to {output_dir}/t10_*, t11_*")


DATA_DIR = "data/fars/"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Forecast-based causal effect estimation for FARS album release study"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Days to exclude around releases (default: 10)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ridge",
        choices=["ridge", "gbm"],
        help="Model type: ridge or gbm (default: ridge)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include 2023-2024 albums (tier 3) in analysis",
    )
    args = parser.parse_args()

    from src.s01_load import load_local_fars
    from src.s02_preprocess import build_daily_series, residualize
    from src.s03_core import decomposition_analysis

    albums = ALBUMS_EXTENDED if args.extended else ALBUMS_ALL

    print("=" * 70)
    print("FARS Album Release Analysis - Forecast Estimator")
    print(f"Sample: {'EXTENDED (27 albums)' if args.extended else 'STANDARD (20 albums)'}")
    print("=" * 70)

    accidents = load_local_fars(DATA_DIR)
    print(f"\nLoaded {len(accidents)} crash records")

    daily = build_daily_series(accidents)
    print(
        f"Data period: {daily['date'].min().date()} to {daily['date'].max().date()} "
        f"({len(daily)} days)"
    )

    df = residualize(daily)

    results = forecast_estimate(df, window=args.window, model_type=args.model, albums=albums)
    print_forecast_results(results)

    df_global, local_df = decomposition_analysis(df, window=args.window)
    compare_estimators(local_df, df_global, results)

    save_forecast_tables(results)


if __name__ == "__main__":
    main()
