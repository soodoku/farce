"""Shared utilities for FARS analysis."""

import datetime

import numpy as np
import pandas as pd

from src.constants import us_holidays


def save_table(df, filepath, decimals=2, caption=None):
    """
    Save DataFrame as markdown table (kable-style).

    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    filepath : str
        Output path (e.g., "tabs/t01_local_estimates.md")
    decimals : int
        Round numeric columns to this many decimals (default: 2)
    caption : str, optional
        Table caption to include above the table
    """
    df_out = df.copy()
    for col in df_out.select_dtypes(include=["float64", "float32"]).columns:
        df_out[col] = df_out[col].round(decimals)

    md = df_out.to_markdown(index=False)

    if caption:
        md = f"**{caption}**\n\n{md}"

    with open(filepath, "w") as f:
        f.write(md)


def add_time_features(df):
    """
    Add time-based features to dataframe.

    Adds: dow, month, year, holiday, holiday_adj
    Requires: df must have 'date' column
    """
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

    return df


def build_design_matrix(df, controls=None, use_week_of_year=False):
    """
    Build design matrix with fixed effects.

    Parameters
    ----------
    df : DataFrame
        Must have: dow, month/week_of_year, year, holiday, holiday_adj columns
    controls : list of str, optional
        Additional control columns to include
    use_week_of_year : bool
        If True, use week-of-year (52 levels) instead of month (12 levels)
    """
    if use_week_of_year:
        df = df.copy()
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        X = pd.get_dummies(
            df[["dow", "week_of_year", "year"]],
            columns=["dow", "week_of_year", "year"],
            drop_first=True,
            dtype=float,
        )
    else:
        X = pd.get_dummies(
            df[["dow", "month", "year"]],
            columns=["dow", "month", "year"],
            drop_first=True,
            dtype=float,
        )

    X["holiday"] = df["holiday"].values
    X["holiday_adj"] = df["holiday_adj"].values

    if controls:
        for col in controls:
            if col in df.columns:
                X[col] = df[col].fillna(0).values

    X["const"] = 1.0
    return X


def ols_fit(X, y, return_se=False, ridge_lambda=0, robust=False):
    """
    OLS regression via normal equations.

    Parameters
    ----------
    X : array-like
        Design matrix (n x k)
    y : array-like
        Response vector (n,)
    return_se : bool
        If True, also return standard errors
    ridge_lambda : float
        Ridge regularization parameter (0 = no regularization).
        Note: If ridge_lambda > 0 and return_se=True, SEs are computed
        ignoring shrinkage bias. Use ridge only for numerical stability.
    robust : bool
        If True, use HC1 heteroskedasticity-robust standard errors

    Returns
    -------
    If return_se=False: (beta, fitted, residuals)
    If return_se=True: (beta, se, fitted, residuals)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n, k = X.shape

    XtX = X.T @ X
    Xty = X.T @ y

    if ridge_lambda > 0:
        XtX = XtX + ridge_lambda * np.eye(k)

    beta = np.linalg.solve(XtX, Xty)
    fitted = X @ beta
    residuals = y - fitted

    if return_se:
        XtX_inv = np.linalg.inv(XtX)
        dof = max(n - k, 1)

        if robust:
            hc1_adj = n / dof
            meat = np.zeros((k, k))
            for i in range(n):
                xi = X[i, :].reshape(-1, 1)
                meat += (residuals[i] ** 2) * (xi @ xi.T)
            meat *= hc1_adj
            var_beta = XtX_inv @ meat @ XtX_inv
        else:
            sigma2 = np.sum(residuals**2) / dof
            var_beta = sigma2 * XtX_inv

        se = np.sqrt(np.maximum(np.diag(var_beta), 0))
        return beta, se, fitted, residuals

    return beta, fitted, residuals


def cluster_robust_se(X, residuals, clusters):
    """
    Compute cluster-robust standard errors (CR1).

    Parameters
    ----------
    X : array-like
        Design matrix (n x k)
    residuals : array-like
        OLS residuals (n,)
    clusters : array-like
        Cluster identifiers (n,)

    Returns
    -------
    se : array
        Cluster-robust standard errors for each coefficient.
        Falls back to OLS SE if matrix is singular.
    """
    X = np.asarray(X)
    residuals = np.asarray(residuals)
    clusters = np.asarray(clusters)

    n, k = X.shape
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)

    meat = np.zeros((k, k))

    for c in unique_clusters:
        mask = clusters == c
        X_c = X[mask]
        e_c = residuals[mask]
        score = X_c.T @ e_c
        meat += np.outer(score, score)

    if G <= 1:
        dof = max(n - k, 1)
        sigma2 = np.sum(residuals**2) / dof
        var_beta = sigma2 * XtX_inv
    else:
        dof_adj = (G / (G - 1)) * ((n - 1) / (n - k))
        sandwich = XtX_inv @ meat @ XtX_inv
        var_beta = dof_adj * sandwich

    return np.sqrt(np.maximum(np.diag(var_beta), 0))
