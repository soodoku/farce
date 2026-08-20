"""Shared utilities for FARS analysis."""

import datetime
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import us_holidays

# Below this many treated units, a cross-album standard error is not worth
# reporting: with n = 2 the SE is |x1 - x2| / (2 * sqrt(2)) and the implied
# t-statistic is arithmetic rather than evidence.
MIN_ALBUMS_FOR_SE = 5

# Write a LaTeX twin of every table so ms/ms.tex can \input it. Turn off if the
# manuscript is not being built.
EMIT_TEX = True


def album_stats(values, alpha=0.05):
    """
    Summarise a set of per-album effects.

    Album-level effects are averages over a handful of treated units, so the
    standard error uses the sample standard deviation (ddof=1) and critical
    values come from the t distribution with n-1 degrees of freedom. Using
    np.std's default ddof=0 and a normal critical value inflates every
    t-statistic by sqrt(n/(n-1)) — 5.4% at n=10 — and understates every
    interval.

    Returns a dict with effect, se, t_stat, p_value, ci_lower, ci_upper, n.
    Effects computed from fewer than MIN_ALBUMS_FOR_SE units get NaN for
    everything except the point estimate and n.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    out = {
        "effect": v.mean() if n else np.nan,
        "se": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n": n,
    }
    if n < MIN_ALBUMS_FOR_SE:
        return out

    se = v.std(ddof=1) / sqrt(n)
    out["se"] = se
    if se > 0:
        t_stat = out["effect"] / se
        crit = stats.t.ppf(1 - alpha / 2, n - 1)
        out["t_stat"] = t_stat
        out["p_value"] = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        out["ci_lower"] = out["effect"] - crit * se
        out["ci_upper"] = out["effect"] + crit * se
    return out


def save_table(df, filepath, decimals=2, caption=None):
    """
    Save DataFrame as a markdown table, and as a LaTeX fragment beside it.

    Markdown is the artifact. A LaTeX twin is written alongside it when
    EMIT_TEX is on, so ms/ms.tex can \\input tables instead of carrying its own
    copies. Set EMIT_TEX to False if the manuscript is not being built and the
    extra files are clutter.

    The .tex twin exists so a manuscript can \\input{} the table instead of
    having its numbers retyped. Every numerical error found in the April 2026
    draft of this paper came from hand transcription: a standard error of 4.8
    where the code produced 7.45, an out-of-sample table matching no estimator
    in the repository, and Friday counts belonging to a different sample
    period. None of them could have survived an \\input.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    filepath : str
        Output path (e.g., "tabs/t01_local_estimates.md")
    decimals : int
        Round numeric columns to this many decimals (default: 2)
    caption : str, optional
        Table caption. Naming the estimator here is what keeps four different
        estimates of "the effect" from reading as four answers.
    """
    df_out = df.copy()
    for col in df_out.select_dtypes(include=["float64", "float32"]).columns:
        df_out[col] = df_out[col].round(decimals)

    md = df_out.to_markdown(index=False)
    if caption:
        md = f"**{caption}**\n\n{md}"

    with open(filepath, "w") as f:
        f.write(md)

    if not EMIT_TEX:
        return

    # pandas escapes the table body but not the caption, and captions here name
    # column identifiers with underscores in them
    tex_caption = None
    if caption:
        tex_caption = caption
        for ch, esc in [
            ("\\", r"\textbackslash{}"),
            ("_", r"\_"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("#", r"\#"),
            ("$", r"\$"),
        ]:
            tex_caption = tex_caption.replace(ch, esc)

    tex_path = str(filepath).rsplit(".", 1)[0] + ".tex"
    body = df_out.to_latex(
        index=False,
        escape=True,
        caption=tex_caption,
        float_format=f"%.{decimals}f".__mod__,
    )
    with open(tex_path, "w") as f:
        f.write(body)


def rubin_combine(estimates, variances, alpha=0.05):
    """
    Combine estimates across multiply-imputed datasets by Rubin's rules.

    With m imputations the pooled estimate is the mean of the m estimates and
    the pooled variance is the within-imputation variance plus the
    between-imputation variance inflated by (1 + 1/m). The second term is the
    price of not knowing the imputed values, and it can only add variance, so
    a pooled standard error smaller than the mean within-imputation one is a
    bug rather than a finding.

    Degrees of freedom use Rubin's original formula. Critical values come from
    t on those degrees of freedom, not from the normal: the same substitution
    has been the source of three separate defects in this repository.

    Returns a dict with estimate, se, df, t_stat, p_value, ci_lower, ci_upper,
    and the fraction of missing information.
    """
    est = np.asarray(estimates, dtype=float)
    var = np.asarray(variances, dtype=float)
    m = len(est)
    if m < 2:
        raise ValueError("Rubin's rules need at least two imputations")
    if len(var) != m:
        raise ValueError("estimates and variances must be the same length")

    within = float(np.mean(var))
    between = float(np.var(est, ddof=1))
    total = within + (1 + 1 / m) * between
    se = sqrt(total)

    if between <= 0:
        # every imputation agreed; degrees of freedom are unbounded above, so
        # fall back on the within-imputation inference
        df = float("inf")
        frac_missing = 0.0
    else:
        df = (m - 1) * (1 + within / ((1 + 1 / m) * between)) ** 2
        frac_missing = ((1 + 1 / m) * between) / total

    crit = stats.t.ppf(1 - alpha / 2, df)
    pooled = float(est.mean())
    t_stat = pooled / se if se > 0 else np.nan

    return {
        "estimate": pooled,
        "se": se,
        "df": df,
        "t_stat": t_stat,
        "p_value": float(2 * (1 - stats.t.cdf(abs(t_stat), df))) if se > 0 else np.nan,
        "ci_lower": pooled - crit * se,
        "ci_upper": pooled + crit * se,
        "within_var": within,
        "between_var": between,
        "frac_missing_info": frac_missing,
        "n_imputations": m,
    }


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
        missing = [c for c in controls if c not in df.columns]
        if missing:
            raise KeyError(
                f"requested control column(s) absent from the frame: {missing}. "
                "Silently dropping them turns a controlled model into the base "
                "model without changing the label on the row."
            )
        for col in controls:
            if df[col].isna().any():
                raise ValueError(
                    f"control column '{col}' has missing values; fill or drop them "
                    "explicitly rather than letting fillna(0) invent zeros."
                )
            X[col] = df[col].values

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
