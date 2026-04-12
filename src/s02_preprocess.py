"""
Data preprocessing: build daily series and residualize.
"""

import datetime

import numpy as np
import pandas as pd

from src.constants import us_holidays


def build_daily_series(accidents):
    """
    From the FARS Accident table, build a daily national fatality count.

    The Accident table has one row per crash. Key columns:
      - FATALS: number of fatalities in that crash
      - MONTH: month (1-12)
      - DAY: day of month (1-31)  [pre-2019: DAY; 2019+: sometimes DAY_OF_CRASH]
      - YEAR or CaseYear: year
    """
    df = accidents.copy()

    # Harmonize column names (FARS changes these across years)
    cols = {c.upper(): c for c in df.columns}

    # Year
    for candidate in ["YEAR", "CASEYEAR"]:
        if candidate in cols:
            df["_year"] = df[cols[candidate]]
            break

    # Month
    if "MONTH" in cols:
        df["_month"] = df[cols["MONTH"]]

    # Day of month
    for candidate in ["DAY", "DAY_OF_CRASH"]:
        if candidate in cols:
            df["_day"] = df[cols[candidate]]
            break

    # Fatalities per crash
    if "FATALS" in cols:
        df["_fatals"] = df[cols["FATALS"]]
    else:
        df["_fatals"] = 1

    # Drop rows with missing date components
    df = df.dropna(subset=["_year", "_month", "_day"])
    df = df[(df["_month"] >= 1) & (df["_month"] <= 12)]
    df = df[(df["_day"] >= 1) & (df["_day"] <= 31)]

    def safe_date(row):
        try:
            return datetime.date(int(row["_year"]), int(row["_month"]), int(row["_day"]))
        except ValueError:
            return None

    df["date"] = df.apply(safe_date, axis=1)
    df = df.dropna(subset=["date"])

    daily = df.groupby("date")["_fatals"].sum().reset_index()
    daily.columns = ["date", "fatalities"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily


def _build_design(df):
    """Build DOW + month + year + holiday design matrix from a df with those cols."""
    X = pd.get_dummies(
        df[["dow", "month", "year"]],
        columns=["dow", "month", "year"],
        drop_first=True,
        dtype=float,
    )
    X["holiday"] = df["holiday"].values
    X["holiday_adj"] = df["holiday_adj"].values
    X["const"] = 1.0
    return X


def residualize(daily):
    """
    Regress daily fatalities on day-of-week, month, year, and holiday FEs.
    Return the DataFrame with residuals attached.
    """
    df = daily.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    holidays = us_holidays(df["year"].unique())
    df["holiday"] = df["date"].dt.date.isin(holidays).astype(int)
    # Day before/after holiday
    hol_adj = set()
    for h in holidays:
        hol_adj.add(h - datetime.timedelta(1))
        hol_adj.add(h + datetime.timedelta(1))
    df["holiday_adj"] = df["date"].dt.date.isin(hol_adj).astype(int)

    # OLS with dummies
    X = pd.get_dummies(
        df[["dow", "month", "year"]],
        columns=["dow", "month", "year"],
        drop_first=True,
        dtype=float,
    )
    X["holiday"] = df["holiday"].values
    X["holiday_adj"] = df["holiday_adj"].values
    X["const"] = 1.0

    y = df["fatalities"].values.astype(float)

    # Solve normal equations
    XtX = X.values.T @ X.values
    Xty = X.values.T @ y
    beta = np.linalg.solve(XtX, Xty)
    yhat = X.values @ beta

    df["fitted"] = yhat
    df["residual"] = y - yhat
    df["z_score"] = (df["residual"] - df["residual"].mean()) / df["residual"].std()

    return df
