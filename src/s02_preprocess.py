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
      - LGT_COND: Light condition (1=Daylight, 2-3=Dark, 4=Dawn, 5=Dusk)
      - WEATHER: Weather (1=Clear, 10=Cloudy, 2=Rain, 3-4=Sleet/Snow, etc.)
      - RUR_URB: Rural (1) vs Urban (2)
      - HOUR: Hour of crash (0-23)
      - DRUNK_DR: Number of drunk drivers in crash
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

    # Extract crash-level predictors
    # Dark conditions: LGT_COND in (2=Dark-Not Lighted, 3=Dark-Lighted, 6=Dark-Unknown)
    if "LGT_COND" in cols:
        df["_dark"] = df[cols["LGT_COND"]].isin([2, 3, 6]).astype(int)
    else:
        df["_dark"] = np.nan

    # Rural: RUR_URB == 1
    if "RUR_URB" in cols:
        df["_rural"] = (df[cols["RUR_URB"]] == 1).astype(int)
    else:
        df["_rural"] = np.nan

    # Bad weather: WEATHER in (2=Rain, 3=Sleet/Hail, 4=Snow, 5=Fog, 11=Blowing Snow, 12=Freezing Rain)
    if "WEATHER" in cols:
        df["_bad_weather"] = df[cols["WEATHER"]].isin([2, 3, 4, 5, 11, 12]).astype(int)
    else:
        df["_bad_weather"] = np.nan

    # Night: HOUR in 21-23 or 0-5 (9pm to 6am)
    if "HOUR" in cols:
        hour = df[cols["HOUR"]]
        df["_night"] = ((hour >= 21) | (hour <= 5)).astype(int)
    else:
        df["_night"] = np.nan

    # Alcohol: DRUNK_DR >= 1
    if "DRUNK_DR" in cols:
        df["_alcohol"] = (df[cols["DRUNK_DR"]] >= 1).astype(int)
    else:
        df["_alcohol"] = np.nan

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

    # Aggregate to daily level
    daily = df.groupby("date").agg(
        fatalities=("_fatals", "sum"),
        n_crashes=("_fatals", "count"),
        n_dark=("_dark", "sum"),
        n_rural=("_rural", "sum"),
        n_bad_weather=("_bad_weather", "sum"),
        n_night=("_night", "sum"),
        n_alcohol=("_alcohol", "sum"),
    ).reset_index()

    # Compute proportions
    daily["pct_dark"] = daily["n_dark"] / daily["n_crashes"]
    daily["pct_rural"] = daily["n_rural"] / daily["n_crashes"]
    daily["pct_bad_weather"] = daily["n_bad_weather"] / daily["n_crashes"]
    daily["pct_night"] = daily["n_night"] / daily["n_crashes"]
    daily["pct_alcohol"] = daily["n_alcohol"] / daily["n_crashes"]

    # Drop intermediate columns
    daily = daily.drop(columns=["n_crashes", "n_dark", "n_rural", "n_bad_weather", "n_night", "n_alcohol"])

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily


def _build_design(df, use_week_of_year=False):
    """
    Build design matrix with fixed effects.

    Parameters
    ----------
    df : DataFrame
        Daily data with dow, month, year columns
    use_week_of_year : bool
        If True, use week-of-year (52 levels) instead of month (12 levels).
        Paper uses week-of-year FEs.
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

    predictor_cols = ["pct_dark", "pct_rural", "pct_bad_weather", "pct_night", "pct_alcohol"]
    for col in predictor_cols:
        if col in df.columns:
            X[col] = df[col].fillna(0).values

    X["const"] = 1.0
    return X


def residualize(daily):
    """
    Regress daily fatalities on day-of-week, month, year, holiday FEs,
    and crash-level predictors (dark, rural, bad weather, night, alcohol).
    Return the DataFrame with residuals attached.
    """
    df = daily.copy()
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

    X = _build_design(df)
    y = df["fatalities"].values.astype(float)

    XtX = X.values.T @ X.values
    Xty = X.values.T @ y
    beta = np.linalg.solve(XtX, Xty)
    yhat = X.values @ beta

    df["fitted"] = yhat
    df["residual"] = y - yhat
    df["z_score"] = (df["residual"] - df["residual"].mean()) / df["residual"].std()

    return df
