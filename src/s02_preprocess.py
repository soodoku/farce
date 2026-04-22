"""Data preprocessing: build daily series from FARS crash data."""

import datetime

import numpy as np
import pandas as pd


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

    df["_fatals"] = df[cols["FATALS"]]

    if "LGT_COND" in cols:
        df["_dark"] = df[cols["LGT_COND"]].isin([2, 3, 6]).astype(int)
    else:
        df["_dark"] = np.nan

    if "RUR_URB" in cols:
        df["_rural"] = (df[cols["RUR_URB"]] == 1).astype(int)
    else:
        df["_rural"] = np.nan

    if "WEATHER" in cols:
        df["_bad_weather"] = df[cols["WEATHER"]].isin([2, 3, 4, 5, 11, 12]).astype(int)
    else:
        df["_bad_weather"] = np.nan

    if "HOUR" in cols:
        hour = df[cols["HOUR"]]
        df["_night"] = ((hour >= 21) | (hour <= 5)).astype(int)
    else:
        df["_night"] = np.nan

    if "DRUNK_DR" in cols:
        df["_alcohol"] = (df[cols["DRUNK_DR"]] >= 1).astype(int)
    else:
        df["_alcohol"] = np.nan

    df = df.dropna(subset=["_year", "_month", "_day"])
    df = df[(df["_month"] >= 1) & (df["_month"] <= 12)]
    df = df[(df["_day"] >= 1) & (df["_day"] <= 31)]

    def safe_date(row):
        try:
            return datetime.date(
                int(row["_year"]), int(row["_month"]), int(row["_day"])
            )
        except ValueError:
            return None

    df["date"] = df.apply(safe_date, axis=1)
    df = df.dropna(subset=["date"])

    daily = (
        df.groupby("date")
        .agg(
            fatalities=("_fatals", "sum"),
            n_crashes=("_fatals", "count"),
            n_dark=("_dark", "sum"),
            n_rural=("_rural", "sum"),
            n_bad_weather=("_bad_weather", "sum"),
            n_night=("_night", "sum"),
            n_alcohol=("_alcohol", "sum"),
        )
        .reset_index()
    )

    daily["pct_dark"] = daily["n_dark"] / daily["n_crashes"]
    daily["pct_rural"] = daily["n_rural"] / daily["n_crashes"]
    daily["pct_bad_weather"] = daily["n_bad_weather"] / daily["n_crashes"]
    daily["pct_night"] = daily["n_night"] / daily["n_crashes"]
    daily["pct_alcohol"] = daily["n_alcohol"] / daily["n_crashes"]

    daily = daily.drop(
        columns=[
            "n_crashes",
            "n_dark",
            "n_rural",
            "n_bad_weather",
            "n_night",
            "n_alcohol",
        ]
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily
