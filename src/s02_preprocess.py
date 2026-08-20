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

    # RUR_URB is absent from the accident file before 2015 and DRUNK_DR was
    # dropped after 2020. Coercing an absent column to 0 makes a structurally
    # missing year look like a year with no rural crashes and no drinking
    # drivers, which then shows up as "balance" in the covariate table. Keep
    # them NaN so downstream code has to decide what to do.
    if "RUR_URB" in cols:
        df["_rural"] = np.where(
            df[cols["RUR_URB"]].notna(),
            (df[cols["RUR_URB"]] == 1).astype(float),
            np.nan,
        )
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
        df["_alcohol"] = np.where(
            df[cols["DRUNK_DR"]].notna(),
            (df[cols["DRUNK_DR"]] >= 1).astype(float),
            np.nan,
        )
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

    # each share uses its own denominator: the number of crashes on that day
    # for which the underlying variable was actually recorded
    daily = (
        df.groupby("date")
        .agg(
            fatalities=("_fatals", "sum"),
            n_crashes=("_fatals", "count"),
            n_dark=("_dark", "sum"),
            d_dark=("_dark", "count"),
            n_rural=("_rural", "sum"),
            d_rural=("_rural", "count"),
            n_bad_weather=("_bad_weather", "sum"),
            d_bad_weather=("_bad_weather", "count"),
            n_night=("_night", "sum"),
            d_night=("_night", "count"),
            n_alcohol=("_alcohol", "sum"),
            d_alcohol=("_alcohol", "count"),
        )
        .reset_index()
    )

    drop_cols = ["n_crashes"]
    for name in ["dark", "rural", "bad_weather", "night", "alcohol"]:
        denom = daily[f"d_{name}"].replace(0, np.nan)
        daily[f"pct_{name}"] = daily[f"n_{name}"] / denom
        drop_cols += [f"n_{name}", f"d_{name}"]

    daily = daily.drop(columns=drop_cols)

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily
