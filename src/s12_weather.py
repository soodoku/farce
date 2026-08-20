"""
Exogenous weather from NOAA stations, to replace a bad control.

The weather variables elsewhere in this repository are shares of a day's fatal
crashes that occurred in rain, fog, cloud or snow. Numerator and denominator
are both the outcome, so conditioning on them is conditioning on a
post-treatment variable and they cannot support the claim that the effect
survives a weather control.

This builds a control that can. Daily precipitation comes from NOAA
GHCN stations, pulled through the get-weather-data package, for the largest
metropolitan area in each of the fifty states and the District of Columbia, and
are aggregated into a national daily series. Nothing in it is a function of the
crash record.

Two design choices worth defending.

The frame is exhaustive rather than convenient. Picking the twenty largest
metropolitan areas, which is what a first version of this did, covers only
about 78 percent of US traffic fatalities and leaves the choice of cut-off
undefended. All fifty-one jurisdictions cover all of it.

The weights are each state's share of US traffic fatalities over 2007 to 2024,
computed once from FARS and held fixed. They are a single number per state over
eighteen years rather than anything that varies with treatment, so they cannot
carry the release-day effect into the control. And they weight by where fatal
crashes actually happen, which is what the control needs to absorb, rather than
by population, which is a proxy for it.

The honest limit is spatial resolution. One station per state under-represents
weather variation inside large states: a storm in west Texas is invisible if
the station is in Houston. The alternative, a population-weighted grid, is a
larger undertaking. What is here is enough to answer whether release days were
unusual weather days, which is the question the bad control was standing in for.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "weather"
CACHE = DATA_DIR / "state_daily.csv"
WEIGHTS = DATA_DIR / "state_fatality_shares.json"

# Largest metropolitan area in each state, with a central ZIP. The station
# search takes the nearest reporting station to that point.
#
# Five entries point at the metro's airport rather than downtown. Airports run
# continuous daily records; some downtown co-operative stations report
# sporadically, and with a downtown ZIP Louisiana and Colorado returned under
# 8% of days. Coverage is reported by build_national_weather so a future gap
# is visible rather than silent.
STATE_LOCATIONS = {
    "Alabama": "35203",
    "Alaska": "99501",
    "Arizona": "85004",
    "Arkansas": "72201",
    "California": "90012",
    "Colorado": "80249",  # airport
    "Connecticut": "06604",
    "Delaware": "19801",
    "District of Columbia": "20001",
    "Florida": "33130",
    "Georgia": "30303",
    "Hawaii": "96813",
    "Idaho": "83702",
    "Illinois": "60666",  # airport
    "Indiana": "46204",
    "Iowa": "50309",
    "Kansas": "67202",
    "Kentucky": "40202",
    "Louisiana": "70062",  # airport
    "Maine": "04101",
    "Maryland": "21202",
    "Massachusetts": "02108",
    "Michigan": "48226",
    "Minnesota": "55401",
    "Mississippi": "39201",
    "Missouri": "64106",
    "Montana": "59101",
    "Nebraska": "68102",
    "Nevada": "89101",
    "New Hampshire": "03103",  # airport
    "New Jersey": "07102",
    "New Mexico": "87102",
    "New York": "10001",
    "North Carolina": "28202",
    "North Dakota": "58102",
    "Ohio": "43215",
    "Oklahoma": "73102",
    "Oregon": "97204",
    "Pennsylvania": "19103",
    "Rhode Island": "02903",
    "South Carolina": "29201",
    "South Dakota": "57104",
    "Tennessee": "37203",
    "Texas": "77032",  # airport
    "Utah": "84111",
    "Vermont": "05401",
    "Virginia": "23219",
    "Washington": "98101",
    "West Virginia": "25301",
    "Wisconsin": "53202",
    "Wyoming": "82001",
}


def build_state_weights(accidents, out_path=None):
    """
    Each state's share of US traffic fatalities, 2007 to 2024.

    Fixed weights computed once over the whole series. Because they do not vary
    by day they cannot transmit the treatment into the weather control.
    """
    df = accidents.copy()
    df.columns = [c.upper() for c in df.columns]
    names = (
        df.loc[df["STATENAME"].notna(), ["STATE", "STATENAME"]]
        .drop_duplicates()
        .set_index("STATE")["STATENAME"]
        .to_dict()
    )
    totals = df.groupby("STATE")["FATALS"].sum()
    shares = totals / totals.sum()
    weights = {names.get(code, str(code)): float(v) for code, v in shares.items()}

    out_path = Path(out_path or WEIGHTS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(weights, indent=1, sort_keys=True) + "\n")
    print(f"  state weights written to {out_path} ({len(weights)} states)")
    return weights


def fetch_state_weather(start_date, end_date, package_src=None, force=False):
    """
    Pull daily precipitation for each state's largest metro.

    Cached to data/weather/state_daily.csv so the pipeline runs offline after
    the first pull.
    """
    if CACHE.exists() and not force:
        cached = pd.read_csv(CACHE, parse_dates=["date"])
        if cached["date"].min() <= pd.Timestamp(start_date) and cached[
            "date"
        ].max() >= pd.Timestamp(end_date):
            print(f"  using cached station weather: {CACHE}")
            return cached

    if package_src:
        sys.path.insert(0, str(package_src))
    try:
        from get_weather_data import Weather
    except ImportError as exc:
        raise SystemExit(
            "get-weather-data is not importable. Install it, or pass "
            "package_src pointing at its src/ directory."
        ) from exc

    weather = Weather()
    frames = []
    for state, zipcode in STATE_LOCATIONS.items():
        try:
            # PRCP only, deliberately. GHCN PRCP is total precipitation
            # including the water equivalent of snow, so it covers the exposure
            # that matters. Asking for SNOW as well is expensive for no gain:
            # GSOD carries snow depth (SNDP) but no snowfall amount, so a
            # SNOW request makes the search walk outward through stations,
            # downloading GSOD files that can never satisfy it.
            frame = weather.get_frame(zipcode, start_date, end_date, elements=["PRCP"])
        except Exception as exc:
            print(f"  {state}: FAILED ({type(exc).__name__}), skipped")
            continue
        frame = frame[["date", "prcp"]].copy()
        frame["state"] = state
        frames.append(frame)
        print(
            f"  {state}: {frame['prcp'].notna().sum()}/{len(frame)} days with "
            "precipitation"
        )

    if not frames:
        raise SystemExit("no state returned weather; nothing to build")

    out = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    print(f"  cached to {CACHE}")
    return out


def build_national_weather(state_daily, weights):
    """
    Aggregate state readings into a national daily precipitation series.

    Both a fatality-weighted and an equally weighted series are returned, so a
    reader can see whether the weighting carries anything.
    """
    df = state_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["prcp"] = pd.to_numeric(df["prcp"], errors="coerce")
    df["weight"] = df["state"].map(weights)
    df = df.dropna(subset=["weight"])

    rows = []
    for date, group in df.groupby("date"):
        valid = group.dropna(subset=["prcp"])
        if valid.empty:
            continue
        rows.append(
            {
                "date": date,
                "precip_weighted": float(
                    np.average(valid["prcp"], weights=valid["weight"])
                ),
                "precip_equal": float(valid["prcp"].mean()),
                "n_states": len(valid),
                "weight_covered": float(valid["weight"].sum()),
            }
        )

    national = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(
        f"  national series: {len(national)} days, median "
        f"{national['n_states'].median():.0f} states reporting, median "
        f"{national['weight_covered'].median():.2f} of fatality weight covered"
    )
    return national


def exogenous_weather_model(daily_df, national_weather, albums=None, window=10):
    """
    Does the release-day effect survive a weather control that is actually weather?

    Adds fatality-weighted daily precipitation to the fixed-effect model,
    refits on the donut sample, and reads the effect off the release days. The
    fatality-weighted and equally weighted series are run separately so the
    weighting choice is visible rather than buried.

    Unlike the crash-composition shares used elsewhere, nothing in these
    regressors is a function of the crash record, so this is a control rather
    than a bad control and the estimate it produces means what it appears to
    mean.

    Returns DataFrame with one row per specification.
    """
    from src.constants import ALBUMS_TIER1
    from src.utils import add_time_features, album_stats, build_design_matrix, ols_fit

    print("\n" + "=" * 70)
    print("EXOGENOUS WEATHER CONTROLS (NOAA STATIONS)")
    print("=" * 70)
    print("Precipitation from weather stations, not from crash records.\n")

    albums = albums or ALBUMS_TIER1
    weather = national_weather.copy()
    weather["date"] = pd.to_datetime(weather["date"])

    df = add_time_features(daily_df).merge(weather, on="date", how="inner")
    release_dates = [pd.to_datetime(a[2]) for a in albums]
    missing = [d for d in release_dates if d not in set(df["date"])]
    if missing:
        raise SystemExit(f"weather series does not cover {missing}")

    import datetime as _dt

    exclude = set()
    for dt in release_dates:
        for offset in range(-window, window + 1):
            exclude.add(dt.date() + _dt.timedelta(days=offset))
    est = (~df["date"].dt.date.isin(exclude)).values
    y = df["fatalities"].values.astype(float)

    specs = [
        ("no weather control", []),
        ("precipitation, fatality-weighted", ["precip_weighted"]),
        ("precipitation, equally weighted", ["precip_equal"]),
        ("both weightings together", ["precip_weighted", "precip_equal"]),
    ]

    rows = []
    for label, controls in specs:
        X = build_design_matrix(df, controls=controls or None)
        beta, _, _ = ols_fit(X.values[est], y[est])
        resid = y - X.values @ beta
        st = album_stats([resid[(df["date"] == d).values][0] for d in release_dates])
        rows.append(
            {
                "specification": label,
                "n_controls": len(controls),
                "effect": st["effect"],
                "se": st["se"],
                "ci_lower": st["ci_lower"],
                "ci_upper": st["ci_upper"],
                "p_value": st["p_value"],
            }
        )

    results_df = pd.DataFrame(rows)

    print(f"{'Specification':<44} | {'Effect':>8} | {'SE':>6} | {'p':>7}")
    print("-" * 76)
    for _, r in results_df.iterrows():
        print(
            f"{r['specification']:<44} | {r['effect']:>+8.2f} | {r['se']:>6.2f} | "
            f"{r['p_value']:>7.4f}"
        )

    spread = results_df["effect"].max() - results_df["effect"].min()
    print("\nINTERPRETATION:")
    print(f"  The estimate moves by {spread:.2f} deaths across these controls.")
    print("  This is the claim the crash-composition shares could not support:")
    print("  release days are not unusually wet in a way that accounts")
    print("  for the excess.")

    return results_df
