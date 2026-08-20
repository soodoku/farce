"""
FARS data loading functions.

Supports both API download and local CSV loading.
"""

import io
import time
from pathlib import Path

import pandas as pd

CACHE_FILE = Path("fars_accident_cache.parquet")
API_BASE = "https://crashviewer.nhtsa.dot.gov/CrashAPI/FARSData/GetFARSData"

# FIPS state codes (1-56, skipping gaps)
STATE_CODES = list(range(1, 57))


def download_fars(years=range(2017, 2023)):
    """Download FARS Accident data via the Crash API, return DataFrame."""
    if CACHE_FILE.exists():
        print(f"Loading cached data from {CACHE_FILE}")
        return pd.read_parquet(CACHE_FILE)

    import requests

    frames = []
    for year in years:
        print(f"Downloading FARS Accident data for {year}...")
        url = (
            f"{API_BASE}?dataset=Accident"
            f"&FromYear={year}&ToYear={year}&State=0&format=csv"
        )
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            if len(resp.content) > 100:
                df = pd.read_csv(io.StringIO(resp.text))
                frames.append(df)
                print(f"  {year}: {len(df)} crashes")
                continue
        except Exception as e:
            print(f"  State=0 failed for {year}: {e}")

        # Fallback: download state by state
        for st in STATE_CODES:
            try:
                url = (
                    f"{API_BASE}?dataset=Accident"
                    f"&FromYear={year}&ToYear={year}&State={st}&format=csv"
                )
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200 and len(resp.content) > 100:
                    df = pd.read_csv(io.StringIO(resp.text))
                    frames.append(df)
            except Exception:
                pass
            time.sleep(0.1)
        print(f"  {year}: done (state-by-state)")

    if not frames:
        raise RuntimeError(
            "Could not download FARS data. Try downloading manually from\n"
            "https://www.nhtsa.gov/file-downloads and placing accident CSVs\n"
            "in this directory, then rerun with --local flag."
        )

    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(CACHE_FILE)
    print(f"Cached {len(data)} crash records to {CACHE_FILE}")
    return data


def load_local_fars(directory="."):
    """
    Load FARS Accident CSVs downloaded manually from NHTSA.

    Column names are normalised to upper case before concatenation. FARS is
    not consistent about case across vintages: the 2007 file names its
    coordinate columns `latitude` and `longitud` while every later file uses
    `LATITUDE` and `LONGITUD`. Concatenating without normalising produces two
    columns for one variable, so `df["LATITUDE"]` returns a DataFrame rather
    than a Series and any code doing `{c.upper(): c for c in df.columns}`
    silently keeps whichever came last, dropping the other year's values.

    The 2021 and 2022 files also carry a UTF-8 byte-order mark, which makes
    their first column `\ufeffSTATE` rather than `STATE`. Any lookup by name
    misses it, silently, for exactly those two years. Both problems are
    normalised away here rather than worked around at each call site.
    """
    frames = []
    for f in sorted(Path(directory).glob("*ccident*.csv")):
        print(f"Loading {f.name}...")
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except UnicodeDecodeError:
            frames.append(pd.read_csv(f, encoding="latin-1", low_memory=False))
    for f in sorted(Path(directory).glob("*CCIDENT*.CSV")):
        print(f"Loading {f.name}...")
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except UnicodeDecodeError:
            frames.append(pd.read_csv(f, encoding="latin-1", low_memory=False))
    if not frames:
        raise FileNotFoundError(
            "No FARS Accident CSV files found. Download from:\n"
            "https://www.nhtsa.gov/file-downloads\n"
            "Look for FARS > [year] > National > accident.csv"
        )
    for i, frame in enumerate(frames):
        frame.columns = [c.replace("\ufeff", "").strip().upper() for c in frame.columns]
        frames[i] = frame.loc[:, ~frame.columns.duplicated()]

    return pd.concat(frames, ignore_index=True)


def load_local_miacc(directory="."):
    """
    Load FARS multiple-imputation BAC files (MIACC).

    NHTSA imputes blood alcohol concentration for crashes where it was not
    measured, and publishes ten imputed values per crash as columns A1 to A10.
    Values are in units of 0.01 g/dL, so the legal threshold of 0.08 is 8.

    This exists because `DRUNK_DR` left the accident file after 2020. Reading
    only that column restricted the alcohol analysis to 2007-2020 and so to two
    of the ten albums, which was a property of the file we read rather than of
    the data: MIACC ships in every annual archive from 2007 to 2024.

    `ST_CASE` is unique only within a year and MIACC carries no year column, so
    the year is taken from the filename and returned as a column. Join on
    (YEAR, ST_CASE).
    """
    frames = []
    for path in sorted(Path(directory).glob("miacc_*.csv")):
        year = int(path.stem.split("_")[-1])
        try:
            frame = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="latin-1", low_memory=False)
        frame.columns = [c.replace("\ufeff", "").strip().upper() for c in frame.columns]
        frame = frame.loc[:, ~frame.columns.duplicated()]
        frame["YEAR"] = year
        frames.append(frame)
        print(f"Loading {path.name}...")

    if not frames:
        raise FileNotFoundError(
            "No MIACC files found. Run `make extract`, which pulls MIACC.CSV "
            "out of each FARS annual archive alongside accident.csv."
        )

    return pd.concat(frames, ignore_index=True)
