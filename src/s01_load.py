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
    """Load FARS Accident CSVs downloaded manually from NHTSA."""
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
    return pd.concat(frames, ignore_index=True)
