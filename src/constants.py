"""
Album release dates and holiday definitions for FARS analysis.

Album data is loaded from data/albums.csv.
See data/albums_sources.md for data provenance.

Tiers:
  0 = Pre-2018 streaming era (2015-2017) - test whether effect exists before paper's cutoff
  1 = Paper's original 10 albums (2018-2022)
  2 = Extended analysis albums 11-20
  3 = Post-2022 and additional albums for extended analysis
"""

import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_albums():
    """Load album data from CSV."""
    df = pd.read_csv(DATA_DIR / "albums.csv")
    albums = []
    for _, row in df.iterrows():
        albums.append(
            (
                row["artist"],
                row["album"],
                row["release_date"],
                row["day_of_week"],
                row["streams_millions"],
                row["tier"],
                row["paper_sample"],
            )
        )
    return albums


_albums = load_albums()

# Pre-2018 streaming era (tier 0) - test whether effect exists before paper's cutoff
ALBUMS_TIER0 = [(a[0], a[1], a[2], a[3], a[4]) for a in _albums if a[5] == 0]

# Paper's original sample (tier 1, paper_sample=TRUE)
ALBUMS_TIER1 = [(a[0], a[1], a[2], a[3], a[4]) for a in _albums if a[5] == 1]

# Extended analysis (tier 2)
ALBUMS_TIER2 = [(a[0], a[1], a[2], a[3], a[4]) for a in _albums if a[5] == 2]

# Post-2022 and additional albums (tier 3)
ALBUMS_TIER3 = [(a[0], a[1], a[2], a[3], a[4]) for a in _albums if a[5] == 3]

# Backward compatibility
ALBUMS = [(a[0], a[1], a[2], a[3]) for a in ALBUMS_TIER1]
ALBUMS_ALL = ALBUMS_TIER1 + ALBUMS_TIER2
ALBUMS_EXTENDED = ALBUMS_TIER0 + ALBUMS_TIER1 + ALBUMS_TIER2 + ALBUMS_TIER3
ALBUMS_FULL = ALBUMS_EXTENDED

# Release date sets
RELEASE_DATES_TIER0 = {d.date() for d in pd.to_datetime([a[2] for a in ALBUMS_TIER0])}
RELEASE_DATES = {d.date() for d in pd.to_datetime([a[2] for a in ALBUMS])}
RELEASE_DATES_TIER2 = {d.date() for d in pd.to_datetime([a[2] for a in ALBUMS_TIER2])}
RELEASE_DATES_TIER3 = {d.date() for d in pd.to_datetime([a[2] for a in ALBUMS_TIER3])}
RELEASE_DATES_ALL = RELEASE_DATES | RELEASE_DATES_TIER2
RELEASE_DATES_EXTENDED = (
    RELEASE_DATES_TIER0 | RELEASE_DATES | RELEASE_DATES_TIER2 | RELEASE_DATES_TIER3
)


def us_holidays(years):
    """Generate US federal holiday dates for given years."""
    holidays = []
    for y in years:
        holidays.append(datetime.date(y, 1, 1))
        holidays.append(datetime.date(y, 7, 4))
        holidays.append(datetime.date(y, 12, 25))
        holidays.append(datetime.date(y, 11, 11))
        d = datetime.date(y, 1, 1)
        while d.weekday() != 0:
            d += datetime.timedelta(1)
        holidays.append(d + datetime.timedelta(14))
        d = datetime.date(y, 2, 1)
        while d.weekday() != 0:
            d += datetime.timedelta(1)
        holidays.append(d + datetime.timedelta(14))
        d = datetime.date(y, 5, 31)
        while d.weekday() != 0:
            d -= datetime.timedelta(1)
        holidays.append(d)
        d = datetime.date(y, 9, 1)
        while d.weekday() != 0:
            d += datetime.timedelta(1)
        holidays.append(d)
        d = datetime.date(y, 11, 1)
        while d.weekday() != 3:
            d += datetime.timedelta(1)
        holidays.append(d + datetime.timedelta(21))
    return set(holidays)
