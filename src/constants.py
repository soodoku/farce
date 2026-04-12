"""
Album release dates and holiday definitions for FARS analysis.

The Tier 1 albums are the 10 most-streamed albums on first day (2017-2022),
which form the treatment group in Patel et al. (2026).
"""

import datetime
import pandas as pd

# Tier 1: Top 10 most-streamed albums on first day, 2017-2022
# (the paper's treatment group)
ALBUMS_TIER1 = [
    # (Artist, Album, Release Date, Day of Week, ~First-day streams M)
    ("Taylor Swift", "Midnights", "2022-10-21", "Friday", 184.6),
    ("Drake", "Certified Lover Boy", "2021-09-03", "Friday", 153.4),
    ("Bad Bunny", "Un Verano Sin Ti", "2022-05-06", "Friday", 145.8),
    ("Drake", "Scorpion", "2018-06-29", "Friday", 132.3),
    ("Kendrick Lamar", "Mr. Morale & The Big Steppers", "2022-05-13", "Friday", 100),
    ("Kanye West", "Donda", "2021-08-29", "Sunday", 95),
    ("Taylor Swift", "Red (Taylor's Version)", "2021-11-12", "Friday", 91),
    ("Taylor Swift", "Folklore", "2020-07-24", "Friday", 80),
    ("Harry Styles", "Harry's House", "2022-05-20", "Friday", 76),
    ("Drake & 21 Savage", "Her Loss", "2022-11-04", "Friday", 75),
]

# Tier 2: Albums 11-20 by approximate first-day streams, 2017-2022
# These are the "dose-response" control: big albums, but smaller streaming
# spikes than the top 10. If the mechanism is streaming-induced distraction,
# we should see a positive but attenuated fatality effect.
ALBUMS_TIER2 = [
    ("Travis Scott", "ASTROWORLD", "2018-08-03", "Friday", 72),
    ("Post Malone", "beerbongs & bentleys", "2018-04-27", "Friday", 70),
    ("Post Malone", "Hollywood's Bleeding", "2019-09-06", "Friday", 65),
    ("Billie Eilish", "Happier Than Ever", "2021-07-30", "Friday", 60),
    ("Juice WRLD", "Legends Never Die", "2020-07-10", "Friday", 58),
    ("Ariana Grande", "thank u, next", "2019-02-08", "Friday", 56),
    ("Olivia Rodrigo", "SOUR", "2021-05-21", "Friday", 55),
    ("The Weeknd", "After Hours", "2020-03-20", "Friday", 53),
    ("Ed Sheeran", "= (Equals)", "2021-10-29", "Friday", 51),
    ("Ariana Grande", "Positions", "2020-10-30", "Friday", 50),
]

# Backward-compatible: ALBUMS is tier 1 without the streams column
ALBUMS = [(a[0], a[1], a[2], a[3]) for a in ALBUMS_TIER1]
ALBUMS_ALL = ALBUMS_TIER1 + ALBUMS_TIER2

RELEASE_DATES = {d.date() for d in pd.to_datetime([a[2] for a in ALBUMS])}
RELEASE_DATES_TIER2 = {d.date() for d in pd.to_datetime([a[2] for a in ALBUMS_TIER2])}
RELEASE_DATES_ALL = RELEASE_DATES | RELEASE_DATES_TIER2


def us_holidays(years):
    """Generate US federal holiday dates for given years."""
    holidays = []
    for y in years:
        holidays.append(datetime.date(y, 1, 1))  # New Year
        holidays.append(datetime.date(y, 7, 4))  # Independence Day
        holidays.append(datetime.date(y, 12, 25))  # Christmas
        holidays.append(datetime.date(y, 11, 11))  # Veterans Day
        # MLK: 3rd Monday of January
        d = datetime.date(y, 1, 1)
        while d.weekday() != 0:
            d += datetime.timedelta(1)
        holidays.append(d + datetime.timedelta(14))
        # Presidents Day: 3rd Monday of February
        d = datetime.date(y, 2, 1)
        while d.weekday() != 0:
            d += datetime.timedelta(1)
        holidays.append(d + datetime.timedelta(14))
        # Memorial Day: last Monday of May
        d = datetime.date(y, 5, 31)
        while d.weekday() != 0:
            d -= datetime.timedelta(1)
        holidays.append(d)
        # Labor Day: 1st Monday of September
        d = datetime.date(y, 9, 1)
        while d.weekday() != 0:
            d += datetime.timedelta(1)
        holidays.append(d)
        # Thanksgiving: 4th Thursday of November
        d = datetime.date(y, 11, 1)
        while d.weekday() != 3:
            d += datetime.timedelta(1)
        holidays.append(d + datetime.timedelta(21))
    return set(holidays)
