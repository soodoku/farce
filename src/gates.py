"""
Build-time assertions.

These are cheap checks that would have caught real defects in this repository:
an album list that drifts from the paper's Table 1, a daily series with holes
in it, a FARS extract from the wrong vintage, and treated days quietly serving
as controls for other albums. They run at the top of the pipeline and raise
rather than warn.
"""

import pandas as pd

from src.constants import ALBUMS_TIER1

# Patel et al. (2026), Table 1: the ten most-streamed albums in a single day,
# 2017-2022, with first-day US Spotify streams. Transcribed from w34866.pdf.
PAPER_TABLE1 = {
    "2022-10-21": ("Midnights", 184.695609),
    "2021-09-03": ("Certified Lover Boy", 153.441565),
    "2022-05-06": ("Un Verano Sin Ti", 145.811373),
    "2018-06-29": ("Scorpion", 132.384203),
    "2022-05-13": ("Mr. Morale & The Big Steppers", 99.582729),
    "2022-05-20": ("Harry's House", 97.621794),
    "2022-11-04": ("Her Loss", 97.390844),
    "2021-08-29": ("Donda", 94.455883),
    "2021-11-12": ("Red (Taylor's Version)", 90.556180),
    "2020-07-24": ("Folklore", 79.443136),
}

# FARS Final File national fatality counts, from NHTSA's published annual
# totals. The Annual Report File figures differ by a few hundred; using the
# wrong vintage is easy to do and invisible in a day-level analysis.
FARS_FINAL_FILE_TOTALS = {
    2015: 35484,
    2016: 37806,
    2017: 37473,
    2018: 36835,
    2019: 36355,
    2020: 39007,
    2021: 43230,
    2022: 42721,
    2023: 41025,
    2024: 39254,
}


def check_album_list():
    """Tier 1 must be the paper's ten albums, at the paper's streaming counts."""
    ours = {a[2]: (a[1], a[4]) for a in ALBUMS_TIER1}

    missing = set(PAPER_TABLE1) - set(ours)
    extra = set(ours) - set(PAPER_TABLE1)
    if missing or extra:
        raise AssertionError(
            f"Tier 1 does not match the paper's Table 1. missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )

    for date, (album, streams) in PAPER_TABLE1.items():
        our_album, our_streams = ours[date]
        if abs(our_streams - streams) > 1.5:
            raise AssertionError(
                f"{album} ({date}): albums.csv has {our_streams}M first-day "
                f"streams, the paper's Table 1 reports {streams:.1f}M"
            )

    return len(PAPER_TABLE1)


def check_daily_series(daily):
    """One row per calendar day, no gaps, no duplicates, totals from the Final File."""
    if daily["date"].duplicated().any():
        dupes = daily.loc[daily["date"].duplicated(), "date"].tolist()
        raise AssertionError(f"duplicate dates in the daily series: {dupes[:5]}")

    span = (daily["date"].max() - daily["date"].min()).days + 1
    if len(daily) != span:
        gaps = pd.date_range(daily["date"].min(), daily["date"].max()).difference(
            daily["date"]
        )
        raise AssertionError(
            f"daily series has {len(daily)} rows for a {span}-day span; "
            f"missing {list(gaps[:5])}"
        )

    annual = daily.assign(y=daily["date"].dt.year).groupby("y")["fatalities"].sum()
    for year, expected in FARS_FINAL_FILE_TOTALS.items():
        if year not in annual.index:
            continue
        if int(annual.loc[year]) != expected:
            raise AssertionError(
                f"{year} fatalities = {int(annual.loc[year])}, FARS Final File "
                f"reports {expected}. Wrong data vintage or a dropped record."
            )

    return len(daily)


def window_overlap(albums=None, window=10):
    """
    Treated days that fall inside another album's control window.

    In the stacked design each album contributes its own +/-window of control
    days, so a release that lands within `window` days of another release is
    counted as a control for it. This deflates the pooled estimate. The paper
    has the same issue; the point of reporting it is that neither analysis can
    call the control window clean.

    Returns a DataFrame of (album, contaminating album, day offset).
    """
    albums = albums or ALBUMS_TIER1
    dates = [(a[1], pd.to_datetime(a[2])) for a in albums]

    rows = []
    for name_i, di in dates:
        for name_j, dj in dates:
            offset = (dj - di).days
            if name_i != name_j and 0 < abs(offset) <= window:
                rows.append(
                    {
                        "album": name_i,
                        "contaminated_by": name_j,
                        "day_offset": offset,
                    }
                )

    return pd.DataFrame(rows, columns=["album", "contaminated_by", "day_offset"])


def run_all(daily, window=10):
    """Run every gate and print a one-line summary of each."""
    print(f"\n{'='*70}")
    print("BUILD GATES")
    print(f"{'='*70}")

    n_albums = check_album_list()
    print(f"  albums: Tier 1 matches Patel et al. Table 1 ({n_albums} albums)")

    n_days = check_daily_series(daily)
    print(
        f"  daily series: {n_days} rows, {daily['date'].min().date()} to "
        f"{daily['date'].max().date()}, annual totals match FARS Final File"
    )

    overlap = window_overlap(window=window)
    n_control_rows = 2 * window * len(ALBUMS_TIER1)
    print(
        f"  control window: {len(overlap)} of {n_control_rows} control-day rows "
        "are themselves release days"
    )
    for _, r in overlap.iterrows():
        print(
            f"    {r['album']} <- {r['contaminated_by']} " f"(day {r['day_offset']:+d})"
        )

    return overlap
