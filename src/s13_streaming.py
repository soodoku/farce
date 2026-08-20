"""
US streaming volume from Spotify Charts, and what it can and cannot control for.

The paper's own streaming series is the daily US top-200 chart. We reproduce it
closely: its control-day mean is 86.1 million streams and ours is 86.8, which
is the check that this is the same data rather than something that resembles it.

Three things become possible with it, and one of them needs care.

The alternative the paper cannot address with fatalities alone is that a week
in which more people are out and about produces both more streaming and more
driving. Controlling for total daily streaming does not test that, because
total streaming on a release day is caused by the release: conditioning on it
would absorb the mechanism rather than the confounder. What is needed is
ambient listening that the release did not cause, so the control here is
background streaming, the top-200 total with every track released that day
removed.

The other two are simpler. Measured first-day streams replace the counts
estimated from chart position, and a ranking of all albums by measured
first-day streams gives a sampling frame anyone can reproduce.

One limit throughout: the chart is the top 200 songs, so an album contributes
only its charting tracks. That is the same censoring the paper works under.
"""

import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import ALBUMS_TIER1
from src.utils import add_time_features, album_stats, build_design_matrix, ols_fit

DATA_DIR = Path(__file__).parent.parent / "data" / "spotify"
US_DAILY = DATA_DIR / "us_daily.csv"


def load_streaming():
    """Daily US top-200 rows, with a background series that excludes new releases."""
    if not US_DAILY.exists():
        raise FileNotFoundError(
            f"{US_DAILY} not found. See data/spotify/README.md for the pull."
        )
    df = pd.read_csv(
        US_DAILY,
        usecols=[
            "date",
            "rank",
            "uri",
            "artist_names",
            "track_name",
            "streams",
            "release_date",
        ],
        low_memory=False,
    )
    df["date"] = pd.to_datetime(df["date"])
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["is_new"] = df["release_date"] == df["date"]

    total = df.groupby("date")["streams"].sum().rename("streams_total")
    background = (
        df[~df["is_new"]].groupby("date")["streams"].sum().rename("streams_background")
    )
    daily = pd.concat([total, background], axis=1).reset_index()
    daily["streams_new"] = daily["streams_total"] - daily["streams_background"]
    for col in ["streams_total", "streams_background", "streams_new"]:
        daily[col] = daily[col] / 1e6

    print(
        f"  streaming series: {len(daily)} days, "
        f"{daily['date'].min().date()} to {daily['date'].max().date()}, "
        f"median background {daily['streams_background'].median():.1f}M"
    )
    return df, daily


def benchmark_against_paper(streaming_daily, albums=None, window=10):
    """
    Two numbers the paper publishes that our reconstruction has to reproduce.

    Patel et al. report 123.3 million top-200 US streams on the ten release
    days and 86.1 million (95% CI 85.3 to 87.0) on the ten days surrounding
    each release. Both are means over release windows only, not over the whole
    series, so the comparison has to use the same day set or it is not a check
    at all: the series-wide control mean is a different quantity and lands
    somewhere else.
    """
    albums = albums or ALBUMS_TIER1
    d = streaming_daily.set_index("date")["streams_total"]
    rel, ctl = [], []
    for a in albums:
        dt = pd.Timestamp(a[2])
        if dt not in d.index:
            continue
        rel.append(d.loc[dt])
        span = pd.date_range(
            dt - pd.Timedelta(days=window), dt + pd.Timedelta(days=window)
        )
        ctl.extend(d.reindex(span).drop(index=dt, errors="ignore").dropna())
    rel, ctl = np.asarray(rel), np.asarray(ctl)
    ctl_se = ctl.std(ddof=1) / np.sqrt(len(ctl))
    return {
        "n_albums": len(rel),
        "release_mean": rel.mean(),
        "control_mean": ctl.mean(),
        "control_lo": ctl.mean() - 1.96 * ctl_se,
        "control_hi": ctl.mean() + 1.96 * ctl_se,
        "n_control_days": len(ctl),
    }


def shared_shock_test(fatalities, streaming_daily, albums=None, window=10):
    """
    Does ambient listening explain the release-day effect?

    Two questions, asked in order. First, whether background streaming predicts
    fatalities at all once the usual fixed effects are in: if busy days really
    do produce both more streaming and more driving, that association has to
    exist somewhere. Second, whether adding it as a control moves the
    release-day estimate.

    Background streaming excludes tracks released that day, so it is ambient
    listening rather than the release itself. Total streaming is reported
    alongside it purely to show how differently the two behave; it is a bad
    control and its row should not be read as a robustness check.
    """
    print("\n" + "=" * 70)
    print("SHARED-SHOCK TEST: DOES AMBIENT LISTENING EXPLAIN IT?")
    print("=" * 70)

    albums = albums or ALBUMS_TIER1
    df = add_time_features(fatalities).merge(streaming_daily, on="date", how="inner")
    release_dates = [pd.to_datetime(a[2]) for a in albums]
    covered = [d for d in release_dates if (df["date"] == d).any()]
    print(
        f"  {len(df)} days with both fatalities and streaming; "
        f"{len(covered)} of {len(albums)} release days covered\n"
    )

    exclude = set()
    for dt in release_dates:
        for offset in range(-window, window + 1):
            exclude.add(dt.date() + datetime.timedelta(days=offset))
    est = (~df["date"].dt.date.isin(exclude)).values
    y = df["fatalities"].values.astype(float)

    # does background streaming predict fatalities on ordinary days?
    X = build_design_matrix(df, controls=["streams_background"])
    beta, se, _, _ = ols_fit(X.values[est], y[est], return_se=True)
    idx = list(X.columns).index("streams_background")
    print("Association of background streaming with fatalities, release windows")
    print("excluded from the fit:")
    print(
        f"  {beta[idx]:+.4f} deaths per million streams "
        f"(SE {se[idx]:.4f}, t = {beta[idx]/se[idx]:+.2f})"
    )
    print(
        f"  a one standard deviation move in background streaming "
        f"({df['streams_background'].std():.1f}M) is "
        f"{beta[idx] * df['streams_background'].std():+.2f} deaths\n"
    )

    rows = [
        {"specification": "no streaming control", "controls": []},
        {"specification": "background streaming", "controls": ["streams_background"]},
        {
            "specification": "total streaming (BAD CONTROL, shown for contrast)",
            "controls": ["streams_total"],
        },
    ]
    out = []
    for row in rows:
        X = build_design_matrix(df, controls=row["controls"] or None)
        b, _, _ = ols_fit(X.values[est], y[est])
        resid = y - X.values @ b
        st = album_stats([resid[(df["date"] == d).values][0] for d in covered])
        out.append(
            {
                "specification": row["specification"],
                "n_albums": st["n"],
                "effect": st["effect"],
                "se": st["se"],
                "ci_lower": st["ci_lower"],
                "ci_upper": st["ci_upper"],
                "p_value": st["p_value"],
            }
        )

    results = pd.DataFrame(out)
    results["bg_coef"] = beta[idx]
    results["bg_se"] = se[idx]
    results["bg_sd_move"] = beta[idx] * df["streams_background"].std()
    results["bg_sd"] = df["streams_background"].std()
    results["control_day_streams"] = df.loc[est, "streams_total"].mean()
    print(f"{'Specification':<48} | {'Effect':>8} | {'SE':>6} | {'p':>6}")
    print("-" * 78)
    for _, r in results.iterrows():
        print(
            f"{r['specification']:<48} | {r['effect']:>+8.2f} | {r['se']:>6.2f} | "
            f"{r['p_value']:>6.3f}"
        )

    print("\nINTERPRETATION:")
    print("  Background streaming is ambient listening the release did not")
    print("  cause. If the shared-shock story were driving the result, adding")
    print("  it would move the estimate. The total-streaming row is included")
    print("  only to show what conditioning on a consequence of treatment does.")
    return results


def measured_first_day(streaming_rows, albums=None, min_tracks=3):
    """
    First-day US streams for every album, measured rather than estimated.

    An album's first day is the set of charting tracks whose own release date
    equals the chart date. Albums are identified by lead artist and date, and
    a minimum charting-track count keeps single releases out.

    The paper's Table 1 counts are roughly twice these. Its largest,
    184,695,609 for Midnights, is the widely reported global first-day figure,
    so Table 1 appears to be global while the paper's streaming series is US.
    A US dose is the better match for a US fatality outcome.
    """
    df = streaming_rows[streaming_rows["is_new"]].copy()
    df["lead_artist"] = df["artist_names"].str.split("|").str[0]
    out = (
        df.groupby(["date", "lead_artist"])
        .agg(first_day_streams=("streams", "sum"), n_tracks=("uri", "nunique"))
        .reset_index()
    )
    out = out[out["n_tracks"] >= min_tracks]
    out["first_day_millions"] = out["first_day_streams"] / 1e6
    return out.sort_values("first_day_streams", ascending=False).reset_index(drop=True)


def paper_album_keys(first_day, albums=None):
    """
    Locate the paper's ten albums inside the measured first-day frame.

    Matching on release date alone is wrong and was wrong here for one figure:
    nineteen albums charted on those ten dates, so a date-only match circles
    nine albums the paper never studied, at doses an order of magnitude below
    the real ones. The key is lead artist and date together.

    Lead artist needs normalising because a joint credit is stored two ways.
    Her Loss is "Drake & 21 Savage" in the album list and "Drake|21 Savage" in
    the chart, whose lead is "Drake".
    """
    albums = albums or ALBUMS_TIER1
    keys, missing = [], []
    for artist, title, date, *_ in albums:
        lead = re.split(r"\s*(?:&|,| and )\s*", artist)[0].strip()
        hit = first_day[
            (first_day["lead_artist"] == lead)
            & (first_day["date"] == pd.Timestamp(date))
        ]
        if len(hit):
            keys.append((lead, pd.Timestamp(date), title))
        else:
            missing.append(title)
    if missing:
        print(f"  no measured first day for: {', '.join(missing)}")
    return keys


def dose_matched_contrast(per_album, first_day, albums=None, years=(2018, 2022)):
    """
    Compare the paper's ten with every other album of comparable size.

    The dose-response correlation across all charting albums is near zero, and
    the decile means look like a threshold only because the paper's own ten
    occupy the top of the dose distribution. The test that separates the two
    readings is to hold dose fixed: take every album whose measured first-day
    US streams reach the smallest of the paper's ten, and set the paper's
    albums against the rest of that set.

    This is a rule, not a list. Nothing about it depends on which albums anyone
    considered major, and the comparison albums are matched on the paper's own
    treatment intensity rather than on a judgement about fame.

    The year restriction is reported because the comparison albums run to 2024
    while the paper's stop in 2022. It is a check, not the headline: the
    residual model already carries year effects.
    """
    d = per_album.dropna(subset=["effect", "first_day_millions"]).copy()
    keys = {(a, dt) for a, dt, _ in paper_album_keys(first_day, albums)}
    d["is_paper"] = [
        (a, pd.Timestamp(x)) in keys for a, x in zip(d["lead_artist"], d["date"])
    ]
    cut = d.loc[d["is_paper"], "first_day_millions"].min()
    big = d[d["first_day_millions"] >= cut - 1e-9]
    other = big[~big["is_paper"]]
    in_window = other["date"].dt.year.between(*years)

    rows = [
        ("the paper's ten", big.loc[big["is_paper"], "effect"]),
        (f"every other album reaching {cut:.1f}M", other["effect"]),
        (
            f"the same, {years[0]}-{years[1]} only",
            other.loc[in_window, "effect"],
        ),
        (f"albums below {cut:.1f}M", d.loc[~d.index.isin(big.index), "effect"]),
    ]
    out = []
    for label, x in rows:
        st = album_stats(list(x))
        out.append(
            {
                "group": label,
                **{
                    k: st[k]
                    for k in ("n", "effect", "se", "ci_lower", "ci_upper", "p_value")
                },
            }
        )
    results = pd.DataFrame(out)

    print("\n" + "=" * 70)
    print("DOSE-MATCHED CONTRAST")
    print("=" * 70)
    print(f"  cutoff {cut:.2f}M first-day US streams, the smallest of the ten")
    print(
        f"  {len(big)} albums reach it, {int(big['is_paper'].sum())} of them the paper's\n"
    )
    print(f"{'Group':<40} | {'n':>4} | {'Effect':>7} | {'SE':>6}")
    print("-" * 66)
    for _, r in results.iterrows():
        print(
            f"{r['group']:<40} | {r['n']:>4.0f} | {r['effect']:>+7.2f} | {r['se']:>6.2f}"
        )
    print("\nINTERPRETATION:")
    print("  Holding measured dose at or above the paper's own smallest release,")
    print("  the albums the paper did not study show no release-day excess. The")
    print("  effect is a property of the ten albums, not of releases that size.")
    return results, cut


def reproducible_frame(
    streaming_rows,
    fatality_residuals,
    start="2018-01-01",
    end="2022-12-31",
    sizes=(10, 15, 20, 30, 50),
    min_tracks=3,
):
    """
    A sampling frame anyone can rebuild, and what the estimate does inside it.

    The album panel in this repository was assembled by hand, which is the
    weakest thing about it: no rule generates it, so no one can check whether a
    different rule would give a different answer. Ranking every album in the
    period by measured first-day US streams supplies the rule. "The top N
    albums released between these dates" is reproducible from the chart alone.

    Reporting several N is the point rather than a robustness afterthought. If
    the effect is a property of the biggest releases it should persist as the
    frame widens; if it is a property of the extreme tail it should decay.

    Returns (frame, results) where frame is the ranked album list.
    """
    print("\n" + "=" * 70)
    print("REPRODUCIBLE SAMPLING FRAME")
    print("=" * 70)
    print("Albums ranked by measured first-day US streams, not chosen by hand.\n")

    frame = measured_first_day(streaming_rows, min_tracks=min_tracks)
    frame = frame[
        (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
    ].copy()
    frame = frame.sort_values("first_day_streams", ascending=False).reset_index(
        drop=True
    )
    frame["frame_rank"] = frame.index + 1

    print(f"{'Rank':>5} | {'Date':<12} | {'Lead artist':<18} | {'First-day M':>11}")
    print("-" * 58)
    for _, r in frame.head(10).iterrows():
        print(
            f"{int(r['frame_rank']):>5} | {r['date'].date()!s:<12} | "
            f"{r['lead_artist'][:17]:<18} | {r['first_day_millions']:>11.1f}"
        )

    rows = []
    for n in sizes:
        sub = frame.head(n)
        effects = [
            fatality_residuals[d] for d in sub["date"] if d in fatality_residuals.index
        ]
        st = album_stats(effects)
        rows.append(
            {
                "frame_size": n,
                "n_albums": st["n"],
                "min_first_day_millions": float(sub["first_day_millions"].min()),
                "effect": st["effect"],
                "se": st["se"],
                "ci_lower": st["ci_lower"],
                "ci_upper": st["ci_upper"],
                "p_value": st["p_value"],
            }
        )

    results = pd.DataFrame(rows)
    print(f"\n{'Frame':>6} | {'n':>3} | {'Cutoff M':>9} | {'Effect':>8} | {'p':>6}")
    print("-" * 50)
    for _, r in results.iterrows():
        print(
            f"top {int(r['frame_size']):>2} | {int(r['n_albums']):>3} | "
            f"{r['min_first_day_millions']:>9.1f} | {r['effect']:>+8.2f} | "
            f"{r['p_value']:>6.3f}"
        )

    print("\nINTERPRETATION:")
    print("  The effect is present at every frame size and falls as the frame")
    print("  widens. That is what selection on the largest releases looks like:")
    print("  real, and largest where the sample is most extreme.")
    return frame, results


def dose_response_all(streaming_rows, fatality_residuals, min_tracks=3, n_bins=10):
    """
    Dose-response across every album on the chart, not just the chosen ten.

    The dose-response test in the original critique used ten albums, then
    twenty, and reported a negative correlation as though a null were
    informative. At ten albums a correlation must exceed 0.63 to register, so
    it could not have found anything. Every album with a measured first-day
    total and a fatality residual gives several hundred, which is the first
    version of this test with any power.

    The measured dose also differs from the stated one in a way that mattered:
    the paper's figures are global and the outcome is national, and the gap is
    largest for artists whose audience is least American.

    Returns (per-album frame, binned frame, summary dict).
    """
    print("\n" + "=" * 70)
    print("DOSE-RESPONSE ACROSS EVERY CHARTING ALBUM")
    print("=" * 70)

    fd = measured_first_day(streaming_rows, min_tracks=min_tracks)
    lo, hi = fatality_residuals.index.min(), fatality_residuals.index.max()
    fd = fd[(fd["date"] >= lo) & (fd["date"] <= hi)].copy()
    fd["effect"] = [fatality_residuals.get(d, np.nan) for d in fd["date"]]
    fd = fd.dropna(subset=["effect"]).reset_index(drop=True)

    n = len(fd)
    r, p = stats.pearsonr(fd["first_day_millions"], fd["effect"])
    rho, rho_p = stats.spearmanr(fd["first_day_millions"], fd["effect"])
    t_crit = stats.t.ppf(0.975, n - 2)
    r_min = t_crit / np.sqrt(t_crit**2 + n - 2)
    slope, intercept = np.polyfit(fd["first_day_millions"], fd["effect"], 1)

    fd["dose_bin"] = pd.qcut(fd["first_day_millions"], n_bins, labels=False)
    binned = (
        fd.groupby("dose_bin")
        .agg(
            n=("effect", "size"),
            dose_median=("first_day_millions", "median"),
            effect_mean=("effect", "mean"),
            effect_se=("effect", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        )
        .reset_index()
    )

    print(f"  {n} albums with a measured dose and a fatality residual")
    print(
        f"  dose range {fd['first_day_millions'].min():.1f}M to "
        f"{fd['first_day_millions'].max():.1f}M, median "
        f"{fd['first_day_millions'].median():.1f}M"
    )
    print(f"\n  Pearson r = {r:+.3f} (p = {p:.3f}), Spearman = {rho:+.3f}")
    print(f"  detectable at this n: |r| > {r_min:.3f}")
    print(f"  slope {slope:+.4f} deaths per million first-day streams,")
    print(f"  so an 80M release implies {slope * 80:+.1f} deaths")
    print(
        f"\n  top decile (median {binned['dose_median'].iloc[-1]:.1f}M): "
        f"{binned['effect_mean'].iloc[-1]:+.2f} deaths"
    )
    print(f"  deciles 1-9 mean: {binned['effect_mean'].iloc[:-1].mean():+.2f}")

    print("\nINTERPRETATION:")
    print("  The sign is positive here where the ten-album version was")
    print("  negative. That earlier result was a null from a test with no")
    print("  power, paired with a global dose and a national outcome. This one")
    print("  still does not clear significance, but it is the first version")
    print("  that could have.")

    return (
        fd,
        binned,
        {
            "n": n,
            "pearson": r,
            "pearson_p": p,
            "spearman": rho,
            "r_min_detectable": r_min,
            "slope_per_million": slope,
            "top_decile_effect": float(binned["effect_mean"].iloc[-1]),
        },
    )
