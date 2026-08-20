# Spotify Charts, United States

`us_daily.csv` holds one row per track per day on the US Spotify top 200,
2017-01-01 onward. Everything in `src/s13_streaming.py` is built from it.
It is 160 MB, so it is not in the repository.

## Getting it

```bash
make streaming   # needs a Kaggle API token in ~/.kaggle/kaggle.json
                 # and the Kaggle CLI (pipx install kaggle, or uv on PATH)
```

The source is Kaggle dataset `gonzalopezgil/spotify-charts-daily-updated`,
file `charts_songs_daily.csv`, which covers every country Spotify publishes.
The target keeps `country == "us"` and drops the rest.

## What the columns mean

`streams` is that track's plays in the United States on `date`.
`release_date` is the track's own release date, which is what separates a
release-day track from the ambient catalogue: a track is new on a given chart
day when `release_date == date`.

An album's measured first-day volume is the sum of `streams` over the charting
tracks by one lead artist whose release date is the chart date. Three charting
tracks are required, which keeps singles and features out of the album frame.

## Why not the paper's numbers

Table 1 of Patel et al. reports first-day counts roughly twice what the US
chart shows, and its largest, 184,695,609 for *Midnights*, is the widely
reported **global** first-day figure. The paper's dose is therefore global
while its outcome is American. This file is the American measure.
