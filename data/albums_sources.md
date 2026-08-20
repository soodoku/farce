# Album Data Sources

First-day Spotify streaming numbers for albums used in the analysis.

## Tier 1: Paper's Original Sample (2018-2022)

The 10 albums analyzed in Patel et al. (2026). **Streaming counts are taken
directly from the paper's Table 1**, not from press coverage. The paper's
figures are first-day streams on the Spotify US daily top-200 chart; press
reports usually quote global first-day totals, which are larger and not
comparable. Using a press figure for one album and the paper's for the rest
puts two different measures in the same column — this repository did exactly
that for *Un Verano Sin Ti* (183M global vs 145.8M in the paper) until the
`check_album_list` gate in `src/gates.py` caught it.

The paper's sample window is described as 2017-2022; the ten selected albums
all fall in 2018-2022.

| Album | Streams (paper Table 1) | Date |
|-------|------------------------|------|
| Taylor Swift - Midnights | 184,695,609 | 2022-10-21 |
| Drake - Certified Lover Boy | 153,441,565 | 2021-09-03 |
| Bad Bunny - Un Verano Sin Ti | 145,811,373 | 2022-05-06 |
| Drake - Scorpion | 132,384,203 | 2018-06-29 |
| Kendrick Lamar - Mr. Morale & the Big Steppers | 99,582,729 | 2022-05-13 |
| Harry Styles - Harry's House | 97,621,794 | 2022-05-20 |
| Drake & 21 Savage - Her Loss | 97,390,844 | 2022-11-04 |
| Kanye West - Donda | 94,455,883 | 2021-08-29 |
| Taylor Swift - Red (Taylor's Version) | 90,556,180 | 2021-11-12 |
| Taylor Swift - Folklore | 79,443,136 | 2020-07-24 |

Source: Patel, Worsham, Liu & Jena (2026), Table 1 ([w34866.pdf](../w34866.pdf)).

## Tier 2: Extended Analysis (Albums 11-20)

Dose-response comparison group with lower streaming numbers.

> **These streaming counts are estimates, not measurements.** Eight of the ten
> are inferred from chart position because first-day figures were never
> published for them. The resulting series is a smooth descending sequence of
> round numbers, which is a property of how it was constructed rather than of
> the albums. Any dose-response result that leans on Tier 2 is leaning on that
> construction; see the caveat in the README.

| Album | Streams | Date | Source |
|-------|---------|------|--------|
| Travis Scott - ASTROWORLD | 72M | 2018-08-03 | Estimated from chart position |
| Post Malone - beerbongs & bentleys | 70M | 2018-04-27 | Estimated from chart position |
| Post Malone - Hollywood's Bleeding | 65M | 2019-09-06 | Estimated from chart position |
| Billie Eilish - Happier Than Ever | 60M | 2021-07-30 | Estimated from chart position |
| Juice WRLD - Legends Never Die | 58M | 2020-07-10 | Estimated from chart position |
| Ariana Grande - thank u, next | 56M | 2019-02-08 | [Guinness](https://www.guinnessworldrecords.com/news/2020/7/taylor-swift-breaks-24-hour-streaming-record-on-spotify-for-8th-album-folklore-625253) (cited as previous record) |
| Olivia Rodrigo - SOUR | 55M | 2021-05-21 | Estimated from chart position |
| The Weeknd - After Hours | 53M | 2020-03-20 | Estimated from chart position |
| Ed Sheeran - = (Equals) | 51M | 2021-10-29 | Estimated from chart position |
| Ariana Grande - Positions | 50M | 2020-10-30 | Estimated from chart position |

## Tier 0: Pre-2018 Streaming Era (2015-2017)

Used only to ask whether the same estimator finds an effect before the paper's
sample window opens.

> **No first-day streaming figures exist for these albums.** Systematic
> first-day reporting begins around 2018, and several of these releases were
> subject to exclusivity deals (Apple, Tidal) that make any Spotify-based count
> meaningless. The `streams_millions` values in `albums.csv` for tier 0 are
> placeholders with no source. They must not be used in any dose-response
> analysis, and are not.

| Album | Date |
|-------|------|
| The Weeknd - Beauty Behind the Madness | 2015-08-28 |
| Justin Bieber - Purpose | 2015-11-13 |
| Rihanna - ANTI | 2016-01-28 |
| Beyoncé - Lemonade | 2016-04-23 |
| Drake - Views | 2016-04-29 |
| Frank Ocean - Blonde | 2016-08-20 |
| J. Cole - 4 Your Eyez Only | 2016-12-09 |
| Ed Sheeran - ÷ | 2017-03-03 |
| Drake - More Life | 2017-03-18 |
| Kendrick Lamar - DAMN. | 2017-04-14 |

## Tier 3: Extended Analysis (2023-2024)

Post-paper albums for extended replication.

| Album | Streams | Date | Source |
|-------|---------|------|--------|
| Taylor Swift - The Tortured Poets Department | 313M | 2024-04-19 | [Billboard](https://www.billboard.com/music/music-news/taylor-swift-tortured-poets-department-spotify-record-300-million-streams-single-day-1235661939/) |
| Travis Scott - UTOPIA | 128M | 2023-07-28 | [Chart Data / X](https://x.com/chartdata/status/1685220769901555712) |
| Drake - For All The Dogs | 109M | 2023-10-06 | [Chart Data / X](https://x.com/chartdata/status/1710597835652211037) |
| Beyoncé - Cowboy Carter | 76.1M | 2024-03-29 | [Chart Data / X](https://x.com/chartdata/status/1774033973951299796) |
| Billie Eilish - Hit Me Hard and Soft | 72.7M | 2024-05-17 | [Spotify Stats / X](https://x.com/StatsSpotify/status/1791785591832785213) |
| SZA - SOS | 68M | 2022-12-09 | [That Grape Juice](https://thatgrapejuice.net/2022/12/szas-sos-shatters-spotifys-single-day-streaming-record-for-album/) |
| Morgan Wallen - One Thing at a Time | 52.3M | 2023-03-03 | [Billboard](https://www.billboard.com/business/streaming/morgan-wallen-spotify-single-day-streaming-record-one-thing-at-a-time-1235279973/) |

## Notes

- All dates verified against Wikipedia album pages
- Tier 1 counts come from the paper's Table 1 and are gated in `src/gates.py`
- Tier 2 streaming numbers are estimates from chart position, not measurements
- Tier 0 streaming numbers are placeholders with no source
- Donda released at 8am (not midnight), so its 95M represents ~16 hours of streaming
- "First-day streams" = first 24 hours on Spotify global chart
- `paper_sample` column indicates albums in the original Patel et al. (2026) analysis

## Data Limitations

- Pre-2017 albums often had streaming exclusivity deals (Apple, Tidal)
- First-day streaming data not systematically reported before ~2018
- Some estimates based on relative chart positions

## Last Updated

2026-08-19 (Tier 1 counts realigned to the paper's Table 1)
