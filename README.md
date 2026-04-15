# FARCE: FARS Album Release Coincidence Examination

A replication and critique of Patel, Worsham, Liu & Jena (2026), "[Smartphones, Online Music Streaming, and Traffic Fatalities](https://www.nber.org/papers/w34866)," NBER Working Paper 34866. [[Local PDF]](w34866.pdf)

## The Paper's Claims

Patel et al. (2026) analyze 10 major album releases from 2017-2022 and report:

- **139.1 deaths** on album release days vs **120.9** on control days (+18.2 deaths, +15.1%)
- 123.3M streams on release days vs 86.1M control (+43%)
- Proposed mechanism: smartphone distraction from streaming while driving

> "We find an additional 18.2 traffic fatalities (139.1 versus 120.9; p < 0.01) on album release days compared to control days..." — Patel et al. (2026), Figure 2B

## Replication

**We successfully replicate the paper's main result:**

| Source | Effect | SE | % Effect |
|--------|--------|-----|----------|
| Paper (Figure 2B) | +18.2 deaths | ~5.5 | +15.1% |
| Our replication | +17.6 deaths | 4.8 | +14.4% |

- **Difference: 0.6 deaths** (< 1 SE)
- Same methodology: week-of-year fixed effects, day-of-week, year, holiday indicators
- Same sample: Tier 1 albums, 2017-2022

The statistical effect is real. Randomization inference confirms significance (p < 0.001).

## Issues Identified

### Issue 1: No Dose-Response Relationship

If streaming causes distracted driving deaths, more streams should produce more deaths. The data show the opposite:

| Album | Streams (M) | Effect |
|-------|-------------|--------|
| Tortured Poets (2024) | 313 | -2 deaths |
| Midnights (2022) | 185 | +5 deaths |
| Her Loss (2022) | 97 | +63 deaths |

**Pearson r = -0.17** (negative correlation — more streams → *smaller* effects)

The largest streaming day in Spotify history (Tortured Poets, 313M first-day streams) shows a *negative* effect on fatalities.

### Issue 2: Out-of-Sample Failure

The paper analyzed 2017-2022 releases. We extended the analysis to 2023-2024 (7 additional albums):

| Sample | Estimator | Effect | SE |
|--------|-----------|--------|-----|
| Tier 1 (2017-2022) | Paper spec | +17.6 | 4.8 |
| Tier 3 (2023-2024) | Paper spec | -8.0 | 7.0 |

Key out-of-sample results:

| Album | Streams (M) | Effect |
|-------|-------------|--------|
| Tortured Poets | 313 | -2.1 |
| UTOPIA | 128 | +10.5 |
| For All The Dogs | 109 | -12.8 |
| Cowboy Carter | 76 | -0.4 |
| Hit Me Hard and Soft | 73 | +7.0 |
| SOS | 68 | +9.4 |
| One Thing at a Time | 52 | -1.5 |

**Average out-of-sample effect: +1.4 deaths** (vs. +17.6 for original sample). The pattern found in 2017-2022 does not replicate forward.

### Issue 3: Outlier Dependence

The effect is driven by a single release:

- **Her Loss** (Drake & 21 Savage, 2022): +59.5 deaths
- Total Tier 1 effect: 229.8 deaths across 10 albums
- **Her Loss accounts for 26% of the total effect**

Leave-one-out analysis shows removing Her Loss reduces the average per-album effect from +23.0 to +18.9 deaths.

## Data

| Dataset | Coverage | N |
|---------|----------|---|
| FARS fatalities | 2007-2024 | Extended beyond paper's 2017-2022 |
| Albums | 27 total | 10 Tier 1 + 10 Tier 2 + 7 Tier 3 |

- **FARS**: [NHTSA Fatality Analysis Reporting System](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars)
- **Streaming data**: Spotify Newsroom, Billboard, Chart Data (see [albums_sources.md](data/albums_sources.md))

## Methodology

| Analysis | Description |
|----------|-------------|
| Paper's specification | Week-of-year FEs, DOW, year, holiday indicators |
| Forecast estimator | Train model on non-release days, predict counterfactual |
| Donut-global | Regression excluding ±10 days around releases |
| Dose-response | Correlation between streams and fatality effect |
| Randomization inference | Placebo tests, year permutation, window sensitivity |

## Output Tables

| File | Description |
|------|-------------|
| [t01_local_estimates.csv](tabs/t01_local_estimates.csv) | Per-album local effects |
| [t02_global_estimates.csv](tabs/t02_global_estimates.csv) | Per-album global effects |
| [t03_dose_response.csv](tabs/t03_dose_response.csv) | Streams vs effect |
| [t04_tier_comparison.csv](tabs/t04_tier_comparison.csv) | Tier 1 vs Tier 2 |
| [t05_randomization_inference.csv](tabs/t05_randomization_inference.csv) | RI p-values |
| [t06_leave_one_out.csv](tabs/t06_leave_one_out.csv) | Jackknife analysis |
| [t07_summary.csv](tabs/t07_summary.csv) | Summary statistics |
| [t08_placebo_tests.csv](tabs/t08_placebo_tests.csv) | Placebo results |
| [t09_window_sensitivity.csv](tabs/t09_window_sensitivity.csv) | Window sensitivity |
| [t10_forecast_estimates.csv](tabs/t10_forecast_estimates.csv) | Forecast estimates |
| [t11_forecast_summary.csv](tabs/t11_forecast_summary.csv) | Forecast summary |
| [t12_paper_replication.csv](tabs/t12_paper_replication.csv) | Paper replication comparison |

## Usage

```bash
# Install dependencies
pip install pandas numpy matplotlib scipy requests scikit-learn

# Run analysis
make extract        # Extract FARS CSVs from zips
make run            # Run main analysis
make run-forecast   # Run forecast estimator (standard sample)

# Extended analysis (includes 2023-2024 albums)
python3 -m src.s06_forecast --extended
```

### Data Setup

1. Download FARS zip files from [NHTSA](https://www.nhtsa.gov/file-downloads) → `data/raw/`
2. Run `make extract` to extract accident CSVs
3. Album data in `data/albums.csv` with sources in `data/albums_sources.md`

## Repository Structure

```
farce/
├── Makefile
├── README.md
├── w34866.pdf              # Paper
│
├── data/
│   ├── albums.csv          # Album release dates & streams
│   ├── albums_sources.md   # Data provenance
│   ├── fars/               # Extracted accident CSVs (not tracked)
│   └── raw/                # FARS zip files (not tracked)
│
├── src/
│   ├── constants.py        # Load albums from CSV
│   ├── s01_load.py         # FARS data loading
│   ├── s02_preprocess.py   # Daily aggregation, residualization
│   ├── s03_core.py         # Local/global estimators, RI, dose-response
│   ├── s04_placebo.py      # Placebo tests
│   ├── s05_visualize.py    # Plotting
│   ├── s06_forecast.py     # Forecast-based estimator
│   └── pipeline.py         # Main entry point
│
├── tabs/                   # Output tables (CSV)
└── figs/                   # Output figures (PNG)
```

## Visualization

![Analysis Results](figs/analysis.png)

## References

- Patel, Worsham, Liu & Jena (2026). "[Smartphones, Online Music Streaming, and Traffic Fatalities](https://www.nber.org/papers/w34866)." NBER Working Paper 34866. [[PDF]](w34866.pdf)
- [Harvard Gazette coverage](https://news.harvard.edu/gazette/story/2026/02/streaming-a-new-album-release-while-driving-may-increase-risk-of-fatal-car-accidents/)
- [Freakonomics podcast](https://freakonomics.com/podcast/do-taylor-swift-and-bad-bunny-have-blood-on-their-hands/)
- [New York Times](https://www.nytimes.com/2026/04/10/well/car-crashes-streaming-friday-harvard.html)
