# FARCE: FARS Album Release Coincidence Examination

A replication and extended analysis of Patel, Worsham, Liu & Jena (2026), "Smartphones, Online Music Streaming, and Traffic Fatalities," NBER Working Paper 34866.

## What We Find

**The basic pattern is real**: Traffic fatalities are elevated on the release days of major streaming albums. Using FARS data from 2007-2024, we estimate an average excess of **+16.1 deaths** on the 10 release days (donut-global estimator, SE=5.0). This is statistically significant under multiple testing strategies.

**But the causal interpretation is questionable**: Several findings undermine the claim that streaming-induced distraction causes the excess deaths:

1. **No dose-response**: Albums with more streams show *smaller* effects (r = -0.22)
2. **Single outlier dominance**: Her Loss accounts for 34% of the total effect
3. **Tier 2 inconsistency**: Albums 11-20 show 80% of Tier 1's effect despite having ~60% of streams

## What The Authors Claim

Patel et al. analyze whether major album releases on Spotify lead to increased traffic fatalities, hypothesizing that streaming music while driving causes distraction. Using FARS data from 2017-2022, they compare fatalities on release days of the 10 most-streamed albums to a ±10 day control window.

**Key claims:**
- Streaming surge: 123.3M streams on release days vs 86.1M on ±10 day window (43% increase)
- Fatality increase: 139.1 deaths on release days vs 120.9 on control days (+18.2 deaths, +15.1%)
- Mechanism: Smartphone distraction from streaming music while driving
- Supporting evidence: Effects larger for younger drivers, single-occupant vehicles, sober drivers

## What We Did

We replicate and extend the analysis:

1. **Expanded data**: FARS 2007-2024 (vs. 2017-2022), providing better fixed effect estimates
2. **Global counterfactual**: Compared ±10 day window to a global model prediction
3. **Placebo tests**: Pre-trends, window sensitivity, year permutation
4. **Dose-response analysis**: Correlated streaming intensity with fatality effects
5. **Tier 2 comparison**: Tested whether albums 11-20 show proportionally smaller effects

### Why These Extensions Matter

- **Global counterfactual**: The ±10 day window may be unrepresentative of "normal" days
- **Placebo tests**: If effects appear before release or on wrong-year dates, causation is doubtful
- **Dose-response**: A causal mechanism should show more streams → more deaths
- **Heterogeneity**: Averaging conceals whether the effect is consistent or driven by outliers

## Detailed Results

### The Effect Is Real (Statistically)

| Estimator | Pooled δ | SE | t-stat |
|-----------|----------|-----|--------|
| Local (paper's ±10 day) | +23.0 deaths | 5.1 | 4.5 |
| Donut-global | +16.1 deaths | 5.0 | 3.2 |

Randomization inference p-values:
- iid (10 random days): p = 0.0003
- 9 Fridays + 1 Sunday: p = 0.0001
- Block bootstrap (7-day blocks): p = 0.0009

The block bootstrap p-value is 3× larger than iid, suggesting some autocorrelation.

### Placebo Tests Pass

| Test | Result | Interpretation |
|------|--------|----------------|
| Pre-trends (days -5 to -1) | Avg -0.8 deaths | No anticipation effects |
| Window sensitivity | Shrinkage < 1% | Effect stable across window sizes |
| Year permutation | p < 0.0001 | Effect is year-specific |

These placebo tests support the claim that *something* happens on release days. They do not establish *why*.

### But No Dose-Response

If distraction from streaming causes fatalities, more streams should mean more deaths:

| Album | Streams (M) | δ (deaths) |
|-------|-------------|------------|
| Midnights | 185 | -1.8 |
| Certified Lover Boy | 153 | +11.0 |
| Her Loss | 75 | +57.2 |

- **Pearson r = -0.22** (negative correlation, opposite of prediction)
- Regression: β₁(log-streams) = -10.5 (SE=10.8)
- Her Loss (lowest Tier 1 streams) shows largest effect
- Midnights (highest streams) shows negative effect

### Extreme Heterogeneity

Individual effects range from +3.3 to +59.5 deaths (local):

| Album | δ_local | δ_global |
|-------|---------|----------|
| Her Loss | +59.5 | +56.8 |
| Red (Taylor's Version) | +33.2 | +25.3 |
| Scorpion | +30.5 | +16.1 |
| Midnights | +3.3 | -2.1 |

Leave-one-out analysis:
- Dropping Her Loss: pooled estimate falls from +23.0 to +18.9
- Her Loss accounts for 34% of total Tier 1 effect
- Jackknife SE = 5.1, 95% CI: [+12.9, +33.0]

### Tier 2 Inconsistent with Mechanism

If effects are proportional to streaming intensity, albums 11-20 (~60% of Tier 1 streams) should show ~50% of Tier 1's effect:

| Tier | Avg δ | Streams (relative) | Expected δ |
|------|-------|-------------------|------------|
| Tier 1 (top 10) | +16.7 | 100% | — |
| Tier 2 (11-20) | +13.4 | ~60% | ~+8.4 |

Actual Tier2/Tier1 ratio: **0.80** (expected: ~0.50)

## Interpretation

**What the data support:**
- Elevated fatalities on major album release days (robust to placebo tests)
- Effect is year-specific (not a calendar artifact)
- No pre-release anticipation effects

**What the data do not support:**
- Streaming intensity as the mechanism (negative dose-response)
- A consistent effect across albums (extreme heterogeneity)
- Proportionality to streaming volume (Tier 2 too large)

**Alternative explanations to consider:**
- Coincidence with other Friday events (concerts, parties, album release events)
- Her Loss as an outlier (34% of effect from one album)
- Residual confounding not captured by fixed effects

## Running the Analysis

### Requirements

```bash
pip install pandas numpy matplotlib scipy requests
```

### Usage

```bash
make extract        # Extract CSVs from zips
make run            # Run analysis
make run-placebos   # Run with placebo tests
make clean          # Remove extracted files
```

### Downloading FARS Data

1. Visit [NHTSA FARS Data](https://www.nhtsa.gov/file-downloads)
2. Download FARS zip files for desired years → place in `data/raw/`
3. Extract accident CSVs → place in `data/fars/`
4. Run with `--local data/fars/` flag

## Project Structure

```
farce/
├── Makefile
├── data/
│   ├── fars/              # Extracted accident CSVs (not tracked)
│   └── raw/               # FARS zip files (not tracked)
├── figs/                  # Output figures (not tracked)
├── tabs/                  # Output tables (not tracked)
└── src/
    ├── constants.py       # Album lists, release dates
    ├── s01_load.py        # Data loading
    ├── s02_preprocess.py  # Residualization
    ├── s03_core.py        # Analysis functions
    ├── s04_placebo.py     # Placebo tests
    ├── s05_visualize.py   # Plotting
    └── pipeline.py        # Main entry point
```

## Visualization

![Analysis Results](figs/analysis.png)

Six-panel figure showing:
1. Time series of residualized fatalities with release dates marked
2. Individual event study curves (showing heterogeneity)
3. Local vs global counterfactual decomposition by album
4. Randomization inference null distribution
5. Leave-one-out sensitivity analysis
6. Dose-response scatter (streams vs effect)

## Data Sources

- **FARS**: [NHTSA Fatality Analysis Reporting System](https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars)
- **Streaming estimates**: ChartMasters, RouteNote, Wikipedia first-day/first-week data

## References

- Patel, Worsham, Liu & Jena (2026). "Smartphones, Online Music Streaming, and Traffic Fatalities." NBER Working Paper 34866.
- [Harvard Gazette coverage](https://news.harvard.edu/gazette/story/2026/02/streaming-a-new-album-release-while-driving-may-increase-risk-of-fatal-car-accidents/)
