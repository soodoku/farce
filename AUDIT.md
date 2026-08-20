# Correctness audit — `farce`

> **All findings below have been repaired.** See the status table at the end. The
> numbers quoted as "published" are what the repository reported before the audit;
> the corrected values are what `make run` produces now. `ms/ms.pdf` is the one
> artifact that could not be fixed here — it has no committed source — and is flagged
> as stale in [`ms/README.md`](ms/README.md).

Audit of the replication/critique repository and manuscript (`ms/ms.pdf`, 26 Apr 2026) against
Patel, Worsham, Liu & Jena (2026), NBER WP 34866.

Method: full re-run of `python3 -m src.pipeline`, independent re-derivation of every headline
number from the FARS extract, and line-level reading of the estimators. Every finding below has a
computed old value and a computed new value.

---

## Verdict

**The replication is right. Most of the critique is not.**

The pipeline reproduces all 33 committed tables byte-identically, the FARS extract matches NHTSA's
Final File annual totals exactly, the ten Tier-1 albums and dates match the paper's Table 1
exactly, and the headline estimate (+17.6, cluster SE 7.45) is what the code computes.

Of the five pillars of the critique, two are wrong, one does not survive correct inference, one is
mislabeled, and one stands:

| Pillar | Status |
|---|---|
| Friday placebo shows 100% false-positive rate | **Wrong.** The code cherry-picks the top 10 of 100 Fridays. The random-Friday placebo it claims to run gives p = 0.0003. |
| Day −6 spike is "mechanically the previous Friday" | **Wrong.** Day −6 is Saturday. Day −7 is the previous Friday: +3.8, t = 0.92, p = 0.38. |
| Pre-trends are significant (F = 2.04, p = 0.03) | **Does not survive.** Correct df: p = 0.051. Album-level permutation: p = 0.19. Drop day −6: p = 0.55. |
| Post-2022 reversal | **Stands, estimator-dependent.** −8.0 (SE 7.56) by the paper's spec; +5.6 by the raw release-vs-window comparison the pipeline also prints. |
| No dose-response | **Overstated.** r = −0.18, p = 0.46, 95% CI [−0.57, +0.29]; half the x-variable is admittedly made up. |

The manuscript's own Figure 1, panel 4, shows the correct Friday-restricted randomization inference
rejecting the null at **p = 0.0001**. Section 4.1, two pages later, reports that 100% of random
Friday placebos exceed the observed effect. Both are in the same manuscript. Only the first is real.

Net effect of the audit: the paper comes out of this **better** than the current draft says, not worse.

---

## Claim → estimand map

| # | Claim (source) | Unit | Comparison | Estimand actually computed | Verdict |
|---|---|---|---|---|---|
| C1 | "+17.6 excess deaths (SE 7.45)" (abstract, §2, README) | album-day | day 0 vs ±10-day window, DOW+week+year+holiday FE, CR1 by album | same | ✔ reproduces (17.612, 7.450) |
| C2 | "14.4% increase on a release-day mean of 122.1" (§2) | national day | effect ÷ mean | effect ÷ **control** mean (121.93) | label wrong, ratio right |
| C3 | "Every set of ten random Fridays … produces an effect at least as large" (§4.1) | day | 10 random Fridays vs observed | **mean of the top 10 residuals out of 100 bootstrap Fridays** | ✘ not the stated estimand |
| C4 | "day −6 … is the previous Friday" (abstract, §4.2, §6) | day | — | day −6 = Saturday (9/10 albums) | ✘ arithmetic error |
| C5 | "larger for sober than for drunk crashes" (abstract, §3) | album | release-day residual, split by DRUNK_DR | **n = 2 albums** | ✘ not a sample-wide claim |
| C6 | "unchanged by weather controls" (abstract, §3) | national day | + rain/fog/cloudy shares | controls silently dropped (3 rows = base model) | ✘ as run; ✔ when wired |
| C7 | "Tier 3 pooled −8.0 (SE 7.56); per-album mean −2.8" (§4.3, Table 2) | album | paper spec / residual mean | Table 2's own column averages **+1.4** | ✘ internally contradictory |
| C8 | "Fridays average 110.6 … in FARS 2017–2022, overall 101.4"; "939 Fridays in 2017–2022" (§4.1) | day | — | those are **2007–2024** figures | ✘ wrong period |
| C9 | "placebo outcomes … shift" (abstract, §4.6) | day | latitude residual on release days | real (+0.37), but latitude is not a placebo | ✘ construct |

---

## Findings, ranked

### F1 — The Friday placebo does not do what the paper says it does. *Kills the central claim.*

`src/s07_falsification.py:361` `best_fridays_false_positive_rate`. Each of the 10,000 iterations
draws **100** Fridays with replacement and keeps the **ten largest residuals**:

```python
sample_idx = np.random.choice(len(all_fridays), size=n_sample, replace=True)  # n_sample = 100
sample = all_fridays.iloc[sample_idx]
top_n = sample.nlargest(n_pick, "resid_global")                                # n_pick = 10
effect = top_n["resid_global"].mean()
```

The function's own name and docstring say "cherry-pick the *best* 10 Fridays". The manuscript and
README describe it as ten Fridays drawn at random.

| Quantity | Published | Corrected |
|---|---|---|
| Mean placebo effect | **+25.9** | **+0.07** |
| 95th percentile | +31.9 | +7.4 |
| Share of placebos ≥ observed (16.1) | **100%** | **0.03%** (3 draws in 10,000) |
| Same, pool restricted to 2017–2022 as claimed | — | 0.08% (8 draws in 10,000) |

Three further checks, all pointing the same way:

- **It has nothing to do with Fridays.** The identical top-10-of-100 procedure applied to *Tuesdays*
  returns a 98.7% "false positive rate", and to all days 99.8%. It measures an order statistic.
- **There is no Friday contrast left to confound.** `resid_global` is the residual after
  day-of-week, month, year and holiday fixed effects. Mean residual on Fridays = **+0.14 deaths**.
  The quantity §4.1 says the design cannot separate the effect from is 0.14, not 16.
- **The repo already ran the right test.** `tabs/t05_randomization_inference.md`, row
  `9_fridays_only`, p ≈ 0; and manuscript Figure 1, panel 4, is captioned
  "Randomization Inference: p = 0.0001 (9 Fri + 1 Sun, 10,000 draws)". The original paper ran the same
  falsification and got 20/1000. Three independent versions of the correct test agree; the fourth,
  broken one is the one the argument rests on.

Affects: abstract, §4.1 (whole subsection and its table), §5 first limitation, §6 conclusion, README
"The Friday Problem" and summary table, `tabs/t18_friday_fpr.md`.

### F2 — Day −6 is Saturday, not the previous Friday.

Nine of ten releases are Fridays; Friday − 6 days = **Saturday**. The previous Friday is day −7.

| Day | Weekday (9/10 albums) | Effect | t (correct df) | p |
|---|---|---|---|---|
| −7 | **Friday** | +3.84 | 0.92 | 0.38 |
| −6 | Saturday | +16.51 | 3.36 | 0.009 |
| +7 | **Friday** | −0.34 | −0.07 | 0.95 |

Both same-day-of-week neighbours are flat. The event study therefore contains no Friday pattern at
all, which is the opposite of what §4.2 concludes from it. Whatever day −6 is, it is not evidence
for the Friday story — and with 21 day-coefficients estimated, one t = 3.4 is roughly what noise
produces.

Affects: abstract, §4.2, §6.

### F3 — The pre-trend F test does not survive correct inference.

`src/s03_design.py:409`. The statistic is `sum(effect² / se²) / 10` referred to F(10, 1000), where
each `se` is `np.std(x)` (ddof = 0) over ten albums.

| Version | F | p |
|---|---|---|
| As published | 2.04 | **0.027** |
| ddof = 1 | 1.84 | 0.051 |
| Album-level sign-flip permutation (20,000 draws) | — | **0.19** |
| Excluding day −6 | 0.79 | 0.55 |

Three problems: ddof = 0 inflates every t² by 10/9; ten album means are not ten independent normal
draws (the same ten albums generate every day's coefficient); and F(10, 1000) is the wrong
reference for statistics built from ten observations.

### F4 — Every album-level SE in the repo is ~5% too small, and all critical values are normal.

`np.std(day_resids) / sqrt(n)` (ddof = 0) appears in `compute_dynamic_effects`,
`parallel_trends_test`, `extended_series_analysis`, `multiverse_analysis`,
`structural_fars_placebos`, `drunk_vs_sober_analysis`, `covid_sensitivity`. CIs use 1.96 and
p-values use `stats.norm.cdf`, with n = 10.

The one claim this changes:

| §3 claim | Published | Corrected |
|---|---|---|
| Day +1 | +8.8, t = **2.05**, "consistent with lingering release-day behavior" | +8.8, t = **1.94**, p = 0.084, 95% CI **[−1.4, +19.0]** |

Day 0 survives (t = 3.20 → 3.04, p = 0.014). The multiverse 86% survives exactly (30/35 either way).

### F5 — Manuscript Table 1 contradicts the manuscript.

| Location | SE |
|---|---|
| Abstract, §2 prose, README, `t12`, code output | **7.45** |
| **Table 1, "Our Replication"** | **4.8** |

No estimator in the repo produces 4.8. Same paragraph: "a 14.4% increase on a release-day mean of
122.1 fatalities" — 121.93 is the **control-day** mean (the analogue of the paper's 120.9); the
release-day mean is **144.9**. The ratio is right, the noun is wrong, and the digit is off by 0.2.

Also missing: with G = 10 clusters the headline is 17.61 ± 7.45 → **t₉ p = 0.042, 95% CI [0.8, 34.5]**.
The draft never states an interval for its own replication.

### F6 — Manuscript Table 2 is not reproducible from this repo.

| Album | Table 2 "Excess Deaths" | Repo (`t20` / residual) | Repo (raw local, `pipeline.py:339`) |
|---|---|---|---|
| Tortured Poets | −2.1 | −4.72 | −1.5 |
| UTOPIA | 10.5 | +4.63 | +15.5 |
| For All The Dogs | −12.8 | −18.94 | −11.2 |
| Cowboy Carter | −0.4 | −1.38 | +8.2 |
| Hit Me Hard and Soft | 7.0 | +2.81 | +8.2 |
| SOS | 9.4 | +3.68 | +12.2 |
| One Thing at a Time | −1.5 | −5.38 | +7.7 |
| **Mean** | **+1.44** | **−2.76** | **+5.59** |

The Table 2 column matches neither estimator. Its own average (+1.4) contradicts the prose two
paragraphs above it ("the unweighted mean of per-album effects is −2.8"). Its pooled SE (7.0)
contradicts the prose (7.56) and `t12` (7.56).

Substantively: the raw release-day-vs-window comparison — the unadjusted analogue of the paper's
own 139.1 vs 120.9 headline — is **+5.6 out of sample**, and the draft does not mention it. The
"no corresponding positive effect" claim in the abstract is estimator-dependent and should say so.

### F7 — The sober-vs-drunk mechanism test uses two albums.

`DRUNK_DR` was dropped from the FARS accident file after 2020 (verified: present 2007–2020, absent
2021–2024), so `drunk_vs_sober_analysis` keeps only Tier-1 albums released ≤ 2020: **Scorpion and
Folklore**. `tabs/t22` records `n_albums = 2`; the manuscript, the README and the abstract present
it as a sample-wide finding. `np.std` over two points makes SE = |x₁−x₂|/(2√2), so the reported
t = 5.13 is arithmetic, not evidence.

Second error in the same sentence: `DRUNK_DR` counts drivers with **BAC ≥ 0.08**, not BAC > 0. "Sober
crashes (driver BAC = 0)" therefore includes every crash with a driver at 0.01–0.07.

### F8 — Weather controls were never added.

`pipeline.py:450` calls `weather_controlled_model(df_global, …)`. `df_global` has no
`pct_rain`/`pct_fog`/`pct_cloudy` columns; `build_design_matrix` drops absent controls silently
(`if col in df.columns`). Rows 2–4 of `t21` are byte-identical to the base row. The weather frame
*is* built (`build_daily_weather_controls`) — and handed to a different function.

| Model | `t21` as published | Actually wired |
|---|---|---|
| Base | 15.83 | 15.83 |
| +Rain | 15.83 | 15.90 |
| +Rain+Fog | 15.83 | 15.80 |
| +Rain+Fog+Cloudy | 15.83 | **15.64** |
| +All bad weather | 15.95 | 15.95 |

The conclusion holds either way. But §3 attributes +15.95 to "rain, fog, and cloudy" when that row
is the `pct_bad_weather` model, and the README's "15.8 → 15.6" is right for a reason nobody can
reproduce from the current code — it reflects an earlier run in which the controls were connected.

### F9 — The Friday descriptives are for the wrong period.

| §4.1 / README | Stated | 2017–2022 (as labeled) | 2007–2024 (what was computed) |
|---|---|---|---|
| Friday mean fatalities | 110.6 | **117.6** | 110.58 |
| Overall daily mean | 101.4 | **107.5** | 101.39 |
| Number of Fridays | 939 | **313** | 939 |

The placebo pool is the full 2007–2024 series, not the paper's window.

### F10 — Latitude is not a placebo outcome.

Mean crash latitude is a composition statistic of the same crashes whose count is the treated
outcome. If release days carry ~16 extra fatal crashes and those are not distributed uniformly over
the country, mean latitude moves as a mechanical consequence of the effect. Calling it an outcome
that "should not respond under any plausible streaming mechanism" (§4.6) is wrong: it is
post-treatment, not a placebo.

The number itself is real (+0.37°, 9/10 albums positive, t = 2.9 with correct df) and worth
reporting — as a composition shift, not as evidence the estimator is noisy.

Related: `tabs/t28_placebo_outcomes.md` publishes "School bus involved: effect −0.37, SE 0.01,
t = **−27.05**, n_albums = 3". That row should not be in a table.

### F11 — The dose-response null is reported as if it were a negative finding.

r = −0.177 over 20 albums, **p = 0.46, 95% CI [−0.57, +0.29]**. The interval contains everything
from a strong negative to a moderate positive relationship. §4.4 says "the sign is opposite to what
the proposed mechanism would naturally imply" and calls the two possible defenses "ways to preserve
the hypothesis"; the honest statement is that with 20 events the test has no power to distinguish
any of it.

Undisclosed forking: Tier-1 only gives **r = −0.50**; the log-streams slope is −7.1 (§4.4 says −7.3).

And the x-variable is partly invented. `data/albums_sources.md` marks eight of ten Tier-2 values
"Estimated from chart position", and the resulting series is a monotone sequence of round numbers —
72, 70, 65, 60, 58, 56, 55, 53, 51, 50. The ten Tier-0 albums have **no entry in the sources file at
all** and carry equally round values (50, 55, 45, 70, 80, 35, 40, 60, 65, 70). Half the regressor in
the headline correlation, and all of the pre-2018 tier, is guessed.

### F12 — One album's streaming count disagrees with the paper's Table 1, which is in this repo.

| Album | `albums.csv` | Paper Table 1 | Δ |
|---|---|---|---|
| Un Verano Sin Ti | 183.0M | **145.8M** | +37.2M |

Every other Tier-1 value agrees to ≤ 1.2M. The paper's counts are US Spotify top-200 daily streams;
the cited Billboard figure is a global measure. Consequence for the headline correlation: r goes
−0.177 → −0.169. No conclusion changes — but the authoritative numbers are in `w34866.pdf` and
should be used.

### F13 — Two silently-zero covariates.

| Column | Zero (not missing) for | Cause |
|---|---|---|
| `pct_alcohol` | 2021–2024 | `DRUNK_DR` absent → `(NaN >= 1).astype(int)` = 0 |
| `pct_rural` | 2007–2014 | `RUR_URB` absent → same pattern |

Both flow into `tabs/t24_balance_check.md`. Six of the ten Tier-1 release days fall in 2021+, so the
`pct_alcohol` balance row compares structural zeros against a mixture. The headline estimate is
unaffected — `build_design_matrix` is called with `controls=None` everywhere that matters.

### F14 — Reproducibility gaps.

- `make run-forecast` calls `python3 -m src.s06_forecast`; no such module exists (it is
  `s06_specification`). The target has never worked.
- No `requirements.txt`, `pyproject.toml`, or pinned versions. README says `pip install pandas numpy
  matplotlib scipy requests scikit-learn`.
- `ms/` contains only `ms.pdf`. With no LaTeX source, no automated check can tie a manuscript
  number to the table that produced it — which is how F5 and F6 survived.
- No tests, no CI.
- `.DS_Store` and `.Rhistory` are in the working tree.

### F15 — "35 specifications" overstates the multiverse.

The grid is 7 distinct album-set × sample-period configurations × 5 window widths. Widening the
window from ±5 to ±21 moves the Tier-1 estimate from 14.78 to 14.29. The honest sentence is
"6 of 7 distinct configurations are significant; the only null is pre-2018", which says the same
thing without borrowing credibility from a count of 35.

### F16 — Overlapping event windows (affects the paper too, unremarked).

Six of the 200 control-day rows in the stacked panel are themselves other albums' release days
(Donda ↔ Certified Lover Boy, 5 days apart; Un Verano ↔ Mr. Morale ↔ Harry's House, 7 days apart).
Treated days serve as controls, biasing the pooled estimate toward zero. Small here, but it is a
real design point the critique could make and does not.

---

## Rejected candidates

Written down so they are not chased again.

| Candidate | Why it failed |
|---|---|
| Latitude placebo driven by FARS `77.7777` "Not Reported" code | Real hygiene bug — the filter is `lat > 90`, which misses 77.7777, leaving 630 records — but cleaning it moves the effect +0.365 → +0.372, t 3.08 → 3.07. No change. |
| FARS annual totals disagree with published figures | They match the **Final File** (2018 = 36,835). The 36,560 figure is the Annual Report File. Extract is correct. |
| Multiverse 86% inflated by the ddof = 0 SEs | Recomputed with ddof = 1 and t₍ₙ₋₁₎: still exactly 30/35. |
| Un Verano stream error changes the dose-response conclusion | r −0.177 → −0.169. |
| Puerto Rico inflating counts | State 43 absent from every file. |
| `unzip -p "*/accident.csv"` concatenating multiple archive members | One header line per file, zero duplicate (STATE, ST_CASE). |
| Committed tables are stale | Full pipeline re-run reproduces all 33 tables byte-identically. |
| Album list drifts from the paper's | All ten albums and dates match Table 1 of `w34866.pdf` exactly. |
| Ridge (`ridge_lambda=1e-8`) biasing the reported SE | Negligible at that magnitude. |

## Untestable

| Check | Missing artifact |
|---|---|
| Whether the paper's 18.2 uses the identical FE set | Paper's replication package not public; only the text description is available. |
| Whether ms Table 2 came from an older data vintage or was typed | No LaTeX source, no build log, no git history for `ms/`. |
| Tier-0 and Tier-2 streaming counts | No sources recorded for Tier 0; Tier 2 self-described as estimates. |

---

## Repairs — status

All applied and verified by a full `make run` that reproduces `tabs/` byte-identically,
plus `make lint` (black, isort, flake8) clean and 15 regression tests passing.

| # | Repair | Where |
|---|---|---|
| F1 | `best_fridays_false_positive_rate` replaced by `friday_placebo_test` (three draw schemes, exact draw counts). The cherry-pick kept as `cherry_pick_benchmark`, with a Tuesday row so it cannot be misread as a Friday result. | `src/s07_falsification.py`, `tabs/t18_friday_placebo.md`, `tabs/t18b_cherry_pick_benchmark.md` |
| F2 | Event-time tables now carry a `weekday` column naming the modal weekday of each offset. | `src/s03_design.py`, `tabs/t32_parallel_trends.md` |
| F3 | Ad-hoc F test replaced by an album-level sign-flip permutation, reported alongside the same test with the largest pre-day dropped. | `_sign_flip_joint_test` |
| F4 | Every album-level SE routed through `album_stats`: ddof = 1, t critical values, NaN below five treated units. | `src/utils.py` and all seven call sites |
| F5, F6 | Cannot be fixed in-repo (no manuscript source). Every affected claim is enumerated. | `ms/README.md` |
| F7 | `n = 2` surfaced in the table, the console output and the docstring; BAC ≥ 0.08 terminology corrected. | `drunk_vs_sober_analysis` |
| F8 | Weather frame passed to the weather model; `build_design_matrix` now raises on an absent control instead of dropping it. | `src/pipeline.py`, `src/utils.py` |
| F9 | Period labels corrected throughout the README; placebo pools labeled by the years they actually cover. | `README.md`, `tabs/t18_friday_placebo.md` |
| F10 | Renamed to `structural_fars_composition`; documented as post-treatment, not placebo. FARS 77.7777 latitude code now filtered. | `src/s07_falsification.py` |
| F11 | Fisher-z interval on the correlation and a t interval on the log-streams slope; null framed as uninformative. Tier 0 and Tier 2 stream counts labeled as estimates. | `stream_effect_correlation`, `data/albums_sources.md` |
| F12 | Tier 1 streaming counts taken from the paper's Table 1, gated. | `data/albums.csv`, `src/gates.py` |
| F13 | Structurally absent covariates are NaN, each share gets its own denominator, and the balance check skips any covariate not recorded for every treated day. | `src/s02_preprocess.py`, `covariate_balance_check` |
| F14 | `make run-forecast` fixed, `requirements.txt` pinned, `setup.cfg` added, `make lint` target, `tests/`. | repo root |
| F15 | README states 6 of 7 distinct configurations rather than leaning on the count of 35. | `README.md` |
| F16 | `window_overlap` gate reports contaminated control rows on every run. | `src/gates.py` |

## Gates now enforced

| Gate | Catches |
|---|---|
| `check_album_list` | Tier 1 drifting from the paper's Table 1 — it caught F12 on its first run |
| `check_daily_series` | Gaps, duplicate dates, or the wrong FARS vintage |
| `window_overlap` | Treated days serving as controls |
| `build_design_matrix` | A requested control column that does not exist, or one with missing values |
| `album_stats` | Standard errors from fewer than five treated units |
| `tests/test_guards.py` | Each of the above, as a regression test tied to the defect it replaced |

## Still open

- `ms/ms.pdf` needs a hand revision against the current tables, and its LaTeX source
  should be committed so prose numbers can be checked against `tabs/` automatically.
- Tier 0 and Tier 2 streaming counts remain estimates. Any dose-response conclusion is
  limited by that, not by the analysis.
