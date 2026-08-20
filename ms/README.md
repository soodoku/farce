# Status of `ms.pdf`

**`ms.pdf` (26 April 2026) predates the corrections in [`../AUDIT.md`](../AUDIT.md) and
several of its claims no longer hold. Do not circulate it.** No LaTeX source is
committed, so it cannot be regenerated here; it has to be revised by hand against the
current tables in [`../tabs/`](../tabs/).

Claims in the PDF that the corrected code contradicts:

| Location | Claim in `ms.pdf` | Current result |
|---|---|---|
| Abstract, §4.1, §5, §6 | "In placebo exercises that draw ten Fridays at random … the resulting effects are routinely as large as or larger than the observed estimate" (100%) | The code drew the **top ten of a hundred** Fridays. The random-Friday placebo gives p = 0.0003; see `tabs/t18_friday_placebo.md`. The cherry-picking number is `tabs/t18b_cherry_pick_benchmark.md`, where the Tuesday row shows it is an order statistic, not a Friday effect. |
| Abstract, §4.2, §6 | "a pronounced day −6 coefficient, which in this sample is mechanically the previous Friday" | Day −6 is the **Saturday** before release. The previous Friday is day −7: +3.8, not distinguishable from zero. `tabs/t32_parallel_trends.md` now names the modal weekday of every event-time offset. |
| §4.2 | "The joint F-test of flatness across the ten pre-treatment days returns F = 2.04 (p = 0.03)" | That test treated ten album-level day means as independent. The album-level sign-flip permutation gives p ≈ 0.19, and dropping day −6 gives p ≈ 0.6. |
| §3 | "a positive day-1 estimate (+8.8, t = 2.05)" | With n − 1 degrees of freedom: t = 1.94, p = 0.08, 95% CI [−1.4, +19.0]. |
| §3, abstract | "larger for sober than for drunk crashes" | Rests on **two albums**. `DRUNK_DR` leaves the FARS accident file after 2020, so six of the ten releases drop out. No standard error is reportable. |
| §3 | "Adding indicators for the share of crashes under rain, fog, and cloudy conditions moves the point estimate from +15.83 to +15.95" | Those controls were silently dropped; three rows of the old table were identical to the base model. Wired properly the estimate moves 15.83 → 15.64, and +15.95 is the `pct_bad_weather` model. |
| Table 1 | "Our Replication … SE 4.8" | 7.45, as the abstract and §2 of the same document say. |
| §2 | "a 14.4% increase on a release-day mean of 122.1 fatalities" | 121.9 is the **control-day** mean; the release-day mean is 144.9. |
| Table 2 | Per-album out-of-sample effects averaging +1.4 | Matches no estimator in the repo, and contradicts the −2.8 in the prose above it. See `tabs/t20_extended_series.md`. |
| §4.1 | "Fridays average 110.6 fatalities in FARS 2017–2022, against an overall daily mean of 101.4"; "the 939 Fridays in 2017–2022" | Those are 2007–2024 figures. For 2017–2022: 117.6, 107.5, and 313 Fridays. |
| §4.6 | Latitude etc. as "placebo outcomes … that should be invariant to album releases" | They are post-treatment summaries of the same crashes, so they move with the effect rather than falsifying it. The table is now `t28b_structural_fars_composition.md`. |

What survives unchanged: the replication itself (+17.6, SE 7.45, clustered by album), the
specification multiverse, the concentration on day 0, the out-of-sample Tier 3 result, the
dependence on *Her Loss*, and the small number of treated units.
