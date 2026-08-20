**Full-series regression on a release-day indicator with day-of-week, month, year and holiday fixed effects, plus the listed weather shares. Not the stacked album-day design, so the level differs from t12.**

| model                 |   effect |   se |   t_stat |   pct_effect | controls                      |
|:----------------------|---------:|-----:|---------:|-------------:|:------------------------------|
| Base (DOW+Month+Year) |    15.83 | 4.38 |     3.62 |        15.61 | None                          |
| +Rain                 |    15.9  | 4.37 |     3.64 |        15.68 | pct_rain                      |
| +Rain+Fog             |    15.8  | 4.37 |     3.62 |        15.58 | pct_rain, pct_fog             |
| +Rain+Fog+Cloudy      |    15.64 | 4.36 |     3.59 |        15.42 | pct_rain, pct_fog, pct_cloudy |
| +All bad weather      |    15.95 | 4.37 |     3.65 |        15.73 | pct_bad_weather               |