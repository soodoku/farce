| model                 |   effect |   se |   t_stat |   pct_effect | controls                      |
|:----------------------|---------:|-----:|---------:|-------------:|:------------------------------|
| Base (DOW+Month+Year) |    15.83 | 4.38 |     3.62 |        15.61 | None                          |
| +Rain                 |    15.83 | 4.38 |     3.62 |        15.61 | pct_rain                      |
| +Rain+Fog             |    15.83 | 4.38 |     3.62 |        15.61 | pct_rain, pct_fog             |
| +Rain+Fog+Cloudy      |    15.83 | 4.38 |     3.62 |        15.61 | pct_rain, pct_fog, pct_cloudy |
| +All bad weather      |    15.95 | 4.37 |     3.65 |        15.73 | pct_bad_weather               |