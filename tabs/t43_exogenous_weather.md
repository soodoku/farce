**Release-day effect with daily precipitation from NOAA stations in all 51 states, weighted by each state's share of US fatalities. Unlike the crash-composition shares, nothing here is a function of the crash record.**

| specification                    |   n_controls |   effect |     se |   ci_lower |   ci_upper |   p_value |
|:---------------------------------|-------------:|---------:|-------:|-----------:|-----------:|----------:|
| no weather control               |            0 |  14.0333 | 5.2411 |     2.1771 |    25.8895 |    0.0253 |
| precipitation, fatality-weighted |            1 |  14.2374 | 5.2287 |     2.4093 |    26.0655 |    0.0235 |
| precipitation, equally weighted  |            1 |  14.5388 | 5.2938 |     2.5633 |    26.5142 |    0.0226 |
| both weightings together         |            2 |  14.5067 | 5.2848 |     2.5517 |    26.4617 |    0.0227 |