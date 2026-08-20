**Residual estimator with the additive day-of-week term replaced by the listed interaction. mean_friday_resid shows how little additive Friday signal the baseline leaves.**

| model                         |   n_params |   effect |   se |   t_stat |   p_value |   mean_friday_resid |
|:------------------------------|-----------:|---------:|-----:|---------:|----------:|--------------------:|
| DOW + month + year (baseline) |         37 |    16.1  | 5.3  |     3.04 |      0.01 |                0.14 |
| DOW x year                    |        139 |    15.95 | 5.37 |     2.97 |      0.02 |                0.13 |
| DOW x month                   |        103 |    16.24 | 5.8  |     2.8  |      0.02 |                0.15 |
| DOW x year + DOW x month      |        211 |    16.16 | 5.88 |     2.75 |      0.02 |                0.14 |
| DOW x week-of-year            |        390 |    15.07 | 4.73 |     3.18 |      0.01 |                0.12 |