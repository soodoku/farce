| test                          |   p_raw |   p_bonferroni |   p_bh | significant_raw   | significant_bonf   | significant_bh   |
|:------------------------------|--------:|---------------:|-------:|:------------------|:-------------------|:-----------------|
| RI (all days)                 |    0    |           0.01 |   0    | True              | True               | True             |
| RI (9 Fri + 1 Sun)            |    0    |           0    |   0    | True              | True               | True             |
| RI (Fridays only)             |    0    |           0.01 |   0    | True              | True               | True             |
| RI (block bootstrap)          |    0    |           0.02 |   0    | True              | True               | True             |
| Studentized RI                |    0.01 |           0.11 |   0.01 | True              | False              | True             |
| Clustered RI                  |    0.01 |           0.1  |   0.01 | True              | False              | True             |
| Main effect                   |    0    |           0.02 |   0    | True              | True               | True             |
| Placebo: Mean crash latitude  |    0    |           0.05 |   0.01 | True              | True               | True             |
| Placebo: Mean crash longitude |    0.56 |           1    |   0.68 | False             | False              | False            |
| Placebo: Mean vehicles per cr |    0.97 |           1    |   0.97 | False             | False              | False            |
| Placebo: Mean persons per cra |    0.56 |           1    |   0.68 | False             | False              | False            |
| Placebo: % railroad crossing  |    0.04 |           0.68 |   0.07 | True              | False              | False            |
| Placebo: % school bus involve |    0.69 |           1    |   0.77 | False             | False              | False            |
| Placebo: % work zone          |    0.07 |           1    |   0.12 | False             | False              | False            |
| Placebo: Weather-related only |    0.43 |           1    |   0.59 | False             | False              | False            |
| Placebo: Work zone crashes    |    0.77 |           1    |   0.81 | False             | False              | False            |
| Placebo: School bus involved  |    0    |           0    |   0    | True              | True               | True             |
| Placebo: National Highway Sys |    0.18 |           1    |   0.26 | False             | False              | False            |