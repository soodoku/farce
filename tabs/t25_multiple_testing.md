| test                          |   p_raw |   p_bonferroni |   p_bh | significant_raw   | significant_bonf   | significant_bh   |
|:------------------------------|--------:|---------------:|-------:|:------------------|:-------------------|:-----------------|
| RI (all days)                 |    0    |           0.01 |   0    | True              | True               | True             |
| RI (9 Fri + 1 Sun)            |    0    |           0    |   0    | True              | True               | True             |
| RI (Fridays only)             |    0    |           0.01 |   0    | True              | True               | True             |
| RI (block bootstrap)          |    0    |           0.02 |   0    | True              | True               | True             |
| Studentized RI                |    0.01 |           0.11 |   0.01 | True              | False              | True             |
| Clustered RI                  |    0.01 |           0.1  |   0.01 | True              | False              | True             |
| Main effect                   |    0    |           0.02 |   0    | True              | True               | True             |
| Placebo: Mean crash latitude  |    0    |           0.06 |   0.01 | True              | False              | True             |
| Placebo: Mean crash longitude |    0.56 |           1    |   0.75 | False             | False              | False            |
| Placebo: Mean vehicles per cr |    0.97 |           1    |   1    | False             | False              | False            |
| Placebo: Mean persons per cra |    0.58 |           1    |   0.75 | False             | False              | False            |
| Placebo: % railroad crossing  |    0.05 |           0.88 |   0.1  | True              | False              | False            |
| Placebo: % school bus involve |    0.7  |           1    |   0.84 | False             | False              | False            |
| Placebo: % work zone          |    0.09 |           1    |   0.16 | False             | False              | False            |
| Placebo: Weather-related only |    0.45 |           1    |   0.68 | False             | False              | False            |
| Placebo: Work zone crashes    |    0.78 |           1    |   0.88 | False             | False              | False            |
| Placebo: School bus involved  |  nan    |         nan    | nan    | False             | False              | False            |
| Placebo: National Highway Sys |    0.2  |           1    |   0.33 | False             | False              | False            |