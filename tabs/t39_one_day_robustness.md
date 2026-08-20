**Statistics a single large day cannot move: rank tests, a trimmed mean, and the estimate with the most influential album dropped.**

| statistic                |    value |   p_value | note                                            |
|:-------------------------|---------:|----------:|:------------------------------------------------|
| mean                     |  16.1027 |    0.014  | the headline estimator                          |
| median                   |  12.902  |    0.0044 | randomization p over 20,000 draws of 10 days    |
| 20% trimmed mean         |  12.9611 |  nan      | discards the two largest and two smallest       |
| positive count (9 of 10) |   9      |    0.0215 | sign test, ignores magnitudes entirely          |
| signed-rank              | nan      |    0.0039 | Wilcoxon, ignores magnitudes beyond their order |
| mean without Her Loss    |  11.5822 |    0.0056 | most influential album, delta +56.8             |