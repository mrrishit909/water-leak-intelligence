# Evaluation

Everything here is measured on synthetic utilities generated with seeds the thresholds were not tuned on (`(101, 102, 103)`), and
every model sits beside a simple baseline. Reproduce with `python -m wa.evaluate`. The numbers say how well the methods
recover what the simulator hid. They are not a claim about any real network: real hydraulics are non-linear, real demand
is messier, and a real hydraulic model is never a perfect copy of the pipes in the ground, as this one is.

## 1. Is it a leak, a faulty sensor, or just demand?

600 district-fortnights: a quarter with a leak of 0.2 to 3.0 L/s at a random junction, a quarter with one pressure sensor drifting
0.001 to 0.006 m an hour, a quarter with one night of 1.5 to 6 L/s extra demand, a quarter with nothing. Rows are the truth.

| truth / verdict | leak | sensor fault | transient demand | normal | unexplained |
|---|---|---|---|---|---|
| leak | 137 | 0 | 0 | 12 | 1 |
| sensor fault | 0 | 122 | 0 | 28 | 0 |
| transient demand | 0 | 0 | 150 | 0 | 0 |
| normal | 0 | 0 | 0 | 150 | 0 |

Districts wrongly called a leak (a crew sent for nothing): **0 of 450**.

Leaks found, by size (night flow in these districts is about 9.5 L/s):

| leak size (L/s) | leaks | found |
|---|---|---|
| 0.2 to 0.5 | 21 | 38.1% |
| 0.5 to 1.0 | 31 | 100.0% |
| 1.0 to 2.0 | 51 | 100.0% |
| 2.0 to 3.0 | 47 | 100.0% |

Small leaks are missed on purpose: the alarm needs night flow well clear of normal on the last two nights, because a
false alarm costs a crew a day. Slow sensor drifts below about 0.25 m in total are also left as normal.

## 2. Where is the leak?

For the 137 leaks found. Distance is counted in junctions along the grid (one junction = 60 m). The baseline is the
obvious rule: the leak is at the sensor that lost the most pressure.

| method | median distance | mean | within 3 junctions | exactly right |
|---|---|---|---|---|
| signature fit (this system) | 2 | 2.12 | 83.9% | 21.2% |
| best of its top 5 candidates | 0 | 0.99 | 89.1% | 62.0% |
| nearest sensor (baseline) | 5 | 6.09 | 28.5% | 2.2% |

By leak size, share located within 3 junctions (the area a work order asks a crew to survey):

| leak size (L/s) | found | within 3 junctions | baseline |
|---|---|---|---|
| 0.2 to 0.5 | 8 | 37.5% | 12.5% |
| 0.5 to 1.0 | 31 | 67.7% | 29.0% |
| 1.0 to 2.0 | 51 | 90.2% | 25.5% |
| 2.0 to 3.0 | 47 | 95.7% | 34.0% |

Leak size from the rise in night flow: mean error **5.6%** of the true size.

This is the best case for the method: the model used to locate the leak is the same one that generated the pressures.
With a real network the model's pipe roughness and valve states are partly wrong, and accuracy falls with them.

## 3. Which pipes fail next?

Trained on pipe-years up to 2022, tested on 2023 to 2025. The baseline ranks pipes by age alone.

| utility (seed) | failures in test | AUC | AUC, age only | failures caught in riskiest 5% | same, age only | predicted rate | actual rate |
|---|---|---|---|---|---|---|---|
| 101 | 1095 | 0.792 | 0.758 | 24.5% | 16.0% | 1.1% | 1.1% |
| 102 | 1134 | 0.778 | 0.753 | 25.0% | 16.1% | 1.0% | 1.1% |
| 103 | 1026 | 0.783 | 0.748 | 23.5% | 14.4% | 1.2% | 1.0% |

The model beats age alone by a modest margin. Failures are rare (1.1% of pipes a year) and largely chance, which
caps what any ranking can do: section 4 shows that planning with the true hidden rates would gain only about a tenth more.

## 4. Does the capital plan spend the money better?

$4 million a year for 5 years. Every plan is scored the same way: the failure cost it avoids under the hidden failure
rates the data was generated from, which no plan was shown. "Perfect knowledge" plans with those hidden rates.

| utility (seed) | this system | oldest first | random pipes | perfect knowledge | upper bound |
|---|---|---|---|---|---|
| 101 | $5.13M | $2.30M | $0.72M | $5.44M | $7.84M |
| 102 | $4.48M | $2.22M | $0.63M | $5.32M | $7.67M |
| 103 | $5.21M | $2.21M | $0.67M | $5.53M | $7.96M |

Against oldest-first the plan avoids **2.2 times** the failure cost, and reaches 90.9% of what perfect knowledge
of the failure rates would. (The upper bound is looser than it looks: it spends all five years of money in year one.) Note what the table also says: $20 million of replacements avoids about $5 million of failure
cost in five years. On these cost assumptions replacing pipes does not pay for itself within the plan; the plan only
decides how to lose least. A real case rests on longer horizons and on costs this model leaves out.
