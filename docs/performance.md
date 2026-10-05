# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 40 districts, 33,572 pipes, 591,360 readings.
Reproduce with `make demo`, then `make load-test`.

## Reads: network summary, events, riskiest pipes, latest plan (in rotation), closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 431 | 43.0 req/s | 35.5 ms | 50.0 ms | 56.4 ms | 0.00% |
| 8 | 2672 | 266.1 req/s | 38.9 ms | 62.7 ms | 71.4 ms | 0.00% |
| 32 | 2241 | 221.7 req/s | 108.2 ms | 156.1 ms | 173.5 ms | 0.00% |

Target from the blueprint: p95 under 400 ms for reads: met. The network summary aggregates all 33,572 pipes on every
call; it would be the first thing to cache.

## Compute (in-process, one core)

| Work | Time |
|---|---|
| Generate the utility: 40 sparse linear systems factorised, 3,185 failures drawn | 0.2 s |
| Load it: 33,572 pipes and 591,360 readings through COPY | 5.2 s |
| Detect: read and classify 40 districts, 500 candidate junctions each | 0.95 s |
| Train on 232,193 pipe-years, test, refit on all years, score 33,572 pipes | 1.8 s |
| Capital plan and the oldest-first comparison over 33,572 pipes | 0.26 s |
| Hydraulic what-if for one junction | under 10 ms |
| Evaluation: 600 district-fortnights and three utilities | about 10 s |

Not measured: continuous ingestion. Detection re-reads a whole fortnight per district on each run; with live data it would keep
running nightly aggregates instead. The blueprint's scale (millions of meters, sub-minute telemetry) is far beyond this slice.
