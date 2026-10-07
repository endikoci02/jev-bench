# jev-bench summary

Accuracy and macro-F1 count failures as wrong. Calibration metrics use answered
predictions. Brier is top-label. 95% CIs: percentile bootstrap (1000 resamples). Cost per 1k decisions from logged usage.
Claude has no native probabilities here: its confidence is *verbalized* or *self-consistency*, a weaker signal than Jev's distribution.

## banking77

| model (confidence) | n | accuracy [95% CI] | macro-F1 | ECE [95% CI] | MCE | Brier | AUROC | wrong@≥0.8 | wrong@≥0.9 | hard ≥0.99 | failures | $/1k std | $/1k batch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev (native max-prob) | 200 | 79.5% [74.0%, 85.0%] | 0.780 | 0.116 [0.080, 0.170] | 0.445 | 0.130 | 0.847 | 9.0% | 4.5% | 56.5% | 0.0% | $0.087 | n/a |
| Jev (API confidence) | 200 | 79.5% [74.0%, 85.0%] | 0.780 | 0.116 [0.081, 0.172] | 0.435 | 0.129 | 0.846 | 9.0% | 4.5% | 56.5% | 0.0% | $0.087 | n/a |
| Haiku 4.5 (verbalized) | 200 | 78.0% [72.0%, 83.0%] | 0.772 | 0.090 [0.063, 0.155] | 0.350 | 0.150 | 0.809 | 9.0% | 3.5% | 0.5% | 0.0% | $2.492 | $1.246 |
| Haiku 4.5 (self-consistency) | 100 | 76.0% [67.0%, 84.0%] | 0.716 | 0.238 [0.164, 0.320] | 0.400 | 0.233 | 0.535 | 24.0% | 21.0% | 93.0% | 0.0% | $12.463 | $6.232 |
| Sonnet 5.5 (verbalized) | 200 | 82.5% [77.5%, 88.0%] | 0.821 | 0.095 [0.065, 0.143] | 0.800 | 0.110 | 0.854 | 3.5% | 1.5% | 0.0% | 0.0% | $0.983 | $0.492 |

## clinc150

| model (confidence) | n | accuracy [95% CI] | macro-F1 | ECE [95% CI] | MCE | Brier | AUROC | wrong@≥0.8 | wrong@≥0.9 | hard ≥0.99 | failures | $/1k std | $/1k batch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev (native max-prob) | 200 | 91.5% [87.5%, 95.5%] | 0.883 | 0.044 [0.028, 0.082] | 0.460 | 0.054 | 0.902 | 4.0% | 2.5% | 63.5% | 0.0% | $0.145 | n/a |
| Jev (API confidence) | 200 | 91.5% [87.5%, 95.5%] | 0.883 | 0.044 [0.028, 0.082] | 0.460 | 0.054 | 0.901 | 4.0% | 2.5% | 63.5% | 0.0% | $0.145 | n/a |
| Haiku 4.5 (verbalized) | 200 | 91.0% [86.5%, 94.5%] | 0.872 | 0.020 [0.014, 0.062] | 0.450 | 0.073 | 0.717 | 7.0% | 4.0% | 5.0% | 0.0% | $3.577 | $1.788 |
| Sonnet 5.5 (verbalized) | 200 | 93.5% [90.0%, 97.0%] | 0.910 | 0.056 [0.038, 0.085] | 0.500 | 0.050 | 0.857 | 2.0% | 1.5% | 3.0% | 0.0% | $1.229 | $0.614 |

## offensive

| model (confidence) | n | accuracy [95% CI] | macro-F1 | ECE [95% CI] | MCE | Brier | AUROC | wrong@≥0.8 | wrong@≥0.9 | hard ≥0.99 | failures | $/1k std | $/1k batch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev (native max-prob) | 200 | 74.0% [68.0%, 80.0%] | 0.701 | 0.180 [0.135, 0.245] | 0.352 | 0.212 | 0.672 | 16.5% | 12.5% | 37.0% | 0.0% | $0.016 | n/a |
| Jev (API confidence) | 200 | 74.0% [68.0%, 80.0%] | 0.701 | 0.201 [0.161, 0.267] | 0.760 | 0.229 | 0.672 | 12.5% | 8.5% | 22.5% | 0.0% | $0.016 | n/a |
| Haiku 4.5 (verbalized) | 200 | 74.0% [68.0%, 80.0%] | 0.666 | 0.157 [0.107, 0.213] | 0.303 | 0.196 | 0.733 | 18.0% | 8.0% | 12.5% | 0.0% | $0.435 | $0.217 |
| Sonnet 5.5 (verbalized) | 200 | 74.5% [68.0%, 80.5%] | 0.685 | 0.119 [0.077, 0.178] | 0.620 | 0.172 | 0.771 | 15.0% | 5.0% | 15.0% | 0.0% | $1.114 | $0.557 |

## sst2

| model (confidence) | n | accuracy [95% CI] | macro-F1 | ECE [95% CI] | MCE | Brier | AUROC | wrong@≥0.8 | wrong@≥0.9 | hard ≥0.99 | failures | $/1k std | $/1k batch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev (native max-prob) | 200 | 98.0% [96.0%, 99.5%] | 0.980 | 0.023 [0.013, 0.046] | 0.380 | 0.022 | 0.575 | 2.0% | 1.5% | 71.5% | 0.0% | $0.015 | n/a |
| Jev (API confidence) | 200 | 98.0% [96.0%, 99.5%] | 0.980 | 0.047 [0.031, 0.073] | 0.760 | 0.031 | 0.575 | 1.5% | 1.5% | 65.0% | 0.0% | $0.015 | n/a |
| Haiku 4.5 (verbalized) | 200 | 98.0% [96.0%, 99.5%] | 0.980 | 0.086 [0.065, 0.104] | 0.350 | 0.032 | 0.687 | 1.5% | 0.5% | 3.5% | 0.0% | $0.394 | $0.197 |
| Sonnet 5.5 (verbalized) | 200 | 98.0% [96.0%, 99.5%] | 0.980 | 0.070 [0.052, 0.090] | 0.400 | 0.029 | 0.733 | 1.0% | 1.0% | 7.5% | 0.0% | $0.992 | $0.496 |

## Latency (sync Messages API / Jev API)

Client: a single client in Europe. Concurrency per provider: {'typesafe': 4, 'anthropic': 4}. 5 warmup calls excluded. Dataset: sst2.

| model | n | p50 ms | p95 ms | p99 ms | failures |
|---|---|---|---|---|---|
| Jev | 50 | 266 | 397 | 653 | 0.0% |
| Haiku 4.5 | 50 | 767 | 1,286 | 1,437 | 0.0% |
| Sonnet 5.5 | 50 | 1,527 | 2,410 | 3,902 | 0.0% |

## Same-subset comparison: banking77 (100 examples used by self-consistency)

| model (confidence) | accuracy | ECE | AUROC | wrong@≥0.9 | hard ≥0.99 |
|---|---|---|---|---|---|
| Jev (native max-prob) | 79.0% | 0.123 | 0.808 | 6.0% | 57.0% |
| Jev (API confidence) | 79.0% | 0.123 | 0.808 | 6.0% | 57.0% |
| Haiku 4.5 (verbalized) | 75.0% | 0.112 | 0.790 | 4.0% | 0.0% |
| Haiku 4.5 (self-consistency) | 76.0% | 0.238 | 0.535 | 21.0% | 93.0% |
| Sonnet 5.5 (verbalized) | 81.0% | 0.121 | 0.855 | 3.0% | 0.0% |
