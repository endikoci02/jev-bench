# jev-bench summary

Accuracy and macro-F1 count failures as wrong. Calibration metrics use answered
predictions. Brier is top-label. 95% CIs: percentile bootstrap (1000 resamples). Cost per 1k decisions from logged usage.
Claude has no native probabilities here: its confidence is *verbalized* or *self-consistency*, a weaker signal than Jev's distribution.

## sst2

| model (confidence) | n | accuracy [95% CI] | macro-F1 | ECE [95% CI] | MCE | Brier | AUROC | wrong@≥0.8 | wrong@≥0.9 | hard ≥0.99 | failures | $/1k std | $/1k batch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev (native max-prob) | 20 | 100.0% [100.0%, 100.0%] | 1.000 | 0.038 [0.011, 0.073] | 0.230 | 0.006 | n/a | 0.0% | 0.0% | 60.0% | 0.0% | $0.015 | n/a |
| Jev (API confidence) | 20 | 100.0% [100.0%, 100.0%] | 1.000 | 0.077 [0.021, 0.147] | 0.520 | 0.026 | n/a | 0.0% | 0.0% | 50.0% | 0.0% | $0.015 | n/a |
| Haiku 4.5 (verbalized) | 20 | 100.0% [100.0%, 100.0%] | 1.000 | 0.144 [0.102, 0.196] | 0.450 | 0.031 | n/a | 0.0% | 0.0% | 10.0% | 0.0% | $0.393 | $0.197 |

