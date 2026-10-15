# Environment responsiveness report (M6)

Deterministic, non-RL probe policies on the real network, averaged over 5 evaluation seeds. **These are not the pre-registered baselines and nothing here is a reportable result.** Their purpose is to establish that the environment responds to dispatch decisions before any PPO training begins.

Reward reference scales: ref_W=2729.64 passenger-minutes/epoch, ref_U=6.63 passengers/epoch, ref_E=45.36 energy units/epoch, ref_C=1.83 cost units/epoch (source policy `constant_1`).

## Scenario `normal`

| policy | reward | inductions | wait (pass-h) | p95 wait (s) | left behind | left-behind frac | energy | cost | served |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_0_no_service` | -56,301.0 | 0 | 2,561,353 | nan | 0 | 0.0000 | 0 | 0 | 0.000 |
| `constant_1` | -822.3 | 228 | 10,226 | 210 | 937 | 0.0033 | 10,344 | 416 | 1.000 |
| `constant_2` | -1,547.6 | 236 | 12,221 | 330 | 5,351 | 0.0190 | 10,673 | 431 | 1.000 |
| `queue_threshold_100` | -837.5 | 226 | 10,499 | 210 | 1,021 | 0.0036 | 10,275 | 413 | 1.000 |
| `queue_threshold_400` | -2,884.3 | 188 | 18,442 | 894 | 13,906 | 0.0495 | 8,652 | 346 | 0.997 |
| `queue_threshold_1000` | -8,661.4 | 163 | 45,061 | 2,376 | 48,635 | 0.1731 | 7,518 | 300 | 0.997 |

## Scenario `surge`

| policy | reward | inductions | wait (pass-h) | p95 wait (s) | left behind | left-behind frac | energy | cost | served |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_0_no_service` | -59,081.2 | 0 | 2,687,838 | nan | 0 | 0.0000 | 0 | 0 | 0.000 |
| `constant_1` | -853.3 | 228 | 10,654 | 210 | 1,080 | 0.0037 | 10,345 | 416 | 1.000 |
| `constant_2` | -1,550.8 | 238 | 12,637 | 330 | 5,293 | 0.0182 | 10,741 | 433 | 1.000 |
| `queue_threshold_100` | -877.8 | 226 | 10,963 | 210 | 1,220 | 0.0042 | 10,275 | 413 | 1.000 |
| `queue_threshold_400` | -2,760.2 | 190 | 18,529 | 858 | 13,043 | 0.0449 | 8,747 | 349 | 0.995 |
| `queue_threshold_1000` | -9,028.0 | 164 | 46,112 | 2,370 | 50,894 | 0.1751 | 7,577 | 303 | 0.998 |
