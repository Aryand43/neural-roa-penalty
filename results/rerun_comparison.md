# Rerun Comparison

- Old: `results/summary_nl_v0.4.2_2026-05-28.csv`
- New: `results/summary.csv`

## Per-run

| penalty | sigmoid | log_scale | area old | area new | Δ area | max dV/dt old | max dV/dt new | time old (s) | time new (s) | NaN-grad skips new | accuracy new | eval time new (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_one` | `default` | `false` | 2.1204 | 2.1204 | +0.0 | 0.0 | 0.0 | 14.77 | 11.02 | 0 | 0.336 | 0.04 |
| `constant_one` | `default` | `true` | 2.1204 | 2.1204 | +0.0 | 0.0 | 0.0 | 13.8 | 11.07 | 0 | 0.336 | 0.03 |
| `constant_one` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0087 | 0.0087 | 15.69 | 11.89 | 0 | 0.402 | 0.05 |
| `constant_one` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0081 | 0.0081 | 14.86 | 12.06 | 0 | 0.402 | 0.03 |
| `control_zero` | `default` | `false` | 2.0932 | 2.128 | +0.0348 | 0.0 | 0.0029 | 60.67 | 8.02 | 0 | 0.336 | 0.03 |
| `control_zero` | `default` | `true` | 2.0932 | 2.128 | +0.0348 | 0.0 | 0.0029 | 14.8 | 7.14 | 0 | 0.336 | 0.41 |
| `control_zero` | `logistic` | `false` | 2.2172 | 2.2172 | +0.0 | 0.0 | 0.0 | 15.75 | 8.08 | 0 | 0.344 | 0.03 |
| `control_zero` | `logistic` | `true` | 2.086 | 2.086 | +0.0 | 0.0 | 0.0 | 14.58 | 7.72 | 0 | 0.334 | 0.05 |
| `inv_V_clipped` | `default` | `false` | 0.7816 | 0.7812 | -0.0004 | 2.5367 | 2.5541 | 14.31 | 10.87 | 0 | 0.26 | 0.05 |
| `inv_V_clipped` | `default` | `true` | 0.7816 | 0.7812 | -0.0004 | 2.5367 | 2.5541 | 14.08 | 11.36 | 0 | 0.26 | 0.04 |
| `inv_V_clipped` | `logistic` | `false` | 0.5884 | 0.5884 | +0.0 | 2.5621 | 2.5643 | 16.95 | 11.88 | 570 | 0.252 | 0.03 |
| `inv_V_clipped` | `logistic` | `true` | 1.0696 | 1.0716 | +0.002 | 1.117 | 2.1403 | 15.32 | 11.56 | 0 | 0.278 | 0.04 |
| `inv_V_rho` | `default` | `false` | 0.0104 | 0.0104 | +0.0 | 0.1471 | 0.147 | 13.91 | 10.49 | 0 | 0.217 | 0.42 |
| `inv_V_rho` | `default` | `true` | 0.0104 | 0.0104 | +0.0 | 0.1471 | 0.147 | 14.38 | 10.46 | 0 | 0.217 | 0.03 |
| `inv_V_rho` | `logistic` | `false` | 0.5924 | 0.5924 | +0.0 | 0.3914 | 0.3914 | 17.4 | 11.31 | 591 | 0.249 | 0.05 |
| `inv_V_rho` | `logistic` | `true` | 0.0068 | 0.0092 | +0.0024 | 0.0353 | 0.0186 | 15.06 | 11.29 | 0 | 0.216 | 0.03 |
| `inv_V_small` | `default` | `false` | 0.3384 | 0.3864 | +0.048 | 2.1996 | 2.4969 | 14.0 | 11.45 | 0 | 0.232 | 0.03 |
| `inv_V_small` | `default` | `true` | 0.3384 | 0.3864 | +0.048 | 2.1996 | 2.4969 | 14.08 | 11.28 | 0 | 0.232 | 0.05 |
| `inv_V_small` | `logistic` | `false` | 0.5108 | 0.5108 | +0.0 | 4.9103 | 4.9124 | 18.07 | 12.14 | 569 | 0.245 | 0.05 |
| `inv_V_small` | `logistic` | `true` | 0.8312 | 0.7396 | -0.0916 | 1.7514 | 1.4942 | 15.3 | 11.95 | 0 | 0.256 | 0.04 |
| `inv_dist` | `default` | `false` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 13.19 | 10.42 | 0 | 0.395 | 0.03 |
| `inv_dist` | `default` | `true` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 12.99 | 10.5 | 0 | 0.395 | 0.05 |
| `inv_dist` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0123 | 0.0123 | 13.74 | 12.38 | 0 | 0.402 | 0.05 |
| `inv_dist` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0095 | 0.0095 | 13.86 | 11.97 | 0 | 0.402 | 0.04 |
| `inv_dist_sq` | `default` | `false` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 13.55 | 11.13 | 0 | 0.395 | 0.05 |
| `inv_dist_sq` | `default` | `true` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 12.76 | 10.0 | 0 | 0.395 | 0.05 |
| `inv_dist_sq` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0106 | 0.0106 | 14.37 | 11.5 | 0 | 0.402 | 0.03 |
| `inv_dist_sq` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0081 | 0.0081 | 14.13 | 10.96 | 0 | 0.402 | 0.04 |
| `quadratic_over_rho` | `default` | `false` | 3.1172 | 3.1172 | +0.0 | 0.1335 | 0.1335 | 14.35 | 11.59 | 0 | 0.401 | 0.03 |
| `quadratic_over_rho` | `default` | `true` | 3.1172 | 3.1172 | +0.0 | 0.1335 | 0.1335 | 14.5 | 11.36 | 0 | 0.401 | 0.05 |
| `quadratic_over_rho` | `logistic` | `false` | 3.116 | 3.116 | +0.0 | 0.1359 | 0.1359 | 16.12 | 11.85 | 0 | 0.401 | 0.07 |
| `quadratic_over_rho` | `logistic` | `true` | 3.116 | 3.116 | +0.0 | 0.1359 | 0.1359 | 15.86 | 12.64 | 0 | 0.401 | 0.03 |
| `scaled_inv_dist_sq` | `default` | `false` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 13.25 | 10.19 | 0 | 0.395 | 0.04 |
| `scaled_inv_dist_sq` | `default` | `true` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 13.28 | 10.38 | 0 | 0.395 | 0.04 |
| `scaled_inv_dist_sq` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0028 | 0.0028 | 14.69 | 11.43 | 0 | 0.402 | 0.05 |
| `scaled_inv_dist_sq` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0054 | 0.0054 | 14.16 | 11.35 | 0 | 0.402 | 0.04 |

## log_scale effect (log(V/ρ) − (V − ρ)) per penalty/sigmoid

| penalty | sigmoid | Δ area old | Δ area new | Δ time old (s) | Δ time new (s) |
|---|---|---:|---:|---:|---:|
| `constant_one` | `default` | 0.0 | 0.0 | -0.97 | 0.04 |
| `constant_one` | `logistic` | 0.0 | 0.0 | -0.83 | 0.17 |
| `control_zero` | `default` | 0.0 | 0.0 | -45.87 | -0.88 |
| `control_zero` | `logistic` | -0.1312 | -0.1312 | -1.17 | -0.36 |
| `inv_V_clipped` | `default` | 0.0 | 0.0 | -0.23 | 0.49 |
| `inv_V_clipped` | `logistic` | 0.4812 | 0.4832 | -1.62 | -0.32 |
| `inv_V_rho` | `default` | 0.0 | 0.0 | 0.47 | -0.02 |
| `inv_V_rho` | `logistic` | -0.5856 | -0.5832 | -2.33 | -0.03 |
| `inv_V_small` | `default` | 0.0 | 0.0 | 0.08 | -0.17 |
| `inv_V_small` | `logistic` | 0.3204 | 0.2288 | -2.76 | -0.19 |
| `inv_dist` | `default` | 0.0 | 0.0 | -0.2 | 0.07 |
| `inv_dist` | `logistic` | 0.0 | 0.0 | 0.12 | -0.41 |
| `inv_dist_sq` | `default` | 0.0 | 0.0 | -0.79 | -1.13 |
| `inv_dist_sq` | `logistic` | 0.0 | 0.0 | -0.24 | -0.54 |
| `quadratic_over_rho` | `default` | 0.0 | 0.0 | 0.15 | -0.23 |
| `quadratic_over_rho` | `logistic` | 0.0 | 0.0 | -0.25 | 0.79 |
| `scaled_inv_dist_sq` | `default` | 0.0 | 0.0 | 0.03 | 0.19 |
| `scaled_inv_dist_sq` | `logistic` | 0.0 | 0.0 | -0.52 | -0.08 |

## Timing summary

- old: mean 15.9s over all runs; log_scale=false mean 17.49s; log_scale=true mean 14.32s; max 60.67s
- new: mean 10.91s over all runs; log_scale=false mean 10.98s; log_scale=true mean 10.84s; max 12.64s
