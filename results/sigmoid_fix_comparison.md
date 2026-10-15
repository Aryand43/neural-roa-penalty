# Rerun Comparison

- Old: `results/summary_nl_v0.4.9_naive_sigmoid.csv`
- New: `results/summary.csv`

## Per-run

| penalty | sigmoid | log_scale | area old | area new | Δ area | max dV/dt old | max dV/dt new | time old (s) | time new (s) | NaN-grad skips new | accuracy new | eval time new (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_one` | `default` | `false` | 2.1204 | 2.1204 | +0.0 | 0.0 | 0.0 | 11.02 | 10.63 | 0 | 0.336 | 0.04 |
| `constant_one` | `default` | `true` | 2.1204 | 2.1204 | +0.0 | 0.0 | 0.0 | 11.07 | 10.73 | 0 | 0.336 | 0.03 |
| `constant_one` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0087 | 0.0087 | 11.89 | 12.68 | 0 | 0.402 | 0.04 |
| `constant_one` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0081 | 0.0081 | 12.06 | 12.17 | 0 | 0.402 | 0.04 |
| `control_zero` | `default` | `false` | 2.128 | 2.128 | +0.0 | 0.0029 | 0.0029 | 8.02 | 6.82 | 0 | 0.336 | 0.04 |
| `control_zero` | `default` | `true` | 2.128 | 2.128 | +0.0 | 0.0029 | 0.0029 | 7.14 | 7.19 | 0 | 0.336 | 0.03 |
| `control_zero` | `logistic` | `false` | 2.2172 | 2.2172 | +0.0 | 0.0 | 0.0 | 8.08 | 9.68 | 0 | 0.344 | 0.03 |
| `control_zero` | `logistic` | `true` | 2.086 | 2.086 | +0.0 | 0.0 | 0.0 | 7.72 | 8.04 | 0 | 0.334 | 0.04 |
| `inv_V_clipped` | `default` | `false` | 0.7812 | 0.7812 | +0.0 | 2.5541 | 2.5541 | 10.87 | 11.0 | 0 | 0.26 | 0.03 |
| `inv_V_clipped` | `default` | `true` | 0.7812 | 0.7812 | +0.0 | 2.5541 | 2.5541 | 11.36 | 11.16 | 0 | 0.26 | 0.05 |
| `inv_V_clipped` | `logistic` | `false` | 0.5884 | 0.6644 | +0.076 | 2.5643 | 6.9007 | 11.88 | 13.77 | 0 | 0.252 | 0.04 |
| `inv_V_clipped` | `logistic` | `true` | 1.0716 | 0.786 | -0.2856 | 2.1403 | 2.2878 | 11.56 | 12.11 | 0 | 0.267 | 0.05 |
| `inv_V_rho` | `default` | `false` | 0.0104 | 0.0104 | +0.0 | 0.147 | 0.147 | 10.49 | 10.92 | 0 | 0.217 | 0.03 |
| `inv_V_rho` | `default` | `true` | 0.0104 | 0.0104 | +0.0 | 0.147 | 0.147 | 10.46 | 11.05 | 0 | 0.217 | 0.03 |
| `inv_V_rho` | `logistic` | `false` | 0.5924 | 0.008 | -0.5844 | 0.3914 | 0.3761 | 11.31 | 12.22 | 0 | 0.217 | 0.03 |
| `inv_V_rho` | `logistic` | `true` | 0.0092 | 0.0112 | +0.002 | 0.0186 | 0.4764 | 11.29 | 11.82 | 0 | 0.217 | 0.03 |
| `inv_V_small` | `default` | `false` | 0.3864 | 0.3864 | +0.0 | 2.4969 | 2.4969 | 11.45 | 11.52 | 0 | 0.232 | 0.04 |
| `inv_V_small` | `default` | `true` | 0.3864 | 0.3864 | +0.0 | 2.4969 | 2.4969 | 11.28 | 10.94 | 0 | 0.232 | 0.07 |
| `inv_V_small` | `logistic` | `false` | 0.5108 | 1.1596 | +0.6488 | 4.9124 | 3.1487 | 12.14 | 11.82 | 0 | 0.289 | 0.03 |
| `inv_V_small` | `logistic` | `true` | 0.7396 | 0.8088 | +0.0692 | 1.4942 | 2.4856 | 11.95 | 11.95 | 0 | 0.267 | 0.05 |
| `inv_dist` | `default` | `false` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 10.42 | 10.16 | 0 | 0.395 | 0.03 |
| `inv_dist` | `default` | `true` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 10.5 | 10.37 | 0 | 0.395 | 0.04 |
| `inv_dist` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0123 | 0.0123 | 12.38 | 10.72 | 0 | 0.402 | 0.04 |
| `inv_dist` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0095 | 0.0095 | 11.97 | 10.58 | 0 | 0.402 | 0.05 |
| `inv_dist_sq` | `default` | `false` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 11.13 | 10.43 | 0 | 0.395 | 0.03 |
| `inv_dist_sq` | `default` | `true` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 10.0 | 10.12 | 0 | 0.395 | 0.03 |
| `inv_dist_sq` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0106 | 0.0106 | 11.5 | 10.95 | 0 | 0.402 | 0.05 |
| `inv_dist_sq` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0081 | 0.0081 | 10.96 | 10.24 | 0 | 0.402 | 0.04 |
| `quadratic_over_rho` | `default` | `false` | 3.1172 | 3.1172 | +0.0 | 0.1335 | 0.1335 | 11.59 | 11.39 | 0 | 0.401 | 0.03 |
| `quadratic_over_rho` | `default` | `true` | 3.1172 | 3.1172 | +0.0 | 0.1335 | 0.1335 | 11.36 | 10.87 | 0 | 0.401 | 0.03 |
| `quadratic_over_rho` | `logistic` | `false` | 3.116 | 3.116 | +0.0 | 0.1359 | 0.1359 | 11.85 | 11.97 | 0 | 0.401 | 0.03 |
| `quadratic_over_rho` | `logistic` | `true` | 3.116 | 3.116 | +0.0 | 0.1359 | 0.1359 | 12.64 | 12.12 | 0 | 0.401 | 0.04 |
| `scaled_inv_dist_sq` | `default` | `false` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 10.19 | 9.62 | 0 | 0.395 | 0.04 |
| `scaled_inv_dist_sq` | `default` | `true` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 10.38 | 10.24 | 0 | 0.395 | 0.03 |
| `scaled_inv_dist_sq` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0028 | 0.0028 | 11.43 | 11.25 | 0 | 0.402 | 0.05 |
| `scaled_inv_dist_sq` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0054 | 0.0054 | 11.35 | 10.7 | 0 | 0.402 | 0.03 |

## log_scale effect (log(V/ρ) − (V − ρ)) per penalty/sigmoid

| penalty | sigmoid | Δ area old | Δ area new | Δ time old (s) | Δ time new (s) |
|---|---|---:|---:|---:|---:|
| `constant_one` | `default` | 0.0 | 0.0 | 0.04 | 0.1 |
| `constant_one` | `logistic` | 0.0 | 0.0 | 0.17 | -0.51 |
| `control_zero` | `default` | 0.0 | 0.0 | -0.88 | 0.37 |
| `control_zero` | `logistic` | -0.1312 | -0.1312 | -0.36 | -1.63 |
| `inv_V_clipped` | `default` | 0.0 | 0.0 | 0.49 | 0.15 |
| `inv_V_clipped` | `logistic` | 0.4832 | 0.1216 | -0.32 | -1.66 |
| `inv_V_rho` | `default` | 0.0 | 0.0 | -0.02 | 0.12 |
| `inv_V_rho` | `logistic` | -0.5832 | 0.0032 | -0.03 | -0.39 |
| `inv_V_small` | `default` | 0.0 | 0.0 | -0.17 | -0.58 |
| `inv_V_small` | `logistic` | 0.2288 | -0.3508 | -0.19 | 0.13 |
| `inv_dist` | `default` | 0.0 | 0.0 | 0.07 | 0.21 |
| `inv_dist` | `logistic` | 0.0 | 0.0 | -0.41 | -0.14 |
| `inv_dist_sq` | `default` | 0.0 | 0.0 | -1.13 | -0.31 |
| `inv_dist_sq` | `logistic` | 0.0 | 0.0 | -0.54 | -0.71 |
| `quadratic_over_rho` | `default` | 0.0 | 0.0 | -0.23 | -0.52 |
| `quadratic_over_rho` | `logistic` | 0.0 | 0.0 | 0.79 | 0.15 |
| `scaled_inv_dist_sq` | `default` | 0.0 | 0.0 | 0.19 | 0.62 |
| `scaled_inv_dist_sq` | `logistic` | 0.0 | 0.0 | -0.08 | -0.55 |

## Timing summary

- old: mean 10.91s over all runs; log_scale=false mean 10.98s; log_scale=true mean 10.84s; max 12.64s
- new: mean 10.83s over all runs; log_scale=false mean 10.97s; log_scale=true mean 10.69s; max 13.77s
