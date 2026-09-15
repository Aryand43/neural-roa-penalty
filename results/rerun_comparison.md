# Rerun Comparison

- Old: `results/summary_nl_v0.4.2_2026-05-28.csv`
- New: `results/summary.csv`

## Per-run

| penalty | sigmoid | log_scale | area old | area new | Δ area | max dV/dt old | max dV/dt new | time old (s) | time new (s) | NaN-grad skips new |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `constant_one` | `default` | `false` | 2.1204 | 2.1204 | +0.0 | 0.0 | 0.0 | 14.77 | 11.83 | 0 |
| `constant_one` | `default` | `true` | 2.1204 | 2.1204 | +0.0 | 0.0 | 0.0 | 13.8 | 12.17 | 0 |
| `constant_one` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0087 | 0.0087 | 15.69 | 12.54 | 0 |
| `constant_one` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0081 | 0.0081 | 14.86 | 12.41 | 0 |
| `control_zero` | `default` | `false` | 2.0932 | 2.128 | +0.0348 | 0.0 | 0.0029 | 60.67 | 7.89 | 0 |
| `control_zero` | `default` | `true` | 2.0932 | 2.128 | +0.0348 | 0.0 | 0.0029 | 14.8 | 7.84 | 0 |
| `control_zero` | `logistic` | `false` | 2.2172 | 2.2172 | +0.0 | 0.0 | 0.0 | 15.75 | 8.99 | 0 |
| `control_zero` | `logistic` | `true` | 2.086 | 2.086 | +0.0 | 0.0 | 0.0 | 14.58 | 8.3 | 0 |
| `inv_V_clipped` | `default` | `false` | 0.7816 | 0.7812 | -0.0004 | 2.5367 | 2.5541 | 14.31 | 12.83 | 0 |
| `inv_V_clipped` | `default` | `true` | 0.7816 | 0.7812 | -0.0004 | 2.5367 | 2.5541 | 14.08 | 11.36 | 0 |
| `inv_V_clipped` | `logistic` | `false` | 0.5884 | 0.5884 | +0.0 | 2.5621 | 2.5643 | 16.95 | 13.21 | 570 |
| `inv_V_clipped` | `logistic` | `true` | 1.0696 | 1.0716 | +0.002 | 1.117 | 2.1403 | 15.32 | 12.53 | 0 |
| `inv_V_rho` | `default` | `false` | 0.0104 | 0.0104 | +0.0 | 0.1471 | 0.147 | 13.91 | 11.5 | 0 |
| `inv_V_rho` | `default` | `true` | 0.0104 | 0.0104 | +0.0 | 0.1471 | 0.147 | 14.38 | 13.18 | 0 |
| `inv_V_rho` | `logistic` | `false` | 0.5924 | 0.5924 | +0.0 | 0.3914 | 0.3914 | 17.4 | 13.67 | 591 |
| `inv_V_rho` | `logistic` | `true` | 0.0068 | 0.0092 | +0.0024 | 0.0353 | 0.0186 | 15.06 | 12.54 | 0 |
| `inv_V_small` | `default` | `false` | 0.3384 | 0.3864 | +0.048 | 2.1996 | 2.4969 | 14.0 | 11.57 | 0 |
| `inv_V_small` | `default` | `true` | 0.3384 | 0.3864 | +0.048 | 2.1996 | 2.4969 | 14.08 | 12.03 | 0 |
| `inv_V_small` | `logistic` | `false` | 0.5108 | 0.5108 | +0.0 | 4.9103 | 4.9124 | 18.07 | 12.54 | 569 |
| `inv_V_small` | `logistic` | `true` | 0.8312 | 0.7396 | -0.0916 | 1.7514 | 1.4942 | 15.3 | 12.85 | 0 |
| `inv_dist` | `default` | `false` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 13.19 | 10.06 | 0 |
| `inv_dist` | `default` | `true` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 12.99 | 10.7 | 0 |
| `inv_dist` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0123 | 0.0123 | 13.74 | 11.74 | 0 |
| `inv_dist` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0095 | 0.0095 | 13.86 | 12.52 | 0 |
| `inv_dist_sq` | `default` | `false` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 13.55 | 11.4 | 0 |
| `inv_dist_sq` | `default` | `true` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 12.76 | 10.44 | 0 |
| `inv_dist_sq` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0106 | 0.0106 | 14.37 | 12.57 | 0 |
| `inv_dist_sq` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0081 | 0.0081 | 14.13 | 11.49 | 0 |
| `quadratic_over_rho` | `default` | `false` | 3.1172 | 3.1172 | +0.0 | 0.1335 | 0.1335 | 14.35 | 11.82 | 0 |
| `quadratic_over_rho` | `default` | `true` | 3.1172 | 3.1172 | +0.0 | 0.1335 | 0.1335 | 14.5 | 11.73 | 0 |
| `quadratic_over_rho` | `logistic` | `false` | 3.116 | 3.116 | +0.0 | 0.1359 | 0.1359 | 16.12 | 13.18 | 0 |
| `quadratic_over_rho` | `logistic` | `true` | 3.116 | 3.116 | +0.0 | 0.1359 | 0.1359 | 15.86 | 12.62 | 0 |
| `scaled_inv_dist_sq` | `default` | `false` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 13.25 | 10.66 | 0 |
| `scaled_inv_dist_sq` | `default` | `true` | 3.0008 | 3.0008 | +0.0 | 0.664 | 0.664 | 13.28 | 11.59 | 0 |
| `scaled_inv_dist_sq` | `logistic` | `false` | 3.13 | 3.13 | +0.0 | 0.0028 | 0.0028 | 14.69 | 13.09 | 0 |
| `scaled_inv_dist_sq` | `logistic` | `true` | 3.13 | 3.13 | +0.0 | 0.0054 | 0.0054 | 14.16 | 13.51 | 0 |

## log_scale effect (log(V/ρ) − (V − ρ)) per penalty/sigmoid

| penalty | sigmoid | Δ area old | Δ area new | Δ time old (s) | Δ time new (s) |
|---|---|---:|---:|---:|---:|
| `constant_one` | `default` | 0.0 | 0.0 | -0.97 | 0.34 |
| `constant_one` | `logistic` | 0.0 | 0.0 | -0.83 | -0.13 |
| `control_zero` | `default` | 0.0 | 0.0 | -45.87 | -0.05 |
| `control_zero` | `logistic` | -0.1312 | -0.1312 | -1.17 | -0.69 |
| `inv_V_clipped` | `default` | 0.0 | 0.0 | -0.23 | -1.47 |
| `inv_V_clipped` | `logistic` | 0.4812 | 0.4832 | -1.62 | -0.68 |
| `inv_V_rho` | `default` | 0.0 | 0.0 | 0.47 | 1.68 |
| `inv_V_rho` | `logistic` | -0.5856 | -0.5832 | -2.33 | -1.13 |
| `inv_V_small` | `default` | 0.0 | 0.0 | 0.08 | 0.46 |
| `inv_V_small` | `logistic` | 0.3204 | 0.2288 | -2.76 | 0.31 |
| `inv_dist` | `default` | 0.0 | 0.0 | -0.2 | 0.64 |
| `inv_dist` | `logistic` | 0.0 | 0.0 | 0.12 | 0.77 |
| `inv_dist_sq` | `default` | 0.0 | 0.0 | -0.79 | -0.96 |
| `inv_dist_sq` | `logistic` | 0.0 | 0.0 | -0.24 | -1.09 |
| `quadratic_over_rho` | `default` | 0.0 | 0.0 | 0.15 | -0.08 |
| `quadratic_over_rho` | `logistic` | 0.0 | 0.0 | -0.25 | -0.55 |
| `scaled_inv_dist_sq` | `default` | 0.0 | 0.0 | 0.03 | 0.94 |
| `scaled_inv_dist_sq` | `logistic` | 0.0 | 0.0 | -0.52 | 0.42 |

## Timing summary

- old: mean 15.9s over all runs; log_scale=false mean 17.49s; log_scale=true mean 14.32s; max 60.67s
- new: mean 11.69s over all runs; log_scale=false mean 11.73s; log_scale=true mean 11.66s; max 13.67s
