# Seed study: `inv_dist_sq` vs `constant_one`

Paired comparison of the two configurations that tied for largest RoA area on the 36-run grid. Both use the logistic gate (k = 20.0) and `log_scale = false`; everything else matches `scripts/run_experiments.jl` (Van der Pol, 3x32 tanh MLP + `AdditiveLyapunovNet`, `QuasiRandomTraining(1024)`, `Adam(0.01)` x300 then `Adam()` x300, rho = 1.0, `NeuralLyapunov.benchmark` for training and evaluation).

- Seeds: 2026-2045 (20 paired seeds, 40 runs)
- Generated: 2026-10-22 10:13
- All deltas below are **`constant_one` - `inv_dist_sq`**, so a positive time delta means `constant_one` is slower.
- Area is counted on the 201x201 grid, so it is quantized in steps of one cell = 0.0004.

## Per-seed

| seed | area `inv_dist_sq` | area `constant_one` | Δ area | max dV/dt in `inv_dist_sq` | max dV/dt in `constant_one` | train `inv_dist_sq` (s) | train `constant_one` (s) | Δ train (s) | skips A/B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 3.13 | 3.13 | +0.0 | 0.0106 | 0.0087 | 12.05 | 11.76 | -0.28 | 0/0 |
| 2027 | 3.13 | 3.13 | +0.0 | 0.0182 | 0.0298 | 12.88 | 11.07 | -1.82 | 0/0 |
| 2028 | 3.13 | 3.13 | +0.0 | 0.0142 | 0.0 | 11.0 | 11.41 | +0.4 | 0/0 |
| 2029 | 3.13 | 3.13 | +0.0 | 0.0197 | 0.0043 | 11.7 | 11.42 | -0.28 | 0/0 |
| 2030 | 3.13 | 3.13 | +0.0 | 0.0071 | 0.001 | 11.54 | 11.26 | -0.28 | 0/0 |
| 2031 | 3.13 | 3.13 | +0.0 | 0.0041 | 0.0005 | 10.94 | 11.6 | +0.67 | 0/0 |
| 2032 | 3.13 | 3.13 | +0.0 | 0.0104 | 0.0084 | 12.66 | 12.34 | -0.33 | 0/0 |
| 2033 | 3.13 | 3.13 | +0.0 | 0.0093 | 0.01 | 12.23 | 13.03 | +0.8 | 0/0 |
| 2034 | 3.13 | 3.13 | +0.0 | 0.0056 | 0.0113 | 11.52 | 11.76 | +0.23 | 0/0 |
| 2035 | 3.13 | 3.13 | +0.0 | 0.0003 | 0.0002 | 14.24 | 12.76 | -1.49 | 0/0 |
| 2036 | 3.13 | 3.1276 | -0.0024 | 0.0007 | 0.0023 | 11.93 | 12.82 | +0.89 | 0/0 |
| 2037 | 3.13 | 3.13 | +0.0 | 0.003 | 0.0048 | 14.08 | 11.49 | -2.59 | 0/0 |
| 2038 | 3.13 | 3.13 | +0.0 | 0.0054 | 0.0098 | 11.9 | 11.38 | -0.51 | 0/0 |
| 2039 | 3.13 | 3.13 | +0.0 | 0.0001 | 0.0108 | 11.84 | 13.03 | +1.19 | 0/0 |
| 2040 | 3.13 | 3.13 | +0.0 | 0.0015 | 0.005 | 11.75 | 11.84 | +0.09 | 0/0 |
| 2041 | 3.13 | 3.13 | +0.0 | 0.0006 | 0.0072 | 9.13 | 8.88 | -0.24 | 0/0 |
| 2042 | 3.13 | 3.13 | +0.0 | 0.0009 | 0.0026 | 8.8 | 9.1 | +0.31 | 0/0 |
| 2043 | 3.13 | 3.13 | +0.0 | 0.0027 | 0.0001 | 9.3 | 10.14 | +0.83 | 0/0 |
| 2044 | 3.13 | 3.13 | +0.0 | 0.0118 | 0.0057 | 8.99 | 9.43 | +0.44 | 0/0 |
| 2045 | 3.13 | 3.13 | +0.0 | 0.0052 | 0.0073 | 9.17 | 9.05 | -0.12 | 0/0 |

## RoA area

- `inv_dist_sq`: mean 3.13, sd 0.0, median 3.13, range 3.13-3.13
- `constant_one`: mean 3.1299, sd 0.0005, median 3.13, range 3.1276-3.13
- Paired Δ (`constant_one` - `inv_dist_sq`): mean -0.0001, sd 0.0005, median +0.0
- Bootstrap 95% CI for the mean Δ: [-0.0004, 0.0]
- Sign test: `constant_one` larger in 0 of 1 non-tied seeds (p = 1.0)
- Wilcoxon signed-rank (exact, n = 1): W = 0.0, p = 1.0 [exact]
- Identical to within one grid cell in **19 of 20** seeds

## Training time

- `inv_dist_sq`: mean 11.38 s, sd 1.6 s, median 11.73 s, range 8.8-14.24 s
- `constant_one`: mean 11.28 s, sd 1.32 s, median 11.46 s, range 8.88-13.03 s
- Paired Δ (`constant_one` - `inv_dist_sq`): mean -0.1 s, sd 0.95 s, median -0.01 s
- Bootstrap 95% CI for the mean Δ: [-0.53, 0.27] s
- Sign test: `constant_one` larger in 10 of 20 non-tied seeds (p = 1.0)
- Wilcoxon signed-rank (exact, n = 20): W = 108.0, p = 0.9273 [exact]

## Decrease condition inside the claimed set

`max_dVdt_inside` is the max of V̇ over `{V <= rho}` on the grid. V̇ vanishes at the origin, which is itself a grid point, so this quantity can never be negative and a tiny positive value may only reflect the origin neighbourhood. The column below excludes ||x|| < 0.05 to separate that from a genuine violation.

| penalty | mean max dV/dt | max over seeds | mean max dV/dt excl. origin ball | max excl. over seeds | seeds with excl. max > 1e-3 | mean ||argmax|| |
|---|---:|---:|---:|---:|---:|---:|
| `inv_dist_sq` | 0.00657 | 0.01971 | 0.00657 | 0.01971 | 15/20 | 0.52 |
| `constant_one` | 0.00649 | 0.02979 | 0.00649 | 0.02979 | 16/20 | 0.376 |

## Classifier, loss and NaN gradients

| penalty | mean accuracy | mean tp | mean fp | mean loss (last 100 iters) | mean within-run loss sd | total NaN-grad skips | mean eval time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `inv_dist_sq` | 0.4365 | 193.8 | 0.0 | 0.1479 | 0.0044 | 0 | 0.048 |
| `constant_one` | 0.4365 | 193.8 | 0.0 | 0.7936 | 0.0077 | 0 | 0.044 |

## Timing drift check

Runs are listed in execution order; the two configurations alternate which goes first, so a trend here would show up as drift rather than as a penalty effect.

- First half of runs: mean 11.96 s; second half: mean 10.7 s
- Ran first within its seed: mean 11.23 s; ran second: mean 11.43 s
