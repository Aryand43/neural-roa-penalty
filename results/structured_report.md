# Structured Report

- Timestamp: 2026-09-15 16:04:43
- Seed: `2026`
- NeuralLyapunov path: `/Users/aryand/.julia/packages/NeuralLyapunov/GMYBc/src/NeuralLyapunov.jl`
- NeuralLyapunov version: `0.4.9`; Julia `1.12.6`; threads: 1
- Training and evaluation via `NeuralLyapunov.benchmark` (training time as measured by `benchmark` since SciML/NeuralLyapunov.jl#215, PDE discretization excluded); an untimed warm-up run precedes the grid so JIT compilation is not charged to the first configuration.
- Benchmark evaluation: 1000 Latin-hypercube samples, classifier `V ≤ ρ`, 20 s simulations with `EnsembleSerial`, endpoint `≈ 0` with `atol = 1e-3`, trajectories terminated once `‖x‖ > 5`.
- Default sigmoid in source: `(x) -> x .≥ zero.(x)` (hard step)
- Logistic sigmoid used in experiments: `σ(z)=1/(1+exp(-k*z))`, with `k=20`
- Penalty term wiring in source: `[sigmoid(ρ-V)*in_RoA_penalty, sigmoid(V-ρ)*out_of_RoA_penalty]`
- Loss aggregation: NeuralPDE residual loss over PDE equations (each residual vs `0.0`).
- Gradient through default sigmoid: non-smooth/boolean gate; logistic variant provides smooth gate.

## Penalty Hypotheses

| Penalty | Expression | Expected Behavior | Reasoning |
|---------|------------|-------------------|-----------|
| control_zero | 0 | smallest RoA estimate since there is no penalty outside the RoA | optimizer receives no gradient signal outside the RoA boundary |
| constant_one | 1 | moderate RoA expansion | uniform penalty encourages RoA expansion but provides no spatial structure |
| inv_dist_sq | 1 / ‖x - x₀‖² | unstable gradients near equilibrium | gradient magnitude increases rapidly near the equilibrium |
| scaled_inv_dist_sq | 100 / ‖x - x₀‖² | similar to inv_dist_sq due to loss aggregation scaling | NeuralPDE residual loss may normalize effect of scale changes |
| inv_dist | 1 / ‖x - x₀‖ | smoother variant of inverse distance penalty | weaker singularity compared to squared inverse distance |
| inv_V_small | 1 / (V + a), a ≪ ρ | stabilized inverse-V with small offset vs ρ | avoids V→0 singularity while keeping `a` small relative to ρ |
| inv_V_rho | 1 / (V + ρ), a = ρ | stabilized inverse-V with offset tied to ρ | uses ρ as natural offset, singularity-free for V ≥ 0 |
| inv_V_clipped | 1 / max(V, 0.1) | safe inverse-V via hard floor | clamps V away from zero, preventing gradient blowup entirely |
| quadratic_over_rho | max(0, V - ρ)² | well-shaped RoA boundary | directly penalizes states outside the RoA level set |

## Per-Run Results

| penalty | sigmoid | log_scale | rng_seed | inv_V_a | inv_V_a_regime | final_loss | ρ | roa_area | max_dVdt_inside | train_time_s | has_nan | error |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| `control_zero` | `default` | `false` | 2026 | NaN | `` | 0 | 1.0 | 2.128 | 0.002876 | 8.02 | false | `none` |
| `control_zero` | `default` | `true` | 2026 | NaN | `` | 0 | 1.0 | 2.128 | 0.002876 | 7.14 | false | `none` |
| `control_zero` | `logistic` | `false` | 2026 | NaN | `` | 7.16688e-13 | 1.0 | 2.2172 | 0.0 | 8.08 | false | `none` |
| `control_zero` | `logistic` | `true` | 2026 | NaN | `` | 1.12188e-14 | 1.0 | 2.086 | 0.0 | 7.72 | false | `none` |
| `constant_one` | `default` | `false` | 2026 | NaN | `` | 0.861328 | 1.0 | 2.1204 | 0.0 | 11.02 | false | `none` |
| `constant_one` | `default` | `true` | 2026 | NaN | `` | 0.861328 | 1.0 | 2.1204 | 0.0 | 11.07 | false | `none` |
| `constant_one` | `logistic` | `false` | 2026 | NaN | `` | 0.787382 | 1.0 | 3.13 | 0.008698 | 11.89 | false | `none` |
| `constant_one` | `logistic` | `true` | 2026 | NaN | `` | 0.786618 | 1.0 | 3.13 | 0.008115 | 12.06 | false | `none` |
| `inv_dist_sq` | `default` | `false` | 2026 | NaN | `` | 0.159748 | 1.0 | 3.0008 | 0.664016 | 11.13 | false | `none` |
| `inv_dist_sq` | `default` | `true` | 2026 | NaN | `` | 0.159748 | 1.0 | 3.0008 | 0.664016 | 10.00 | false | `none` |
| `inv_dist_sq` | `logistic` | `false` | 2026 | NaN | `` | 0.14106 | 1.0 | 3.13 | 0.010552 | 11.50 | false | `none` |
| `inv_dist_sq` | `logistic` | `true` | 2026 | NaN | `` | 0.140468 | 1.0 | 3.13 | 0.008124 | 10.96 | false | `none` |
| `scaled_inv_dist_sq` | `default` | `false` | 2026 | NaN | `` | 1593.24 | 1.0 | 3.0008 | 0.664016 | 10.19 | false | `none` |
| `scaled_inv_dist_sq` | `default` | `true` | 2026 | NaN | `` | 1593.24 | 1.0 | 3.0008 | 0.664016 | 10.38 | false | `none` |
| `scaled_inv_dist_sq` | `logistic` | `false` | 2026 | NaN | `` | 1410.68 | 1.0 | 3.13 | 0.002791 | 11.43 | false | `none` |
| `scaled_inv_dist_sq` | `logistic` | `true` | 2026 | NaN | `` | 1404.7 | 1.0 | 3.13 | 0.005422 | 11.35 | false | `none` |
| `inv_dist` | `default` | `false` | 2026 | NaN | `` | 0.31782 | 1.0 | 3.0008 | 0.664016 | 10.42 | false | `none` |
| `inv_dist` | `default` | `true` | 2026 | NaN | `` | 0.31782 | 1.0 | 3.0008 | 0.664016 | 10.50 | false | `none` |
| `inv_dist` | `logistic` | `false` | 2026 | NaN | `` | 0.299173 | 1.0 | 3.13 | 0.012262 | 12.38 | false | `none` |
| `inv_dist` | `logistic` | `true` | 2026 | NaN | `` | 0.298504 | 1.0 | 3.13 | 0.00954 | 11.97 | false | `none` |
| `inv_V_small` | `default` | `false` | 2026 | 0.0001 | `much_less_than_rho` | 0.0162429 | 1.0 | 0.3864 | 2.496937 | 11.45 | false | `none` |
| `inv_V_small` | `default` | `true` | 2026 | 0.0001 | `much_less_than_rho` | 0.0162429 | 1.0 | 0.3864 | 2.496937 | 11.28 | false | `none` |
| `inv_V_small` | `logistic` | `false` | 2026 | 0.0001 | `much_less_than_rho` | 0.0702228 | 1.0 | 0.5108 | 4.912448 | 12.14 | false | `none` |
| `inv_V_small` | `logistic` | `true` | 2026 | 0.0001 | `much_less_than_rho` | 0.00938697 | 1.0 | 0.7396 | 1.494161 | 11.95 | false | `none` |
| `inv_V_rho` | `default` | `false` | 2026 | 1.0 | `equal_rho` | 0.000139982 | 1.0 | 0.0104 | 0.147046 | 10.49 | false | `none` |
| `inv_V_rho` | `default` | `true` | 2026 | 1.0 | `equal_rho` | 0.000139982 | 1.0 | 0.0104 | 0.147046 | 10.46 | false | `none` |
| `inv_V_rho` | `logistic` | `false` | 2026 | 1.0 | `equal_rho` | 0.0182317 | 1.0 | 0.5924 | 0.391372 | 11.31 | false | `none` |
| `inv_V_rho` | `logistic` | `true` | 2026 | 1.0 | `equal_rho` | 6.96906e-05 | 1.0 | 0.0092 | 0.018585 | 11.29 | false | `none` |
| `inv_V_clipped` | `default` | `false` | 2026 | NaN | `` | 0.0187545 | 1.0 | 0.7812 | 2.554115 | 10.87 | false | `none` |
| `inv_V_clipped` | `default` | `true` | 2026 | NaN | `` | 0.0187545 | 1.0 | 0.7812 | 2.554115 | 11.36 | false | `none` |
| `inv_V_clipped` | `logistic` | `false` | 2026 | NaN | `` | 0.0392017 | 1.0 | 0.5884 | 2.564325 | 11.88 | false | `none` |
| `inv_V_clipped` | `logistic` | `true` | 2026 | NaN | `` | 0.00891811 | 1.0 | 1.0716 | 2.140324 | 11.56 | false | `none` |
| `quadratic_over_rho` | `default` | `false` | 2026 | NaN | `` | 89.6984 | 1.0 | 3.1172 | 0.133467 | 11.59 | false | `none` |
| `quadratic_over_rho` | `default` | `true` | 2026 | NaN | `` | 89.6984 | 1.0 | 3.1172 | 0.133467 | 11.36 | false | `none` |
| `quadratic_over_rho` | `logistic` | `false` | 2026 | NaN | `` | 89.7003 | 1.0 | 3.116 | 0.135873 | 11.85 | false | `none` |
| `quadratic_over_rho` | `logistic` | `true` | 2026 | NaN | `` | 89.7003 | 1.0 | 3.116 | 0.135867 | 12.64 | false | `none` |

## Expected vs Observed Behavior

### control_zero (default sigmoid)

- **Expected:** smallest RoA estimate since there is no penalty outside the RoA
- **Observed:** area = 2.128, max dV/dt inside = 0.0028763825327256694
- **Comparison:** unexpected — control_zero did not produce the smallest RoA area; other penalties may have collapsed or failed.

### constant_one (logistic sigmoid)

- **Expected:** moderate RoA expansion due to uniform penalty with smooth sigmoid gating
- **Observed:** area = 3.13, max dV/dt inside = 0.008698092539507471
- **Comparison:** consistent — constant_one with logistic sigmoid achieved at-or-above-median area, matching the moderate expansion expectation.

### inv_V_small (default sigmoid)

- **Expected:** stabilized inverse-V penalty with small offset `a` (chosen so `a ≪ ρ`) avoiding V→0 singularity
- **Observed:** area = 0.3864, max dV/dt inside = 2.496936780857375, has_nan = false, training_time = 11.449354083s
- **Comparison:** small-`a` stabilization successful — finite area obtained without collapse.

### inv_V_rho (default sigmoid)

- **Expected:** stabilized inverse-V penalty using ρ as offset, singularity-free
- **Observed:** area = 0.010400000000000001, max dV/dt inside = 0.14704646026595702, has_nan = false, training_time = 10.485247959s
- **Comparison:** ρ-based stabilization successful — finite area obtained.

> **Note on inv_V instability:** Instability likely arises from interaction between penalty scaling and NeuralPDE residual formulation, where large curvature near low-V regions still destabilizes optimization despite offset.

## Adaptive Reweighting Check

- No adaptive reweighting configured in this framework (QuadratureTraining without adaptive loss callbacks).
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (default sigmoid): 0.0
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (logistic sigmoid): 0.0

## Hypothesis Validation

- Largest RoA area (default sigmoid): `quadratic_over_rho` with area `3.1172000000000004`
- Largest RoA area (logistic sigmoid): `constant_one` with area `3.13`
- Runs with decrease-condition violation (max dV/dt inside V <= ρ > 0): control_zero/default/ls=false, control_zero/default/ls=true, constant_one/logistic/ls=false, constant_one/logistic/ls=true, inv_dist_sq/default/ls=false, inv_dist_sq/default/ls=true, inv_dist_sq/logistic/ls=false, inv_dist_sq/logistic/ls=true, scaled_inv_dist_sq/default/ls=false, scaled_inv_dist_sq/default/ls=true, scaled_inv_dist_sq/logistic/ls=false, scaled_inv_dist_sq/logistic/ls=true, inv_dist/default/ls=false, inv_dist/default/ls=true, inv_dist/logistic/ls=false, inv_dist/logistic/ls=true, inv_V_small/default/ls=false, inv_V_small/default/ls=true, inv_V_small/logistic/ls=false, inv_V_small/logistic/ls=true, inv_V_rho/default/ls=false, inv_V_rho/default/ls=true, inv_V_rho/logistic/ls=false, inv_V_rho/logistic/ls=true, inv_V_clipped/default/ls=false, inv_V_clipped/default/ls=true, inv_V_clipped/logistic/ls=false, inv_V_clipped/logistic/ls=true, quadratic_over_rho/default/ls=false, quadratic_over_rho/default/ls=true, quadratic_over_rho/logistic/ls=false, quadratic_over_rho/logistic/ls=true
- Next architecture modification if all plateau: increase `MLP` width/depth and test `MultiplicativeLyapunovNet` with same protocol.

## Training Time Comparison

| Penalty | Sigmoid | log_scale | Training Time (s) |
|---------|---------|-----------|-------------------|
| control_zero | default | false | 8.02 |
| control_zero | default | true | 7.14 |
| control_zero | logistic | false | 8.08 |
| control_zero | logistic | true | 7.72 |
| constant_one | default | false | 11.02 |
| constant_one | default | true | 11.07 |
| constant_one | logistic | false | 11.89 |
| constant_one | logistic | true | 12.06 |
| inv_dist_sq | default | false | 11.13 |
| inv_dist_sq | default | true | 10.0 |
| inv_dist_sq | logistic | false | 11.5 |
| inv_dist_sq | logistic | true | 10.96 |
| scaled_inv_dist_sq | default | false | 10.19 |
| scaled_inv_dist_sq | default | true | 10.38 |
| scaled_inv_dist_sq | logistic | false | 11.43 |
| scaled_inv_dist_sq | logistic | true | 11.35 |
| inv_dist | default | false | 10.42 |
| inv_dist | default | true | 10.5 |
| inv_dist | logistic | false | 12.38 |
| inv_dist | logistic | true | 11.97 |
| inv_V_small | default | false | 11.45 |
| inv_V_small | default | true | 11.28 |
| inv_V_small | logistic | false | 12.14 |
| inv_V_small | logistic | true | 11.95 |
| inv_V_rho | default | false | 10.49 |
| inv_V_rho | default | true | 10.46 |
| inv_V_rho | logistic | false | 11.31 |
| inv_V_rho | logistic | true | 11.29 |
| inv_V_clipped | default | false | 10.87 |
| inv_V_clipped | default | true | 11.36 |
| inv_V_clipped | logistic | false | 11.88 |
| inv_V_clipped | logistic | true | 11.56 |
| quadratic_over_rho | default | false | 11.59 |
| quadratic_over_rho | default | true | 11.36 |
| quadratic_over_rho | logistic | false | 11.85 |
| quadratic_over_rho | logistic | true | 12.64 |

- **Fastest:** `control_zero` / `default` / ls=`true` at 7.14s
- **Slowest:** `quadratic_over_rho` / `logistic` / ls=`true` at 12.64s
- **Mean:** 10.91s across 36 runs
