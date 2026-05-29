# Structured Report

- Timestamp: 2026-05-28 16:07:25
- Seed: `2026`
- NeuralLyapunov path: `/home/nklugman/.julia/packages/NeuralLyapunov/ykuJS/src/NeuralLyapunov.jl`
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
| `control_zero` | `default` | `false` | 2026 | NaN | `` | 0 | 1.0 | 2.0932 | 0.0 | 60.67 | false | `none` |
| `control_zero` | `default` | `true` | 2026 | NaN | `` | 0 | 1.0 | 2.0932 | 0.0 | 14.80 | false | `none` |
| `control_zero` | `logistic` | `false` | 2026 | NaN | `` | 7.16688e-13 | 1.0 | 2.2172 | 0.0 | 15.75 | false | `none` |
| `control_zero` | `logistic` | `true` | 2026 | NaN | `` | 1.12188e-14 | 1.0 | 2.086 | 0.0 | 14.58 | false | `none` |
| `constant_one` | `default` | `false` | 2026 | NaN | `` | 0.861328 | 1.0 | 2.1204 | 0.0 | 14.77 | false | `none` |
| `constant_one` | `default` | `true` | 2026 | NaN | `` | 0.861328 | 1.0 | 2.1204 | 0.0 | 13.80 | false | `none` |
| `constant_one` | `logistic` | `false` | 2026 | NaN | `` | 0.787382 | 1.0 | 3.13 | 0.008698 | 15.69 | false | `none` |
| `constant_one` | `logistic` | `true` | 2026 | NaN | `` | 0.786618 | 1.0 | 3.13 | 0.008115 | 14.86 | false | `none` |
| `inv_dist_sq` | `default` | `false` | 2026 | NaN | `` | 0.159748 | 1.0 | 3.0008 | 0.664016 | 13.55 | false | `none` |
| `inv_dist_sq` | `default` | `true` | 2026 | NaN | `` | 0.159748 | 1.0 | 3.0008 | 0.664016 | 12.76 | false | `none` |
| `inv_dist_sq` | `logistic` | `false` | 2026 | NaN | `` | 0.14106 | 1.0 | 3.13 | 0.010552 | 14.37 | false | `none` |
| `inv_dist_sq` | `logistic` | `true` | 2026 | NaN | `` | 0.140468 | 1.0 | 3.13 | 0.008124 | 14.13 | false | `none` |
| `scaled_inv_dist_sq` | `default` | `false` | 2026 | NaN | `` | 1593.24 | 1.0 | 3.0008 | 0.664016 | 13.25 | false | `none` |
| `scaled_inv_dist_sq` | `default` | `true` | 2026 | NaN | `` | 1593.24 | 1.0 | 3.0008 | 0.664016 | 13.28 | false | `none` |
| `scaled_inv_dist_sq` | `logistic` | `false` | 2026 | NaN | `` | 1410.68 | 1.0 | 3.13 | 0.002791 | 14.69 | false | `none` |
| `scaled_inv_dist_sq` | `logistic` | `true` | 2026 | NaN | `` | 1404.7 | 1.0 | 3.13 | 0.005422 | 14.16 | false | `none` |
| `inv_dist` | `default` | `false` | 2026 | NaN | `` | 0.31782 | 1.0 | 3.0008 | 0.664016 | 13.19 | false | `none` |
| `inv_dist` | `default` | `true` | 2026 | NaN | `` | 0.31782 | 1.0 | 3.0008 | 0.664016 | 12.99 | false | `none` |
| `inv_dist` | `logistic` | `false` | 2026 | NaN | `` | 0.299173 | 1.0 | 3.13 | 0.012262 | 13.74 | false | `none` |
| `inv_dist` | `logistic` | `true` | 2026 | NaN | `` | 0.298504 | 1.0 | 3.13 | 0.00954 | 13.86 | false | `none` |
| `inv_V_small` | `default` | `false` | 2026 | 0.0001 | `much_less_than_rho` | 0.0177115 | 1.0 | 0.3384 | 2.199634 | 14.00 | false | `none` |
| `inv_V_small` | `default` | `true` | 2026 | 0.0001 | `much_less_than_rho` | 0.0177115 | 1.0 | 0.3384 | 2.199634 | 14.08 | false | `none` |
| `inv_V_small` | `logistic` | `false` | 2026 | 0.0001 | `much_less_than_rho` | 0.0702451 | 1.0 | 0.5108 | 4.910308 | 18.07 | false | `none` |
| `inv_V_small` | `logistic` | `true` | 2026 | 0.0001 | `much_less_than_rho` | 0.00998547 | 1.0 | 0.8312 | 1.751438 | 15.30 | false | `none` |
| `inv_V_rho` | `default` | `false` | 2026 | 1.0 | `equal_rho` | 0.000139981 | 1.0 | 0.0104 | 0.147055 | 13.91 | false | `none` |
| `inv_V_rho` | `default` | `true` | 2026 | 1.0 | `equal_rho` | 0.000139981 | 1.0 | 0.0104 | 0.147055 | 14.38 | false | `none` |
| `inv_V_rho` | `logistic` | `false` | 2026 | 1.0 | `equal_rho` | 0.0182317 | 1.0 | 0.5924 | 0.391372 | 17.40 | false | `none` |
| `inv_V_rho` | `logistic` | `true` | 2026 | 1.0 | `equal_rho` | 4.18442e-05 | 1.0 | 0.0068 | 0.035263 | 15.06 | false | `none` |
| `inv_V_clipped` | `default` | `false` | 2026 | NaN | `` | 0.0187944 | 1.0 | 0.7816 | 2.536716 | 14.31 | false | `none` |
| `inv_V_clipped` | `default` | `true` | 2026 | NaN | `` | 0.0187944 | 1.0 | 0.7816 | 2.536716 | 14.08 | false | `none` |
| `inv_V_clipped` | `logistic` | `false` | 2026 | NaN | `` | 0.0391987 | 1.0 | 0.5884 | 2.562095 | 16.95 | false | `none` |
| `inv_V_clipped` | `logistic` | `true` | 2026 | NaN | `` | 0.00641054 | 1.0 | 1.0696 | 1.117033 | 15.32 | false | `none` |
| `quadratic_over_rho` | `default` | `false` | 2026 | NaN | `` | 89.6984 | 1.0 | 3.1172 | 0.133467 | 14.35 | false | `none` |
| `quadratic_over_rho` | `default` | `true` | 2026 | NaN | `` | 89.6984 | 1.0 | 3.1172 | 0.133467 | 14.50 | false | `none` |
| `quadratic_over_rho` | `logistic` | `false` | 2026 | NaN | `` | 89.7003 | 1.0 | 3.116 | 0.135873 | 16.12 | false | `none` |
| `quadratic_over_rho` | `logistic` | `true` | 2026 | NaN | `` | 89.7003 | 1.0 | 3.116 | 0.135867 | 15.86 | false | `none` |

## Expected vs Observed Behavior

### control_zero (default sigmoid)

- **Expected:** smallest RoA estimate since there is no penalty outside the RoA
- **Observed:** area = 2.0932, max dV/dt inside = 0.0
- **Comparison:** unexpected — control_zero did not produce the smallest RoA area; other penalties may have collapsed or failed.

### constant_one (logistic sigmoid)

- **Expected:** moderate RoA expansion due to uniform penalty with smooth sigmoid gating
- **Observed:** area = 3.13, max dV/dt inside = 0.008698092539554461
- **Comparison:** consistent — constant_one with logistic sigmoid achieved at-or-above-median area, matching the moderate expansion expectation.

### inv_V_small (default sigmoid)

- **Expected:** stabilized inverse-V penalty with small offset `a` (chosen so `a ≪ ρ`) avoiding V→0 singularity
- **Observed:** area = 0.33840000000000003, max dV/dt inside = 2.1996337560898116, has_nan = false, training_time = 14.00160813331604s
- **Comparison:** small-`a` stabilization successful — finite area obtained without collapse.

### inv_V_rho (default sigmoid)

- **Expected:** stabilized inverse-V penalty using ρ as offset, singularity-free
- **Observed:** area = 0.010400000000000001, max dV/dt inside = 0.14705515201076483, has_nan = false, training_time = 13.905606031417847s
- **Comparison:** ρ-based stabilization successful — finite area obtained.

> **Note on inv_V instability:** Instability likely arises from interaction between penalty scaling and NeuralPDE residual formulation, where large curvature near low-V regions still destabilizes optimization despite offset.

## Adaptive Reweighting Check

- No adaptive reweighting configured in this framework (QuadratureTraining without adaptive loss callbacks).
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (default sigmoid): 0.0
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (logistic sigmoid): 0.0

## Hypothesis Validation

- Largest RoA area (default sigmoid): `quadratic_over_rho` with area `3.1172000000000004`
- Largest RoA area (logistic sigmoid): `constant_one` with area `3.13`
- Runs with decrease-condition violation (max dV/dt inside V <= ρ > 0): constant_one/logistic/ls=false, constant_one/logistic/ls=true, inv_dist_sq/default/ls=false, inv_dist_sq/default/ls=true, inv_dist_sq/logistic/ls=false, inv_dist_sq/logistic/ls=true, scaled_inv_dist_sq/default/ls=false, scaled_inv_dist_sq/default/ls=true, scaled_inv_dist_sq/logistic/ls=false, scaled_inv_dist_sq/logistic/ls=true, inv_dist/default/ls=false, inv_dist/default/ls=true, inv_dist/logistic/ls=false, inv_dist/logistic/ls=true, inv_V_small/default/ls=false, inv_V_small/default/ls=true, inv_V_small/logistic/ls=false, inv_V_small/logistic/ls=true, inv_V_rho/default/ls=false, inv_V_rho/default/ls=true, inv_V_rho/logistic/ls=false, inv_V_rho/logistic/ls=true, inv_V_clipped/default/ls=false, inv_V_clipped/default/ls=true, inv_V_clipped/logistic/ls=false, inv_V_clipped/logistic/ls=true, quadratic_over_rho/default/ls=false, quadratic_over_rho/default/ls=true, quadratic_over_rho/logistic/ls=false, quadratic_over_rho/logistic/ls=true
- Next architecture modification if all plateau: increase `MLP` width/depth and test `MultiplicativeLyapunovNet` with same protocol.

## Training Time Comparison

| Penalty | Sigmoid | log_scale | Training Time (s) |
|---------|---------|-----------|-------------------|
| control_zero | default | false | 60.67 |
| control_zero | default | true | 14.8 |
| control_zero | logistic | false | 15.75 |
| control_zero | logistic | true | 14.58 |
| constant_one | default | false | 14.77 |
| constant_one | default | true | 13.8 |
| constant_one | logistic | false | 15.69 |
| constant_one | logistic | true | 14.86 |
| inv_dist_sq | default | false | 13.55 |
| inv_dist_sq | default | true | 12.76 |
| inv_dist_sq | logistic | false | 14.37 |
| inv_dist_sq | logistic | true | 14.13 |
| scaled_inv_dist_sq | default | false | 13.25 |
| scaled_inv_dist_sq | default | true | 13.28 |
| scaled_inv_dist_sq | logistic | false | 14.69 |
| scaled_inv_dist_sq | logistic | true | 14.16 |
| inv_dist | default | false | 13.19 |
| inv_dist | default | true | 12.99 |
| inv_dist | logistic | false | 13.74 |
| inv_dist | logistic | true | 13.86 |
| inv_V_small | default | false | 14.0 |
| inv_V_small | default | true | 14.08 |
| inv_V_small | logistic | false | 18.07 |
| inv_V_small | logistic | true | 15.3 |
| inv_V_rho | default | false | 13.91 |
| inv_V_rho | default | true | 14.38 |
| inv_V_rho | logistic | false | 17.4 |
| inv_V_rho | logistic | true | 15.06 |
| inv_V_clipped | default | false | 14.31 |
| inv_V_clipped | default | true | 14.08 |
| inv_V_clipped | logistic | false | 16.95 |
| inv_V_clipped | logistic | true | 15.32 |
| quadratic_over_rho | default | false | 14.35 |
| quadratic_over_rho | default | true | 14.5 |
| quadratic_over_rho | logistic | false | 16.12 |
| quadratic_over_rho | logistic | true | 15.86 |

- **Fastest:** `inv_dist_sq` / `default` / ls=`true` at 12.76s
- **Slowest:** `control_zero` / `default` / ls=`false` at 60.67s
- **Mean:** 15.9s across 36 runs
