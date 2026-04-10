# Structured Report

- Timestamp: 2026-04-10 11:36:33
- NeuralLyapunov path: `C:\Users\AD\.julia\packages\NeuralLyapunov\oKiqr\src\NeuralLyapunov.jl`
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
| inv_V_small | 1 / (V + 1e-3) | stabilized inverse-V with small epsilon | avoids V→0 singularity via fixed offset |
| inv_V_rho | 1 / (V + ρ) | stabilized inverse-V scaled by ρ | uses ρ as natural offset, singularity-free for V ≥ 0 |
| quadratic_over_rho | max(0, V - ρ)² | well-shaped RoA boundary | directly penalizes states outside the RoA level set |

## Per-Run Results

- penalty: `control_zero` | sigmoid: `default` | final_loss: `0.0` | ρ: `1.0` | area: `2.9064` | max_dVdt_inside: `0.003130671393868094` | training_time: `55.58400011062622` | has_nan: `false` | error: `none`
- penalty: `control_zero` | sigmoid: `logistic` | final_loss: `2.0426950068473098e-11` | ρ: `1.0` | area: `2.9672000000000005` | max_dVdt_inside: `0.0039741924952764055` | training_time: `5.794999837875366` | has_nan: `false` | error: `none`
- penalty: `constant_one` | sigmoid: `default` | final_loss: `0.808597481760882` | ρ: `1.0` | area: `2.9104` | max_dVdt_inside: `0.046152988843279986` | training_time: `3.8320000171661377` | has_nan: `false` | error: `none`
- penalty: `constant_one` | sigmoid: `logistic` | final_loss: `0.7993816143137428` | ρ: `1.0` | area: `3.072` | max_dVdt_inside: `0.09924844106356528` | training_time: `4.224999904632568` | has_nan: `false` | error: `none`
- penalty: `inv_dist_sq` | sigmoid: `default` | final_loss: `0.1552937318199587` | ρ: `1.0` | area: `2.918` | max_dVdt_inside: `0.023539065504716163` | training_time: `2.0820000171661377` | has_nan: `false` | error: `none`
- penalty: `inv_dist_sq` | sigmoid: `logistic` | final_loss: `0.15425024660380138` | ρ: `1.0` | area: `3.1084000000000005` | max_dVdt_inside: `0.023726920333461857` | training_time: `5.437999963760376` | has_nan: `false` | error: `none`
- penalty: `scaled_inv_dist_sq` | sigmoid: `default` | final_loss: `1552.9373181995877` | ρ: `1.0` | area: `2.918` | max_dVdt_inside: `0.023539065504716163` | training_time: `1.5210001468658447` | has_nan: `false` | error: `none`
- penalty: `scaled_inv_dist_sq` | sigmoid: `logistic` | final_loss: `1502.4840541512265` | ρ: `1.0` | area: `3.0707999999999998` | max_dVdt_inside: `0.15771116754693446` | training_time: `1.9470000267028809` | has_nan: `false` | error: `none`
- penalty: `inv_dist` | sigmoid: `default` | final_loss: `0.31553499619999803` | ρ: `1.0` | area: `2.918` | max_dVdt_inside: `0.023291029088036444` | training_time: `2.4149999618530273` | has_nan: `false` | error: `none`
- penalty: `inv_dist` | sigmoid: `logistic` | final_loss: `0.2972606281946578` | ρ: `1.0` | area: `3.1296` | max_dVdt_inside: `0.011495752290922215` | training_time: `5.608000040054321` | has_nan: `false` | error: `none`
- penalty: `inv_V_small` | sigmoid: `default` | final_loss: `10.73090584339218` | ρ: `1.0` | area: `0.0404` | max_dVdt_inside: `84.11204668715888` | training_time: `1.4609999656677246` | has_nan: `false` | error: `none`
- penalty: `inv_V_small` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | training_time: `NaN` | has_nan: `true` | error: `AssertionError: isfinite(phi_c) && isfinite(dphi_c)`
- penalty: `inv_V_rho` | sigmoid: `default` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | training_time: `NaN` | has_nan: `true` | error: `AssertionError: b > a`
- penalty: `inv_V_rho` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | training_time: `NaN` | has_nan: `true` | error: `AssertionError: isfinite(phi_d) && isfinite(gphi)`
- penalty: `quadratic_over_rho` | sigmoid: `default` | final_loss: `85.23712543906419` | ρ: `1.0` | area: `3.0484000000000004` | max_dVdt_inside: `0.13978296082359337` | training_time: `2.491000175476074` | has_nan: `false` | error: `none`
- penalty: `quadratic_over_rho` | sigmoid: `logistic` | final_loss: `83.48128896399297` | ρ: `1.0` | area: `3.0336000000000003` | max_dVdt_inside: `0.16822368973977558` | training_time: `2.746999979019165` | has_nan: `false` | error: `none`

## Expected vs Observed Behavior

### control_zero (default sigmoid)

- **Expected:** smallest RoA estimate since there is no penalty outside the RoA
- **Observed:** area = 2.9064, max dV/dt inside = 0.003130671393868094
- **Comparison:** unexpected — control_zero did not produce the smallest RoA area; other penalties may have collapsed or failed.

### constant_one (logistic sigmoid)

- **Expected:** moderate RoA expansion due to uniform penalty with smooth sigmoid gating
- **Observed:** area = 3.072, max dV/dt inside = 0.09924844106356528
- **Comparison:** consistent — constant_one with logistic sigmoid achieved at-or-above-median area, matching the moderate expansion expectation.

### inv_V_small (default sigmoid)

- **Expected:** stabilized inverse-V penalty with small epsilon (1e-3) avoiding V→0 singularity
- **Observed:** area = 0.0404, max dV/dt inside = 84.11204668715888, has_nan = false, training_time = 1.4609999656677246s
- **Comparison:** epsilon stabilization successful — finite area obtained without collapse.

### inv_V_rho (default sigmoid)

- **Expected:** stabilized inverse-V penalty using ρ as offset, singularity-free
- **Observed:** area = NaN, max dV/dt inside = NaN, has_nan = true, training_time = NaNs
- **Comparison:** training still unstable despite ρ-based stabilization.

## Adaptive Reweighting Check

- No adaptive reweighting configured in this framework (QuadratureTraining without adaptive loss callbacks).
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (default sigmoid): 0.0
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (logistic sigmoid): 0.037600000000000744

## Hypothesis Validation

- Largest RoA area (default sigmoid): `quadratic_over_rho` with area `3.0484000000000004`
- Largest RoA area (logistic sigmoid): `inv_dist` with area `3.1296`
- Runs with decrease-condition violation (max dV/dt inside V <= ρ > 0): control_zero/default, control_zero/logistic, constant_one/default, constant_one/logistic, inv_dist_sq/default, inv_dist_sq/logistic, scaled_inv_dist_sq/default, scaled_inv_dist_sq/logistic, inv_dist/default, inv_dist/logistic, inv_V_small/default, quadratic_over_rho/default, quadratic_over_rho/logistic
- Next architecture modification if all plateau: increase `MLP` width/depth and test `MultiplicativeLyapunovNet` with same protocol.

## Training Time Comparison

| Penalty | Sigmoid | Training Time (s) |
|---------|---------|-------------------|
| control_zero | default | 55.58 |
| control_zero | logistic | 5.79 |
| constant_one | default | 3.83 |
| constant_one | logistic | 4.22 |
| inv_dist_sq | default | 2.08 |
| inv_dist_sq | logistic | 5.44 |
| scaled_inv_dist_sq | default | 1.52 |
| scaled_inv_dist_sq | logistic | 1.95 |
| inv_dist | default | 2.41 |
| inv_dist | logistic | 5.61 |
| inv_V_small | default | 1.46 |
| quadratic_over_rho | default | 2.49 |
| quadratic_over_rho | logistic | 2.75 |

- **Fastest:** `inv_V_small` / `default` at 1.46s
- **Slowest:** `control_zero` / `default` at 55.58s
- **Mean:** 7.32s across 13 runs
