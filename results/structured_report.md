# Structured Report

- Timestamp: 2026-02-26 22:48:42
- NeuralLyapunov path: `C:\Users\AD\.julia\packages\NeuralLyapunov\oKiqr\src\NeuralLyapunov.jl`
- Default sigmoid in source: `(x) -> x .≥ zero.(x)` (hard step)
- Logistic sigmoid used in experiments: `σ(z)=1/(1+exp(-k*z))`, with `k=20`
- Penalty term wiring in source: `[sigmoid(ρ-V)*in_RoA_penalty, sigmoid(V-ρ)*out_of_RoA_penalty]`
- Loss aggregation: NeuralPDE residual loss over PDE equations (each residual vs `0.0`).
- Gradient through default sigmoid: non-smooth/boolean gate; logistic variant provides smooth gate.

## Per-Run Results

- penalty: `control_zero` | sigmoid: `default` | final_loss: `5.934729841099874e-67` | ρ: `1.0` | area: `2.3368` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`
- penalty: `control_zero` | sigmoid: `logistic` | final_loss: `1.0804917134453544e-6` | ρ: `1.0` | area: `1.8988` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`
- penalty: `constant_one` | sigmoid: `default` | final_loss: `0.84375` | ρ: `1.0` | area: `2.3368` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`
- penalty: `constant_one` | sigmoid: `logistic` | final_loss: `0.7997294614670528` | ρ: `1.0` | area: `3.0644` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`
- penalty: `inv_dist_sq` | sigmoid: `default` | final_loss: `0.26274792950832` | ρ: `1.0` | area: `2.3368` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`
- penalty: `inv_dist_sq` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `NaN` | has_nan: `true` | error: `AssertionError: B > A`
- penalty: `scaled_inv_dist_sq` | sigmoid: `default` | final_loss: `2071.9350230876857` | ρ: `1.0` | area: `2.3368` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`
- penalty: `scaled_inv_dist_sq` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `NaN` | has_nan: `true` | error: `AssertionError: slopes[ib] >= zeroT`
- penalty: `inv_dist` | sigmoid: `default` | final_loss: `0.38669887437855266` | ρ: `1.0` | area: `2.3368` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`
- penalty: `inv_dist` | sigmoid: `logistic` | final_loss: `0.27599991667382623` | ρ: `1.0` | area: `3.1252` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`
- penalty: `inv_V` | sigmoid: `default` | final_loss: `1.1442877638580552e-86` | ρ: `1.0` | area: `0.1668` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`
- penalty: `inv_V` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `NaN` | has_nan: `true` | error: `AssertionError: isfinite(phi_d) && isfinite(gphi)`
- penalty: `quadratic_over_rho` | sigmoid: `default` | final_loss: `99.46261909070823` | ρ: `1.0` | area: `3.0348` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`
- penalty: `quadratic_over_rho` | sigmoid: `logistic` | final_loss: `85.76407764540255` | ρ: `1.0` | area: `3.1008` | leak_outside_true_RoA: `false` | max_dVdt_inside_V_le_ρ: `0.0` | has_nan: `false` | error: `none`

## Adaptive Reweighting Check

- No adaptive reweighting configured in this framework (QuadratureTraining without adaptive loss callbacks).
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (default sigmoid): 0.0
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (logistic sigmoid): NaN

## Hypothesis Validation

- Largest RoA area (default sigmoid): `quadratic_over_rho` with area `3.0348`
- Largest RoA area (logistic sigmoid): `inv_dist` with area `3.1252`
- No run had positive max dV/dt inside V <= ρ on the evaluation grid.
- Next architecture modification if all plateau: increase `MLP` width/depth and test `MultiplicativeLyapunovNet` with same protocol.
