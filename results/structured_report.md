# Structured Report

- Timestamp: 2026-03-10 14:02:35
- NeuralLyapunov path: `C:\Users\AD\.julia\packages\NeuralLyapunov\oKiqr\src\NeuralLyapunov.jl`
- Default sigmoid in source: `(x) -> x .≥ zero.(x)` (hard step)
- Logistic sigmoid used in experiments: `σ(z)=1/(1+exp(-k*z))`, with `k=20`
- Penalty term wiring in source: `[sigmoid(ρ-V)*in_RoA_penalty, sigmoid(V-ρ)*out_of_RoA_penalty]`
- Loss aggregation: NeuralPDE residual loss over PDE equations (each residual vs `0.0`).
- Gradient through default sigmoid: non-smooth/boolean gate; logistic variant provides smooth gate.

## Per-Run Results

- penalty: `control_zero` | sigmoid: `default` | final_loss: `0.0` | ρ: `1.0` | area: `2.9064` | max_dVdt_inside: `0.003130671393868094` | has_nan: `false` | error: `none`
- penalty: `control_zero` | sigmoid: `logistic` | final_loss: `1.5216549746737513e-11` | ρ: `1.0` | area: `2.9164` | max_dVdt_inside: `0.0007391496307473034` | has_nan: `false` | error: `none`
- penalty: `constant_one` | sigmoid: `default` | final_loss: `0.8164062671232701` | ρ: `1.0` | area: `2.9004000000000003` | max_dVdt_inside: `0.042320306416454394` | has_nan: `false` | error: `none`
- penalty: `constant_one` | sigmoid: `logistic` | final_loss: `0.8084920458825822` | ρ: `1.0` | area: `3.1276` | max_dVdt_inside: `0.015466163134244088` | has_nan: `false` | error: `none`
- penalty: `inv_dist_sq` | sigmoid: `default` | final_loss: `0.15123321377798002` | ρ: `1.0` | area: `2.9175999999999997` | max_dVdt_inside: `0.0032593504696101176` | has_nan: `false` | error: `none`
- penalty: `inv_dist_sq` | sigmoid: `logistic` | final_loss: `0.15430589237610345` | ρ: `1.0` | area: `3.0840000000000005` | max_dVdt_inside: `0.17844331466696872` | has_nan: `false` | error: `none`
- penalty: `scaled_inv_dist_sq` | sigmoid: `default` | final_loss: `1748.4465871427296` | ρ: `1.0` | area: `2.9495999999999998` | max_dVdt_inside: `0.05415694760656234` | has_nan: `false` | error: `none`
- penalty: `scaled_inv_dist_sq` | sigmoid: `logistic` | final_loss: `1697.4048477169388` | ρ: `1.0` | area: `3.0940000000000003` | max_dVdt_inside: `0.05423647247687331` | has_nan: `false` | error: `none`
- penalty: `inv_dist` | sigmoid: `default` | final_loss: `0.3139007281136562` | ρ: `1.0` | area: `2.9072000000000005` | max_dVdt_inside: `0.02041326841571523` | has_nan: `false` | error: `none`
- penalty: `inv_dist` | sigmoid: `logistic` | final_loss: `0.30941833431062626` | ρ: `1.0` | area: `3.1068000000000002` | max_dVdt_inside: `0.01874912468803494` | has_nan: `false` | error: `none`
- penalty: `inv_V` | sigmoid: `default` | final_loss: `1.1226784589775953e-5` | ρ: `1.0` | area: `0.0016` | max_dVdt_inside: `52.32431285778799` | has_nan: `false` | error: `none`
- penalty: `inv_V` | sigmoid: `logistic` | final_loss: `0.15060974536345745` | ρ: `1.0` | area: `1.3516` | max_dVdt_inside: `2.779808936432459` | has_nan: `false` | error: `none`
- penalty: `quadratic_over_rho` | sigmoid: `default` | final_loss: `105.90212524702662` | ρ: `1.0` | area: `3.0356` | max_dVdt_inside: `0.23240711672411768` | has_nan: `false` | error: `none`
- penalty: `quadratic_over_rho` | sigmoid: `logistic` | final_loss: `96.9347660333508` | ρ: `1.0` | area: `3.052` | max_dVdt_inside: `0.12404547271298605` | has_nan: `false` | error: `none`

## Adaptive Reweighting Check

- No adaptive reweighting configured in this framework (QuadratureTraining without adaptive loss callbacks).
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (default sigmoid): 0.03200000000000003
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (logistic sigmoid): 0.009999999999999787

## Hypothesis Validation

- Largest RoA area (default sigmoid): `quadratic_over_rho` with area `3.0356`
- Largest RoA area (logistic sigmoid): `constant_one` with area `3.1276`
- Runs with decrease-condition violation (max dV/dt inside V <= ρ > 0): control_zero/default, control_zero/logistic, constant_one/default, constant_one/logistic, inv_dist_sq/default, inv_dist_sq/logistic, scaled_inv_dist_sq/default, scaled_inv_dist_sq/logistic, inv_dist/default, inv_dist/logistic, inv_V/default, inv_V/logistic, quadratic_over_rho/default, quadratic_over_rho/logistic
- Next architecture modification if all plateau: increase `MLP` width/depth and test `MultiplicativeLyapunovNet` with same protocol.
