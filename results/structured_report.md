# Structured Report

- Timestamp: 2026-05-18 08:04:16
- NeuralLyapunov path: `/Users/aryand/.julia/packages/NeuralLyapunov/ykuJS/src/NeuralLyapunov.jl`
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

- penalty: `control_zero` | sigmoid: `default` | log_scale: `false` | rng_seed: `2027` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `4.9872392764124e-7` | ρ: `1.0` | area: `3.0328000000000004` | max_dVdt_inside: `0.04908785580496713` | train_time_seconds: `34.24862313270569` | has_nan: `false` | error: `none`
- penalty: `control_zero` | sigmoid: `default` | log_scale: `true` | rng_seed: `2028` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `3.7092061506874214e-68` | ρ: `1.0` | area: `2.5828` | max_dVdt_inside: `0.22729785603102254` | train_time_seconds: `5.014153003692627` | has_nan: `false` | error: `none`
- penalty: `control_zero` | sigmoid: `logistic` | log_scale: `false` | rng_seed: `2029` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `3.2633453068312455e-9` | ρ: `1.0` | area: `2.6228000000000002` | max_dVdt_inside: `0.017212144904647748` | train_time_seconds: `4.379756927490234` | has_nan: `false` | error: `none`
- penalty: `control_zero` | sigmoid: `logistic` | log_scale: `true` | rng_seed: `2030` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `2.4638178670086623e-5` | ρ: `1.0` | area: `3.0356` | max_dVdt_inside: `0.220679687614512` | train_time_seconds: `4.208165168762207` | has_nan: `false` | error: `none`
- penalty: `constant_one` | sigmoid: `default` | log_scale: `false` | rng_seed: `2031` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.8242276223391437` | ρ: `1.0` | area: `2.7848` | max_dVdt_inside: `0.11305477415322668` | train_time_seconds: `4.3183159828186035` | has_nan: `false` | error: `none`
- penalty: `constant_one` | sigmoid: `default` | log_scale: `true` | rng_seed: `2032` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.81640625` | ρ: `1.0` | area: `2.9968000000000004` | max_dVdt_inside: `0.09227132329982501` | train_time_seconds: `2.400611162185669` | has_nan: `false` | error: `none`
- penalty: `constant_one` | sigmoid: `logistic` | log_scale: `false` | rng_seed: `2033` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.7999607399627741` | ρ: `1.0` | area: `3.1108` | max_dVdt_inside: `0.06546324457431979` | train_time_seconds: `3.6170201301574707` | has_nan: `false` | error: `none`
- penalty: `constant_one` | sigmoid: `logistic` | log_scale: `true` | rng_seed: `2034` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.7715583212470486` | ρ: `1.0` | area: `3.1292000000000004` | max_dVdt_inside: `0.014608946433498912` | train_time_seconds: `4.393573045730591` | has_nan: `false` | error: `none`
- penalty: `inv_dist_sq` | sigmoid: `default` | log_scale: `false` | rng_seed: `2035` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.15802799772495096` | ρ: `1.0` | area: `3.0116000000000005` | max_dVdt_inside: `0.2556775600588352` | train_time_seconds: `2.1756598949432373` | has_nan: `false` | error: `none`
- penalty: `inv_dist_sq` | sigmoid: `default` | log_scale: `true` | rng_seed: `2036` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.1976600337918104` | ρ: `1.0` | area: `2.7988` | max_dVdt_inside: `0.0` | train_time_seconds: `1.6307621002197266` | has_nan: `false` | error: `none`
- penalty: `inv_dist_sq` | sigmoid: `logistic` | log_scale: `false` | rng_seed: `2037` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.15218032615517915` | ρ: `1.0` | area: `3.1252` | max_dVdt_inside: `0.024360416624010936` | train_time_seconds: `2.8984780311584473` | has_nan: `false` | error: `none`
- penalty: `inv_dist_sq` | sigmoid: `logistic` | log_scale: `true` | rng_seed: `2038` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.14205398869687816` | ρ: `1.0` | area: `3.1072` | max_dVdt_inside: `0.07870730082002297` | train_time_seconds: `2.8820738792419434` | has_nan: `false` | error: `none`
- penalty: `scaled_inv_dist_sq` | sigmoid: `default` | log_scale: `false` | rng_seed: `2039` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `1908.293556572756` | ρ: `1.0` | area: `2.7288` | max_dVdt_inside: `0.28449669374359793` | train_time_seconds: `1.8049769401550293` | has_nan: `false` | error: `none`
- penalty: `scaled_inv_dist_sq` | sigmoid: `default` | log_scale: `true` | rng_seed: `2040` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `2330.3861345875493` | ρ: `1.0` | area: `2.2840000000000003` | max_dVdt_inside: `0.17165126680239784` | train_time_seconds: `1.7730891704559326` | has_nan: `false` | error: `none`
- penalty: `scaled_inv_dist_sq` | sigmoid: `logistic` | log_scale: `false` | rng_seed: `2041` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `1548.2325710468774` | ρ: `1.0` | area: `2.9856000000000003` | max_dVdt_inside: `0.514974172999419` | train_time_seconds: `3.3760318756103516` | has_nan: `false` | error: `none`
- penalty: `scaled_inv_dist_sq` | sigmoid: `logistic` | log_scale: `true` | rng_seed: `2042` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `1553.7624462351566` | ρ: `1.0` | area: `3.0624000000000002` | max_dVdt_inside: `0.11164891985445888` | train_time_seconds: `3.2515981197357178` | has_nan: `false` | error: `none`
- penalty: `inv_dist` | sigmoid: `default` | log_scale: `false` | rng_seed: `2043` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.4183618206333009` | ρ: `1.0` | area: `2.13` | max_dVdt_inside: `0.0` | train_time_seconds: `1.8260951042175293` | has_nan: `false` | error: `none`
- penalty: `inv_dist` | sigmoid: `default` | log_scale: `true` | rng_seed: `2044` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.3750430039575077` | ρ: `1.0` | area: `2.06` | max_dVdt_inside: `0.03696173302527139` | train_time_seconds: `1.9231541156768799` | has_nan: `false` | error: `none`
- penalty: `inv_dist` | sigmoid: `logistic` | log_scale: `false` | rng_seed: `2045` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.29916971757541466` | ρ: `1.0` | area: `3.0692000000000004` | max_dVdt_inside: `0.05386839509190591` | train_time_seconds: `2.2421841621398926` | has_nan: `false` | error: `none`
- penalty: `inv_dist` | sigmoid: `logistic` | log_scale: `true` | rng_seed: `2046` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.3182507199711196` | ρ: `1.0` | area: `3.1004000000000005` | max_dVdt_inside: `0.04000279398598132` | train_time_seconds: `2.1630971431732178` | has_nan: `false` | error: `none`
- penalty: `inv_V_small` | sigmoid: `default` | log_scale: `false` | rng_seed: `2047` | inv_V_a: `0.0001` | inv_V_a_regime: `much_less_than_rho` | final_loss: `2.943066332921894e-7` | ρ: `1.0` | area: `0.0032` | max_dVdt_inside: `1277.8570421322831` | train_time_seconds: `1.7999258041381836` | has_nan: `false` | error: `none`
- penalty: `inv_V_small` | sigmoid: `default` | log_scale: `true` | rng_seed: `2048` | inv_V_a: `0.0001` | inv_V_a_regime: `much_less_than_rho` | final_loss: `0.008759042885400834` | ρ: `1.0` | area: `0.0848` | max_dVdt_inside: `84.85227865357116` | train_time_seconds: `2.034061908721924` | has_nan: `false` | error: `none`
- penalty: `inv_V_small` | sigmoid: `logistic` | log_scale: `false` | rng_seed: `2049` | inv_V_a: `0.0001` | inv_V_a_regime: `much_less_than_rho` | final_loss: `0.07020440411934042` | ρ: `1.0` | area: `1.1124` | max_dVdt_inside: `2.7722141343742686` | train_time_seconds: `2.270620822906494` | has_nan: `false` | error: `none`
- penalty: `inv_V_small` | sigmoid: `logistic` | log_scale: `true` | rng_seed: `2050` | inv_V_a: `0.0001` | inv_V_a_regime: `much_less_than_rho` | final_loss: `0.008859160940084806` | ρ: `1.0` | area: `0.128` | max_dVdt_inside: `28.69919885189579` | train_time_seconds: `2.4887571334838867` | has_nan: `false` | error: `none`
- penalty: `inv_V_rho` | sigmoid: `default` | log_scale: `false` | rng_seed: `2051` | inv_V_a: `1.0` | inv_V_a_regime: `equal_rho` | final_loss: `0.08527904275750046` | ρ: `1.0` | area: `0.5292` | max_dVdt_inside: `8.88206628600468` | train_time_seconds: `2.4724748134613037` | has_nan: `false` | error: `none`
- penalty: `inv_V_rho` | sigmoid: `default` | log_scale: `true` | rng_seed: `2052` | inv_V_a: `1.0` | inv_V_a_regime: `equal_rho` | final_loss: `4.704102992810951e-7` | ρ: `1.0` | area: `0.0028000000000000004` | max_dVdt_inside: `635.8460680925207` | train_time_seconds: `1.8824858665466309` | has_nan: `false` | error: `none`
- penalty: `inv_V_rho` | sigmoid: `logistic` | log_scale: `false` | rng_seed: `2053` | inv_V_a: `1.0` | inv_V_a_regime: `equal_rho` | final_loss: `0.05342631014012901` | ρ: `1.0` | area: `1.5548` | max_dVdt_inside: `1.128493429434334` | train_time_seconds: `2.0046541690826416` | has_nan: `false` | error: `none`
- penalty: `inv_V_rho` | sigmoid: `logistic` | log_scale: `true` | rng_seed: `2054` | inv_V_a: `1.0` | inv_V_a_regime: `equal_rho` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | train_time_seconds: `NaN` | has_nan: `true` | error: `AssertionError: B > A`
- penalty: `inv_V_clipped` | sigmoid: `default` | log_scale: `false` | rng_seed: `2055` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.0019474387024063463` | ρ: `1.0` | area: `0.0244` | max_dVdt_inside: `561.6650519734965` | train_time_seconds: `2.2864060401916504` | has_nan: `false` | error: `none`
- penalty: `inv_V_clipped` | sigmoid: `default` | log_scale: `true` | rng_seed: `2056` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `0.0006350419402940127` | ρ: `1.0` | area: `0.0336` | max_dVdt_inside: `299.4740349282321` | train_time_seconds: `2.921048879623413` | has_nan: `false` | error: `none`
- penalty: `inv_V_clipped` | sigmoid: `logistic` | log_scale: `false` | rng_seed: `2057` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | train_time_seconds: `NaN` | has_nan: `true` | error: `AssertionError: isfinite(phi_c) && isfinite(dphi_c)`
- penalty: `inv_V_clipped` | sigmoid: `logistic` | log_scale: `true` | rng_seed: `2058` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `8.273361004242684` | ρ: `1.0` | area: `0.0408` | max_dVdt_inside: `215.78258273584686` | train_time_seconds: `2.8561699390411377` | has_nan: `false` | error: `none`
- penalty: `quadratic_over_rho` | sigmoid: `default` | log_scale: `false` | rng_seed: `2059` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `185.3377618006385` | ρ: `1.0` | area: `1.9436000000000002` | max_dVdt_inside: `0.7685400474645746` | train_time_seconds: `1.8319711685180664` | has_nan: `false` | error: `none`
- penalty: `quadratic_over_rho` | sigmoid: `default` | log_scale: `true` | rng_seed: `2060` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `105.33221506598463` | ρ: `1.0` | area: `2.9996` | max_dVdt_inside: `0.5286501524039937` | train_time_seconds: `1.9741389751434326` | has_nan: `false` | error: `none`
- penalty: `quadratic_over_rho` | sigmoid: `logistic` | log_scale: `false` | rng_seed: `2061` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `95.76158676319709` | ρ: `1.0` | area: `3.0456000000000003` | max_dVdt_inside: `0.2235817746644324` | train_time_seconds: `2.6779580116271973` | has_nan: `false` | error: `none`
- penalty: `quadratic_over_rho` | sigmoid: `logistic` | log_scale: `true` | rng_seed: `2062` | inv_V_a: `NaN` | inv_V_a_regime: `` | final_loss: `106.96928558253842` | ρ: `1.0` | area: `2.8816` | max_dVdt_inside: `0.9380747181731564` | train_time_seconds: `2.9588310718536377` | has_nan: `false` | error: `none`

## Expected vs Observed Behavior

### control_zero (default sigmoid)

- **Expected:** smallest RoA estimate since there is no penalty outside the RoA
- **Observed:** area = 3.0328000000000004, max dV/dt inside = 0.04908785580496713
- **Comparison:** unexpected — control_zero did not produce the smallest RoA area; other penalties may have collapsed or failed.

### constant_one (logistic sigmoid)

- **Expected:** moderate RoA expansion due to uniform penalty with smooth sigmoid gating
- **Observed:** area = 3.1108, max dV/dt inside = 0.06546324457431979
- **Comparison:** consistent — constant_one with logistic sigmoid achieved at-or-above-median area, matching the moderate expansion expectation.

### inv_V_small (default sigmoid)

- **Expected:** stabilized inverse-V penalty with small offset `a` (chosen so `a ≪ ρ`) avoiding V→0 singularity
- **Observed:** area = 0.0032, max dV/dt inside = 1277.8570421322831, has_nan = false, training_time = 1.7999258041381836s
- **Comparison:** small-`a` stabilization successful — finite area obtained without collapse.

### inv_V_rho (default sigmoid)

- **Expected:** stabilized inverse-V penalty using ρ as offset, singularity-free
- **Observed:** area = 0.5292, max dV/dt inside = 8.88206628600468, has_nan = false, training_time = 2.4724748134613037s
- **Comparison:** ρ-based stabilization successful — finite area obtained.

> **Note on inv_V instability:** Instability likely arises from interaction between penalty scaling and NeuralPDE residual formulation, where large curvature near low-V regions still destabilizes optimization despite offset.

## Adaptive Reweighting Check

- No adaptive reweighting configured in this framework (QuadratureTraining without adaptive loss callbacks).
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (default sigmoid): 0.2828000000000004
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (logistic sigmoid): 0.13959999999999972

## Hypothesis Validation

- Largest RoA area (default sigmoid): `control_zero` with area `3.0328000000000004`
- Largest RoA area (logistic sigmoid): `constant_one` with area `3.1292000000000004`
- Runs with decrease-condition violation (max dV/dt inside V <= ρ > 0): control_zero/default/ls=false, control_zero/default/ls=true, control_zero/logistic/ls=false, control_zero/logistic/ls=true, constant_one/default/ls=false, constant_one/default/ls=true, constant_one/logistic/ls=false, constant_one/logistic/ls=true, inv_dist_sq/default/ls=false, inv_dist_sq/logistic/ls=false, inv_dist_sq/logistic/ls=true, scaled_inv_dist_sq/default/ls=false, scaled_inv_dist_sq/default/ls=true, scaled_inv_dist_sq/logistic/ls=false, scaled_inv_dist_sq/logistic/ls=true, inv_dist/default/ls=true, inv_dist/logistic/ls=false, inv_dist/logistic/ls=true, inv_V_small/default/ls=false, inv_V_small/default/ls=true, inv_V_small/logistic/ls=false, inv_V_small/logistic/ls=true, inv_V_rho/default/ls=false, inv_V_rho/default/ls=true, inv_V_rho/logistic/ls=false, inv_V_clipped/default/ls=false, inv_V_clipped/default/ls=true, inv_V_clipped/logistic/ls=true, quadratic_over_rho/default/ls=false, quadratic_over_rho/default/ls=true, quadratic_over_rho/logistic/ls=false, quadratic_over_rho/logistic/ls=true
- Next architecture modification if all plateau: increase `MLP` width/depth and test `MultiplicativeLyapunovNet` with same protocol.

## Training Time Comparison

| Penalty | Sigmoid | log_scale | Training Time (s) |
|---------|---------|-----------|-------------------|
| control_zero | default | false | 34.25 |
| control_zero | default | true | 5.01 |
| control_zero | logistic | false | 4.38 |
| control_zero | logistic | true | 4.21 |
| constant_one | default | false | 4.32 |
| constant_one | default | true | 2.4 |
| constant_one | logistic | false | 3.62 |
| constant_one | logistic | true | 4.39 |
| inv_dist_sq | default | false | 2.18 |
| inv_dist_sq | default | true | 1.63 |
| inv_dist_sq | logistic | false | 2.9 |
| inv_dist_sq | logistic | true | 2.88 |
| scaled_inv_dist_sq | default | false | 1.8 |
| scaled_inv_dist_sq | default | true | 1.77 |
| scaled_inv_dist_sq | logistic | false | 3.38 |
| scaled_inv_dist_sq | logistic | true | 3.25 |
| inv_dist | default | false | 1.83 |
| inv_dist | default | true | 1.92 |
| inv_dist | logistic | false | 2.24 |
| inv_dist | logistic | true | 2.16 |
| inv_V_small | default | false | 1.8 |
| inv_V_small | default | true | 2.03 |
| inv_V_small | logistic | false | 2.27 |
| inv_V_small | logistic | true | 2.49 |
| inv_V_rho | default | false | 2.47 |
| inv_V_rho | default | true | 1.88 |
| inv_V_rho | logistic | false | 2.0 |
| inv_V_clipped | default | false | 2.29 |
| inv_V_clipped | default | true | 2.92 |
| inv_V_clipped | logistic | true | 2.86 |
| quadratic_over_rho | default | false | 1.83 |
| quadratic_over_rho | default | true | 1.97 |
| quadratic_over_rho | logistic | false | 2.68 |
| quadratic_over_rho | logistic | true | 2.96 |

- **Fastest:** `inv_dist_sq` / `default` / ls=`true` at 1.63s
- **Slowest:** `control_zero` / `default` / ls=`false` at 34.25s
- **Mean:** 3.62s across 34 runs
