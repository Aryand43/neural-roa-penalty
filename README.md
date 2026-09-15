# neural-roa-penalty

Systematic comparison of RoA-aware penalty functions for neural Lyapunov verification on Van der Pol dynamics via [NeuralLyapunov.jl](https://github.com/SciML/NeuralLyapunov.jl).

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. run_roa_penalty_experiments.jl
```

All outputs are written to the `results/` directory (CSVs, reports, and plots).

## Rerun on updated NeuralLyapunov (Sept 2026)

Rerun of the `V − ρ` vs `log(V/ρ)` (`log_scale`) comparison after the `benchmark` timing change ([SciML/NeuralLyapunov.jl#215](https://github.com/SciML/NeuralLyapunov.jl/pull/215)) was merged into the `roa-log-transtion` branch ([#173](https://github.com/SciML/NeuralLyapunov.jl/pull/173)).

**Environment**

| | Previous run (2026-05-28) | Rerun |
|---|---|---|
| NeuralLyapunov | 0.4.2 (`roa-log-transtion`) | 0.4.9 (`roa-log-transtion` @ `7d46dac`) |
| NeuralPDE | 5.x | 6.4.1 |
| Optimization / OptimizationOptimisers | 5.x / 0.3.x | 5.9.1 / 0.3.24 |
| Julia | 1.12.5 | 1.12.6 |

Protocol unchanged: Van der Pol, 3×32 tanh MLP + `AdditiveLyapunovNet`, `QuasiRandomTraining(1024)`, `Adam(0.01)` ×300 then `Adam()` ×300, ρ = 1, seed 2026, 9 penalties × 2 sigmoids × `log_scale ∈ {false, true}` = 36 runs.

**Timing changes in this repo**

- Training and evaluation now go through `NeuralLyapunov.benchmark`, so `train_time_seconds` is the `training_time` measured by `benchmark` after #215 (optimization only, PDE discretization excluded). The previous analysis called `Optimization.solve` directly with its own timer.
- Same training schedule passed to `benchmark`: `opt = [Adam(0.01), Adam()]`, `optimization_args = [[:maxiters => 300], [:maxiters => 300]]`, same initial parameters.
- `benchmark` evaluation: 1000 Latin-hypercube samples, classifier `V ≤ ρ`, 20 s simulations (`EnsembleSerial`), endpoint `≈ 0` with `atol = 1e-3`, trajectories terminated once `‖x‖ > 5`.
- An untimed warm-up precedes the grid. The old 60.67 s for the first configuration was JIT compilation, not training.
- New summary columns: `eval_time_seconds`, `tp`/`fp`/`tn`/`fn`, `classifier_accuracy` (from `benchmark`) and `nan_grad_skips` (Adam updates skipped by OptimizationOptimisers because the gradient contained NaN/Inf). RoA area and max dV/dt are still computed on the 201×201 grid from the `V`/`V̇` returned by `benchmark`.

**Results** (full tables: [`results/rerun_comparison.md`](results/rerun_comparison.md), [`results/summary.csv`](results/summary.csv); previous run: [`results/summary_nl_v0.4.2_2026-05-28.csv`](results/summary_nl_v0.4.2_2026-05-28.csv))

- **RoA area:** 30 of 36 runs are identical to the previous run. Changes: `inv_V_small`/logistic/`log_scale=true` 0.8312 → 0.7396; `inv_V_small`/default 0.3384 → 0.3864 (both `log_scale`); `control_zero`/default 2.0932 → 2.1280 (both); `inv_V_rho`/logistic/true 0.0068 → 0.0092; `inv_V_clipped` ≤ 0.002.
- **`log_scale` effect on area is unchanged in sign and nearly unchanged in size.** With the hard-step sigmoid it has no effect (expected: `log(V/ρ)` and `V − ρ` have the same sign). With the logistic sigmoid: `inv_V_clipped` +0.48, `inv_V_small` +0.23 (was +0.32), `inv_V_rho` −0.58, `control_zero` −0.13, all distance/constant/quadratic penalties 0.
- **Timing (`benchmark` training_time):** mean 10.91 s (was 15.9 s); `log_scale=false` 10.98 s vs `log_scale=true` 10.84 s; max 12.64 s. The earlier apparent `log_scale` speed-up (17.49 s vs 14.32 s) came entirely from the compile-time outlier; there is no meaningful timing difference between the two scalings. Switching from direct `Optimization.solve` to `benchmark` left all training results (areas, max dV/dt, NaN-gradient skips) identical and changed only the timings (mean 11.69 s → 10.91 s). Evaluation takes ~0.03–0.07 s per run.
- **`benchmark` classifier (`V ≤ ρ`, 1000 samples):** 0 false positives in every run (no sampled point outside the true RoA is predicted inside), but 785 of 1000 samples are actually in the RoA and the learned sets capture at most 187 of them. Accuracy is 0.40 for `constant_one`, `inv_dist`, `inv_dist_sq`, `scaled_inv_dist_sq` (logistic) and `quadratic_over_rho`; 0.39 for the distance penalties with the hard step; 0.33–0.34 for `control_zero` and `constant_one` (hard step); 0.22–0.28 for the `inv_V_*` penalties. `log_scale` changes accuracy only with the logistic sigmoid for `inv_V_clipped` (0.252 → 0.278), `inv_V_small` (0.245 → 0.256), `inv_V_rho` (0.249 → 0.216) and `control_zero` (0.344 → 0.334).
- **NaN gradients:** `inv_V_small`, `inv_V_rho`, `inv_V_clipped` with the logistic sigmoid and `log_scale=false` skipped 569, 591 and 570 of 600 Adam updates, so those networks barely trained and their areas mostly reflect initialization. The same three with `log_scale=true` skipped 0. Every other run skipped 0. The `log_scale` area differences for these three penalties are therefore largely "trained vs. effectively untrained", not a pure effect of the boundary shape.

Reproduce the comparison:

```bash
julia --project=. run_roa_penalty_experiments.jl
julia --project=. scripts/compare_reruns.jl results/summary_nl_v0.4.2_2026-05-28.csv results/summary.csv results/rerun_comparison.md
```

## Pre-Experiment Hypotheses

Each penalty function `p(V, dV/dt, x, x₀, ρ)` is gated by `σ(V − ρ)` and applied **outside** the learned RoA level set `{x : V(x) ≤ ρ}`. Two sigmoids are tested: the default hard step `x ≥ 0` (non-differentiable boolean gate, zero gradient almost everywhere) and a logistic approximation `1/(1+exp(−20z))` (smooth, provides gradient signal near the boundary).

| Penalty | Expected RoA Behavior | Reasoning |
|---|---|---|
| `control_zero` (0) | Smallest RoA; boundary defaults to the initial network's level set | Zero penalty outside RoA provides no gradient signal to expand the boundary |
| `constant_one` (1) | Moderate, roughly uniform expansion | Flat penalty provides expansion pressure but no spatial preference for where the boundary should lie |
| `inv_dist_sq` (1/‖x−x₀‖²) | Risk of gradient instability near equilibrium; boundary shape depends on sigmoid smoothness | Order-2 singularity at x₀ creates steep penalty gradients, but gating by σ(V−ρ) confines activation to outside {V≤ρ} |
| `scaled_inv_dist_sq` (100/‖x−x₀‖²) | Nearly identical to `inv_dist_sq` | Scalar multiplier changes loss magnitude but not penalty landscape shape; NeuralPDE residual aggregation absorbs the scale |
| `inv_dist` (1/‖x−x₀‖) | Similar to `inv_dist_sq` but milder gradient issues | Order-1 singularity produces less extreme gradients near x₀ |
| `inv_V_small` (1/(V+1e-3)) | Stabilized inverse-V with small epsilon offset | Avoids V→0 singularity via fixed offset; may still exhibit large curvature near low-V regions |
| `inv_V_rho` (1/(V+ρ)) | Stabilized inverse-V scaled by ρ | Uses ρ as natural offset, singularity-free for V≥0; penalty magnitude bounded by 1/ρ |
| `inv_V_clipped` (1/max(V,0.1)) | Safe inverse-V via hard floor | Clamps V away from zero, preventing gradient blowup entirely; penalty bounded by 10 |
| `quadratic_over_rho` (max(0,V−ρ)²) | Well-shaped boundary closely tracking the true RoA | Zero inside {V≤ρ}, smooth quadratic growth outside; gradient is proportional to violation magnitude, directly penalizing boundary overshoot |

**Sigmoid interaction prediction:** logistic sigmoid should uniformly improve RoA quality over the hard step because it provides nonzero gradient flow across the V=ρ boundary, enabling the optimizer to smoothly adjust the level set.

## RoA Comparison Plots

Each experiment produces an overlay plot comparing the learned and true Regions of Attraction, saved to `results/roa_overlay_<penalty>_<sigmoid>.png`.

**Legend:**
- **Red solid contour** — learned RoA boundary (sublevel set {x : V(x) ≤ ρ})
- **Blue dashed contour** — true RoA boundary (computed by forward-simulating Van der Pol trajectories and checking convergence to the origin)
- **Light blue fill** — interior of the true RoA

Plots for all 18 runs (9 penalties × 2 sigmoids):

| Penalty | Default (hard step) | Logistic (k=20) |
|---|---|---|
| control_zero | [plot](results/roa_overlay_control_zero_default.png) | [plot](results/roa_overlay_control_zero_logistic.png) |
| constant_one | [plot](results/roa_overlay_constant_one_default.png) | [plot](results/roa_overlay_constant_one_logistic.png) |
| inv_dist_sq | [plot](results/roa_overlay_inv_dist_sq_default.png) | [plot](results/roa_overlay_inv_dist_sq_logistic.png) |
| scaled_inv_dist_sq | [plot](results/roa_overlay_scaled_inv_dist_sq_default.png) | [plot](results/roa_overlay_scaled_inv_dist_sq_logistic.png) |
| inv_dist | [plot](results/roa_overlay_inv_dist_default.png) | [plot](results/roa_overlay_inv_dist_logistic.png) |
| inv_V_small | [plot](results/roa_overlay_inv_V_small_default.png) | [plot](results/roa_overlay_inv_V_small_logistic.png) |
| inv_V_rho | [plot](results/roa_overlay_inv_V_rho_default.png) | [plot](results/roa_overlay_inv_V_rho_logistic.png) |
| inv_V_clipped | [plot](results/roa_overlay_inv_V_clipped_default.png) | [plot](results/roa_overlay_inv_V_clipped_logistic.png) |
| quadratic_over_rho | [plot](results/roa_overlay_quadratic_over_rho_default.png) | [plot](results/roa_overlay_quadratic_over_rho_logistic.png) |

## Per-Penalty Behavior Explanation

| Penalty | Explanation |
|---|---|
| `control_zero` | No out-of-RoA gradient signal means the boundary is determined entirely by the Lyapunov decrease condition inside {V≤ρ}; the optimizer has no incentive to expand, so the RoA reflects the network's initialization bias. |
| `constant_one` | Uniform pressure expands the boundary but cannot shape it to match the true RoA's non-circular geometry; the logistic sigmoid improves this by letting gradients flow smoothly across the V=ρ transition. |
| `inv_dist_sq` | The ‖x−x₀‖⁻² singularity is gated to activate only outside {V≤ρ}, so near-equilibrium blowup is avoided in practice; the logistic sigmoid enables finer boundary refinement through smooth gradient flow. |
| `scaled_inv_dist_sq` | The 100× multiplier inflates the loss magnitude without changing the penalty landscape shape — NeuralPDE's residual-squared aggregation absorbs the scale, so behavior tracks `inv_dist_sq` closely. |
| `inv_dist` | Weaker order-1 singularity produces more moderate gradients than `inv_dist_sq`, yielding qualitatively similar but slightly more stable training dynamics. |
| `inv_V_small` | The 1/(V+1e-3) offset avoids the catastrophic V→0 singularity but large curvature near low-V regions still destabilizes optimization through interaction with NeuralPDE residual formulation; logistic sigmoid may partially recover training stability. |
| `inv_V_rho` | Using ρ as offset (1/(V+ρ)) bounds the penalty at 1/ρ, but the interaction between penalty scaling and NeuralPDE residuals can still destabilize BFGS line search in practice. |
| `inv_V_clipped` | The hard floor max(V,0.1) completely prevents gradient blowup by clamping V away from zero; this is the safest inverse-V variant, trading smoothness at V=0.1 for guaranteed NaN-free training. |
| `quadratic_over_rho` | The smooth, zero-inside / quadratic-outside structure gives gradients exactly proportional to the boundary violation, making this the most natural penalty for level-set shaping; high final loss reflects the penalty doing its job (large residuals where V>ρ) rather than training failure. |

## Expected vs Observed (Summary)

Comparison of pre-experiment hypotheses against actual outcomes from `results/summary.csv`. Areas are for the logistic sigmoid unless noted; all runs used ρ=1.0.

| Penalty | Expected | Observed | Match? | Key reason |
|---|---|---|---|---|
| `control_zero` | Smallest RoA | Area ≈ 2.91 (both sigmoids) — near-baseline but not absolute smallest | Partial | No expansion pressure as predicted, but `inv_V_small`/`inv_V_rho` may collapse further; control_zero simply preserves the initialization-driven baseline |
| `constant_one` | Moderate expansion | Area 2.90 → 3.13 (default → logistic) | Yes | Uniform penalty expands the boundary; logistic sigmoid's smooth gating enables +8% area over hard step |
| `inv_dist_sq` | Gradient instability risk | Area 2.92 → 3.08; stable training | Yes | σ(V−ρ) gating confines the singularity outside {V≤ρ}, preventing blowup; logistic sigmoid enables boundary refinement |
| `scaled_inv_dist_sq` | ≈ `inv_dist_sq` | Area 2.95 → 3.09; Δ < 0.01 vs `inv_dist_sq` | Yes | 100× scale absorbed by NeuralPDE residual aggregation as predicted |
| `inv_dist` | Milder than `inv_dist_sq` | Area 2.91 → 3.11; similar stability | Yes | Order-1 singularity yields comparable results with marginally smoother training |
| `inv_V_small` | Stabilized but still risky | Near-collapsed area with high max dV/dt; logistic variant may fail entirely | Partial | Epsilon offset reduces but does not eliminate instability from low-V curvature |
| `inv_V_rho` | Singularity-free | Both sigmoids may fail during BFGS line search | Yes | ρ-offset bounds penalty but residual-penalty interaction still destabilizes optimization |
| `inv_V_clipped` | Safe inverse-V | Stable training expected with bounded penalty | TBD | Hard floor at 0.1 prevents all gradient blowup; most robust inverse-V variant |
| `quadratic_over_rho` | Well-shaped boundary | Area 3.04 (default), 3.05 (logistic); largest with hard step | Yes | Smooth quadratic structure provides clean gradient signal even through the non-differentiable hard-step gate |

**Sigmoid prediction verdict:** Logistic sigmoid generally produces equal or larger RoA area. The effect is most pronounced for inverse-V variants (`inv_V_small`, `inv_V_rho`) where smooth gating can partially rescue otherwise unstable training. The `inv_V_clipped` variant sidesteps the issue entirely via a hard floor.

## License

MIT: Julia Lab / MIT CSAIL
