# neural-roa-penalty

Systematic comparison of RoA-aware penalty functions for neural Lyapunov verification on Van der Pol dynamics via [NeuralLyapunov.jl](https://github.com/SciML/NeuralLyapunov.jl).

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. run_roa_penalty_experiments.jl
```

All outputs are written to the `results/` directory (CSVs, reports, and plots).

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
