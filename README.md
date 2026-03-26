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
| `inv_V` (1/V) | Collapsed or degenerate RoA, especially with hard-step sigmoid | Singularity at V=0 (the equilibrium) causes catastrophic gradient blowup; hard-step gating amplifies the problem by offering no smooth transition |
| `quadratic_over_rho` (max(0,V−ρ)²) | Well-shaped boundary closely tracking the true RoA | Zero inside {V≤ρ}, smooth quadratic growth outside; gradient is proportional to violation magnitude, directly penalizing boundary overshoot |

**Sigmoid interaction prediction:** logistic sigmoid should uniformly improve RoA quality over the hard step because it provides nonzero gradient flow across the V=ρ boundary, enabling the optimizer to smoothly adjust the level set.

## RoA Comparison Plots

Each experiment produces an overlay plot comparing the learned and true Regions of Attraction, saved to `results/roa_overlay_<penalty>_<sigmoid>.png`.

**Legend:**
- **Red solid contour** — learned RoA boundary (sublevel set {x : V(x) ≤ ρ})
- **Blue dashed contour** — true RoA boundary (computed by forward-simulating Van der Pol trajectories and checking convergence to the origin)
- **Light blue fill** — interior of the true RoA

Plots for all 14 runs (7 penalties × 2 sigmoids):

| Penalty | Default (hard step) | Logistic (k=20) |
|---|---|---|
| control_zero | [plot](results/roa_overlay_control_zero_default.png) | [plot](results/roa_overlay_control_zero_logistic.png) |
| constant_one | [plot](results/roa_overlay_constant_one_default.png) | [plot](results/roa_overlay_constant_one_logistic.png) |
| inv_dist_sq | [plot](results/roa_overlay_inv_dist_sq_default.png) | [plot](results/roa_overlay_inv_dist_sq_logistic.png) |
| scaled_inv_dist_sq | [plot](results/roa_overlay_scaled_inv_dist_sq_default.png) | [plot](results/roa_overlay_scaled_inv_dist_sq_logistic.png) |
| inv_dist | [plot](results/roa_overlay_inv_dist_default.png) | [plot](results/roa_overlay_inv_dist_logistic.png) |
| inv_V | [plot](results/roa_overlay_inv_V_default.png) | [plot](results/roa_overlay_inv_V_logistic.png) |
| quadratic_over_rho | [plot](results/roa_overlay_quadratic_over_rho_default.png) | [plot](results/roa_overlay_quadratic_over_rho_logistic.png) |

## Per-Penalty Behavior Explanation

| Penalty | Explanation |
|---|---|
| `control_zero` | No out-of-RoA gradient signal means the boundary is determined entirely by the Lyapunov decrease condition inside {V≤ρ}; the optimizer has no incentive to expand, so the RoA reflects the network's initialization bias. |
| `constant_one` | Uniform pressure expands the boundary but cannot shape it to match the true RoA's non-circular geometry; the logistic sigmoid improves this by letting gradients flow smoothly across the V=ρ transition. |
| `inv_dist_sq` | The ‖x−x₀‖⁻² singularity is gated to activate only outside {V≤ρ}, so near-equilibrium blowup is avoided in practice; the logistic sigmoid enables finer boundary refinement through smooth gradient flow. |
| `scaled_inv_dist_sq` | The 100× multiplier inflates the loss magnitude without changing the penalty landscape shape — NeuralPDE's residual-squared aggregation absorbs the scale, so behavior tracks `inv_dist_sq` closely. |
| `inv_dist` | Weaker order-1 singularity produces more moderate gradients than `inv_dist_sq`, yielding qualitatively similar but slightly more stable training dynamics. |
| `inv_V` | The 1/V singularity is catastrophic: with the hard-step sigmoid, the non-differentiable gate combined with gradient blowup at V→0 collapses the learned RoA to near-zero area; the logistic sigmoid partially recovers by providing smooth gating, but the fundamental singularity still degrades the estimate. |
| `quadratic_over_rho` | The smooth, zero-inside / quadratic-outside structure gives gradients exactly proportional to the boundary violation, making this the most natural penalty for level-set shaping; high final loss reflects the penalty doing its job (large residuals where V>ρ) rather than training failure. |

## License

MIT: Julia Lab / MIT CSAIL
