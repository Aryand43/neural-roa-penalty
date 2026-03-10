# neural-roa-penalty

RoA penalty comparison experiments using NeuralLyapunov.jl to learn Region of Attraction (RoA) estimates for Van der Pol oscillator dynamics.

## Setup

Dependencies are managed through the Julia project environment. To set up:

```julia
julia --project=.
] instantiate
```

This will install all required packages as specified in `Project.toml`.

## Run

```bash
julia --project=. run_roa_penalty_experiments.jl
```

## Generated outputs (`results/`)
- `source_inspection.md` - inspection of NeuralLyapunov source code methods used
- `summary.csv` - experiment metrics (final loss, RoA area, etc.)
- `training_losses.csv` - iteration-by-iteration loss data
- `structured_report.md` - comprehensive analysis report
- `loss_logscale.png` - training loss curves
- `roa_contours.png` - learned RoA contours for each penalty/sigmoid combination
- `area_comparison.png` - bar chart of estimated RoA areas across experiments

## System Dynamics

This framework uses the **Van der Pol oscillator**:

$$\dot{x}_1 = -x_2$$
$$\dot{x}_2 = \mu(x_1^2 - 1)x_2 + x_1$$

with equilibrium at the origin.
