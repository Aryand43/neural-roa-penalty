# neural-roa-penalty

RoA penalty comparison experiments using local `NeuralLyapunov.jl` source inspection and scripted training sweeps.

## Requirements
- Julia `1.11`
- `NeuralLyapunov` `0.3.0`
- Key packages: `NeuralPDE`, `Optimization`, `OptimizationOptimisers`, `OptimizationOptimJL`, `Boltz`, `Lux`, `Plots`

## Run
```bash
julia --project=. run_roa_penalty_experiments.jl
```

## Generated outputs (`results/`)
- `source_inspection.md`
- `summary.csv`
- `training_losses.csv`
- `structured_report.md`
- `loss_logscale.png`
- `roa_contours_vs_true.png`
- `area_comparison.png`