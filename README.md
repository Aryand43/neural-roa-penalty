# neural-roa-penalty

Systematic comparison of RoA-aware penalty functions for neural Lyapunov verification on Van der Pol dynamics via [NeuralLyapunov.jl](https://github.com/SciML/NeuralLyapunov.jl).

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. run_roa_penalty_experiments.jl
```

All outputs are written to the `results/` directory (CSVs, reports, and plots such as `loss_logscale.png`, `roa_contours.png`, and `area_comparison.png`).[file:1]

## License

MIT: Julia Lab / MIT CSAIL
