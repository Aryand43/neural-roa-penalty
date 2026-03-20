# neural-roa-penalty

Systematic comparison of RoA-aware penalty functions for neural Lyapunov verification on Van der Pol dynamics via [NeuralLyapunov.jl](https://github.com/MIT-REALM/NeuralLyapunov.jl).

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. run_roa_penalty_experiments.jl
```

Outputs land in `results/`.

## License

MIT: Julia Lab / MIT CSAIL
