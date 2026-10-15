# Why the logistic gate produced NaN gradients, and why the tanh form does not.
#
# Usage:
#   julia --project=. scripts/sigmoid_overflow_check.jl
#
# NeuralLyapunov's RoA-aware decrease condition calls the gate at both `scaled_V` and
# `-scaled_V`, so with `log_scale = false` (`scaled_V = V - ρ`) the in-RoA term's gate is
# evaluated at `ρ - V`, which is very negative whenever the network's `V` is large.
# `1/(1+exp(-k*z))` overflows there; `(1+tanh(k*z/2))/2` does not. Derivatives below are the
# chain rule written out, i.e. what AD evaluates — no extra dependency needed.

using LinearAlgebra
include("../src/penalties.jl")

const K = 20.0
const RHO = 1.0

naive = logistic_sigmoid_naive(K)
safe = logistic_sigmoid(K)

# d/dz 1/(1+exp(-k z)) as AD composes it: (dy/du) * (du/dz) with u = exp(-k z).
d_naive(z) = (u = exp(-K * z); -(1.0 / (1.0 + u)^2) * (-K * u))
# d/dz (1+tanh(k z/2))/2
d_safe(z) = (K / 2) * (1 - tanh(K * z / 2)^2) / 2

overflow_z = -log(prevfloat(Inf)) / K

println("k = $(K), ρ = $(RHO)")
println("exp(-k*z) overflows for z < $(round(overflow_z, digits = 3)), ",
    "i.e. the in-RoA gate `σ(ρ - V)` breaks once V > $(round(RHO - overflow_z, digits = 3))")
println("with log_scale = true the gate sees -log(V/ρ), which needs V/ρ > ",
    "$(round(exp(-overflow_z), sigdigits = 3)) to overflow")
println()
println(rpad("V", 10), rpad("z = ρ - V", 12), rpad("naive σ(z)", 14), rpad("naive dσ/dz", 14),
    rpad("tanh σ(z)", 12), "tanh dσ/dz")
for V in (0.0, 1.0, 2.0, 10.0, 36.0, 37.0, 50.0, 100.0, 1.0e4)
    z = RHO - V
    println(rpad(V, 10), rpad(round(z, digits = 2), 12), rpad(naive(z), 14),
        rpad(d_naive(z), 14), rpad(safe(z), 12), d_safe(z))
end
println()
nan_at = [V for V in 0.0:0.5:100.0 if !isfinite(d_naive(RHO - V))]
println("naive gate: non-finite derivative for V ≥ $(minimum(nan_at)) (scanned V ∈ [0, 100])")
println("tanh gate:  finite derivative everywhere scanned; ",
    "max |σ_naive − σ_tanh| over V ∈ [0, 100] where both are finite = ",
    maximum(abs(naive(RHO - V) - safe(RHO - V)) for V in 0.0:0.5:100.0 if isfinite(naive(RHO - V))))
