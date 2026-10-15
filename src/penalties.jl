hard_step_sigmoid(x) = x .>= zero.(x)

# Overflow-safe logistic gate.
#
# `1/(1+exp(-k*z))` is mathematically fine but numerically unsafe here: NeuralLyapunov's
# RoA-aware decrease condition evaluates the gate at both `z` and `-z`
# (`sigmoid(-scaled_V)` for the in-RoA term, `sigmoid(scaled_V)` for the penalty), so with
# `scaled_V = V - ρ` the in-RoA gate sees `ρ - V`. Once `V > ρ + 709.78/k` (≈ 36.5 for
# k = 20, ρ = 1) `exp(-k*z)` overflows to `Inf`; the value still rounds to 0, but the
# derivative evaluates as `Inf/Inf` (equivalently `-0.0 * Inf`) and comes back `NaN`, which
# OptimizationOptimisers then skips, so those runs barely train. `log_scale = true` hides
# the problem rather than fixing it: it needs `V/ρ > exp(709.78/k) ≈ 1.6e15` to overflow.
#
# `(1 + tanh(k*z/2))/2` is the identical function with a bounded intermediate: `tanh`
# saturates to ±1 instead of overflowing, so both value and derivative stay finite for all
# real `z`, and it traces symbolically through NeuralPDE like `exp` does.
logistic_sigmoid(k::Real) = (x -> (one(x) + tanh(k * x / 2)) / 2)

# Previous (unsafe) form, kept for the overflow diagnostic in
# `scripts/sigmoid_overflow_check.jl`; not used by the experiment grid.
logistic_sigmoid_naive(k::Real) = (x -> one(x) / (one(x) + exp(-k * x)))

# Inverse-V penalties use stabilized form `1/(V+a)` (never bare `1/V`).
function inv_V_penalty(a::Real)
    return (V, dVdt, x, x_eq, ρ) -> 1.0 / (V + a)
end

# Each tuple is `(name, fn, inv_V_a, inv_V_a_regime)`; non-inverse-V rows use `NaN` and "".
function make_penalty_list(; ρ::Real = 1.0)
    ρf = Float64(ρ)
    a_lt_rho = max(sqrt(eps(Float64)), ρf * 1e-4)
    inv_na = NaN
    inv_na_regime = ""
    return [
        ("control_zero", (V, dVdt, x, x_eq, ρr) -> 0.0, inv_na, inv_na_regime),
        ("constant_one", (V, dVdt, x, x_eq, ρr) -> 1.0, inv_na, inv_na_regime),
        ("inv_dist_sq", (V, dVdt, x, x_eq, ρr) -> 1.0 / (norm(x - x_eq)^2 + 1.0e-6), inv_na, inv_na_regime),
        ("scaled_inv_dist_sq", (V, dVdt, x, x_eq, ρr) -> 100.0 / (norm(x - x_eq)^2 + 1.0e-6), inv_na, inv_na_regime),
        ("inv_dist", (V, dVdt, x, x_eq, ρr) -> 1.0 / (norm(x - x_eq) + 1.0e-6), inv_na, inv_na_regime),
        ("inv_V_small", inv_V_penalty(a_lt_rho), Float64(a_lt_rho), "much_less_than_rho"),
        ("inv_V_rho", inv_V_penalty(ρf), ρf, "equal_rho"),
        ("inv_V_clipped", (V, dVdt, x, x_eq, ρr) -> 1.0 / max(V, 0.1), inv_na, inv_na_regime),
        ("quadratic_over_rho", (V, dVdt, x, x_eq, ρr) -> max(0.0, V - ρr)^2, inv_na, inv_na_regime),
    ]
end
