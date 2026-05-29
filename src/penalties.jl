hard_step_sigmoid(x) = x .>= zero.(x)
logistic_sigmoid(k::Real) = (x -> one(x) / (one(x) + exp(-k * x)))

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
