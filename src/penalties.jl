hard_step_sigmoid(x) = x .>= zero.(x)
logistic_sigmoid(k::Real) = (x -> one(x) / (one(x) + exp(-k * x)))

function make_penalty_list()
    return [
        ("control_zero", (V, dVdt, x, x_eq, ρ) -> 0.0),
        ("constant_one", (V, dVdt, x, x_eq, ρ) -> 1.0),
        ("inv_dist_sq", (V, dVdt, x, x_eq, ρ) -> 1.0 / (norm(x - x_eq)^2 + 1.0e-6)),
        ("scaled_inv_dist_sq", (V, dVdt, x, x_eq, ρ) -> 100.0 / (norm(x - x_eq)^2 + 1.0e-6)),
        ("inv_dist", (V, dVdt, x, x_eq, ρ) -> 1.0 / (norm(x - x_eq) + 1.0e-6)),
        ("inv_V_small", (V, dVdt, x, x_eq, ρ) -> 1.0 / (V + 1e-3)),
        ("inv_V_rho", (V, dVdt, x, x_eq, ρ) -> 1.0 / (V + ρ)),
        ("inv_V_clipped", (V, dVdt, x, x_eq, ρ) -> 1.0 / max(V, 0.1)),
        ("quadratic_over_rho", (V, dVdt, x, x_eq, ρ) -> max(0.0, V - ρ)^2),
    ]
end
