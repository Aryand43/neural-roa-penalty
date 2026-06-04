include("scaling.jl")

hard_step_sigmoid(x) = x .>= zero.(x)
logistic_sigmoid(k::Real) = (x -> one(x) / (one(x) + exp(-k * x)))

function _base_penalty_fns()
    return Dict{String, Function}(
        "control_zero" => (V, dVdt, x, x_eq, ρ) -> 0.0,
        "constant_one" => (V, dVdt, x, x_eq, ρ) -> 1.0,
        "inv_dist_sq" => (V, dVdt, x, x_eq, ρ) -> 1.0 / (norm(x - x_eq)^2 + 1.0e-6),
        "scaled_inv_dist_sq" => (V, dVdt, x, x_eq, ρ) -> 100.0 / (norm(x - x_eq)^2 + 1.0e-6),
        "inv_dist" => (V, dVdt, x, x_eq, ρ) -> 1.0 / (norm(x - x_eq) + 1.0e-6),
        "inv_V_small" => (V, dVdt, x, x_eq, ρ) -> 1.0 / (V + 1e-3),
        "inv_V_rho" => (V, dVdt, x, x_eq, ρ) -> 1.0 / (V + ρ),
        "inv_V_clipped" => (V, dVdt, x, x_eq, ρ) -> 1.0 / max(V, 0.1),
        "quadratic_over_rho" => (V, dVdt, x, x_eq, ρ) -> begin
            # Placeholder; replaced per scaling mode in make_penalty_list.
            u = violation(V, ρ, :linear)
            return max(0.0, u)^2
        end,
    )
end

function apply_violation_scaling_to_penalty(name::String, base_fn::Function, scaling::Symbol)
    if name == "quadratic_over_rho"
        return (V, dVdt, x, x_eq, ρ) -> begin
            u = violation(V, ρ, scaling)
            return max(0.0, u)^2
        end
    else
        return base_fn
    end
end

const PENALTY_ORDER = [
    "control_zero",
    "constant_one",
    "inv_dist_sq",
    "scaled_inv_dist_sq",
    "inv_dist",
    "inv_V_small",
    "inv_V_rho",
    "inv_V_clipped",
    "quadratic_over_rho",
]

function make_penalty_list(scaling::Symbol = :linear)
    scaling in (:linear, :log) || error("scaling must be :linear or :log, got $(scaling)")
    base = _base_penalty_fns()
    return [(name, apply_violation_scaling_to_penalty(name, base[name], scaling)) for name in PENALTY_ORDER]
end

function make_scaling_modes()
    return [(:linear, :linear), (:log, :log)]
end
