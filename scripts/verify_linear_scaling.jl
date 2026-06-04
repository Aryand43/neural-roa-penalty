# Quick check: :linear scaling matches pre-refactor penalty + gate wiring.
include(joinpath(@__DIR__, "..", "src", "penalties.jl"))

function legacy_quadratic(V, dVdt, x, x_eq, ρ)
    return max(0.0, V - ρ)^2
end

@assert violation(1.0, 1.0, :linear) ≈ 0.0
@assert violation(2.0, 1.0, :linear) ≈ 1.0
@assert isapprox(violation(2.0, 1.0, :log), log(2.0); atol = 1e-10)

penalties = make_penalty_list(:linear)
quad = only(f for (n, f) in penalties if n == "quadratic_over_rho")
@assert quad(1.5, 0.0, [1.0, 1.0], [0.0, 0.0], 1.0) ≈ legacy_quadratic(1.5, 0.0, [1.0, 1.0], [0.0, 0.0], 1.0)

const_one = only(f for (n, f) in penalties if n == "constant_one")
@assert const_one(1.5, 0.0, [1.0, 1.0], [0.0, 0.0], 1.0) == 1.0

base = (V, dVdt, x, x_eq, ρ) -> 1.0
wrapped = adjust_out_of_roa_gate(base, hard_step_sigmoid, :linear)
@assert wrapped === base
@assert wrapped(1.5, 0.0, [0.0, 0.0], [0.0, 0.0], 1.0) == 1.0

log_list = make_penalty_list(:log)
@assert length(penalties) == length(log_list) == length(PENALTY_ORDER)
println("verify_linear_scaling: OK")
