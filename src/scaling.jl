# Violation scaling for RoA-aware penalties and gating.
#
# :linear — violation u = V − ρ (matches pre-refactor behavior).
# :log     — violation u = log(max(V, ρ·(1+ε)) / ρ); u = 0 at V = ρ, smooth near the boundary.

const VIOLATION_RATIO_EPS = 1e-6

"""Out-of-RoA violation measure used for penalty shaping and log-mode gate correction."""
function violation(V, ρ, mode::Symbol)
    if mode === :linear
        return V - ρ
    elseif mode === :log
        ratio = max(V, ρ * (1 + VIOLATION_RATIO_EPS)) / ρ
        return log(ratio)
    else
        error("Unknown violation scaling mode: $(mode). Use :linear or :log.")
    end
end

scaling_mode_label(mode::Symbol) = mode === :linear ? "linear" : mode === :log ? "log" : string(mode)

"""
    adjust_out_of_roa_gate(penalty_fn, sigmoid_fn, scaling)

For `:log`, scales the out-of-RoA penalty so the effective gate is `σ(u_log)` instead of
`σ(V − ρ)`, where `u_log = violation(V, ρ, :log)` and `σ` is `sigmoid_fn`. NeuralLyapunov
still applies `σ(V − ρ)`; this wrapper cancels and replaces that factor on the penalty side.
`:linear` returns `penalty_fn` unchanged.
"""
function adjust_out_of_roa_gate(penalty_fn, sigmoid_fn, scaling::Symbol)
    scaling === :linear && return penalty_fn
    return function (V, dVdt, x, x_eq, ρ)
        pen = penalty_fn(V, dVdt, x, x_eq, ρ)
        z_lin = V - ρ
        z_log = violation(V, ρ, :log)
        σ_lin = sigmoid_fn(z_lin)
        if σ_lin <= 1e-12
            return pen
        end
        σ_log = sigmoid_fn(z_log)
        return pen * (σ_log / σ_lin)
    end
end
