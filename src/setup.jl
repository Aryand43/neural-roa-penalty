function build_setup(; seed = 1234, hidden = 32)
    rng = StableRNG(seed)
    Random.seed!(seed)

    lb = [-2.0, -2.0]
    ub = [2.0, 2.0]
    fixed_point = [0.0, 0.0]
    dim_state = length(lb)

    base = MLP(dim_state, (hidden, hidden, 1), tanh)
    chain = AdditiveLyapunovNet(base; dim_ϕ = 1, fixed_point = fixed_point)
    init_params, init_states = Lux.setup(rng, chain)
    init_params = Lux.f64(init_params)
    init_states = Lux.f64(init_states)

    return (; rng, f = vanderpol, lb, ub, fixed_point, chain, init_params, init_states)
end
