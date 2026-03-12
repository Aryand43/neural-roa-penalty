function run_one_experiment(setup, penalty_name, penalty_fn, sigmoid_name, sigmoid_fn;
        adam_iters = 300, bfgs_iters = 300, ρ = 1.0)
    structure = NoAdditionalStructure()
    minimization_condition = DontCheckNonnegativity()
    decrease_condition = make_RoA_aware(
        AsymptoticStability();
        out_of_RoA_penalty = penalty_fn,
        sigmoid = sigmoid_fn,
        ρ = ρ
    )
    spec = NeuralLyapunovSpecification(structure, minimization_condition, decrease_condition)

    logger = NeuralLyapunov.NeuralLyapunovBenchmarkLogger{Float64}()
    log_options = LogOptions(; log_frequency = 1)

    @named pde_system = NeuralLyapunovPDESystem(
        setup.f,
        setup.lb,
        setup.ub,
        spec;
        fixed_point = setup.fixed_point
    )

    discretization = PhysicsInformedNN(
        setup.chain,
        QuasiRandomTraining(256);
        init_params = setup.init_params,
        init_states = setup.init_states,
        logger,
        log_options
    )
    prob = discretize(pde_system, discretization)

    adam_res = Optimization.solve(prob, Adam(); maxiters = adam_iters)
    prob2 = Optimization.remake(prob, u0 = adam_res.u)
    bfgs_res = Optimization.solve(prob2, BFGS(); maxiters = bfgs_iters)
    u = bfgs_res.u
    θ = hasproperty(u, :depvar) ? u.depvar : u

    V, V̇ = get_numerical_lyapunov_function(
        discretization.phi,
        θ,
        structure,
        setup.f,
        setup.fixed_point
    )

    grid = compute_grid_metrics(V, V̇, decrease_condition.ρ; lb = setup.lb, ub = setup.ub)
    losses = copy(logger.losses)
    iters = copy(logger.iterations)

    has_nan = any(!isfinite, losses) || any(!isfinite, grid.Vvals) || any(!isfinite, grid.dVvals)

    return (
        penalty = penalty_name,
        sigmoid = sigmoid_name,
        final_loss = isempty(losses) ? NaN : losses[end],
        rho = decrease_condition.ρ,
        area = grid.area,
        max_dVdt_inside = grid.max_dVdt_inside,
        has_nan = has_nan,
        losses = losses,
        iterations = iters,
        xs = grid.xs,
        ys = grid.ys,
        Vvals = grid.Vvals,
        roa_mask = grid.roa_mask,
        error_message = ""
    )
end

function failed_result(penalty_name, sigmoid_name, err)
    msg = sprint(showerror, err)
    return (
        penalty = penalty_name,
        sigmoid = sigmoid_name,
        final_loss = NaN,
        rho = NaN,
        area = NaN,
        max_dVdt_inside = NaN,
        has_nan = true,
        losses = Float64[],
        iterations = Int[],
        xs = Float64[],
        ys = Float64[],
        Vvals = Float64[],
        roa_mask = Bool[],
        error_message = msg
    )
end
