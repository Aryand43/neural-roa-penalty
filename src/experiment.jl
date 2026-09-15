function run_one_experiment(setup, penalty_name, penalty_fn, sigmoid_name, sigmoid_fn;
        adam_iters1 = 300, adam_iters2 = 300, ρ = 1.0, log_scale::Bool = false,
        rng_seed::Int = -1,
        inv_V_a::Float64 = NaN,
        inv_V_a_regime::AbstractString = "")
    structure = NoAdditionalStructure()
    minimization_condition = DontCheckNonnegativity()
    decrease_condition = make_RoA_aware(
        AsymptoticStability();
        out_of_RoA_penalty = penalty_fn,
        sigmoid = sigmoid_fn,
        log_scale,
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
        QuasiRandomTraining(1024);
        init_params = setup.init_params,
        init_states = setup.init_states,
        logger,
        log_options
    )
    prob = discretize(pde_system, discretization)

    # Time only the optimization passes, matching `NeuralLyapunov.benchmark` since
    # SciML/NeuralLyapunov.jl#215 (PDE discretization excluded from training time).
    # Adam skips (does not apply) updates whose gradients contain NaN/Inf; count them.
    grad_skips = GradSkipCounter()
    training = with_logger(grad_skips) do
        @timed begin
            adam_res = Optimization.solve(prob, Adam(0.01); maxiters = adam_iters1)
            prob2 = Optimization.remake(prob, u0 = adam_res.u)
            Optimization.solve(prob2, Adam(); maxiters = adam_iters2)
        end
    end
    elapsed = training.time

    u = training.value.u
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
        log_scale = log_scale,
        rng_seed = rng_seed,
        inv_V_a = inv_V_a,
        inv_V_a_regime = String(inv_V_a_regime),
        final_loss = isempty(losses) ? NaN : losses[end],
        rho = decrease_condition.ρ,
        area = grid.area,
        max_dVdt_inside = grid.max_dVdt_inside,
        training_time = elapsed,
        compile_time = training.compile_time,
        nan_grad_skips = grad_skips.count,
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

function failed_result(penalty_name, sigmoid_name, err; log_scale::Bool = false,
        rng_seed::Int = -1,
        inv_V_a::Float64 = NaN,
        inv_V_a_regime::AbstractString = "")
    msg = sprint(showerror, err)
    return (
        penalty = penalty_name,
        sigmoid = sigmoid_name,
        log_scale = log_scale,
        rng_seed = rng_seed,
        inv_V_a = inv_V_a,
        inv_V_a_regime = String(inv_V_a_regime),
        final_loss = NaN,
        rho = NaN,
        area = NaN,
        max_dVdt_inside = NaN,
        training_time = NaN,
        compile_time = NaN,
        nan_grad_skips = -1,
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
