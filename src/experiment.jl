function run_one_experiment(setup, penalty_name, penalty_fn, sigmoid_name, sigmoid_fn;
        adam_iters1 = 300, adam_iters2 = 300, ρ = 1.0, log_scale::Bool = false,
        rng_seed::Int = -1,
        inv_V_a::Float64 = NaN,
        inv_V_a_regime::AbstractString = "",
        n_eval::Int = 1000,
        simulation_time = 20.0,
        endpoint_atol = 1.0e-3)
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

    # Learned RoA estimate is the sublevel set {V ≤ ρ} (same set as the grid area metric).
    classifier = (V, V̇, x) -> V <= ρ

    # Stop evaluation trajectories once they escape ‖x‖ > 5 (same bound as
    # `approximate_true_roa`); they cannot reach the fixed point, and letting them blow up
    # floods the log with solver instability warnings.
    escape = NeuralLyapunov.SciMLBase.DiscreteCallback(
        (u, t, integrator) -> sum(abs2, u) > 25.0,
        NeuralLyapunov.SciMLBase.terminate!
    )

    # Train and evaluate with `NeuralLyapunov.benchmark`, so `training_time` is measured by
    # NeuralLyapunov itself (PDE discretization excluded since SciML/NeuralLyapunov.jl#215).
    # Adam skips (does not apply) updates whose gradients contain NaN/Inf; count them.
    grad_skips = GradSkipCounter()
    res = with_logger(grad_skips) do
        benchmark(
            setup.f,
            setup.lb,
            setup.ub,
            spec,
            setup.chain,
            QuasiRandomTraining(1024),
            [Adam(0.01), Adam()];
            n = n_eval,
            classifier,
            fixed_point = setup.fixed_point,
            optimization_args = [[:maxiters => adam_iters1], [:maxiters => adam_iters2]],
            simulation_time,
            ode_solver_args = [:callback => escape],
            atol = endpoint_atol,
            init_params = setup.init_params,
            init_states = setup.init_states,
            rng = StableRNG(max(rng_seed, 0)),
            ensemble_alg = NeuralLyapunov.SciMLBase.EnsembleSerial(),
            log_frequency = 1
        )
    end

    grid = compute_grid_metrics(res.V, res.V̇, decrease_condition.ρ; lb = setup.lb, ub = setup.ub)
    losses = Float64.(res.training_losses.Loss)
    iters = Int.(res.training_losses.Iteration)
    tp, fp, tn, fn = res.confusion_matrix.Count

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
        training_time = res.training_time,
        evaluation_time = res.evaluation_time,
        tp = tp,
        fp = fp,
        tn = tn,
        fn = fn,
        accuracy = (tp + tn) / (tp + fp + tn + fn),
        nan_grad_skips = grad_skips.count,
        has_nan = has_nan,
        losses = losses,
        iterations = iters,
        xs = grid.xs,
        ys = grid.ys,
        Vvals = grid.Vvals,
        dVvals = grid.dVvals,
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
        evaluation_time = NaN,
        tp = -1,
        fp = -1,
        tn = -1,
        fn = -1,
        accuracy = NaN,
        nan_grad_skips = -1,
        has_nan = true,
        losses = Float64[],
        iterations = Int[],
        xs = Float64[],
        ys = Float64[],
        Vvals = Float64[],
        dVvals = Float64[],
        roa_mask = Bool[],
        error_message = msg
    )
end
