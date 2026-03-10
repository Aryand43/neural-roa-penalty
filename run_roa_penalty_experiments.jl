using NeuralLyapunov
using NeuralPDE
using ModelingToolkit
using StableRNGs
using Random
using LinearAlgebra
using Statistics
using Printf
using Dates
using Optimization
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using Boltz.Layers: MLP
using Lux

const HAVE_PLOTS = let
    try
        @eval using Plots
        true
    catch
        false
    end
end

const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

function extract_function_block(path::AbstractString, start_regex::Regex)
    lines = readlines(path)
    start_idx = findfirst(i -> occursin(start_regex, lines[i]), eachindex(lines))
    start_idx === nothing && error("Could not find function in $(path) with pattern $(start_regex)")

    opens(line) = length(collect(eachmatch(r"\b(function|if|for|while|let|begin|try|quote|struct|macro)\b", line)))
    closes(line) = length(collect(eachmatch(r"\bend\b", line)))

    depth = 0
    block = String[]
    for i in start_idx:length(lines)
        line = lines[i]
        push!(block, line)
        depth += opens(line)
        depth -= closes(line)
        if depth == 0 && i > start_idx
            break
        end
    end
    return join(block, "\n")
end

function source_inspection()
    src_root = dirname(pathof(NeuralLyapunov))
    info = Dict{String, Any}()
    info["pathof(NeuralLyapunov)"] = pathof(NeuralLyapunov)
    info["methods(make_RoA_aware)"] = sprint(io -> show(io, methods(make_RoA_aware)))
    info["methods(AdditiveLyapunovNet)"] = sprint(io -> show(io, methods(AdditiveLyapunovNet)))
    info["methods(AsymptoticStability)"] = sprint(io -> show(io, methods(AsymptoticStability)))
    info["methods(DontCheckNonnegativity)"] = sprint(io -> show(io, methods(DontCheckNonnegativity)))
    info["methods(NoAdditionalStructure)"] = sprint(io -> show(io, methods(NoAdditionalStructure)))
    info["lowered(make_RoA_aware)"] = sprint(io -> show(io, first(code_lowered(make_RoA_aware, Tuple{NeuralLyapunov.AbstractLyapunovDecreaseCondition}))))

    roa_file = joinpath(src_root, "decrease_conditions_RoA_aware.jl")
    dec_file = joinpath(src_root, "decrease_conditions.jl")
    min_file = joinpath(src_root, "minimization_conditions.jl")
    str_file = joinpath(src_root, "structure_specification.jl")
    lux_file = joinpath(src_root, "lux_structures.jl")
    pde_file = joinpath(src_root, "NeuralLyapunovPDESystem.jl")

    defs = Dict{String, String}()
    defs["make_RoA_aware"] = extract_function_block(roa_file, r"^\s*function make_RoA_aware")
    defs["get_decrease_condition(::RoAAwareDecreaseCondition)"] = extract_function_block(roa_file, r"^\s*function get_decrease_condition\(cond::RoAAwareDecreaseCondition\)")
    defs["AsymptoticStability"] = extract_function_block(dec_file, r"^\s*function AsymptoticStability")
    defs["DontCheckNonnegativity"] = extract_function_block(min_file, r"^\s*function DontCheckNonnegativity")
    defs["NoAdditionalStructure"] = extract_function_block(str_file, r"^\s*function NoAdditionalStructure")
    defs["AdditiveLyapunovNet"] = extract_function_block(lux_file, r"^\s*function AdditiveLyapunovNet")
    defs["_NeuralLyapunovPDESystem"] = extract_function_block(pde_file, r"^\s*function _NeuralLyapunovPDESystem")

    return (; info, defs)
end

hard_step_sigmoid(x) = x .>= zero.(x)
logistic_sigmoid(k::Real) = (x -> one(x) / (one(x) + exp(-k * x)))

function make_penalty_list()
    return [
        ("control_zero", (V, dVdt, x, x_eq, ρ) -> 0.0),
        ("constant_one", (V, dVdt, x, x_eq, ρ) -> 1.0),
        ("inv_dist_sq", (V, dVdt, x, x_eq, ρ) -> 1.0 / (norm(x - x_eq)^2 + 1.0e-6)),
        ("scaled_inv_dist_sq", (V, dVdt, x, x_eq, ρ) -> 100.0 / (norm(x - x_eq)^2 + 1.0e-6)),
        ("inv_dist", (V, dVdt, x, x_eq, ρ) -> 1.0 / (norm(x - x_eq) + 1.0e-6)),
        ("inv_V", (V, dVdt, x, x_eq, ρ) -> 1.0 / (V + 1.0e-6)),
        ("quadratic_over_rho", (V, dVdt, x, x_eq, ρ) -> max(0.0, V - ρ)^2),
    ]
end

function build_setup(; seed = 1234, hidden = 32)
    rng = StableRNG(seed)
    Random.seed!(seed)

    function f(x, p, t)
        x1, x2 = x
        dx1 = -x2
        dx2 = p * (x1^2 - 1) * x2 + x1
        return [dx1, dx2]
    end
    lb = [-2.0, -2.0]
    ub = [2.0, 2.0]
    fixed_point = [0.0, 0.0]
    dim_state = length(lb)

    base = MLP(dim_state, (hidden, hidden, 1), tanh)
    chain = AdditiveLyapunovNet(base; dim_ϕ = 1, fixed_point = fixed_point)
    init_params, init_states = Lux.setup(rng, chain)
    init_params = Lux.f64(init_params)
    init_states = Lux.f64(init_states)

    return (; rng, f, lb, ub, fixed_point, chain, init_params, init_states)
end

function compute_grid_metrics(V, V̇, ρ; lb, ub, n = 201)
    xs = range(lb[1], ub[1], length = n)
    ys = range(lb[2], ub[2], length = n)
    dx = step(xs)
    dy = step(ys)

    points = Matrix{Float64}(undef, 2, n * n)
    idx = 1
    for y in ys, x in xs
        points[:, idx] .= (x, y)
        idx += 1
    end

    Vvals = vec(V(points))
    dVvals = vec(V̇(points))
    roa_mask = Vvals .<= ρ

    area = sum(roa_mask) * dx * dy
    max_dVdt_inside = any(roa_mask) ? maximum(dVvals[roa_mask]) : NaN

    return (; xs, ys, points, Vvals, dVvals, roa_mask, area, max_dVdt_inside)
end

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

function save_loss_csv(results)
    out = joinpath(RESULTS_DIR, "training_losses.csv")
    open(out, "w") do io
        println(io, "penalty,sigmoid,iteration,loss")
        for r in results
            for (it, loss) in zip(r.iterations, r.losses)
                println(io, "$(r.penalty),$(r.sigmoid),$(Int(it)),$(Float64(loss))")
            end
        end
    end
end

function save_summary_csv(results)
    rows = [
        (
            penalty = r.penalty,
            sigmoid = r.sigmoid,
            final_loss = Float64(r.final_loss),
            rho = Float64(r.rho),
            roa_area = Float64(r.area),
            leak_outside_true_roa = Bool(r.leak),
            max_dVdt_inside_V_le_rho = Float64(r.max_dVdt_inside),
            has_nan = Bool(r.has_nan),
            error_message = r.error_message,
        ) for r in results
    ]
    out = joinpath(RESULTS_DIR, "summary.csv")
    open(out, "w") do io
        println(io, "penalty,sigmoid,final_loss,rho,roa_area,true_roa_area,leak_outside_true_roa,max_dVdt_inside_V_le_rho,has_nan,error_message")
        for row in rows
            escaped = replace(row.error_message, "\"" => "'")
            println(
                io,
                "$(row.penalty),$(row.sigmoid),$(row.final_loss),$(row.rho),$(row.roa_area),$(row.true_roa_area),$(row.leak_outside_true_roa),$(row.max_dVdt_inside_V_le_rho),$(row.has_nan),\"$(escaped)\""
            )
        end
    end
    return rows
end

function maybe_make_plots(results)
    if !HAVE_PLOTS
        open(joinpath(RESULTS_DIR, "plot_status.txt"), "w") do io
            write(io, "Plots.jl not available in environment; skipped plot generation.\n")
        end
        return
    end

    valid = filter(r -> !isempty(r.iterations) && !isempty(r.losses), results)
    p_loss = plot(; yscale = :log10, xlabel = "Iteration", ylabel = "Loss", title = "Training loss (log-scale)")
    for r in valid
        plot!(p_loss, r.iterations, r.losses; label = "$(r.penalty) / $(r.sigmoid)", lw = 1.5)
    end
    savefig(p_loss, joinpath(RESULTS_DIR, "loss_logscale.png"))

    first_per_penalty = Dict{String, NamedTuple}()
    for r in filter(r -> !isempty(r.Vvals) && isfinite(r.rho), results)
        key = r.penalty
        if !haskey(first_per_penalty, key) || (r.sigmoid == "logistic")
            first_per_penalty[key] = r
        end
    end

    plt_list = Plots.Plot[]
    for key in sort(collect(keys(first_per_penalty)))
        r = first_per_penalty[key]
        n = length(r.xs)
        Vmat = reshape(r.Vvals, n, n)
        p = contour(
            r.xs,
            r.ys,
            Vmat';
            levels = [r.rho],
            xlabel = "x1",
            ylabel = "x2",
            title = "$(r.penalty) ($(r.sigmoid))",
            legend = false
        )
        push!(plt_list, p)
    end
    big = plot(plt_list...; layout = (ceil(Int, length(plt_list) / 3), 3), size = (1200, 900))
    savefig(big, joinpath(RESULTS_DIR, "roa_contours.png"))

    labels = ["$(r.penalty)\n$(r.sigmoid)" for r in results]
    areas = [r.area for r in results]
    p_area = bar(labels, areas; xlabel = "Penalty / Sigmoid", ylabel = "Estimated RoA area", legend = false, xrotation = 45)
    hline!(p_area, [pi]; color = :red, ls = :dash, lw = 2)
    savefig(p_area, joinpath(RESULTS_DIR, "area_comparison.png"))
end

function write_source_report(src)
    out = joinpath(RESULTS_DIR, "source_inspection.md")
    nl_path = src.info["pathof(NeuralLyapunov)"]
    open(out, "w") do io
        println(io, "# Source Inspection")
        println(io)
        println(io, "## Runtime")
        println(io, "- pathof(NeuralLyapunov): `$(nl_path)`")
        println(io)
        for key in [
            "methods(make_RoA_aware)",
            "methods(AdditiveLyapunovNet)",
            "methods(AsymptoticStability)",
            "methods(DontCheckNonnegativity)",
            "methods(NoAdditionalStructure)",
            "lowered(make_RoA_aware)",
        ]
            println(io, "### $(key)")
            println(io, "```julia")
            println(io, src.info[key])
            println(io, "```")
            println(io)
        end
        for key in keys(src.defs)
            println(io, "## Definition: $(key)")
            println(io, "```julia")
            println(io, src.defs[key])
            println(io, "```")
            println(io)
        end
    end
end

function write_structured_report(results, summary_rows, src)
    out = joinpath(RESULTS_DIR, "structured_report.md")
    nl_path = src.info["pathof(NeuralLyapunov)"]
    ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

    default_rows = filter(r -> r.sigmoid == "default", results)
    logistic_rows = filter(r -> r.sigmoid == "logistic", results)

    function pick_max_area(rows)
        rows = filter(r -> isfinite(r.area), rows)
        if isempty(rows)
            return nothing
        end
        return rows[argmax([r.area for r in rows])]
    end

    best_default = pick_max_area(default_rows)
    best_logistic = pick_max_area(logistic_rows)

    inv_default = only(filter(r -> r.penalty == "inv_dist_sq" && r.sigmoid == "default", results))
    scaled_default = only(filter(r -> r.penalty == "scaled_inv_dist_sq" && r.sigmoid == "default", results))
    inv_logistic = only(filter(r -> r.penalty == "inv_dist_sq" && r.sigmoid == "logistic", results))
    scaled_logistic = only(filter(r -> r.penalty == "scaled_inv_dist_sq" && r.sigmoid == "logistic", results))

    adaptive_note = "No adaptive reweighting configured in this framework (QuadratureTraining without adaptive loss callbacks)."
    scale_effect_default = abs(scaled_default.area - inv_default.area)
    scale_effect_logistic = abs(scaled_logistic.area - inv_logistic.area)

    open(out, "w") do io
        println(io, "# Structured Report")
        println(io)
        println(io, "- Timestamp: $(ts)")
        println(io, "- NeuralLyapunov path: `$(nl_path)`")
        println(io, "- Default sigmoid in source: `(x) -> x .≥ zero.(x)` (hard step)")
        println(io, "- Logistic sigmoid used in experiments: `σ(z)=1/(1+exp(-k*z))`, with `k=20`")
        println(io, "- Penalty term wiring in source: `[sigmoid(ρ-V)*in_RoA_penalty, sigmoid(V-ρ)*out_of_RoA_penalty]`")
        println(io, "- Loss aggregation: NeuralPDE residual loss over PDE equations (each residual vs `0.0`).")
        println(io, "- Gradient through default sigmoid: non-smooth/boolean gate; logistic variant provides smooth gate.")
        println(io)

        println(io, "## Per-Run Results")
        println(io)
        for row in summary_rows
            e = isempty(row.error_message) ? "none" : row.error_message
            println(io, "- penalty: `$(row.penalty)` | sigmoid: `$(row.sigmoid)` | final_loss: `$(row.final_loss)` | ρ: `$(row.rho)` | area: `$(row.roa_area)` | leak_outside_true_RoA: `$(row.leak_outside_true_roa)` | max_dVdt_inside_V_le_ρ: `$(row.max_dVdt_inside_V_le_rho)` | has_nan: `$(row.has_nan)` | error: `$(e)`")
        end
        println(io)

        println(io, "## Adaptive Reweighting Check")
        println(io)
        println(io, "- $(adaptive_note)")
        println(io, "- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (default sigmoid): $(scale_effect_default)")
        println(io, "- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (logistic sigmoid): $(scale_effect_logistic)")
        println(io)

        println(io, "## Hypothesis Validation")
        println(io)
        if best_default !== nothing
            println(io, "- Largest RoA area (default sigmoid): `$(best_default.penalty)` with area `$(best_default.area)`")
        end
        if best_logistic !== nothing
            println(io, "- Largest RoA area (logistic sigmoid): `$(best_logistic.penalty)` with area `$(best_logistic.area)`")
        end
        violators = filter(r -> isfinite(r.max_dVdt_inside) && r.max_dVdt_inside > 0.0, results)
        if isempty(violators)
            println(io, "- No run had positive max dV/dt inside V <= ρ on the evaluation grid.")
        else
            names = join(["$(r.penalty)/$(r.sigmoid)" for r in violators], ", ")
            println(io, "- Runs with decrease-condition violation (max dV/dt inside V <= ρ > 0): $(names)")
        end
        println(io, "- Next architecture modification if all plateau: increase `MLP` width/depth and test `MultiplicativeLyapunovNet` with same protocol.")
    end
end

function main()
    println("Running source inspection...")
    src = source_inspection()
    write_source_report(src)

    println("Building controlled setup...")
    setup = build_setup(; seed = 2026, hidden = 32)
    penalties = make_penalty_list()
    sigmoid_list = [
        ("default", hard_step_sigmoid),
        ("logistic", logistic_sigmoid(20.0)),
    ]

    results = NamedTuple[]
    adam_iters = 20
    bfgs_iters = 20
    for (pname, pfn) in penalties
        for (sname, sfn) in sigmoid_list
            println("Training penalty=$(pname), sigmoid=$(sname)")
            res = try
                run_one_experiment(setup, pname, pfn, sname, sfn; adam_iters = adam_iters, bfgs_iters = bfgs_iters, ρ = 1.0)
            catch err
                @warn "Experiment failed" penalty = pname sigmoid = sname error = sprint(showerror, err)
                failed_result(pname, sname, err)
            end
            push!(results, res)
        end
    end

    save_loss_csv(results)
    summary_rows = save_summary_csv(results)
    maybe_make_plots(results)
    write_structured_report(results, summary_rows, src)

    println("Done. Outputs written to: $(RESULTS_DIR)")
end

main()
