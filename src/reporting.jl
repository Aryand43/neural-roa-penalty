const HAVE_PLOTS = let
    try
        @eval using Plots
        true
    catch
        false
    end
end

const RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results"))
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
            max_dVdt_inside = Float64(r.max_dVdt_inside),
            has_nan = Bool(r.has_nan),
            error_message = r.error_message,
        ) for r in results
    ]
    out = joinpath(RESULTS_DIR, "summary.csv")
    open(out, "w") do io
        println(io, "penalty,sigmoid,final_loss,rho,roa_area,max_dVdt_inside,has_nan,error_message")
        for row in rows
            escaped = replace(row.error_message, "\"" => "'")
            println(
                io,
                "$(row.penalty),$(row.sigmoid),$(row.final_loss),$(row.rho),$(row.roa_area),$(row.max_dVdt_inside),$(row.has_nan),\"$(escaped)\""
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
        clamped = max.(r.losses, eps(Float64))
        plot!(p_loss, r.iterations, clamped; label = "$(r.penalty) / $(r.sigmoid)", lw = 1.5)
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
    if !isempty(plt_list)
        big = plot(plt_list...; layout = (ceil(Int, length(plt_list) / 3), 3), size = (1200, 900))
        savefig(big, joinpath(RESULTS_DIR, "roa_contours.png"))
    else
        open(joinpath(RESULTS_DIR, "plot_status.txt"), "a") do io
            write(io, "No valid contour data available; skipped roa_contours.png generation.\n")
        end
    end

    labels = ["$(r.penalty)\n$(r.sigmoid)" for r in results]
    areas = [r.area for r in results]
    p_area = bar(labels, areas; xlabel = "Penalty / Sigmoid", ylabel = "Estimated RoA area", legend = false, xrotation = 45)
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

        println(io, "## Penalty Hypotheses")
        println(io)
        println(io, "| Penalty | Expression | Expected Behavior | Reasoning |")
        println(io, "|---------|------------|-------------------|-----------|")
        println(io, "| control_zero | 0 | smallest RoA estimate since there is no penalty outside the RoA | optimizer receives no gradient signal outside the RoA boundary |")
        println(io, "| constant_one | 1 | moderate RoA expansion | uniform penalty encourages RoA expansion but provides no spatial structure |")
        println(io, "| inv_dist_sq | 1 / ‖x - x₀‖² | unstable gradients near equilibrium | gradient magnitude increases rapidly near the equilibrium |")
        println(io, "| scaled_inv_dist_sq | 100 / ‖x - x₀‖² | similar to inv_dist_sq due to loss aggregation scaling | NeuralPDE residual loss may normalize effect of scale changes |")
        println(io, "| inv_dist | 1 / ‖x - x₀‖ | smoother variant of inverse distance penalty | weaker singularity compared to squared inverse distance |")
        println(io, "| inv_V | 1 / V | unstable or collapsed RoA estimate | singularity when V approaches zero |")
        println(io, "| quadratic_over_rho | max(0, V - ρ)² | well-shaped RoA boundary | directly penalizes states outside the RoA level set |")
        println(io)

        println(io, "## Per-Run Results")
        println(io)
        for row in summary_rows
            e = isempty(row.error_message) ? "none" : row.error_message
            println(io, "- penalty: `$(row.penalty)` | sigmoid: `$(row.sigmoid)` | final_loss: `$(row.final_loss)` | ρ: `$(row.rho)` | area: `$(row.roa_area)` | max_dVdt_inside: `$(row.max_dVdt_inside)` | has_nan: `$(row.has_nan)` | error: `$(e)`")
        end
        println(io)

        println(io, "## Expected vs Observed Behavior")
        println(io)

        r_cz = only(filter(r -> r.penalty == "control_zero" && r.sigmoid == "default", results))
        println(io, "### control_zero (default sigmoid)")
        println(io)
        println(io, "- **Expected:** smallest RoA estimate since there is no penalty outside the RoA")
        println(io, "- **Observed:** area = $(r_cz.area), max dV/dt inside = $(r_cz.max_dVdt_inside)")
        if r_cz.area <= minimum(r.area for r in results if isfinite(r.area))
            println(io, "- **Comparison:** consistent — control_zero produced the smallest (or tied smallest) RoA area, confirming that the absence of an out-of-RoA penalty limits expansion.")
        else
            println(io, "- **Comparison:** unexpected — control_zero did not produce the smallest RoA area; other penalties may have collapsed or failed.")
        end
        println(io)

        r_co = only(filter(r -> r.penalty == "constant_one" && r.sigmoid == "logistic", results))
        println(io, "### constant_one (logistic sigmoid)")
        println(io)
        println(io, "- **Expected:** moderate RoA expansion due to uniform penalty with smooth sigmoid gating")
        println(io, "- **Observed:** area = $(r_co.area), max dV/dt inside = $(r_co.max_dVdt_inside)")
        median_area = sort([r.area for r in results if isfinite(r.area)])[div(count(r -> isfinite(r.area), results), 2) + 1]
        if r_co.area >= median_area
            println(io, "- **Comparison:** consistent — constant_one with logistic sigmoid achieved at-or-above-median area, matching the moderate expansion expectation.")
        else
            println(io, "- **Comparison:** below median area; the uniform penalty may not provide enough spatial signal even with a smooth sigmoid.")
        end
        println(io)

        r_iv = only(filter(r -> r.penalty == "inv_V" && r.sigmoid == "default", results))
        println(io, "### inv_V (default sigmoid)")
        println(io)
        println(io, "- **Expected:** unstable or collapsed RoA estimate due to singularity when V approaches zero")
        println(io, "- **Observed:** area = $(r_iv.area), max dV/dt inside = $(r_iv.max_dVdt_inside), has_nan = $(r_iv.has_nan)")
        if r_iv.has_nan || !isfinite(r_iv.area) || r_iv.area ≈ 0.0
            println(io, "- **Comparison:** consistent — inv_V produced NaN or collapsed area, confirming the V→0 singularity destabilises training.")
        else
            println(io, "- **Comparison:** unexpected — inv_V did not collapse; the network may have learned to keep V bounded away from zero.")
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
