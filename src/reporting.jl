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
            log_scale = r.log_scale,
            final_loss = Float64(r.final_loss),
            rho = Float64(r.rho),
            roa_area = Float64(r.area),
            max_dVdt_inside = Float64(r.max_dVdt_inside),
            training_time = Float64(r.training_time),
            has_nan = Bool(r.has_nan),
            error_message = r.error_message,
        ) for r in results
    ]
    out = joinpath(RESULTS_DIR, "summary.csv")
    open(out, "w") do io
        println(io, "penalty,sigmoid,log_scale,finalloss,rho,roaarea,maxdVdtinside,trainingtime,hasnan,errormessage")
        for row in rows
            escaped = replace(row.error_message, "\"" => "'")
            println(
                io,
                row.penalty,
                ",",
                row.sigmoid,
                ",",
                row.log_scale,
                ",",
                row.final_loss,
                ",",
                row.rho,
                ",",
                row.roa_area,
                ",",
                row.max_dVdt_inside,
                ",",
                row.training_time,
                ",",
                row.has_nan,
                ",",
                "\"$(escaped)\"",
            )
        end
    end
    return rows
end

function approximate_true_roa(xs, ys; T = 20.0, dt = 0.01, bound = 5.0, tol = 0.2)
    nx, ny = length(xs), length(ys)
    mask = falses(nx, ny)
    nsteps = round(Int, T / dt)
    for j in 1:ny
        for i in 1:nx
            x = [Float64(xs[i]), Float64(ys[j])]
            escaped = false
            for _ in 1:nsteps
                dx = vanderpol(x, nothing, 0.0)
                x = x .+ dt .* dx
                if x[1]^2 + x[2]^2 > bound^2
                    escaped = true
                    break
                end
            end
            if !escaped && sqrt(x[1]^2 + x[2]^2) <= tol
                mask[i, j] = true
            end
        end
    end
    return mask
end

function plot_roa_overlay(r, true_roa_mask)
    n = length(r.xs)
    learned = reshape(r.roa_mask, n, n)
    lw = 3

    p = heatmap(
        r.xs, r.ys, Float64.(true_roa_mask)';
        color = cgrad([:white, :lightblue]),
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "$(r.penalty) | $(r.sigmoid) | area=$(round(r.area, digits=3)) | time=$(round(r.training_time, digits=2))s",
        colorbar = false
    )
    contour!(p, r.xs, r.ys, Float64.(learned)';
        levels = [0.5], color = :red, lw = lw, label = "Learned RoA (V ≤ ρ)")
    contour!(p, r.xs, r.ys, Float64.(true_roa_mask)';
        levels = [0.5], color = :blue, lw = lw - 1, ls = :dash, label = "True RoA")

    fname = "roa_overlay_$(r.penalty)_$(r.sigmoid).png"
    savefig(p, joinpath(RESULTS_DIR, fname))
    return p
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

    overlay_candidates = filter(r -> !isempty(r.roa_mask) && !isempty(r.xs) && !isempty(r.ys) && isfinite(r.rho), results)
    if !isempty(overlay_candidates)
        ref = first(overlay_candidates)
        println("Computing approximate true RoA on $(length(ref.xs))×$(length(ref.ys)) grid...")
        true_roa_mask = approximate_true_roa(ref.xs, ref.ys)
        for r in overlay_candidates
            plot_roa_overlay(r, true_roa_mask)
        end
        println("Saved $(length(overlay_candidates)) RoA overlay plots to $(RESULTS_DIR)")
    else
        open(joinpath(RESULTS_DIR, "plot_status.txt"), "a") do io
            write(io, "No valid results with xs/ys/roa_mask; skipped RoA overlay plots.\n")
        end
    end
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

    inv_default = only(filter(r -> r.penalty == "inv_dist_sq" && r.sigmoid == "default" && !r.log_scale, results))
    scaled_default = only(filter(r -> r.penalty == "scaled_inv_dist_sq" && r.sigmoid == "default" && !r.log_scale, results))
    inv_logistic = only(filter(r -> r.penalty == "inv_dist_sq" && r.sigmoid == "logistic" && !r.log_scale, results))
    scaled_logistic = only(filter(r -> r.penalty == "scaled_inv_dist_sq" && r.sigmoid == "logistic" && !r.log_scale, results))

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
        println(io, "| inv_V_small | 1 / (V + 1e-3) | stabilized inverse-V with small epsilon | avoids V→0 singularity via fixed offset |")
        println(io, "| inv_V_rho | 1 / (V + ρ) | stabilized inverse-V scaled by ρ | uses ρ as natural offset, singularity-free for V ≥ 0 |")
        println(io, "| inv_V_clipped | 1 / max(V, 0.1) | safe inverse-V via hard floor | clamps V away from zero, preventing gradient blowup entirely |")
        println(io, "| quadratic_over_rho | max(0, V - ρ)² | well-shaped RoA boundary | directly penalizes states outside the RoA level set |")
        println(io)

        println(io, "## Per-Run Results")
        println(io)
        for row in summary_rows
            e = isempty(row.error_message) ? "none" : row.error_message
            println(io, "- penalty: `$(row.penalty)` | sigmoid: `$(row.sigmoid)` | log_scale: `$(row.log_scale)` | final_loss: `$(row.final_loss)` | ρ: `$(row.rho)` | area: `$(row.roa_area)` | max_dVdt_inside: `$(row.max_dVdt_inside)` | training_time: `$(row.training_time)` | has_nan: `$(row.has_nan)` | error: `$(e)`")
        end
        println(io)

        println(io, "## Expected vs Observed Behavior")
        println(io)

        r_cz = only(filter(r -> r.penalty == "control_zero" && r.sigmoid == "default" && !r.log_scale, results))
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

        r_co = only(filter(r -> r.penalty == "constant_one" && r.sigmoid == "logistic" && !r.log_scale, results))
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

        r_ivs = only(filter(r -> r.penalty == "inv_V_small" && r.sigmoid == "default" && !r.log_scale, results))
        println(io, "### inv_V_small (default sigmoid)")
        println(io)
        println(io, "- **Expected:** stabilized inverse-V penalty with small epsilon (1e-3) avoiding V→0 singularity")
        println(io, "- **Observed:** area = $(r_ivs.area), max dV/dt inside = $(r_ivs.max_dVdt_inside), has_nan = $(r_ivs.has_nan), training_time = $(r_ivs.training_time)s")
        if r_ivs.has_nan || !isfinite(r_ivs.area)
            println(io, "- **Comparison:** training still unstable despite epsilon stabilization.")
        else
            println(io, "- **Comparison:** epsilon stabilization successful — finite area obtained without collapse.")
        end
        println(io)

        r_ivr = only(filter(r -> r.penalty == "inv_V_rho" && r.sigmoid == "default" && !r.log_scale, results))
        println(io, "### inv_V_rho (default sigmoid)")
        println(io)
        println(io, "- **Expected:** stabilized inverse-V penalty using ρ as offset, singularity-free")
        println(io, "- **Observed:** area = $(r_ivr.area), max dV/dt inside = $(r_ivr.max_dVdt_inside), has_nan = $(r_ivr.has_nan), training_time = $(r_ivr.training_time)s")
        if r_ivr.has_nan || !isfinite(r_ivr.area)
            println(io, "- **Comparison:** training still unstable despite ρ-based stabilization.")
        else
            println(io, "- **Comparison:** ρ-based stabilization successful — finite area obtained.")
        end
        println(io)
        println(io, "> **Note on inv_V instability:** Instability likely arises from interaction between penalty scaling and NeuralPDE residual formulation, where large curvature near low-V regions still destabilizes optimization despite offset.")
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
        println(io)

        println(io, "## Training Time Comparison")
        println(io)
        times_valid = filter(r -> isfinite(r.training_time), results)
        if !isempty(times_valid)
            slowest = times_valid[argmax([r.training_time for r in times_valid])]
            fastest = times_valid[argmin([r.training_time for r in times_valid])]
            mean_time = sum(r.training_time for r in times_valid) / length(times_valid)
            println(io, "| Penalty | Sigmoid | Training Time (s) |")
            println(io, "|---------|---------|-------------------|")
            for r in times_valid
                println(io, "| $(r.penalty) | $(r.sigmoid) | $(round(r.training_time, digits=2)) |")
            end
            println(io)
            println(io, "- **Fastest:** `$(fastest.penalty)` / `$(fastest.sigmoid)` at $(round(fastest.training_time, digits=2))s")
            println(io, "- **Slowest:** `$(slowest.penalty)` / `$(slowest.sigmoid)` at $(round(slowest.training_time, digits=2))s")
            println(io, "- **Mean:** $(round(mean_time, digits=2))s across $(length(times_valid)) runs")
        else
            println(io, "No valid training time data available.")
        end
    end
end
