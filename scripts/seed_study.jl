# Paired multi-seed comparison of the two configurations that came out best on the 36-run
# grid: `inv_dist_sq` and `constant_one`, both with the logistic gate and `log_scale = false`.
#
# Usage:
#   julia --project=. scripts/seed_study.jl [N_SEEDS] [FIRST_SEED]
#
# Why this is a *paired* study: a seed fixes the network initialization, the
# `QuasiRandomTraining` samples and `benchmark`'s evaluation samples, and both
# configurations are run on every seed, so the within-seed difference is attributable to
# the penalty alone. Within a seed the two are run in alternating order (by seed parity) so
# machine drift (thermal throttling, background load) cannot load systematically onto one of
# them. An untimed warm-up precedes everything, as in `scripts/run_experiments.jl`, so JIT
# compilation is not charged to the first measured run.

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
using Boltz.Layers: MLP
using Lux

include("../src/dynamics.jl")
include("../src/setup.jl")
include("../src/penalties.jl")
include("../src/grad_skip_counter.jl")
include("../src/experiment.jl")
include("../src/metrics.jl")

const RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "results"))
mkpath(RESULTS_DIR)

const RHO = 1.0
const HIDDEN = 32
# Iteration counts match the grid. `ROA_SEED_STUDY_ITERS` shrinks both stages so the whole
# pipeline (including the report writer) can be smoke-tested in under a minute.
const ADAM_ITERS1 = parse(Int, get(ENV, "ROA_SEED_STUDY_ITERS", "300"))
const ADAM_ITERS2 = parse(Int, get(ENV, "ROA_SEED_STUDY_ITERS", "300"))
const SIGMOID_K = 20.0
# `V̇(x_eq) = 0` exactly (the dynamics have a fixed point there) and the origin is a grid
# point, so `max_dVdt_inside` is bounded below by 0 and a tiny positive value may just be a
# near-origin point where V̇ must vanish. Report the max outside this ball as well, so a
# genuine decrease-condition violation is distinguishable from the origin neighbourhood.
const EXCLUDE_RADIUS = 0.05
# Order matters for the report: deltas below are B − A, i.e. `constant_one − inv_dist_sq`.
const WANTED = ("inv_dist_sq", "constant_one")

function selected_configs()
    penalties = make_penalty_list(; ρ = RHO)
    out = NamedTuple[]
    for name in WANTED
        idx = findfirst(p -> p[1] == name, penalties)
        idx === nothing && error("penalty $(name) not found in make_penalty_list")
        pname, pfn, inv_V_a, regime = penalties[idx]
        push!(out, (;
            penalty = pname,
            pfn,
            inv_V_a = Float64(inv_V_a),
            inv_V_a_regime = String(regime),
            sigmoid = "logistic",
            sfn = logistic_sigmoid(SIGMOID_K),
            log_scale = false,
        ))
    end
    return out
end

############################## dV/dt grid diagnostics ##############################

function dVdt_diagnostics(r; exclude_radius = EXCLUDE_RADIUS)
    nan_result = (; argmax_x = NaN, argmax_y = NaN, argmax_norm = NaN, max_excl = NaN)
    (isempty(r.xs) || isempty(r.dVvals)) && return nan_result
    xs, ys = r.xs, r.ys
    nx = length(xs)
    best_k = 0
    best_v = -Inf
    excl_best = -Inf
    for k in eachindex(r.dVvals)
        r.roa_mask[k] || continue
        v = r.dVvals[k]
        isfinite(v) || continue
        if v > best_v
            best_v = v
            best_k = k
        end
        # `compute_grid_metrics` lays points out as `for y in ys, x in xs`, so the linear
        # index of (xs[i], ys[j]) is i + (j - 1) * length(xs).
        if hypot(xs[mod1(k, nx)], ys[cld(k, nx)]) >= exclude_radius && v > excl_best
            excl_best = v
        end
    end
    best_k == 0 && return nan_result
    ax = xs[mod1(best_k, nx)]
    ay = ys[cld(best_k, nx)]
    return (;
        argmax_x = ax,
        argmax_y = ay,
        argmax_norm = hypot(ax, ay),
        max_excl = isfinite(excl_best) ? excl_best : NaN,
    )
end

############################## statistics (dependency-free) ##############################

# Average ranks for tied values, as the Wilcoxon signed-rank test requires.
function midranks(x)
    n = length(x)
    p = sortperm(x)
    r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[p[j + 1]] == x[p[i]]
            j += 1
        end
        avg = (i + j) / 2
        for k in i:j
            r[p[k]] = avg
        end
        i = j + 1
    end
    return r
end

# Exact two-sided Wilcoxon signed-rank test on paired differences (zeros dropped, as is
# standard). Enumerates every sign assignment, which is exact and fast at these seed counts
# and avoids pulling in HypothesisTests/Distributions.
function wilcoxon_signed_rank(d)
    nz = filter(x -> isfinite(x) && !iszero(x), collect(d))
    n = length(nz)
    n == 0 && return (; n = 0, W = NaN, p = NaN, note = "all differences exactly zero")
    n > 22 && return (; n, W = NaN, p = NaN, note = "n > 22; exact enumeration skipped")
    r = midranks(abs.(nz))
    W = sum(Float64[r[k] for k in 1:n if nz[k] > 0]; init = 0.0)
    ge = 0
    le = 0
    for mask in 0:(2^n - 1)
        s = 0.0
        m = mask
        k = 1
        while m != 0
            (m & 1) == 1 && (s += r[k])
            m >>= 1
            k += 1
        end
        s >= W - 1.0e-9 && (ge += 1)
        s <= W + 1.0e-9 && (le += 1)
    end
    return (; n, W, p = min(1.0, 2 * min(ge, le) / 2^n), note = "exact")
end

# Exact two-sided sign test: of the seeds where the two differ, how lopsided is the split?
function sign_test(d)
    nz = filter(x -> isfinite(x) && !iszero(x), collect(d))
    n = length(nz)
    n == 0 && return (; n = 0, pos = 0, p = NaN)
    pos = count(>(0), nz)
    k = max(pos, n - pos)
    tail = sum(binomial(big(n), big(j)) for j in k:n)
    return (; n, pos, p = min(1.0, 2 * Float64(tail / big(2)^n)))
end

function bootstrap_ci(d; n_boot = 20_000, level = 0.95, seed = 1)
    x = filter(isfinite, collect(d))
    isempty(x) && return (NaN, NaN)
    rng = StableRNG(seed)
    n = length(x)
    means = sort!([mean(@view x[rand(rng, 1:n, n)]) for _ in 1:n_boot])
    α = (1 - level) / 2
    return (means[max(1, floor(Int, α * n_boot))], means[min(n_boot, ceil(Int, (1 - α) * n_boot))])
end

mean_f(x) = (f = filter(isfinite, collect(x)); isempty(f) ? NaN : mean(f))
max_f(x) = (f = filter(isfinite, collect(x)); isempty(f) ? NaN : maximum(f))
sd_f(x) = (f = filter(isfinite, collect(x)); length(f) > 1 ? std(f) : NaN)
median_f(x) = (f = filter(isfinite, collect(x)); isempty(f) ? NaN : median(f))

############################## run bookkeeping ##############################

function summarize_run(r, seed, order)
    d = dVdt_diagnostics(r)
    tail = isempty(r.losses) ? Float64[] : r.losses[max(1, end - 99):end]
    return (;
        penalty = r.penalty,
        sigmoid = r.sigmoid,
        log_scale = r.log_scale,
        seed = seed,
        order_index = order,
        roa_area = Float64(r.area),
        max_dVdt_inside = Float64(r.max_dVdt_inside),
        argmax_x = Float64(d.argmax_x),
        argmax_y = Float64(d.argmax_y),
        argmax_norm = Float64(d.argmax_norm),
        max_dVdt_excl_origin = Float64(d.max_excl),
        train_time_seconds = Float64(r.training_time),
        eval_time_seconds = Float64(r.evaluation_time),
        final_loss = Float64(r.final_loss),
        loss_mean_last100 = isempty(tail) ? NaN : mean(tail),
        loss_sd_last100 = length(tail) > 1 ? std(tail) : NaN,
        tp = r.tp,
        fp = r.fp,
        tn = r.tn,
        fn = r.fn,
        classifier_accuracy = Float64(r.accuracy),
        nan_grad_skips = r.nan_grad_skips,
        has_nan = Bool(r.has_nan),
        error_message = r.error_message,
    )
end

const CSV_COLUMNS = (
    :penalty, :sigmoid, :log_scale, :seed, :order_index, :roa_area, :max_dVdt_inside,
    :argmax_x, :argmax_y, :argmax_norm, :max_dVdt_excl_origin, :train_time_seconds,
    :eval_time_seconds, :final_loss, :loss_mean_last100, :loss_sd_last100,
    :tp, :fp, :tn, :fn, :classifier_accuracy, :nan_grad_skips, :has_nan,
)

function write_seed_csv(rows)
    out = joinpath(RESULTS_DIR, "seed_study.csv")
    open(out, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","), ",error_message")
        for row in rows
            print(io, join([string(getproperty(row, c)) for c in CSV_COLUMNS], ","))
            println(io, ",\"", replace(row.error_message, "\"" => "'"), "\"")
        end
    end
    println("Wrote $(out)")
    return out
end

fmt(x; d = 4) = isfinite(x) ? string(round(x, digits = d)) : "NaN"
fmt_signed(x; d = 4) = isfinite(x) ? (x >= 0 ? "+" : "") * string(round(x, digits = d)) : "NaN"

function write_seed_report(rows, seeds, cfgs)
    A, B = cfgs[1].penalty, cfgs[2].penalty
    pick(p) = Dict(r.seed => r for r in rows if r.penalty == p)
    a, b = pick(A), pick(B)
    paired = [s for s in seeds if haskey(a, s) && haskey(b, s)]

    darea = [b[s].roa_area - a[s].roa_area for s in paired]
    dtime = [b[s].train_time_seconds - a[s].train_time_seconds for s in paired]
    # One 201x201 cell; area differences below this are invisible to the metric.
    cell = 0.0004

    out = joinpath(RESULTS_DIR, "seed_study.md")
    open(out, "w") do io
        println(io, "# Seed study: `$(A)` vs `$(B)`")
        println(io)
        println(io, "Paired comparison of the two configurations that tied for largest RoA area on the ",
            "36-run grid. Both use the logistic gate (k = $(SIGMOID_K)) and `log_scale = false`; ",
            "everything else matches `scripts/run_experiments.jl` (Van der Pol, 3x$(HIDDEN) tanh MLP + ",
            "`AdditiveLyapunovNet`, `QuasiRandomTraining(1024)`, `Adam(0.01)` x$(ADAM_ITERS1) then ",
            "`Adam()` x$(ADAM_ITERS2), rho = $(RHO), `NeuralLyapunov.benchmark` for training and evaluation).")
        println(io)
        println(io, "- Seeds: $(first(seeds))-$(last(seeds)) ($(length(paired)) paired seeds, $(length(rows)) runs)")
        println(io, "- Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM"))")
        println(io, "- All deltas below are **`$(B)` - `$(A)`**, so a positive time delta means `$(B)` is slower.")
        println(io, "- Area is counted on the 201x201 grid, so it is quantized in steps of one cell = $(cell).")
        println(io)

        println(io, "## Per-seed")
        println(io)
        println(io, "| seed | area `$(A)` | area `$(B)` | Δ area | max dV/dt in `$(A)` | max dV/dt in `$(B)` | train `$(A)` (s) | train `$(B)` (s) | Δ train (s) | skips A/B |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in paired
            ra, rb = a[s], b[s]
            println(io, "| $(s) | $(fmt(ra.roa_area)) | $(fmt(rb.roa_area)) | $(fmt_signed(rb.roa_area - ra.roa_area)) | ",
                "$(fmt(ra.max_dVdt_inside)) | $(fmt(rb.max_dVdt_inside)) | ",
                "$(fmt(ra.train_time_seconds, d = 2)) | $(fmt(rb.train_time_seconds, d = 2)) | ",
                "$(fmt_signed(rb.train_time_seconds - ra.train_time_seconds, d = 2)) | ",
                "$(ra.nan_grad_skips)/$(rb.nan_grad_skips) |")
        end
        println(io)

        for (label, vals, unit, d) in (("RoA area", darea, "", 4), ("Training time", dtime, " s", 2))
            xa = [getproperty(a[s], label == "RoA area" ? :roa_area : :train_time_seconds) for s in paired]
            xb = [getproperty(b[s], label == "RoA area" ? :roa_area : :train_time_seconds) for s in paired]
            lo, hi = bootstrap_ci(vals)
            w = wilcoxon_signed_rank(vals)
            st = sign_test(vals)
            println(io, "## $(label)")
            println(io)
            println(io, "- `$(A)`: mean $(fmt(mean_f(xa), d = d))$(unit), sd $(fmt(sd_f(xa), d = d))$(unit), median $(fmt(median_f(xa), d = d))$(unit), range $(fmt(minimum(xa), d = d))-$(fmt(maximum(xa), d = d))$(unit)")
            println(io, "- `$(B)`: mean $(fmt(mean_f(xb), d = d))$(unit), sd $(fmt(sd_f(xb), d = d))$(unit), median $(fmt(median_f(xb), d = d))$(unit), range $(fmt(minimum(xb), d = d))-$(fmt(maximum(xb), d = d))$(unit)")
            println(io, "- Paired Δ (`$(B)` - `$(A)`): mean $(fmt_signed(mean_f(vals), d = d))$(unit), sd $(fmt(sd_f(vals), d = d))$(unit), median $(fmt_signed(median_f(vals), d = d))$(unit)")
            println(io, "- Bootstrap 95% CI for the mean Δ: [$(fmt(lo, d = d)), $(fmt(hi, d = d))]$(unit)")
            println(io, "- Sign test: `$(B)` larger in $(st.pos) of $(st.n) non-tied seeds (p = $(fmt(st.p, d = 4)))")
            println(io, "- Wilcoxon signed-rank (exact, n = $(w.n)): W = $(fmt(w.W, d = 1)), p = $(fmt(w.p, d = 4)) [$(w.note)]")
            if label == "RoA area"
                same = count(x -> abs(x) < cell / 2, vals)
                println(io, "- Identical to within one grid cell in **$(same) of $(length(vals))** seeds")
            end
            println(io)
        end

        println(io, "## Decrease condition inside the claimed set")
        println(io)
        println(io, "`max_dVdt_inside` is the max of V̇ over `{V <= rho}` on the grid. V̇ vanishes at the ",
            "origin, which is itself a grid point, so this quantity can never be negative and a tiny ",
            "positive value may only reflect the origin neighbourhood. The column below excludes ",
            "||x|| < $(EXCLUDE_RADIUS) to separate that from a genuine violation.")
        println(io)
        println(io, "| penalty | mean max dV/dt | max over seeds | mean max dV/dt excl. origin ball | max excl. over seeds | seeds with excl. max > 1e-3 | mean ||argmax|| |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for p in (A, B)
            rs = [r for r in rows if r.penalty == p]
            mx = [r.max_dVdt_inside for r in rs]
            ex = [r.max_dVdt_excl_origin for r in rs]
            println(io, "| `$(p)` | $(fmt(mean_f(mx), d = 5)) | $(fmt(max_f(mx), d = 5)) | ",
                "$(fmt(mean_f(ex), d = 5)) | $(fmt(max_f(ex), d = 5)) | ",
                "$(count(x -> isfinite(x) && x > 1.0e-3, ex))/$(length(rs)) | ",
                "$(fmt(mean_f([r.argmax_norm for r in rs]), d = 3)) |")
        end
        println(io)

        println(io, "## Classifier, loss and NaN gradients")
        println(io)
        println(io, "| penalty | mean accuracy | mean tp | mean fp | mean loss (last 100 iters) | mean within-run loss sd | total NaN-grad skips | mean eval time (s) |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
        for p in (A, B)
            rs = [r for r in rows if r.penalty == p]
            println(io, "| `$(p)` | $(fmt(mean_f([r.classifier_accuracy for r in rs]), d = 4)) | ",
                "$(fmt(mean_f([Float64(r.tp) for r in rs]), d = 1)) | ",
                "$(fmt(mean_f([Float64(r.fp) for r in rs]), d = 1)) | ",
                "$(fmt(mean_f([r.loss_mean_last100 for r in rs]), d = 4)) | ",
                "$(fmt(mean_f([r.loss_sd_last100 for r in rs]), d = 4)) | ",
                "$(sum(r.nan_grad_skips for r in rs)) | ",
                "$(fmt(mean_f([r.eval_time_seconds for r in rs]), d = 3)) |")
        end
        println(io)

        println(io, "## Timing drift check")
        println(io)
        println(io, "Runs are listed in execution order; the two configurations alternate which goes ",
            "first, so a trend here would show up as drift rather than as a penalty effect.")
        println(io)
        first_half = [r.train_time_seconds for r in rows if r.order_index <= length(rows) ÷ 2]
        second_half = [r.train_time_seconds for r in rows if r.order_index > length(rows) ÷ 2]
        println(io, "- First half of runs: mean $(fmt(mean_f(first_half), d = 2)) s; second half: mean $(fmt(mean_f(second_half), d = 2)) s")
        ran_first = [r.train_time_seconds for r in rows if isodd(r.order_index)]
        ran_second = [r.train_time_seconds for r in rows if iseven(r.order_index)]
        println(io, "- Ran first within its seed: mean $(fmt(mean_f(ran_first), d = 2)) s; ran second: mean $(fmt(mean_f(ran_second), d = 2)) s")
    end
    println("Wrote $(out)")
    return out
end

############################## main ##############################

function main()
    n_seeds = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
    first_seed = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2026
    seeds = collect(first_seed:(first_seed + n_seeds - 1))
    cfgs = selected_configs()
    total = n_seeds * length(cfgs)

    println("Seed study: ", join(["$(c.penalty)" for c in cfgs], " vs "),
        "  (sigmoid = logistic(k=$(SIGMOID_K)), log_scale = false)")
    println("Seeds $(first(seeds))-$(last(seeds)): $(n_seeds) seeds x $(length(cfgs)) configs = $(total) runs")
    println("Started $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

    println("Warm-up (untimed, results discarded)...")
    for c in cfgs
        Random.seed!(first_seed)
        warm = build_setup(; seed = first_seed, hidden = HIDDEN)
        run_one_experiment(warm, "warmup", c.pfn, "warmup", c.sfn;
            adam_iters1 = 2, adam_iters2 = 2, ρ = RHO, log_scale = c.log_scale, n_eval = 20)
    end

    rows = NamedTuple[]
    order = 0
    for s in seeds
        # Alternate which configuration runs first so machine drift cannot load onto one.
        ordered = iseven(s - first_seed) ? cfgs : reverse(cfgs)
        for c in ordered
            order += 1
            Random.seed!(s)
            setup = build_setup(; seed = s, hidden = HIDDEN)
            @printf("[%2d/%2d] seed=%d penalty=%s\n", order, total, s, c.penalty)
            r = try
                run_one_experiment(
                    setup, c.penalty, c.pfn, c.sigmoid, c.sfn;
                    adam_iters1 = ADAM_ITERS1,
                    adam_iters2 = ADAM_ITERS2,
                    ρ = RHO,
                    log_scale = c.log_scale,
                    rng_seed = s,
                    inv_V_a = c.inv_V_a,
                    inv_V_a_regime = c.inv_V_a_regime,
                )
            catch err
                @warn "Run failed" seed = s penalty = c.penalty error = sprint(showerror, err)
                failed_result(c.penalty, c.sigmoid, err; log_scale = c.log_scale, rng_seed = s,
                    inv_V_a = c.inv_V_a, inv_V_a_regime = c.inv_V_a_regime)
            end
            row = summarize_run(r, s, order)
            push!(rows, row)
            @printf("        area=%.4f  max dV/dt in=%.5g (excl origin %.5g)  train=%.2fs  skips=%d\n",
                row.roa_area, row.max_dVdt_inside, row.max_dVdt_excl_origin,
                row.train_time_seconds, row.nan_grad_skips)
        end
    end

    write_seed_csv(rows)
    write_seed_report(rows, seeds, cfgs)
    println("Finished $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
end

main()
