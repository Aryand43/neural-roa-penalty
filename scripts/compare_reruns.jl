# Compare two summary.csv files (e.g. before/after a NeuralLyapunov upgrade).
#
# Usage:
#   julia --project=. scripts/compare_reruns.jl OLD.csv NEW.csv [OUT.md]
#
# Uses only Base so it runs without extra dependencies.

function read_summary(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    rows = Dict{Tuple{String, String, String}, Dict{String, String}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        # error_message is the last, quoted column and may contain commas.
        q = findfirst('"', line)
        head = q === nothing ? line : line[1:(q - 2)]
        fields = split(head, ",")
        row = Dict(String(header[i]) => String(fields[i]) for i in eachindex(fields))
        rows[(row["penalty"], row["sigmoid"], row["log_scale"])] = row
    end
    return rows
end

num(s) = something(tryparse(Float64, s), NaN)
fmt(x; d = 4) = isfinite(x) ? string(round(x, digits = d)) : "NaN"
function fmt_delta(old, new; d = 4)
    (isfinite(old) && isfinite(new)) || return "—"
    δ = new - old
    return (δ >= 0 ? "+" : "") * string(round(δ, digits = d))
end

function mean_finite(xs)
    f = filter(isfinite, xs)
    return isempty(f) ? NaN : sum(f) / length(f)
end

function main(old_path, new_path, out_path)
    old = read_summary(old_path)
    new = read_summary(new_path)
    keys_sorted = sort(collect(intersect(keys(old), keys(new))))

    open(out_path, "w") do io
        println(io, "# Rerun Comparison")
        println(io)
        println(io, "- Old: `$(old_path)`")
        println(io, "- New: `$(new_path)`")
        println(io)

        println(io, "## Per-run")
        println(io)
        println(io, "| penalty | sigmoid | log_scale | area old | area new | Δ area | max dV/dt old | max dV/dt new | time old (s) | time new (s) | NaN-grad skips new |")
        println(io, "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k in keys_sorted
            o, n = old[k], new[k]
            ao, an = num(o["roa_area"]), num(n["roa_area"])
            println(io, "| `$(k[1])` | `$(k[2])` | `$(k[3])` | $(fmt(ao)) | $(fmt(an)) | $(fmt_delta(ao, an)) | ",
                "$(fmt(num(o["max_dVdt_inside"]))) | $(fmt(num(n["max_dVdt_inside"]))) | ",
                "$(fmt(num(o["train_time_seconds"]), d = 2)) | $(fmt(num(n["train_time_seconds"]), d = 2)) | ",
                "$(get(n, "nan_grad_skips", "—")) |")
        end
        println(io)

        println(io, "## log_scale effect (log(V/ρ) − (V − ρ)) per penalty/sigmoid")
        println(io)
        println(io, "| penalty | sigmoid | Δ area old | Δ area new | Δ time old (s) | Δ time new (s) |")
        println(io, "|---|---|---:|---:|---:|---:|")
        pairs = sort(unique([(k[1], k[2]) for k in keys_sorted]))
        for (p, s) in pairs
            haskey(old, (p, s, "true")) && haskey(old, (p, s, "false")) || continue
            haskey(new, (p, s, "true")) && haskey(new, (p, s, "false")) || continue
            d(tbl, col) = num(tbl[(p, s, "true")][col]) - num(tbl[(p, s, "false")][col])
            println(io, "| `$(p)` | `$(s)` | $(fmt(d(old, "roa_area"))) | $(fmt(d(new, "roa_area"))) | ",
                "$(fmt(d(old, "train_time_seconds"), d = 2)) | $(fmt(d(new, "train_time_seconds"), d = 2)) |")
        end
        println(io)

        println(io, "## Timing summary")
        println(io)
        for (label, tbl) in (("old", old), ("new", new))
            t = [num(tbl[k]["train_time_seconds"]) for k in keys_sorted]
            tf = [num(tbl[k]["train_time_seconds"]) for k in keys_sorted if k[3] == "false"]
            tt = [num(tbl[k]["train_time_seconds"]) for k in keys_sorted if k[3] == "true"]
            println(io, "- $(label): mean $(fmt(mean_finite(t), d = 2))s over all runs; ",
                "log_scale=false mean $(fmt(mean_finite(tf), d = 2))s; log_scale=true mean $(fmt(mean_finite(tt), d = 2))s; ",
                "max $(fmt(maximum(filter(isfinite, t); init = -Inf), d = 2))s")
        end
    end
    println("Wrote $(out_path)")
end

length(ARGS) >= 2 || error("usage: compare_reruns.jl OLD.csv NEW.csv [OUT.md]")
main(ARGS[1], ARGS[2], length(ARGS) >= 3 ? ARGS[3] : joinpath(dirname(ARGS[2]), "rerun_comparison.md"))
