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

include("../src/dynamics.jl")
include("../src/setup.jl")
include("../src/penalties.jl")
include("../src/experiment.jl")
include("../src/metrics.jl")
include("../src/reporting.jl")

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

            Random.seed!(2026)

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
