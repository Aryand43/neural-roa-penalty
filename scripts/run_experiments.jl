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
include("../src/experiment.jl")
include("../src/metrics.jl")
include("../src/reporting.jl")

function main()
    println("Running source inspection...")
    src = source_inspection()
    write_source_report(src)

    println("Building controlled setup...")
    ρ_exp = 1.0
    penalties = make_penalty_list(; ρ = ρ_exp)
    # Use a single fixed seed across the entire comparison grid by default.
    # Override explicitly by setting `ROA_EXPERIMENT_SEED` in the environment.
    seed = try
        parse(Int, get(ENV, "ROA_EXPERIMENT_SEED", "2026"))
    catch
        2026
    end
    println("Experiment RNG seed: $(seed)")
    sigmoid_list = [
        ("default", hard_step_sigmoid),
        ("logistic", logistic_sigmoid(20.0)),
    ]

    results = NamedTuple[]
    adam_iters1 = 300
    adam_iters2 = 300
    for (pname, pfn, inv_V_a, inv_V_a_regime) in penalties
        for (sname, sfn) in sigmoid_list
            for log_scale in (false, true)
                # Re-seed before every trial so each configuration starts from an identical RNG state.
                rng_seed = seed
                Random.seed!(rng_seed)
                setup = build_setup(; seed = rng_seed, hidden = 32)

                println("Training penalty=$(pname), sigmoid=$(sname), log_scale=$(log_scale), rng_seed=$(rng_seed)")
                res = try
                    run_one_experiment(
                        setup, pname, pfn, sname, sfn;
                        adam_iters1,
                        adam_iters2,
                        ρ = ρ_exp,
                        log_scale = log_scale,
                        rng_seed = rng_seed,
                        inv_V_a = Float64(inv_V_a),
                        inv_V_a_regime = inv_V_a_regime,
                    )
                catch err
                    @warn "Experiment failed" penalty = pname sigmoid = sname error = sprint(showerror, err)
                    failed_result(
                        pname,
                        sname,
                        err;
                        log_scale = log_scale,
                        rng_seed = rng_seed,
                        inv_V_a = Float64(inv_V_a),
                        inv_V_a_regime = inv_V_a_regime,
                    )
                end
                push!(results, res)
            end
        end
    end

    save_loss_csv(results)
    summary_rows = save_summary_csv(results)
    maybe_make_plots(results)
    write_structured_report(results, summary_rows, src; seed = seed)

    println("Done. Outputs written to: $(RESULTS_DIR)")
end

main()
