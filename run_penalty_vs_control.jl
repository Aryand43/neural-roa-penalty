using Random
using StableRNGs
using Printf
using LinearAlgebra
using NNlib
using Lux
using NeuralPDE
using NeuralLyapunov
using DataFrames
using Plots
using OptimizationOptimisers: Adam
using OptimizationOptimJL: BFGS
using OrdinaryDiffEq: ODEProblem, Tsit5, solve, EnsembleSerial
import Boltz.Layers: MLP

const SEED = 200
const MU = 1.0
const RHO = 1.0
const X_EQ = [0.0, 0.0]

const LB = [-3.0, -3.0]
const UB = [3.0, 3.0]

const TRAIN_STRATEGY = QuasiRandomTraining(1500)
const OPT_SCHEDULE = [Adam(1.0e-3), BFGS()]
const OPT_ARGS = [[:maxiters => 120], [:maxiters => 120]]

const GRID_N = 61
const CLASSIFY_SIM_TIME = 20.0
const CLASSIFY_ATOL = 0.15

control_penalty = (V, Vdot, x, x_eq, ρ) -> 0.0
constant_penalty = (V, Vdot, x, x_eq, ρ) -> 1.0

vanderpol(x, p, t) = [x[2], p[1] * (1.0 - x[1]^2) * x[2] - x[1]]

function scalar(v)
    return v isa AbstractArray ? v[] : v
end

function build_chain()
    ϕ = MLP(2, (32, 32, 8), NNlib.hardsigmoid)
    return [AdditiveLyapunovNet(ϕ; dim_ϕ = 8, fixed_point = X_EQ)]
end

function on_true_roa(x)
    prob = ODEProblem(vanderpol, x, (0.0, CLASSIFY_SIM_TIME), [MU])
    sol = solve(prob, Tsit5(); save_everystep = false)
    return norm(sol.u[end] .- X_EQ) <= CLASSIFY_ATOL
end

function run_experiment(run_name, penalty_fn, truth_grid, x1s, x2s, init_ps, init_st)
    spec = NeuralLyapunovSpecification(
        NoAdditionalStructure(),
        DontCheckNonnegativity(),
        make_RoA_aware(
            AsymptoticStability();
            ρ = RHO,
            out_of_RoA_penalty = penalty_fn,
        ),
    )

    chain = build_chain()
    out = benchmark(
        vanderpol,
        LB,
        UB,
        spec,
        chain,
        TRAIN_STRATEGY,
        OPT_SCHEDULE;
        n = 1200,
        classifier = (V, Vdot, x) -> V <= RHO,
        fixed_point = X_EQ,
        p = [MU],
        optimization_args = OPT_ARGS,
        simulation_time = CLASSIFY_SIM_TIME,
        init_params = deepcopy(init_ps),
        init_states = deepcopy(init_st),
        rng = StableRNG(SEED),
        ensemble_alg = EnsembleSerial(),
        log_frequency = 1,
    )

    V_grid = zeros(length(x2s), length(x1s))
    pred_grid = falses(length(x2s), length(x1s))
    for (j, x2) in enumerate(x2s), (i, x1) in enumerate(x1s)
        v = scalar(out.V([x1, x2]))
        V_grid[j, i] = v
        pred_grid[j, i] = v <= RHO
    end

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for idx in eachindex(pred_grid)
        p = pred_grid[idx]
        t = truth_grid[idx]
        if p && t
            tp += 1
        elseif p && !t
            fp += 1
        elseif !p && t
            fn += 1
        else
            tn += 1
        end
    end

    dx = x1s[2] - x1s[1]
    dy = x2s[2] - x2s[1]
    cell_area = dx * dy
    learned_area = count(pred_grid) * cell_area
    true_area = count(truth_grid) * cell_area
    intersection_area = count(pred_grid .& truth_grid) * cell_area
    overestimation_area = count(pred_grid .& .!truth_grid) * cell_area
    underestimation_area = count((.!pred_grid) .& truth_grid) * cell_area

    run_dir = joinpath("results", run_name)
    mkpath(run_dir)

    p1 = contour(
        x1s,
        x2s,
        V_grid;
        levels = [RHO],
        color = :dodgerblue3,
        linewidth = 2,
        linestyle = :solid,
        label = "Learned V(x)=ρ",
        xlabel = "x1",
        ylabel = "x2",
        title = "Learned V(x) = ρ contour (ρ = $(RHO))",
        legend = :topright,
    )
    contour!(
        p1,
        x1s,
        x2s,
        truth_grid;
        levels = [0.5],
        color = :black,
        linewidth = 2,
        linestyle = :dash,
        label = "True RoA boundary",
    )
    scatter!(
        p1,
        [X_EQ[1]],
        [X_EQ[2]];
        marker = :star5,
        markersize = 8,
        color = :red,
        label = "Equilibrium",
    )
    savefig(p1, joinpath(run_dir, "vanderpol_v_contour.png"))

    p2 = plot(
        out.training_losses[!, "Iteration"],
        out.training_losses[!, "Loss"];
        yaxis = :log,
        xlabel = "Iteration",
        ylabel = "Loss",
        title = "Training Loss",
        legend = false,
    )
    savefig(p2, joinpath(run_dir, "loss_curve.png"))

    open(joinpath(run_dir, "confusion_matrix.txt"), "w") do io
        println(io, "TP $tp")
        println(io, "FP $fp")
        println(io, "FN $fn")
        println(io, "TN $tn")
    end

    open(joinpath(run_dir, "metrics.txt"), "w") do io
        println(io, "seed: $SEED")
        println(io, "rho: $RHO")
        println(io, @sprintf("learned_area: %.8f", learned_area))
        println(io, @sprintf("true_area: %.8f", true_area))
        println(io, @sprintf("intersection_area: %.8f", intersection_area))
        println(io, @sprintf("overestimation_area: %.8f", overestimation_area))
        println(io, @sprintf("underestimation_area: %.8f", underestimation_area))
    end

    over_pct = learned_area > 0 ? 100 * (overestimation_area / learned_area) : 0.0
    under_pct = true_area > 0 ? 100 * (underestimation_area / true_area) : 0.0
    println("[$run_name] Learned area: ", @sprintf("%.8f", learned_area))
    println("[$run_name] True area: ", @sprintf("%.8f", true_area))
    println("[$run_name] Overestimation %: ", @sprintf("%.2f%%", over_pct))
    println("[$run_name] Underestimation %: ", @sprintf("%.2f%%", under_pct))

    return (
        learned_area = learned_area,
        true_area = true_area,
        intersection_area = intersection_area,
        overestimation_area = overestimation_area,
        underestimation_area = underestimation_area,
    )
end

function main()
    Random.seed!(SEED)

    chain = build_chain()
    init_ps, init_st = Lux.setup(StableRNG(SEED), chain)
    init_ps = init_ps |> f64
    init_st = init_st |> f64

    x1s = collect(range(LB[1], UB[1]; length = GRID_N))
    x2s = collect(range(LB[2], UB[2]; length = GRID_N))
    truth_grid = falses(length(x2s), length(x1s))
    for (j, x2) in enumerate(x2s), (i, x1) in enumerate(x1s)
        truth_grid[j, i] = on_true_roa([x1, x2])
    end

    control_metrics = run_experiment(
        "control",
        control_penalty,
        truth_grid,
        x1s,
        x2s,
        init_ps,
        init_st,
    )
    constant_metrics = run_experiment(
        "constant",
        constant_penalty,
        truth_grid,
        x1s,
        x2s,
        init_ps,
        init_st,
    )

    println("Control learned area: ", @sprintf("%.8f", control_metrics.learned_area))
    println("Constant learned area: ", @sprintf("%.8f", constant_metrics.learned_area))
    println(
        "Learned area difference: ",
        @sprintf("%.8f", constant_metrics.learned_area - control_metrics.learned_area),
    )
end

main()
