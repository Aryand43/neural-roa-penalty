using Flux
using NNlib
using Optim
using Plots
using Zygote
using Random
using LinearAlgebra
using Printf

const SEED = 200
const MU = 1.0
const RHO = 1.0
const X_EQ = [0.0, 0.0]

const TRAIN_SAMPLES = 3000
const SAMPLE_BOUND = 3.0
const BATCH_SIZE = 256
const ADAM_EPOCHS = 60
const ADAM_LR = 1e-3
const BFGS_ITERS = 80

const GRID_BOUND = 3.0
const GRID_N = 81
const SIM_DT = 0.01
const SIM_STEPS = 1200
const CONVERGENCE_TOL = 0.15
const DIVERGENCE_TOL = 30.0

const FD_EPS = 1e-3

control_penalty = (V, Vdot, x, x_eq, rho) -> 0.0
constant_penalty = (V, Vdot, x, x_eq, rho) -> 1.0

function vdp_dynamics(x::AbstractVector{<:Real})
    x1, x2 = x
    return [x2, MU * (1.0 - x1^2) * x2 - x1]
end

function build_model()
    Random.seed!(SEED)
    m = Chain(
        Dense(2, 32, NNlib.hardsigmoid),
        Dense(32, 32, NNlib.hardsigmoid),
        Dense(32, 1),
    )
    return Flux.f64(m)
end

function V_value(model, x::AbstractVector{<:Real}, x_eq::AbstractVector{<:Real})
    v_raw = model(x)[1] - model(x_eq)[1]
    return NNlib.softplus(v_raw) + 1e-3 * sum(abs2, x .- x_eq)
end

function Vdot_value(model, x::AbstractVector{<:Real}, x_eq::AbstractVector{<:Real})
    f = vdp_dynamics(x)
    x_forward = x .+ FD_EPS .* f
    v = V_value(model, x, x_eq)
    v_forward = V_value(model, x_forward, x_eq)
    return (v_forward - v) / FD_EPS
end

function sample_training_points(seed::Int, n::Int, bound::Float64)
    rng = MersenneTwister(seed)
    x1 = rand(rng, n) .* (2 * bound) .- bound
    x2 = rand(rng, n) .* (2 * bound) .- bound
    return permutedims(hcat(x1, x2))
end

function build_fixed_batches(seed::Int, n_samples::Int, batch_size::Int, epochs::Int)
    rng = MersenneTwister(seed + 1)
    batches = Vector{Vector{Int}}()
    for _ in 1:epochs
        perm = randperm(rng, n_samples)
        for i in 1:batch_size:n_samples
            j = min(i + batch_size - 1, n_samples)
            push!(batches, perm[i:j])
        end
    end
    return batches
end

function batch_loss(model, X_batch, penalty_fn, rho)
    n = size(X_batch, 2)
    inside_loss = 0.0
    outside_loss = 0.0
    positivity_loss = 0.0
    for i in 1:n
        x = X_batch[:, i]
        V = V_value(model, x, X_EQ)
        Vdot = Vdot_value(model, x, X_EQ)
        inside_weight = max(rho - V, 0.0)
        inside_violation = max(Vdot + 0.05, 0.0)^2
        out_margin = max(V - rho, 0.0)^2
        outside_violation = penalty_fn(V, Vdot, x, X_EQ, rho) * out_margin
        positivity_violation = max(1e-3 * sum(abs2, x .- X_EQ) - V, 0.0)^2
        inside_loss += inside_weight * inside_violation
        outside_loss += outside_violation
        positivity_loss += positivity_violation
    end
    eq_loss = V_value(model, X_EQ, X_EQ)^2
    return (inside_loss + 0.2 * outside_loss + 0.1 * positivity_loss) / n + 10.0 * eq_loss
end

function rk4_step(x, dt)
    k1 = vdp_dynamics(x)
    k2 = vdp_dynamics(x .+ 0.5 .* dt .* k1)
    k3 = vdp_dynamics(x .+ 0.5 .* dt .* k2)
    k4 = vdp_dynamics(x .+ dt .* k3)
    return x .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function converges_to_equilibrium(x0)
    x = copy(x0)
    for _ in 1:SIM_STEPS
        x = rk4_step(x, SIM_DT)
        if norm(x) > DIVERGENCE_TOL
            return false
        end
    end
    return norm(x .- X_EQ) <= CONVERGENCE_TOL
end

function make_grid(bound::Float64, n::Int)
    xs = collect(range(-bound, bound; length = n))
    ys = collect(range(-bound, bound; length = n))
    points = Vector{Vector{Float64}}(undef, n * n)
    idx = 1
    for y in ys, x in xs
        points[idx] = [x, y]
        idx += 1
    end
    return xs, ys, points
end

function compute_true_labels(points)
    labels = BitVector(undef, length(points))
    for i in eachindex(points)
        labels[i] = converges_to_equilibrium(points[i])
    end
    return labels
end

function evaluate_model(model, xs, ys, points, true_labels, rho)
    pred_labels = BitVector(undef, length(points))
    V_grid = zeros(length(ys), length(xs))
    idx = 1
    for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        p = [x, y]
        V = V_value(model, p, X_EQ)
        V_grid[j, i] = V
        pred_labels[idx] = V <= rho
        idx += 1
    end

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in eachindex(pred_labels)
        if pred_labels[i] && true_labels[i]
            tp += 1
        elseif pred_labels[i] && !true_labels[i]
            fp += 1
        elseif !pred_labels[i] && true_labels[i]
            fn += 1
        else
            tn += 1
        end
    end

    delta = xs[2] - xs[1]
    area = count(pred_labels) * delta * delta
    return V_grid, (tp = tp, fp = fp, fn = fn, tn = tn), area
end

function save_outputs(out_dir, xs, ys, V_grid, losses, confusion, area)
    mkpath(out_dir)

    p1 = contour(
        xs,
        ys,
        V_grid;
        levels = [RHO],
        linewidth = 2,
        xlabel = "x1",
        ylabel = "x2",
        title = "Van der Pol V(x)=rho contour",
        legend = false,
    )
    scatter!(p1, [X_EQ[1]], [X_EQ[2]]; markersize = 4)
    savefig(p1, joinpath(out_dir, "vanderpol_v_contour.png"))

    p2 = plot(
        1:length(losses),
        losses;
        xlabel = "iteration",
        ylabel = "loss",
        yscale = :log10,
        title = "Training loss",
        legend = false,
    )
    savefig(p2, joinpath(out_dir, "loss_curve.png"))

    open(joinpath(out_dir, "confusion_matrix.txt"), "w") do io
        println(io, "TP $(confusion.tp)")
        println(io, "FP $(confusion.fp)")
        println(io, "FN $(confusion.fn)")
        println(io, "TN $(confusion.tn)")
    end

    open(joinpath(out_dir, "metrics.txt"), "w") do io
        println(io, "seed: $SEED")
        println(io, "rho: $RHO")
        println(io, @sprintf("roa_area: %.8f", area))
    end
end

function train_and_evaluate(run_name, penalty_fn, X_train, fixed_batches, xs, ys, grid_points, true_labels)
    model = build_model()
    losses = Float64[]

    adam = Flux.setup(Flux.Adam(ADAM_LR), model)
    for batch_idx in fixed_batches
        Xb = X_train[:, batch_idx]
        loss, grads = Flux.withgradient(model) do m
            batch_loss(m, Xb, penalty_fn, RHO)
        end
        Flux.update!(adam, model, grads[1])
        push!(losses, loss)
    end

    theta0, re = Flux.destructure(model)
    objective(theta) = batch_loss(re(theta), X_train, penalty_fn, RHO)

    bfgs_losses = Float64[]
    function fg!(F, G, theta)
        if G !== nothing
            l, back = Zygote.pullback(objective, theta)
            G[:] = first(back(1.0))
            if F !== nothing
                return l
            end
            return nothing
        end
        return F === nothing ? nothing : objective(theta)
    end

    result = Optim.optimize(
        Optim.only_fg!(fg!),
        theta0,
        Optim.BFGS(),
        Optim.Options(
            iterations = BFGS_ITERS,
            show_trace = false,
            callback = state -> begin
                push!(bfgs_losses, state.value)
                false
            end,
        ),
    )

    model = re(Optim.minimizer(result))
    append!(losses, bfgs_losses)

    V_grid, confusion, area = evaluate_model(model, xs, ys, grid_points, true_labels, RHO)
    save_outputs(joinpath("results", run_name), xs, ys, V_grid, losses, confusion, area)
    return area
end

function main()
    Random.seed!(SEED)
    X_train = sample_training_points(SEED, TRAIN_SAMPLES, SAMPLE_BOUND)
    fixed_batches = build_fixed_batches(SEED, TRAIN_SAMPLES, BATCH_SIZE, ADAM_EPOCHS)
    xs, ys, grid_points = make_grid(GRID_BOUND, GRID_N)
    true_labels = compute_true_labels(grid_points)

    control_area = train_and_evaluate("control", control_penalty, X_train, fixed_batches, xs, ys, grid_points, true_labels)
    constant_area = train_and_evaluate("constant", constant_penalty, X_train, fixed_batches, xs, ys, grid_points, true_labels)
    diff = constant_area - control_area

    println("Control RoA area: ", @sprintf("%.8f", control_area))
    println("Constant penalty RoA area: ", @sprintf("%.8f", constant_area))
    println("Area difference (constant - control): ", @sprintf("%.8f", diff))
end

main()
