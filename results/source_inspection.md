# Source Inspection

## Runtime
- pathof(NeuralLyapunov): `/Users/aryand/.julia/packages/NeuralLyapunov/ykuJS/src/NeuralLyapunov.jl`

### methods(make_RoA_aware)
```julia
# 1 method for generic function "make_RoA_aware" from NeuralLyapunov:
 [1] make_RoA_aware(cond::NeuralLyapunov.AbstractLyapunovDecreaseCondition; ρ, out_of_RoA_penalty, sigmoid, log_scale)
     @ ~/.julia/packages/NeuralLyapunov/ykuJS/src/decrease_conditions_RoA_aware.jl:202
```

### methods(AdditiveLyapunovNet)
```julia
# 1 method for generic function "AdditiveLyapunovNet" from NeuralLyapunov:
 [1] AdditiveLyapunovNet(ϕ; ψ, m, r, dim_ϕ, kwargs...)
     @ ~/.julia/packages/NeuralLyapunov/ykuJS/src/lux_structures.jl:43
```

### methods(AsymptoticStability)
```julia
# 1 method for generic function "AsymptoticStability" from NeuralLyapunov:
 [1] AsymptoticStability(; C, strength, rectifier)
     @ ~/.julia/packages/NeuralLyapunov/ykuJS/src/decrease_conditions.jl:192
```

### methods(DontCheckNonnegativity)
```julia
# 1 method for generic function "DontCheckNonnegativity" from NeuralLyapunov:
 [1] DontCheckNonnegativity(; check_fixed_point)
     @ ~/.julia/packages/NeuralLyapunov/ykuJS/src/minimization_conditions.jl:219
```

### methods(NoAdditionalStructure)
```julia
# 1 method for generic function "NoAdditionalStructure" from NeuralLyapunov:
 [1] NoAdditionalStructure()
     @ ~/.julia/packages/NeuralLyapunov/ykuJS/src/structure_specification.jl:23
```

### lowered(make_RoA_aware)
```julia
CodeInfo(
1 ─ %1 = NeuralLyapunov.:(var"#make_RoA_aware#104")
│   %2 = NeuralLyapunov.:(var"#make_RoA_aware##0#make_RoA_aware##1")
│        #105 = %new(%2)
│   %4 = #105
│   %5 = NeuralLyapunov.:(var"#make_RoA_aware##2#make_RoA_aware##3")
│        #106 = %new(%5)
│   %7 = #106
│   %8 =   dynamic (%1)(1.0, %4, %7, false, #self#, cond)
└──      return %8
)
```

## Definition: _NeuralLyapunovPDESystem
```julia
function _NeuralLyapunovPDESystem(
        dynamics,
        domains,
        spec::NeuralLyapunovSpecification,
        fixed_point,
        state,
        params,
        initial_conditions,
        policy_search::Bool,
        name
    )::PDESystem
    ########################## Unpack specifications ##########################
    structure = spec.structure
    minimization_condition = spec.minimization_condition
    decrease_condition = spec.decrease_condition
    f_call = structure.f_call
    state_dim = length(domains)

    ################## Define Lyapunov function & derivative ##################
    output_dim = structure.network_dim
    net_syms = [Symbol(:φ, i) for i in 1:output_dim]
    net = [first(@variables $s(..)) for s in net_syms]

    # φ(x) is the symbolic form of neural network output
    φ(x) = Num.([φi(x...) for φi in net])

    # V(x) is the symbolic form of the Lyapunov function
    V(x) = structure.V(φ, x, fixed_point)

    # V̇(x) is the symbolic time derivative of the Lyapunov function
    function V̇(x)
        return structure.V̇(
            φ,
            y -> Symbolics.jacobian(φ(y), y),
            dynamics,
            x,
            params,
            0.0,
            fixed_point
        )
    end

    ################ Define equations and boundary conditions #################
    eqs = Equation[]

    if check_nonnegativity(minimization_condition)
        cond = get_minimization_condition(minimization_condition)::Function
        cond_eq = cond(V, state, fixed_point) .~ 0.0
        if cond_eq isa Equation
            push!(eqs, cond_eq)
        elseif cond_eq isa AbstractVector{Equation}
            append!(eqs, cond_eq)
        else
            error(
                "Minimization condition function must return an Equation or vector of ",
                "Equations. Instead got $(typeof(cond_eq))."
            )
        end
    end

    if check_decrease(decrease_condition)
        cond = get_decrease_condition(decrease_condition)::Function
        cond_eq = cond(V, V̇, state, fixed_point) .~ 0.0
        if cond_eq isa Equation
            push!(eqs, cond_eq)
        elseif cond_eq isa AbstractVector{Equation}
            append!(eqs, cond_eq)
        else
            error(
                "Decrease condition function must return an Equation or vector of ",
                "Equations. Instead got $(typeof(cond_eq))."
            )
        end
    end

    bcs = Equation[]

    if check_minimal_fixed_point(minimization_condition)
        _V = V(fixed_point)
        _V = _V isa AbstractVector ? _V[] : _V
        push!(bcs, _V ~ 0.0)
    end

    if policy_search
        append!(bcs, f_call(dynamics, φ, fixed_point, params, 0.0) .~ zeros(state_dim))
    end

    if isempty(eqs) && isempty(bcs)
        error("No training conditions specified.")
    end

    # NeuralPDE requires an equation and a boundary condition, even if they are
    # trivial like φ(0.0) == φ(0.0), so we remove those trivial equations if they showed up
    # naturally alongside other equations and add them in if we have no other equations
    eqs = filter(eq -> eq != (0.0 ~ 0.0), eqs)
    bcs = filter(eq -> eq != (0.0 ~ 0.0), bcs)

    if isempty(eqs)
        push!(eqs, φ(fixed_point)[1] ~ φ(fixed_point)[1])
    end
    if isempty(bcs)
        push!(bcs, φ(fixed_point)[1] ~ φ(fixed_point)[1])
    end

    ########################### Construct PDESystem ###########################
    return PDESystem(
        eqs,
        bcs,
        domains,
        state,
        φ(state),
        params;
        initial_conditions,
        name
    )
end
```

## Definition: make_RoA_aware
```julia
function make_RoA_aware(
        cond::AbstractLyapunovDecreaseCondition;
        ρ = 1.0,
        out_of_RoA_penalty = (V, dVdt, state, fixed_point, _ρ) -> 0.0,
        sigmoid = (x) -> x .≥ zero.(x),
        log_scale = false
    )::RoAAwareDecreaseCondition
    return RoAAwareDecreaseCondition(
        cond,
        sigmoid,
        ρ,
        out_of_RoA_penalty,
        log_scale
    )
end
```

## Definition: AsymptoticStability
```julia
function AsymptoticStability(;
        C::Real = 1.0e-6,
        strength = (x, x0) -> C * (x - x0) ⋅ (x - x0),
        rectifier = (t) -> max(zero(t), t)
    )::LyapunovDecreaseCondition
    return LyapunovDecreaseCondition(
        true,
        (V, dVdt) -> dVdt,
        strength,
        rectifier
    )
end
```

## Definition: NoAdditionalStructure
```julia
function NoAdditionalStructure()::NeuralLyapunovStructure
    return NeuralLyapunovStructure(
        (net, x, x0) -> net(x),
        (net, grad_net, f, x, p, t, x0) -> grad_net(x) ⋅ f(x, p, t),
        (f, net, x, p, t) -> f(x, p, t),
        1
    )
end
```

## Definition: AdditiveLyapunovNet
```julia
function AdditiveLyapunovNet(
        ϕ;
        ψ = SoSPooling(),
        m = NoOpLayer(),
        r = SoSPooling(),
        dim_ϕ,
        kwargs...
    )
    if :dim_m in keys(kwargs)
        dim_m = kwargs[:dim_m]
        if :fixed_point in keys(kwargs)
            fixed_point = kwargs[:fixed_point]
        else
            fixed_point = zeros(dim_m)
        end
    elseif :fixed_point in keys(kwargs)
        fixed_point = kwargs[:fixed_point]
        dim_m = length(fixed_point)
    else
        throw(ArgumentError("Either `dim_m` or `fixed_point` must be provided."))
    end

    if ψ isa Function
        ψ = WrappedFunction(ψ)
    end
    if m isa Function
        m = WrappedFunction(m)
    end
    if r isa Function
        r = WrappedFunction(r)
    end

    return Parallel(
        +,
        Chain(
            ShiftTo(
                ϕ,
                fixed_point,
                zeros(eltype(fixed_point), dim_ϕ)
            ),
            ψ
        ),
        Chain(
            ShiftTo(
                m,
                fixed_point,
                zeros(eltype(fixed_point), dim_m)
            ),
            r
        )
    )
end
```

## Definition: DontCheckNonnegativity
```julia
function DontCheckNonnegativity(; check_fixed_point = true)::LyapunovMinimizationCondition
    return LyapunovMinimizationCondition(
        false,
        (state, fixed_point) -> 0.0,
        (t) -> zero(t),
        check_fixed_point
    )
end
```

## Definition: get_decrease_condition(::RoAAwareDecreaseCondition)
```julia
function get_decrease_condition(cond::RoAAwareDecreaseCondition)
    if check_decrease(cond)
        in_RoA_penalty = get_decrease_condition(cond.cond)
        log_scale = cond.log_scale
        return function (V, dVdt, x, fixed_point)
            _V = V(x)
            _V = _V isa AbstractVector ? _V[] : _V
            _V̇ = dVdt(x)
            _V̇ = _V̇ isa AbstractVector ? _V̇[] : _V̇
            scaled_V = log_scale ? log(_V / cond.ρ) : _V - cond.ρ
            return [
                cond.sigmoid(-scaled_V) * in_RoA_penalty(V, dVdt, x, fixed_point),
                cond.sigmoid(scaled_V) *
                    cond.out_of_RoA_penalty(_V, _V̇, x, fixed_point, cond.ρ),
            ]
        end
    else
        return nothing
    end
end
```

