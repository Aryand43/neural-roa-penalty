# Structured Report

- Timestamp: 2026-03-10 13:48:49
- NeuralLyapunov path: `C:\Users\AD\.julia\packages\NeuralLyapunov\oKiqr\src\NeuralLyapunov.jl`
- Default sigmoid in source: `(x) -> x .≥ zero.(x)` (hard step)
- Logistic sigmoid used in experiments: `σ(z)=1/(1+exp(-k*z))`, with `k=20`
- Penalty term wiring in source: `[sigmoid(ρ-V)*in_RoA_penalty, sigmoid(V-ρ)*out_of_RoA_penalty]`
- Loss aggregation: NeuralPDE residual loss over PDE equations (each residual vs `0.0`).
- Gradient through default sigmoid: non-smooth/boolean gate; logistic variant provides smooth gate.

## Per-Run Results

- penalty: `control_zero` | sigmoid: `default` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `control_zero` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `constant_one` | sigmoid: `default` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `constant_one` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `inv_dist_sq` | sigmoid: `default` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `inv_dist_sq` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `scaled_inv_dist_sq` | sigmoid: `default` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `scaled_inv_dist_sq` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `inv_dist` | sigmoid: `default` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `inv_dist` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `inv_V` | sigmoid: `default` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `inv_V` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `quadratic_over_rho` | sigmoid: `default` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`
- penalty: `quadratic_over_rho` | sigmoid: `logistic` | final_loss: `NaN` | ρ: `NaN` | area: `NaN` | max_dVdt_inside: `NaN` | has_nan: `true` | error: `MethodError: no method matching +(::Vector{Any}, ::Num)
The function `+` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  +(::Any, ::Any, !Matched::Any, !Matched::Any...)
   @ Base operators.jl:596
  +(!Matched::Complex{Bool}, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\Symbolics\T2Tbs\src\num.jl:59
  +(!Matched::Num, ::Num)
   @ Symbolics C:\Users\AD\.julia\packages\SymbolicUtils\N76BL\src\methods.jl:72
  ...
`

## Adaptive Reweighting Check

- No adaptive reweighting configured in this framework (QuadratureTraining without adaptive loss callbacks).
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (default sigmoid): NaN
- Area delta for `scaled_inv_dist_sq` vs `inv_dist_sq` (logistic sigmoid): NaN

## Hypothesis Validation

- No run had positive max dV/dt inside V <= ρ on the evaluation grid.
- Next architecture modification if all plateau: increase `MLP` width/depth and test `MultiplicativeLyapunovNet` with same protocol.
