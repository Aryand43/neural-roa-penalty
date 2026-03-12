function compute_grid_metrics(V, V̇, ρ; lb, ub, n = 201)
    xs = range(lb[1], ub[1], length = n)
    ys = range(lb[2], ub[2], length = n)
    dx = step(xs)
    dy = step(ys)

    points = Matrix{Float64}(undef, 2, n * n)
    idx = 1
    for y in ys, x in xs
        points[:, idx] .= (x, y)
        idx += 1
    end

    Vvals = vec(V(points))
    dVvals = vec(V̇(points))
    roa_mask = Vvals .<= ρ

    area = sum(roa_mask) * dx * dy
    max_dVdt_inside = any(roa_mask) ? maximum(dVvals[roa_mask]) : NaN

    return (; xs, ys, points, Vvals, dVvals, roa_mask, area, max_dVdt_inside)
end
