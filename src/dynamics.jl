function vanderpol(x, p, t)
    μ = 1.0
    x1, x2 = x
    dx1 = -x2
    dx2 = μ * (x1^2 - 1) * x2 + x1
    return [dx1, dx2]
end
