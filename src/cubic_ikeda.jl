struct CubicIkeda end

function f(::CubicIkeda, u, ϕ, γ)
    f_ = 0.5γ[1]*(component(ϕ, 1) - component(ϕ, 1)^3)
    return Sequence(space(f_)^1, coefficients(f_))
end

function f_nonlinear(::CubicIkeda, u, ϕ, γ)
    f_nl = -0.5γ[1] * component(ϕ, 1)^3
    return Sequence(space(f_nl)^1, coefficients(f_nl))
end

function D₁f(::CubicIkeda, u, ϕ, γ)
    codom = image(Integral(1, 0), space(u))
    return zeros(eltype(u), space(u), codom)
end

function D₂f(::CubicIkeda, u, ϕ, γ)
    D₂f_ = project(Multiplication(0.5γ[1]*(1 - 3component(ϕ, 1)^2)), space(space(ϕ)), image(Integral(1, 0), space(space(ϕ))))
    return LinearOperator(domain(D₂f_)^1, codomain(D₂f_)^1, coefficients(D₂f_))
end

function D₃f(::CubicIkeda, u, ϕ, γ)
    D₃f_ = 0.5*(component(ϕ, 1) - component(ϕ, 1)^3)
    return LinearOperator(ParameterSpace()^1, space(D₃f_)^1, reshape(coefficients(D₃f_), :, 1))
end
