struct MackeyGlass{T}
    a :: T
    b :: T
    ρ :: T
end

function f(eq::MackeyGlass, u, ϕ, γ)
    a, b, ρ = eq.a, eq.b, eq.ρ
    u₁, u₂, u₃, u₄ = eachcomponent(u)
    ϕ₂ = component(ϕ, 2)

    f₁ = 0.5γ[1]*(-a*u₁ + b*ϕ₂)
    f₂ = u₂*(u₄-ρ*u₂*u₃)*f₁ + γ[2]
    f₃ = (ρ-2)*u₃*u₄*f₁ + γ[3]
    f₄ = -u₄^2*f₁ + γ[4]

	return Sequence(space(f₁) × space(f₂) × space(f₃) × space(f₄),
        [coefficients(f₁) ; coefficients(f₂) ; coefficients(f₃) ; coefficients(f₄)])
end

function f_nonlinear(eq::MackeyGlass, u, ϕ, γ)
    a, b, ρ = eq.a, eq.b, eq.ρ
    u₁, u₂, u₃, u₄ = eachcomponent(u)
    ϕ₂ = component(ϕ, 2)

    f₁ = 0.5γ[1]*(-a*u₁ + b*ϕ₂)
    f₂ = u₂*(u₄-ρ*u₂*u₃)*f₁
    f₃ = (ρ-2)*u₃*u₄*f₁
    f₄ = -u₄^2*f₁

	return Sequence(space(f₁) × space(f₂) × space(f₃) × space(f₄),
        [coefficients(zero(f₁)) ; coefficients(f₂) ; coefficients(f₃) ; coefficients(f₄)])
end

function D₁f(eq::MackeyGlass, u, ϕ, γ)
    a, b, ρ = eq.a, eq.b, eq.ρ
    u₁, u₂, u₃, u₄ = eachcomponent(u)
    ϕ₂ = component(ϕ, 2)

    codom = image(Integral(1, 0), space(u))
    D₁f_ = zeros(eltype(u), space(u), codom)

	f₁ = 0.5γ[1]*(-a*u₁ + b*ϕ₂)
    ∂u₁_f₁ = -0.5γ[1]*a
    project!(component(D₁f_, 1, 1), ∂u₁_f₁*I)
    project!(component(D₁f_, 2, 1), Multiplication(u₂*(u₄-ρ*u₂*u₃)*∂u₁_f₁))
    project!(component(D₁f_, 2, 2), Multiplication((u₄-2ρ*u₂*u₃)*f₁))
    project!(component(D₁f_, 2, 3), Multiplication(-ρ*u₂^2*f₁))
    project!(component(D₁f_, 2, 4), Multiplication(u₂*f₁))
    project!(component(D₁f_, 3, 1), Multiplication((ρ-2)*u₃*u₄*∂u₁_f₁))
    project!(component(D₁f_, 3, 3), Multiplication((ρ-2)*u₄*f₁))
    project!(component(D₁f_, 3, 4), Multiplication((ρ-2)*u₃*f₁))
    project!(component(D₁f_, 4, 1), Multiplication(-u₄^2*∂u₁_f₁))
    project!(component(D₁f_, 4, 4), Multiplication(-2u₄*f₁))

	return D₁f_
end

function D₂f(eq::MackeyGlass, u, ϕ, γ)
    a, b, ρ = eq.a, eq.b, eq.ρ
    u₁, u₂, u₃, u₄ = eachcomponent(u)
    ϕ₂ = component(ϕ, 2)

    codom = image(Integral(1, 0), space(ϕ))
    D₂f_ = zeros(eltype(ϕ), space(ϕ), codom)

    ∂ϕ₂_f₁ = 0.5γ[1]*b
    project!(component(D₂f_, 1, 2), ∂ϕ₂_f₁*I)
    project!(component(D₂f_, 2, 2), Multiplication(u₂*(u₄-ρ*u₂*u₃)*∂ϕ₂_f₁))
    project!(component(D₂f_, 3, 2), Multiplication((ρ-2)*u₃*u₄*∂ϕ₂_f₁))
    project!(component(D₂f_, 4, 2), Multiplication(-u₄^2*∂ϕ₂_f₁))

	return D₂f_
end

function D₃f(eq::MackeyGlass, u, ϕ, γ)
    a, b, ρ = eq.a, eq.b, eq.ρ
    u₁, u₂, u₃, u₄ = eachcomponent(u)
    ϕ₂ = component(ϕ, 2)

    ∂τ_f₁ = 0.5*(-a*u₁ + b*ϕ₂)
	∂τ_f₂ = u₂*(u₄-ρ*u₂*u₃)*∂τ_f₁
	∂τ_f₃ = (ρ-2)*u₃*u₄*∂τ_f₁
	∂τ_f₄ = -u₄^2*∂τ_f₁

	codom = space(∂τ_f₁) × space(∂τ_f₂) × space(∂τ_f₃) × space(∂τ_f₄)
	D₃f_ = zeros(eltype(u), ParameterSpace()^4, codom)

	project!(component(D₃f_, 1, 1), ∂τ_f₁)
	project!(component(D₃f_, 2, 1), ∂τ_f₂)
	project!(component(D₃f_, 3, 1), ∂τ_f₃)
	project!(component(D₃f_, 4, 1), ∂τ_f₄)

	component(D₃f_, 2, 2)[(0,0),1] = 1
	component(D₃f_, 3, 3)[(0,0),1] = 1
	component(D₃f_, 4, 4)[(0,0),1] = 1

    return D₃f_
end
