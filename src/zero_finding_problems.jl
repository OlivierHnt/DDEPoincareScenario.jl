function F_periodic_orbit(eq, x, phase)
    γ, c = component(x, 1), component(x, 2)

    m = nspaces(space(c))

    F = zeros(eltype(x), space(x))

    component(F, 1) .= component(c, m)(1, 0) .- phase

    F₂ = component(F, 2)

    u, ϕ = component(c, 1), component(c, m)
    project!(component(F₂, 1), ϕ(1, 0) + Integral(1, 0) * f(eq, u, ϕ, γ) - u)
    for j ∈ 2:m
        u, ϕ = component(c, j), component(c, j-1)
        project!(component(F₂, j), ϕ(1, 0) + Integral(1, 0) * f(eq, u, ϕ, γ) - u)
    end

    return F
end

function DF_periodic_orbit(eq, x)
    γ, c = component(x, 1), component(x, 2)

    m = nspaces(space(c))

    DF = zeros(eltype(x), space(x), space(x))

    project!(component(component(DF, 1, 2), :, m), Evaluation(1, 0))

    D₁F₂ = component(DF, 2, 1)
    D₂F₂ = component(DF, 2, 2)

    u, ϕ = component(c, 1), component(c, m)
    mul!(component(D₁F₂, 1, :), Integral(1, 0), D₃f(eq, u, ϕ, γ))
    project!(component(D₂F₂, 1, 1), Integral(1, 0) * D₁f(eq, u, ϕ, γ) - I)
    project!(component(D₂F₂, 1, m), Evaluation(1, 0) + Integral(1, 0) * D₂f(eq, u, ϕ, γ))
    for j ∈ 2:m
        u, ϕ = component(c, j), component(c, j-1)
        mul!(component(D₁F₂, j, :), Integral(1, 0), D₃f(eq, u, ϕ, γ))
        project!(component(D₂F₂, j, j), Integral(1, 0) * D₁f(eq, u, ϕ, γ) - I)
        project!(component(D₂F₂, j, j-1), Evaluation(1, 0) + Integral(1, 0) * D₂f(eq, u, ϕ, γ))
    end

    return DF
end

#

function ∂λ_scale(P::Sequence{CartesianPower{TensorSpace{Tuple{Chebyshev,Taylor}}}}, λ) # 1D unstable manifold
    ∂λ_Pλ = copy(P)
    for i ∈ 1:nspaces(space(∂λ_Pλ))
        ∂λ_Pᵢλ = component(∂λ_Pλ, i)
        for α ∈ indices(space(∂λ_Pᵢλ))
            ∂λ_Pᵢλ[α] *= α[2]*λ^(α[2]-1)
        end
    end
    return ∂λ_Pλ
end

#

function F_transverse_intersection(eq, x, scale_eig, c, V)
    γ, λ, P = component(x, 1), component(x, 2)[1], component(x, 3)
    Ω, y = component(x, 4), component(x, 5)

    n = length(γ)

    σ, δ, θ = Ω[1], Ω[2:n+1], Sequence(Ω[n+2:end])

    m = nspaces(space(P))
    k = nspaces(space(y))

    F = zeros(eltype(x), space(x))

    # 1D unstable manifold

    component(F, 1) .= component(P, m)(1, 0) .- δ

    component(F, 2) .= Sequence(space(space(space(P)))[1], view(component(component(P, m), 1), (:, 1)))(1) .- scale_eig

    F₃ = component(F, 3)

    for j ∈ 1:m
        j′ = ifelse(j == 1, m, j-1)
        u, ϕ = scale(component(P, j), (1, λ)), component(P, j′)
        project!(component(F₃, j), ϕ(1, nothing) + Integral(1, 0) * f(eq, u, ϕ, γ) - u)
    end

    # connection

    γ_ = [γ[1] ; fill(0, n-1)]

    Q, yₖ = c + V*θ, component(y, k)
    project!(component(F, 4), yₖ(1, 0) + Integral(1, 0) * f(eq, Q, yₖ, γ_) - Q)

    F₅ = component(F, 5)

    project!(component(F₅, 1), component(P, m)(nothing, σ) - component(y, 1))
    for i ∈ 2:k
        yᵢ₊₁, yᵢ = component(y, i), component(y, i-1)
        project!(component(F₅, i), yᵢ(1, 0) + Integral(1, 0) * f(eq, yᵢ₊₁, yᵢ, γ_)- yᵢ₊₁)
    end

    #

    return F
end

function DF_transverse_intersection(eq, x, c, V)
    γ, λ, P = component(x, 1), component(x, 2)[1], component(x, 3)
    Ω, y = component(x, 4), component(x, 5)

    n = length(γ)

    σ, δ, θ = Ω[1], Ω[2:n+1], Sequence(Ω[n+2:end])

    m = nspaces(space(P))
    k = nspaces(space(y))

    DF = zeros(eltype(x), space(x), space(x))

    # 1D unstable manifold

    project!(component(component(DF, 1, 3), :, m), Evaluation(1, 0))
    project!(LinearOperator(view(component(DF, 1, 4), :, 2:n+1)), -I)

    project!(LinearOperator(space(space(space(P)))[1], ParameterSpace(), view(component(component(component(DF, 2, 3), m), 1), 1:1, (:, 1))), Evaluation(1))

    D₁F₃ = component(DF, 3, 1)
    D₂F₃ = component(DF, 3, 2)
    D₃F₃ = component(DF, 3, 3)

    for j ∈ 1:m
        j′ = ifelse(j == 1, m, j-1)
        u, ϕ = scale(component(P, j), (1, λ)), component(P, j′)
        ∂λ_u = ∂λ_scale(component(P, j), λ)
        D₁f_ = D₁f(eq, u, ϕ, γ)
        mul!(component(D₁F₃, j, :), Integral(1, 0), D₃f(eq, u, ϕ, γ))
        project!(component(D₂F₃, j), Integral(1, 0) * (D₁f_ * ∂λ_u) - ∂λ_u)
        mul!(component(D₃F₃, j, j), Integral(1, 0) * D₁f_ - I, Scale(1, λ))
        project!(component(D₃F₃, j, j′), Evaluation(1, nothing) + Integral(1, 0) * D₂f(eq, u, ϕ, γ))
    end

    # connection

    γ_ = [γ[1] ; fill(0, n-1)]

    Q, yₖ = c + V*θ, component(y, k)
    mul!(component(component(DF, 4, 1), :, 1), Integral(1, 0), component(D₃f(eq, Q, yₖ, γ_), :, 1))
    project!(LinearOperator(domain(V), codomain(V), view(component(DF, 4, 4), :, 2+n:length(Ω))), Integral(1, 0) * D₁f(eq, Q, yₖ, γ_) * V - V)
    project!(component(component(DF, 4, 5), :, k), Evaluation(1, 0) + Integral(1, 0) * D₂f(eq, Q, yₖ, γ_))

    project!(component(component(DF, 5, 3), 1, m), Evaluation(nothing, σ))
    project!(Sequence(codomain(component(component(DF, 5, 4), 1, :)), view(component(component(DF, 5, 4), 1, :), :, 1)), differentiate(component(P, m), (0, 1))(nothing, σ))
    project!(component(component(DF, 5, 5), 1, 1), -I)

    D₁F₅ = component(DF, 5, 1)
    D₅F₅ = component(DF, 5, 5)

    for i ∈ 2:k
        yᵢ₊₁, yᵢ = component(y, i), component(y, i-1)
        mul!(component(D₁F₅, i, 1), Integral(1, 0), component(D₃f(eq, yᵢ₊₁, yᵢ, γ_), :, 1))
        project!(component(D₅F₅, i, i), Integral(1, 0) * D₁f(eq, yᵢ₊₁, yᵢ, γ_) - I)
        project!(component(D₅F₅, i, i-1), Evaluation(1, 0) + Integral(1, 0) * D₂f(eq, yᵢ₊₁, yᵢ, γ_))
    end

    #

    return DF
end
