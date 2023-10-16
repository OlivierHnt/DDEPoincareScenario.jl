function eigendecomposition(eq, γ, c, j)
    m = nspaces(space(c))

    j′ = ifelse(j == m, 1, j+1)
    u, ϕ = component(c, j′), component(c, j)
    H = (I - Integral(1, 0) * D₁f(eq, u, ϕ, γ)) \ (Evaluation(1, 0) + Integral(1, 0) * D₂f(eq, u, ϕ, γ))

    for i ∈ j+1:m
        i′ = ifelse(i == m, 1, i+1)
        u, ϕ = component(c, i′), component(c, i)
        H .= (I - Integral(1, 0) * D₁f(eq, u, ϕ, γ)) \ (Evaluation(1, 0) + Integral(1, 0) * D₂f(eq, u, ϕ, γ)) * H
    end

    for i ∈ 1:j-1
        u, ϕ = component(c, i+1), component(c, i)
        H .= (I - Integral(1, 0) * D₁f(eq, u, ϕ, γ)) \ (Evaluation(1, 0) + Integral(1, 0) * D₂f(eq, u, ϕ, γ)) * H
    end

    return H
end

function stable_eigenspace(eq, γ, c, j)
    H = eigendecomposition(eq, γ, c, j)
    V = eigvecs(coefficients(H); sortby = λ -> inv(1 + abs(λ)))
    return LinearOperator(ParameterSpace()^(size(V, 2)-length(γ)-1), space(space(c)), V[:,2+length(γ):end])
end

function parameterization_unstable_manifold(eq, γ, c, λ, scale_eig, N′) # 1D unstable manifold
    n = nspaces(space(space(c)))
    N = order(space(space(space(c)))[1])

    v₁ = Sequence(space(space(c)), view(eigvecs(coefficients(eigendecomposition(eq, γ, c, 1)); sortby = abs), :, n*(N+1)))

    return _parameterization_unstable_manifold(eq, γ, c, λ, v₁, scale_eig, N′)
end

function _parameterization_unstable_manifold(eq, γ, c, λ, v₁, scale_eig, N′) # 1D unstable manifold
    m = nspaces(space(c))
    n = nspaces(space(space(c)))
    N = order(space(space(space(c)))[1])

    # initialize parameterization

    P = zeros(ComplexF64, ((Chebyshev(N) ⊗ Taylor(N′))^n)^m)

    for i ∈ 1:n
        component(component(P, 1), i)[(:,0)] .= component(component(c, 1), i)
        component(component(P, 1), i)[(:,1)] .= component(v₁, i)
    end

    λ⁻¹ = inv(λ)
    vⱼ = v₁
    for j ∈ 2:m
        u, ϕ = component(c, j), component(c, j-1)
        Ĥⱼ₋₁ = (I - Integral(1, 0) * D₁f(eq, u, ϕ, γ)) \ (Evaluation(1, 0) + Integral(1, 0) * D₂f(eq, u, ϕ, γ))
        vⱼ = λ⁻¹ * Ĥⱼ₋₁ * vⱼ
        for i ∈ 1:n
            component(component(P, j), i)[(:,0)] .= component(component(c, j), i)
            component(component(P, j), i)[(:,1)] .= component(vⱼ, i)
        end
    end

    default_scaling = (Derivative(0, 1) * component(component(P, m), 1))(1, 0)
    for j ∈ 1:m, i ∈ 1:n
        component(component(P, j), i)[(:,1)] .= scale_eig .* (component(component(P, j), i)[(:,1)] ./ default_scaling)
    end

    #

    K₁ = zeros(eltype(c), space(c), space(c))
    K₂ = zeros(eltype(c), space(c), space(c))
    u, ϕ = component(c, 1), component(c, m)
    project!(component(K₁, 1, 1), Integral(1, 0) * D₁f(eq, u, ϕ, γ))
    project!(component(K₂, 1, m), Evaluation(1, 0) + Integral(1, 0) * D₂f(eq, u, ϕ, γ))
    for j ∈ 2:m
        u, ϕ = component(c, j), component(c, j-1)
        project!(component(K₁, j, j), Integral(1, 0) * D₁f(eq, u, ϕ, γ))
        project!(component(K₂, j, j-1), Evaluation(1, 0) + Integral(1, 0) * D₂f(eq, u, ϕ, γ))
    end

    tmp = zeros(eltype(P), space(c))
    for α ∈ 2:N′
        u, ϕ = scale(component(P, 1), (1, λ)), component(P, m)
        f_nl = f_nonlinear(eq, u, ϕ, γ)
        for i ∈ 1:n
            component(component(tmp, 1), i) .= (Integral(1, 0) * component(f_nl, i))[(0:N,α)]
        end
        for j ∈ 2:m
            u, ϕ = scale(component(P, j), (1, λ)), component(P, j-1)
            f_nl = f_nonlinear(eq, u, ϕ, γ)
            for i ∈ 1:n
                component(component(tmp, j), i) .= (Integral(1, 0) * component(f_nl, i))[(0:N,α)]
            end
        end
        tmp .= (λ^α * I - λ^α * K₁ - K₂) \ tmp
        for j ∈ 1:m, i ∈ 1:n
            component(component(P, j), i)[(:,α)] .= component(component(tmp, j), i)
        end
    end

    #

    return P
end
