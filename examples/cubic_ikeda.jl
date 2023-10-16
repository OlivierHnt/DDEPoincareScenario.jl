using RadiiPolynomial
using LinearAlgebra
using DDEPoincareScenario

# load data

import JLD2

data = JLD2.load(normpath("data/cubic_ikeda.jld2"));
eq = data["equation"];
τ_init = data["initial_delay"];
c = data["initial_periodic_orbit"];
σ = data["coordinate_unstable_manifold"];
return_periodic_orbit = data["return_periodic_orbit"];
return_window = data["return_window"];
y = data["connecting_orbit"];


#############################
# Section 2: periodic orbit #
#############################

δ = 0;

x_initial_periodic_orbit = Sequence(ParameterSpace()^1 × space(c), [τ_init ; coefficients(c)]);

x_initial_periodic_orbit, _ = newton(x_initial_periodic_orbit) do x
    F = F_periodic_orbit(eq, x, δ)
    DF = DF_periodic_orbit(eq, x)
    return F, DF
end;

τ_init, c = x_initial_periodic_orbit[1], copy(component(x_initial_periodic_orbit, 2));
@show τ_init;


#################################
# Section 3: eigendecomposition #
#################################

m = nspaces(space(c));

eigenvalues = eigvals(coefficients(eigendecomposition(eq, [τ_init], c, 1)); sortby = abs);

@show λᵐ = real(eigenvalues[end]); # unstable Floquet multiplier
λ = abs(λᵐ) ^ (1/m) * cispi(1/m); # since λᵐ < 0, it is of the form abs(λᵐ) ^ (1/m) * cispi(1/m) * cispi(2k/m), for k = 0, ..., m-1


################################
# Section 4: unstable manifold #
################################

N′ = 15;

scale_eig = 0.3;

P = parameterization_unstable_manifold(eq, [τ_init], c, λ, scale_eig, N′);


##########################################
# Section 5: transverse homoclinic orbit #
##########################################

phase_shift = component(return_periodic_orbit, m)(1, 0);

x_return_periodic_orbit = Sequence(ParameterSpace()^1 × space(return_periodic_orbit), [τ_init ; coefficients(return_periodic_orbit)]);

x_return_periodic_orbit, _ = newton(x_return_periodic_orbit) do x
    F = F_periodic_orbit(eq, x, phase_shift)
    DF = DF_periodic_orbit(eq, x)
    return F, DF
end;

τ, return_periodic_orbit = x_return_periodic_orbit[1], copy(component(x_return_periodic_orbit, 2));

@show τ;
@show phase_shift = component(return_periodic_orbit, m)(1, 0);
@show σ;
@show return_window;

# transverse homoclinic orbit

V = stable_eigenspace(eq, [τ], return_periodic_orbit, return_window);

x_co = Sequence(ParameterSpace()^1 × ParameterSpace() × space(P) × space(space(y)) × space(y),
    [τ ; λ ; coefficients(P) ; σ ; δ ; zeros(dimension(space(space(y))) - 2) ; coefficients(y)]);

x_co, _ = newton(x_co) do x
    F = F_transverse_intersection(eq, x, scale_eig, component(return_periodic_orbit, return_window), V)
    DF = DF_transverse_intersection(eq, x, component(return_periodic_orbit, return_window), V)
    return F, DF
end;

# Results

# - delay
τ_pitchfork = real(x_co[1]);
x_co[1] = τ_pitchfork;
# - unstable eigenvalue
λ = abs(x_co[2]) * cispi(1/m);
x_co[2] = λ;
# - unstable manifold
P = copy(component(x_co, 3));
for j ∈ 1:m
    component(component(P, j), 1)[(:,0)] .= real.(component(component(P, j), 1)[(:,0)])
end
component(component(P, m), 1)[(:,1)] .= real.(component(component(P, m), 1)[(:,1)]);
component(x_co, 3) .= P;
# - coordinate unstable manifold
σ = real(component(x_co, 4)[1]);
component(x_co, 4)[1] = σ;
# - phase unstable manifold
δ = real(component(x_co, 4)[2]);
component(x_co, 4)[2] = δ;
# - connection
y = real(component(x_co, 5));
component(x_co, 5) .= y;

# Check that we still have a good approximate zero
F_co = F_transverse_intersection(eq, x_co, scale_eig, component(return_periodic_orbit, return_window), V);
@show norm(F_co, Inf);
@show τ_pitchfork; # new delay
@show abs(τ_pitchfork - τ); # delay defect
@show σ;
@show δ;
@show last_distance = norm(V * Sequence(component(x_co, 4)[3:end]), 1); # distance
