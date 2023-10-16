module DDEPoincareScenario

    using RadiiPolynomial
    import RadiiPolynomial.LinearAlgebra: eigvecs, eigen

include("cubic_ikeda.jl")
    export CubicIkeda

include("mackey_glass.jl")
    export MackeyGlass

include("zero_finding_problems.jl")
    export F_periodic_orbit, DF_periodic_orbit, F_transverse_intersection, DF_transverse_intersection

include("eigen_unstable_manifold.jl")
    export eigendecomposition, parameterization_unstable_manifold, stable_eigenspace

end
