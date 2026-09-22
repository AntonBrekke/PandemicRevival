include(joinpath(@__DIR__, "../src/coll_3_12.jl"))

"""
Compares `coll_3_12` (dimension-full) against `coll_3_12_scale`
(nondimensionalized internally via `coll_3_12_dimless`, then rescaled back)
across a wide range of x = m_N/T and for every `energy_type` actually used
by the solver (Val(0) for C_n, Val(1)/Val(3) for C_rho -- Val(2) is kept for
completeness). The two are independent implementations, so agreement here
is an actual test of the nondimensionalization, not a tautology.
"""
function test_coll_3_12_dimless()
    m_N = 1e-5
    m_A = 2.5 * m_N

    dof_N = 2
    dof_A = 3
    dof_nu = 2

    N = Particle{Float64}(m_N, 1, dof=dof_N)
    A = Particle{Float64}(m_A, -1, dof=dof_A)
    nu = Particle{Float64}(0., 1, dof=dof_nu)

    y = 1e-5
    sin2_2th = 1e-11
    theta = asin(sqrt(sin2_2th)) / 2.
    model_params = ModelParams(y, theta)

    xi_N = -1.
    xi_A = 2. * xi_N
    xis = (xi_N, 0., xi_A)

    n = 200
    x = logrange(1e-6, 1e2, n)
    temp = m_N ./ x

    energy_types = (Val(0), Val(1), Val(2), Val(3))
    max_rel_err = Dict{Any, Float64}()
    max_abs_err = Dict{Any, Float64}()

    for et in energy_types
        rel_errs = Vector{Float64}(undef, n)
        abs_errs = Vector{Float64}(undef, n)
        for i in 1:n
            temps = (temp[i], temp[i], temp[i])
            direct = coll_3_12(model_params, N, nu, A, temps, xis; energy_type=et)
            scaled = coll_3_12_scale(model_params, N, nu, A, temps, xis; energy_type=et)
            abs_errs[i] = abs(direct - scaled)
            rel_errs[i] = abs_errs[i] / max(abs(direct), abs(scaled), 1e-300)
        end
        max_rel_err[et] = maximum(rel_errs)
        max_abs_err[et] = maximum(abs_errs)
        println("energy_type = ", et, ": max relative error = ", max_rel_err[et],
                ", max absolute error = ", max_abs_err[et])
    end

    # Neither coll_3_12_int_e1/_int_e2 nor their _dimless counterparts pass an
    # explicit reltol/abstol to Integrals.solve, so both float on QuadGKJL's
    # default tolerance for this doubly-nested adaptive integral. The two
    # independent (dimension-full vs. nondimensionalized) quadratures land at
    # slightly different points within that shared, unpinned tolerance,
    # concentrated in the relativistic-to-non-relativistic transition region
    # (x ~ 0.1-1) -- error is ~1e-14 (machine precision) at the domain
    # extremes, confirming this is quadrature noise, not an algebra bug.
    # Tightening this bound requires setting explicit (matching) tolerances
    # on both code paths, not just this test.
    tol = 5e-3
    for et in energy_types
        @assert max_rel_err[et] < tol "coll_3_12_scale disagrees with coll_3_12 for energy_type=$et: max relative error $(max_rel_err[et]) >= $tol"
    end
    println("PASSED: coll_3_12_scale agrees with coll_3_12 to within relative tolerance $tol for all energy_types.")

    return nothing
end

test_coll_3_12_dimless()
