import NonlinearSolve as NLS
import DifferentialEquations as DE
import OrdinaryDiffEqRosenbrock as ODER
import LinearAlgebra as LA

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "time_temp_relation.jl"))
include(joinpath(@__DIR__, "dodelson_widrow.jl"))
include(joinpath(@__DIR__, "pandemolator_common.jl"))
include(joinpath(@__DIR__, "dimless_interp.jl"))

"""
pandemolator_z.jl

The pandemolator written in terms of z = log(x) = log(m_N1 / T_nu).

RELATION TO pandemolator.jl
============================
pandemolator.jl already integrates in log(x), but keys its interpolants on
T_nu and rebuilds x and T_nu from the integration variable on every RHS
evaluation. This file is its direct counterpart: same independent variable
and same equation, but z is the native coordinate of the interpolants too,
so a lookup costs one `exp` rather than `exp` -> divide -> `log` -> `exp`
(and `AutoFiniteDiff` calls the RHS ~5x per Jacobian, so this compounds).
`dx/dt` is also tabulated and interpolated directly, rather than
interpolating dT_nu/dt and converting on every call.

An x-as-independent-variable version was tried and dropped: it needed
2x+ the RHS evaluations of this version for the same tolerance (x spans
~6 decades over the integration range, ~3e-5 to ~1e2, so a uniform step in
x resolves the early evolution far more poorly than a uniform step in z),
with no offsetting benefit. See test/test_pandemolator_z.jl for the
comparison against pandemolator.jl this file was validated against.

The state vector [ln Y_n, ln Y_rho, ln x_N, eta], the mass matrix
Diagonal([1,1,0,0]) and the algebraic constraints are unchanged. The
dimension-full physics (`number_density`, `coll_3_12`, ...) is reached by
converting z back to a temperature at the call site only -- `C_n`, `C_rho`,
`num_dens`, `energy_dens`, `rho_3P`, `n_rho_root`, `xi_from_eta` and
`T_N_from_ln_x_N` all live in pandemolator_common.jl and are reused
verbatim (they carry no `Pandemolator`/`PandemolatorZ` type annotation),
which keeps the physics bit-identical so the comparison in the test
isolates the change of parameterization.
"""
mutable struct PandemolatorZ{T<:Real, FT, FdX, FEnt, FH}
    mp::ModelParams{T}

    N1::Particle{T}
    N2::Particle{T}
    A::Particle{T}
    nu::Particle{T}

    fac_n_A::Int64
    verbose::Bool

    # Bounds on T_N/T_nu for trial states, see `in_physical_domain`.
    T_ratio_min::Float64
    T_ratio_max::Float64

    # Wall-clock deadline for the current solve (Inf = none), see `check_deadline`.
    deadline::Float64

    # Interpolants keyed on z = log(N1.m / T_nu)
    t_interp_z::FT
    dx_dt_interp_z::FdX
    ent_interp_z::FEnt
    H_interp_z::FH

    function PandemolatorZ{T}(
        model_params::ModelParams{T},
        N1::Particle{T},
        N2::Particle{T},
        A::Particle{T},
        nu::Particle{T},
        tT_rel::TimeTempRelation{T},
        verbose::Bool=false;
        # Accepted states at the relic-scan roots reach ln(T_N/T_nu) + max(0, z)
        # >= -5.5 and T_N/T_nu <= 2.1, so these leave margins of ~4e3 and ~5e2.
        T_ratio_min::Real=1e-6,
        T_ratio_max::Real=1e3,
    ) where T <: Real
        if A.m > N1.m + N2.m
            fac_n_A_val = 2
        else
            fac_n_A_val = 1
        end

        x_grid = x_grid_of(N1.m, tT_rel.T_nu_grid)
        z_grid = log.(x_grid)
        dx_dt_grid = dx_dt_grid_of(N1.m, tT_rel.T_nu_grid, tT_rel.dT_nu_dt_grid)

        t_interp_z = z_interpolation(z_grid, tT_rel.t_grid)
        dx_dt_interp_z = z_interpolation(z_grid, dx_dt_grid)
        ent_interp_z = z_interpolation(z_grid, tT_rel.ent_grid)
        H_interp_z = z_interpolation(z_grid, tT_rel.hubble_grid)

        new{
            T,
            typeof(t_interp_z),
            typeof(dx_dt_interp_z),
            typeof(ent_interp_z),
            typeof(H_interp_z),
        }(
            model_params,
            N1, N2, A, nu,
            fac_n_A_val,
            verbose,
            T_ratio_min, T_ratio_max,
            Inf,
            t_interp_z, dx_dt_interp_z, ent_interp_z, H_interp_z,
        )
    end
end

"""T_nu from the native coordinate z, for the dimension-full physics calls."""
T_nu_from_z(pan, z) = pan.N1.m / exp(z)

function pandemolate_z(
        tT_rel::TimeTempRelation{T},
        dw::DodelsonWidrow{T},
        pan::PandemolatorZ{T};
        reltol=nothing,
        abstol=nothing,
    ) where T <: Real
    z_pts = log.(pan.N1.m ./ tT_rel.T_nu_grid[dw.i_ic:dw.i_end+1])
    z_lim = (z_pts[1], z_pts[end])

    u0 = initial_conditions_z(tT_rel, dw, pan)
    if pan.verbose
        println("u0 = ", u0)
        println("exp(u0) = ", exp.(u0))
        println("z_lim = ", z_lim, "  (x_lim = ", exp.(z_lim), ")")
    end

    mass_matrix = LA.Diagonal([1., 1., 0., 0.])
    ode_func! = DE.ODEFunction(mm_func_z!, mass_matrix=mass_matrix)
    prob = DE.ODEProblem(ode_func!, u0, z_lim, pan)

    kw = Dict{Symbol,Any}()
    reltol === nothing || (kw[:reltol] = reltol)
    abstol === nothing || (kw[:abstol] = abstol)
    timed_sol = @timed DE.solve(
        prob,
        ODER.Rodas4P(autodiff=AutoFiniteDiff());
        isoutofdomain=(u, p, t) -> !in_physical_domain(p, u, p.N1.m / exp(t)),
        kw...,
    )
    if pan.verbose
        println("solve time: ", timed_sol.time, " s")
    end
    return timed_sol.value
end

function initial_conditions_z(
        tT_rel::TimeTempRelation{T},
        dw::DodelsonWidrow{T},
        pan::PandemolatorZ{T},
    ) where T <: Real
    n0 = dw.n_ic
    rho0 = dw.rho_ic
    z_0 = log(pan.N1.m / tT_rel.T_nu_grid[dw.i_ic])
    ent0 = pan.ent_interp_z(z_0)

    ln_y_n_0 = log(n0 / ent0)
    ln_y_rho_0 = log(rho0 / ent0^(4. / 3.))
    if pan.verbose
        println("z_0 = log(m_N1 / T_dw) = ", z_0)
        println("rho_dw = ", rho0)
        println("y_n_0 = ", exp(ln_y_n_0))
    end

    # ln x_N = z at the Dodelson-Widrow initial condition
    root_params = (pan = pan, n_ic = n0, rho_ic = rho0)
    n_rho_problem = NLS.NonlinearProblem(
        n_rho_root,            # from pandemolator_common.jl, reused unchanged
        [z_0, 0.0],
        root_params,
    )
    n_rho_root_sol = NLS.solve(n_rho_problem)

    return [ln_y_n_0, ln_y_rho_0, n_rho_root_sol.u[1], n_rho_root_sol.u[2]]
end

"""RHS in z = log(x). The factor x relative to the x-parameterization is
d/dz = x d/dx."""
function mm_func_z!(du, u, pan::PandemolatorZ{T}, z) where {T<:Real}
    check_deadline(pan)
    x = exp(z)
    if !in_physical_domain(pan, u, pan.N1.m / x)
        reject_state!(du)
        return nothing
    end
    H = pan.H_interp_z(z)
    ent = pan.ent_interp_z(z)
    dx_dt = pan.dx_dt_interp_z(z)

    ln_y_n = u[1]
    n = exp(ln_y_n) * ent
    ln_y_rho = u[2]
    rho = exp(ln_y_rho) * ent^(4. / 3.)
    ln_x_N = u[3]
    eta = u[4]

    T_nu = pan.N1.m / x
    T_N = T_N_from_ln_x_N(pan, ln_x_N)
    xi_N = xi_from_eta(pan, eta, ln_x_N)
    gap_A = gap_A_from_eta(eta)

    coll_n, coll_rho = collision_terms(pan, T_nu, T_N, xi_N; gap_A=gap_A)

    du[1] = x * coll_n / (n * dx_dt)
    du[2] = x * (H * rho_3P(pan, T_N, xi_N; gap_A=gap_A) + coll_rho) / (rho * dx_dt)

    n_anal = num_dens(pan, T_N, xi_N; gap_A=gap_A)
    rho_anal = energy_dens(pan, T_N, xi_N; gap_A=gap_A)
    du[3] = ln_y_n - log(n_anal / ent)
    du[4] = ln_y_rho - log(rho_anal / ent^(4. / 3.))
    return nothing
end

"""As `transform_sol`, with z = log(x) as the independent variable (same columns)."""
function transform_sol_z(pan::PandemolatorZ, sol)
    z = sol.t
    x_nu = exp.(z)
    hubble = pan.H_interp_z.(z)
    ent = pan.ent_interp_z.(z)

    ln_x_N = sol[3, :]
    x_N = exp.(ln_x_N)
    T_N = pan.N1.m ./ x_N
    T_nu = pan.N1.m ./ x_nu

    y_n = exp.(sol[1, :])
    y_rho = exp.(sol[2, :])

    eta = sol[4, :]
    xi_N = xi_from_eta.(Ref(pan), eta, ln_x_N)
    xi_A = pan.fac_n_A .* xi_N
    gap_A = gap_A_from_eta.(eta)

    coll_n = map((a, b, c, g) -> C_n(pan, a, b, c; gap_A=g), T_nu, T_N, xi_N, gap_A)
    coll_A_N2nu = map(
        (a, b, c, g) -> - A_N2nu_moments(pan, a, b, c; gap_A=g)[1],
        T_nu, T_N, xi_N, gap_A
    )
    coll_AA_NN = map(
        (b, c, g) -> C_n_AA(pan, b, c; gap_A=g),
        T_N, xi_N, gap_A
    )

    y_N1 = number_density.(Ref(pan.N1), T_N, xi_N) ./ ent
    y_N2 = number_density.(Ref(pan.N2), T_N, xi_N) ./ ent
    y_A = map((b, c, g) -> number_density(pan.A, b, c; gap=g), T_N, xi_A, gap_A) ./ ent

    return [
        x_nu;; x_N;;
        hubble;; ent;;
        y_n;; y_rho;;
        xi_N;; xi_A;;
        y_N1;; y_N2;; y_A;;
        coll_n;; coll_A_N2nu;; coll_AA_NN;;
    ]
end
