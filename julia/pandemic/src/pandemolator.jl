import NonlinearSolve as NLS
import BenchmarkTools as BT
import DifferentialEquations as DE
import DiffEqBase as DEB
import OrdinaryDiffEqRosenbrock as ODER
import OrdinaryDiffEqBDF as ODEB
import OrdinaryDiffEqFIRK as ODEF
import LinearAlgebra as LA

include(joinpath(@__DIR__, "../src/pandemic_result.jl"))
include(joinpath(@__DIR__, "../src/pandemolator_common.jl"))

"""
pandemolator.jl

The original pandemolator: integrates in log(x) = log(m_N1/T_nu), with the
background interpolants (`t_interp_T_nu`, `ent_interp_T_nu`, ...) keyed on
T_nu. T is converted to/from log(x) on every RHS evaluation (`mm_func!`).

The dark-sector physics shared with every other parameterization -- the
algebraic constraint (T_N, xi_N from the state), number/energy densities,
and the collision terms C_n/C_rho -- lives in pandemolator_common.jl, not
here; only this file's own solver plumbing (the struct, `pandemolate`,
`initial_conditions`, `mm_func!`, `transform_sol`) is T-specific.

See pandemolator_z.jl for the z-native counterpart (same equation, same
independent variable, interpolants keyed on z instead of T_nu) -- validated
to agree with this file in test/test_pandemolator_z.jl.
"""
mutable struct Pandemolator{T<:Real, FT, FdT, FEnt, FH}
    mp::ModelParams{T}

    N1::Particle{T}
    N2::Particle{T}
    A::Particle{T}
    nu::Particle{T}

    # Collision operator coefficients
    # TODO: Check if comments are correct
    # C_n::Function     # rhs of Boltzmann-eq. for n_N1 + n_N2 + fac_n_A*n_A
    # C_rho::Function   # rhs of Boltzmann-eq. for rho_N1 +rho_N2 + rho_A
    # C_xi0::Function   # part of rhs of Boltzmann-eq. for n setting xi = 0

    # Factor for number of A particles in effective number density (always 2)
    fac_n_A::Int64

    # Guard for the per-evaluation debug prints below (default off; a scan
    # calls pandemolate hundreds/thousands of times, so this must stay quiet
    # unless explicitly requested).
    verbose::Bool

    # Bounds on T_N/T_nu for trial states, see `in_physical_domain`.
    T_ratio_min::Float64
    T_ratio_max::Float64

    # TODO: [13.08.26] These interpolations are not used. Probably incorrect as well, so test if they are needed.
    # # Interpolation functions in t space
    # T_nu_interp::I
    # dT_nu_dt_interp::I
    # ent_interp::I
    # H_interp::I

    # Interpolation functions in T space (log-log)
    t_interp_T_nu::FT
    dT_nu_dt_interp_T_nu::FdT
    ent_interp_T_nu::FEnt
    H_interp_T_nu::FH

    function Pandemolator{T}(
        model_params::ModelParams{T},
        N1::Particle{T},
        N2::Particle{T},
        A::Particle{T},
        nu::Particle{T},
        # C_n::Function, C_rho::Function, C_xi0::Function,
        tT_rel::TimeTempRelation{T},
        verbose::Bool=false;
        # See PandemolatorZ in pandemolator_z.jl for how these were chosen.
        T_ratio_min::Real=1e-6,
        T_ratio_max::Real=1e3,
    ) where T <: Real
        # TODO: [01.07.26] Ask Anton: Why this factor?
        # Calculate factor for A' particle contribution
        if A.m > N1.m + N2.m
            fac_n_A_val = 2
        else
            # TODO: [16.07.26] Not used
            fac_n_A_val = 1
        end

        # Create closure functions for T-space interpolations
        t_interp_T_nu = temp_interpolation(tT_rel.T_nu_grid, tT_rel.t_grid)
        dT_nu_dt_interp_T_nu = temp_interpolation(tT_rel.T_nu_grid, tT_rel.dT_nu_dt_grid, neg=true) # Note the minus signs
        ent_interp_T_nu = temp_interpolation(tT_rel.T_nu_grid, tT_rel.ent_grid)
        H_interp_T_nu = temp_interpolation(tT_rel.T_nu_grid, tT_rel.hubble_grid)

        new{
            T,
            typeof(t_interp_T_nu),
            typeof(dT_nu_dt_interp_T_nu),
            typeof(ent_interp_T_nu),
            typeof(H_interp_T_nu),
        }(
            model_params,
            N1, N2, A, nu,
            # C_n, C_rho, C_xi0,
            fac_n_A_val,
            verbose,
            T_ratio_min, T_ratio_max,
            # T_nu_interp, dT_nu_dt_interp, ent_interp, H_interp,
            t_interp_T_nu, dT_nu_dt_interp_T_nu, ent_interp_T_nu,
            H_interp_T_nu,
        )
    end
end


function pandemolate(
        tT_rel::TimeTempRelation{T},
        dw::DodelsonWidrow{T},
        pan::Pandemolator{T};
        reltol=nothing,
        abstol=nothing,
    ) where T <: Real
    """
    Anton: Not entirely clear how this works. We use the fact that the dark
    sector is in equilibrium to get T_d, xi_d. As we have to unknown variables,
    we must solve two equations.
    We solve for n, rho numerically to get n_num, rho_num. Using equilibirum,
    the analytical expression is known. Hence, we solve
    n_an(T_d, xi_d) = n_num, rho_an(T_d, xi_d) = rho_num
    using root-solvers to obtain T_d, xi_d.
    In the special case of xi_d = 0, the system simplifies to only one
    variable, in which we solve for rho.
    """
    log_x_pts = log.(pan.N1.m ./ tT_rel.T_nu_grid[dw.i_ic:dw.i_end+1])
    n_pts = log_x_pts.size[1]

    res = PandemicResult{Float64}(n_pts)
    res.t = tT_rel.t_grid[dw.i_ic:dw.i_end+1]
    res.T_nu = tT_rel.T_nu_grid[dw.i_ic:dw.i_end+1]
    res.H = tT_rel.hubble_grid[dw.i_ic:dw.i_end+1]

    u0 = initial_conditions(tT_rel, dw, pan)
    if pan.verbose
        println("u0 = ", u0)
        println("exp(u0) = ", exp.(u0))
    end

    log_x_lim = (log_x_pts[1], log_x_pts[end])
    temp_lim = pan.N1.m ./ exp.(log_x_lim)
    if pan.verbose
        println("log_x_lim = ", log_x_lim)
        println("temp_lim = ", temp_lim)
    end

    ode_params = pan

    mass_matrix = LA.Diagonal([1., 1., 0., 0.])
    ode_func! = DE.ODEFunction(mm_func!, mass_matrix=mass_matrix)
    prob = DE.ODEProblem(
        ode_func!,
        u0,
        log_x_lim,
        ode_params,
    )
    # abstol > 1.3e-15. Initial conditions not consistent with smaller abstol.
    # For small reltol: "Warning: Verbosity toggle: dt_epsilon" at t = log(x) = -7.101953893976883
    # Rodas4/4P
    # Rosenbrock23 - Same error as for Rodas with small reltol
    kw = Dict{Symbol,Any}()
    reltol === nothing || (kw[:reltol] = reltol)
    abstol === nothing || (kw[:abstol] = abstol)
    timed_sol = @timed DE.solve(
        prob,
        # Rosenbrock23(autodiff=AutoFiniteDiff()),
        # ODER.Rodas4(autodiff=AutoFiniteDiff()),
        ODER.Rodas4P(autodiff=AutoFiniteDiff());
        # ODER.Rodas5P(autodiff=AutoFiniteDiff()),
        # ODEF.RadauIIA5(autodiff=AutoFiniteDiff()),
        # ODEF.RadauIIA5(),
        # force_dtmin=true,
        isoutofdomain=(u, p, t) -> !in_physical_domain(p, u, p.N1.m / exp(t)),
        kw...,
    )
    if pan.verbose
        println("solve time: ", timed_sol.time, " s")
    end
    return timed_sol.value
end

function initial_conditions(
        tT_rel::TimeTempRelation{T},
        dw::DodelsonWidrow{T},
        pan::Pandemolator{T}
    ) where T <: Real

    dof_fac_N = 7. / 8. * pan.N1.dof
    dof_fac_A = pan.A.dof

    n0 = dw.n_ic
    T_0 = tT_rel.T_nu_grid[dw.i_ic]
    ent0 = pan.ent_interp_T_nu(T_0)
    rho0 = dw.rho_ic

    # y0[1]: Y_n=n/s (Yield)
    # y0[2]: Y_rho = rho/s^(4/3) (Energy density scaled with entropy)
    ln_y_n_0 = log(n0/ent0)
    ln_y_rho_0 = log(rho0 / ent0^(4. / 3.))
    if pan.verbose
        println("T_0 = ", T_0)
        println("x_0 = m_N1 / T_dw = ", pan.N1.m / T_0)
        println("rho_dw = ", rho0)
        println("y_n_0 = ", exp(ln_y_n_0))
    end

    ln_x_N_guess = log(pan.N1.m / T_0)
    eta_guess = 0.0

    root_params = (
        pan = pan,
        n_ic = n0,
        rho_ic = rho0,
    )
    # tol = 1e-8
    n_rho_problem = NLS.NonlinearProblem(
        n_rho_root,
        [ln_x_N_guess, eta_guess],
        root_params,
    )
    n_rho_root_sol = NLS.solve(
        n_rho_problem,
        # abstol=tol,
        # reltol=tol,
    )
    ln_x_N_0 = n_rho_root_sol.u[1]
    # TODO [07.08.26] This initial condition makes little sense. Ask Torsten.
    # T_N_0 = (30. * rho0 / (pi^2 * (2*dof_fac_N + dof_fac_A)))^(1. / 4.)

    eta = n_rho_root_sol.u[2]

    u0 = [ln_y_n_0, ln_y_rho_0, ln_x_N_0, eta]
    return u0
end

"""
Takes the a solution of the Boltzmann solver as argument, transforms to to usual properties and returns and array with these.
"""
function transform_sol(pan, sol)
    x_nu = exp.(sol.t)
    T_nu = pan.N1.m ./ x_nu

    hubble = pan.H_interp_T_nu.(T_nu)
    ent = pan.ent_interp_T_nu.(T_nu)

    ln_x_N = sol[3, :]
    x_N = exp.(ln_x_N)
    T_N = pan.N1.m ./ x_N

    ln_y_n = sol[1, :]
    ln_y_rho = sol[2, :]
    y_n = exp.(ln_y_n)
    y_rho = exp.(ln_y_rho)

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

function dx_dt_interp(pan, x)
    T_nu = pan.N1.m / x
    dT_nu_dt = pan.dT_nu_dt_interp_T_nu(T_nu)
    return - x^2 * dT_nu_dt / pan.N1.m
end

# Anton: RHS of Boltzmann equation
function mm_func!(
        du,
        u,
        pan::Pandemolator{T},
        log_x
    ) where {T<:Real}
    if pan.verbose
        println("u = ", u)
        println("du = ", du)
    end
    x = exp(log_x)
    T_nu = pan.N1.m / x
    H = pan.H_interp_T_nu(T_nu)
    ent = pan.ent_interp_T_nu(T_nu)

    if pan.verbose
        println("log_x = ", log_x, ", x = ", x)
    end
    ln_y_n = u[1]
    n = exp(ln_y_n) * ent
    ln_y_rho = u[2]
    rho = exp(ln_y_rho) * ent^(4. / 3.)
    ln_x_N = u[3]
    T_N = T_N_from_ln_x_N(pan, ln_x_N)
    eta = u[4]
    xi_N = xi_from_eta(pan, eta, ln_x_N)
    gap_A = gap_A_from_eta(eta)

    # ("T_nu = ", T_nu, ", T_N = ", T_N, ", xi_N = ", xi_N)
    # ("x_N = ", exp(ln_x_N))
    # Trial stages can leave the physical domain; reject them.
    if !in_physical_domain(pan, u, T_nu)
        reject_state!(du)
        return nothing
    end

    coll_n, coll_rho = collision_terms(pan, T_nu, T_N, xi_N; gap_A=gap_A)
    if pan.verbose
        println("coll_n = ", coll_n)
        println("coll_rho = ", coll_rho)
    end

    dx_dt = dx_dt_interp(pan, x)

    # TODO: [13.08.26] Double-check the signs.
    der_ln_y_n = x / (n * dx_dt) * coll_n
    der_ln_y_rho = x / (rho * dx_dt) * (H * rho_3P(pan, T_N, xi_N; gap_A=gap_A) + coll_rho)

    n_anal = try
        num_dens(pan, T_N, xi_N; gap_A=gap_A)
    catch err
        println("eta = ", eta)
        println("ln_x_N = ", ln_x_N)
        println("m_A/T_N - xi_A = e^eta = ", gap_A)
        println("u = ", u)
        rethrow(err)
    end
    rho_anal = energy_dens(pan, T_N, xi_N; gap_A=gap_A)

    ln_y_n_anal = log(n_anal / ent)
    ln_y_rho_anal = log(rho_anal / ent^(4. / 3.))
    du[1] = der_ln_y_n
    du[2] = der_ln_y_rho
    du[3] = ln_y_n - ln_y_n_anal
    du[4] = ln_y_rho - ln_y_rho_anal
    return nothing
end

"""
From old pandemolate:
    println("Start solve_ivp xi non-zero")
    # TODO: Jobb med solveren, og beregn Jacobian.
    sol = solve_ivp(
        pan.der,
        [pan.log_x_pts[i_xi_nonzero], pan.log_x_pts[-1]],
        y0,
        t_eval=pan.log_x_pts[i_xi_nonzero:],
        rtol=rtol_ode_pan,
        atol=0.,
        # atol=1e-9,
        # method='BDF',
        method='RK45',
        # first_step=pan.log_x_pts[i_xi_nonzero+1]-pan.log_x_pts[i_xi_nonzero],
        max_step=1.
    )
    end = time.time()
    println('solve_ivp time:', end - start)
    """


    """
    # println("Enter pandemolate while-loop ")
    while i_max < n_pts - 1:
        # println(f'Pandemolator while loop iteration i_max={i_max}')
        if i_max > 0:#pan.event_xi_nonzero(pan.log_x_pts[i_max], [rho0*(sf0**4.)]) > 0.: # xi = 0 at beginning of calculation
            println(f'i_max > 0')
            def event_xi(log_x, y):
                return pan.event_xi_nonzero(log_x, y)
            event_xi.terminal = True
            event_xi.direction = -1
            def event_abund(log_x, y):
                return pan.event_abund_large_xi_0(log_x, y)
            event_abund.terminal = True
            event_abund.direction = -1
            println(f'Start solve_ivp xi zero')
            sol_xi0 = solve_ivp(
                pan.der_xi_0,
                [pan.log_x_pts[i_max], pan.log_x_pts[-1]],
                [rho0*(sf0**4.)],
                t_eval=pan.log_x_pts[i_max:],
                events=(event_xi, event_abund),
                rtol=rtol_ode_pan,
                atol=0.,
                method='RK45',
                first_step=pan.log_x_pts[i_max+1]-pan.log_x_pts[i_max]
            )
            println(f'End solve_ivp xi zero')
            i_xi_nonzero = i_max + sol_xi0.t.size - 1

            pan.T_chi_last = (rho0 / (cf.pi2*(2*dof_fac_chi+dof_fac_A)/30.))**0.25
            pan.xi_chi_last = 0.
            for i in range(i_max, i_xi_nonzero + 1):
                ent = pan.ent_interp_T(pan.T_grid_sol[i])
                sf = pan.sf_interp_T(pan.T_grid_sol[i])
                rho = sol_xi0.y[0, i-i_max]/(sf**4.)
                root_sol = root(
                    pan.rho_root,
                    [log(pan.T_chi_last)],
                    jac=pan.jac_rho_root,
                    args=(rho)
                )
                pan.T_chi_grid_sol[i] = exp(root_sol.x[0])
                pan.xi_chi_grid_sol[i] = 0.
                pan.T_chi_last, pan.xi_chi_last = pan.T_chi_grid_sol[i], pan.xi_chi_grid_sol[i]
                pan.xi_A_grid_sol[i] = pan.fac_n_A*pan.xi_chi_grid_sol[i]
                pan.n_chi_grid_sol[i] = pan.n_chi(pan.T_chi_grid_sol[i], pan.xi_chi_grid_sol[i])
                pan.n_A_grid_sol[i] = pan.n_A(pan.T_chi_grid_sol[i], pan.xi_A_grid_sol[i])

            if sol_xi0.t_events[1].size == 0: # abundance always < fac_abund_stop*DM abundance
                sf0 = pan.sf_grid[pan.i_ic + i_xi_nonzero]
                rho0 = sol_xi0.y[0,-1]/(sf0**4.)
                n0 = 2*pan.n_chi(pan.T_chi_last, 0.) + pan.fac_n_A*pan.n_A(pan.T_chi_last, 0.)
                ent0 = pan.ent_grid[pan.i_ic + i_xi_nonzero]
                sf0 = pan.sf_grid[pan.i_ic + i_xi_nonzero]
            else: # abundance becomes > fac_abund_stop*DM abundance, loop ends then due to new n_pts
                pan.i_end = pan.i_ic + i_xi_nonzero
                n_pts = i_xi_nonzero + 1
        else:




        if i_xi_nonzero < n_pts - 1:
            println('End solve_ivp xi non-zero')
            i_max = i_xi_nonzero + sol.t.size - 1
            println("i_xi = ", i_xi_nonzero)
            println("t_size = ", sol.t.size)
            println("i_max = ", i_max)

            pan.T_chi_last = (rho0 / (cf.pi2*(2*dof_fac_chi+dof_fac_A)/30.))**0.25
            pan.xi_chi_last = 0.
            i_start = i_xi_nonzero + 1 if i_xi_nonzero > 0 else 0
            start = time.time()
            println(f'i_start={i_start}, i_max={i_max}')
            for i in range(i_start, i_max + 1):
                if i%100 == 0:
                    println("i = ", i)
                ent = pan.ent_interp_T(pan.T_grid_sol[i])
                sf = pan.sf_interp_T(pan.T_grid_sol[i])
                n = sol.y[0, i-i_xi_nonzero]*ent
                rho = sol.y[1, i-i_xi_nonzero]/(sf**4.)
                root_sol = root(
                    pan.n_rho_root,
                    [log(pan.T_chi_last), (pan.xi_chi_last-pan.m_chi/pan.T_chi_last)],
                    jac=pan.jac_n_rho_root,
                    args=(n, rho),
                    method='lm'
                )
                # println(exp(root_sol.x[0]))
                pan.T_chi_grid_sol[i] = exp(root_sol.x[0])
                pan.xi_chi_grid_sol[i] = min(root_sol.x[1] + pan.m_chi/pan.T_chi_grid_sol[i], (1.-1e-14)*pan.m_A/(pan.fac_n_A*pan.T_chi_grid_sol[i]))#root_sol.x[1] + pan.m_chi/pan.T_chi_grid_sol[i]
                pan.T_chi_last, pan.xi_chi_last = pan.T_chi_grid_sol[i], pan.xi_chi_grid_sol[i]
                pan.xi_A_grid_sol[i] = pan.fac_n_A*pan.xi_chi_grid_sol[i]
                pan.n_chi_grid_sol[i] = pan.n_chi(pan.T_chi_grid_sol[i], pan.xi_chi_grid_sol[i])
                pan.n_A_grid_sol[i] = pan.n_A(pan.T_chi_grid_sol[i], pan.xi_A_grid_sol[i])
            end = time.time()
            println('Root-solving time:', end - start)

            ent0 = pan.ent_interp_T(pan.T_grid_sol[i_max])
            sf0 = pan.sf_interp_T(pan.T_grid_sol[i_max])
            n0 = sol.y[0,-1]*ent0
            rho0 = sol.y[1,-1]/(sf0**4.)
            pan.T_chi_last = pan.T_chi_grid_sol[i_max]
            pan.xi_chi_last = pan.xi_chi_grid_sol[i_max]

            if sol.t_events[0].size != 0 or sol.t.size < 2: # abundance becomes > fac_abund_stop*DM abundance, loop ends then due to new n_pts
                pan.i_end = pan.i_ic + sol.t.size + i_xi_nonzero - 1
                n_pts = i_max + 1
        else:
            i_max = i_xi_nonzero

    # println("Exit pandemolate while-loop ")
    # shorten solution grids, relevant if integration stopped due to large abundance
    pan.t_grid_sol = pan.t_grid_sol[:n_pts]
    pan.T_grid_sol = pan.T_grid_sol[:n_pts]
    pan.H_grid_sol = pan.H_grid_sol[:n_pts]
    pan.T_chi_grid_sol = pan.T_chi_grid_sol[:n_pts]
    pan.xi_chi_grid_sol = pan.xi_chi_grid_sol[:n_pts]
    pan.xi_A_grid_sol = pan.xi_A_grid_sol[:n_pts]
    pan.n_chi_grid_sol = pan.n_chi_grid_sol[:n_pts]
    pan.n_A_grid_sol = pan.n_A_grid_sol[:n_pts]
"""
