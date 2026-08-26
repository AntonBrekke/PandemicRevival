import NonlinearSolve as NLS
import BenchmarkTools as BT
import DifferentialEquations as DE
import DiffEqBase as DEB
import OrdinaryDiffEqRosenbrock as ODER
import OrdinaryDiffEqBDF as ODEB

include(joinpath(@__DIR__, "../src/utils.jl"))
include(joinpath(@__DIR__, "../src/pandemic_result.jl"))
include(joinpath(@__DIR__, "../src/densities.jl"))
include(joinpath(@__DIR__, "../src/coll_3_12.jl"))
include(joinpath(@__DIR__, "../src/coll_12_34.jl"))
include(joinpath(@__DIR__, "../src/collision_table.jl"))

# """
#     Pandemolator
# 
# A struct representing the pandemolator solver for dark sector evolution.
# """
mutable struct Pandemolator{T<:Real, FT, FdT, FEnt, FH, FSf}
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

    dw::DodelsonWidrow{T}

    # Factor for number of A particles in effective number density (always 2)
    fac_n_A::Int64

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
    sf_interp_T_nu::FSf

    function Pandemolator{T}(
        model_params::ModelParams{T},
        N1::Particle{T},
        N2::Particle{T},
        A::Particle{T},
        nu::Particle{T},
        # C_n::Function, C_rho::Function, C_xi0::Function,
        tT_rel::TimeTempRelation{T},
        dw::DodelsonWidrow{T}
    ) where T <: Real
        # TODO: [01.07.26] Ask Anton: Why this factor?
        # Calculate factor for A' particle contribution
        if A.m > N1.m + N2.m
            fac_n_A_val = 2
        else
            # TODO: [16.07.26] Not used
            fac_n_A_val = 1
        end

        # Calculate scale factor (normalised to value at DW production)
        sf_grid_val = (tT_rel.ent_grid[dw.i_ic] ./ tT_rel.ent_grid) .^ (1. / 3.)

        # Create closure functions for T-space interpolations
        t_interp_T_nu = temp_interpolation(tT_rel.T_nu_grid, tT_rel.t_grid)
        dT_nu_dt_interp_T_nu = temp_interpolation(tT_rel.T_nu_grid, tT_rel.dT_nu_dt_grid, neg=true) # Note the minus signs
        ent_interp_T_nu = temp_interpolation(tT_rel.T_nu_grid, tT_rel.ent_grid)
        H_interp_T_nu = temp_interpolation(tT_rel.T_nu_grid, tT_rel.hubble_grid)
        sf_interp_T_nu = temp_interpolation(tT_rel.T_nu_grid, sf_grid_val)

        new{
            T,
            typeof(t_interp_T_nu),
            typeof(dT_nu_dt_interp_T_nu),
            typeof(ent_interp_T_nu),
            typeof(H_interp_T_nu),
            typeof(sf_interp_T_nu)
        }(
            model_params,
            N1, N2, A, nu,
            # C_n, C_rho, C_xi0,
            dw,
            fac_n_A_val,
            # T_nu_interp, dT_nu_dt_interp, ent_interp, H_interp,
            t_interp_T_nu, dT_nu_dt_interp_T_nu, ent_interp_T_nu, 
            H_interp_T_nu, sf_interp_T_nu,
        )
    end
end


function pandemolate(
        tT_rel::TimeTempRelation{T},
        dw::DodelsonWidrow{T},
        pan::Pandemolator{T},
    ; collision_table::Union{Nothing, CollisionTable}=nothing,
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

    i_max = 0
    # i_xi_nonzero = i_max

    u0, du0 = initial_conditions(tT_rel, dw, pan)
    println("u0 = ", u0)
    println("du0 = ", du0)
    differential_vars = [true, true, false, false]

    # TODO: [10.08.26] Remove! Only for testing.
    # u0[4] = 0.0

    log_x_lim = (log_x_pts[1], log_x_pts[end])
    println("log_x_lim = ", log_x_lim)
    temp_lim = pan.N1.m ./ exp.(log_x_lim)
    println("temp_lim = ", temp_lim)

    dae_params = isnothing(collision_table) ? pan : (pan, collision_table)
    prob = DE.DAEProblem{true}(
        dae_func!,
        du0,
        u0,
        log_x_lim,
        dae_params,
        differential_vars = differential_vars,
    )
    tol = 1e-8
    @time sol = DE.solve(
        prob,
        initializealg = DEB.CheckInit(),
        # initializealg = DEB.DefaultInit(),
        # initializealg = DEB.BrownFullBasicInit(),
        # initializealg = DEB.ShampineCollocationInit(),
        ODEB.DFBDF(),
        # ODEB.DImplicitEuler(),
        # ODEB.DABDF2()
        # saveat=log_x_pts,
        # alg_hints=:stiff,
    )

    return sol
    # return nothing
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
    println("T_0 = ", T_0)
    println("x_0 = m_N1 / T_dw = ", pan.N1.m / T_0)
    ent0 = pan.ent_interp_T_nu(T_0)
    rho0 = dw.rho_ic
    sf0 = pan.sf_interp_T_nu(T_0)

    println("rho_dw = ", rho0)

    # y0[1]: Y_n=n/s (Yield)
    # y0[2]: Y_rho = a^4*rho (Energy density scaled with the scale factor 'a')

    y_n_0 = n0/ent0
    y_rho_0 = rho0*(sf0^4)

    xi_N_guess = 0.0
    root_params = (
        pan = pan,
        n_ic = n0,
        rho_ic = rho0,
        # T_N = T_0
    )
    tol = 1e-8
    n_rho_problem = NLS.NonlinearProblem(
        n_rho_root,
        [T_0, xi_N_guess],
        root_params,
        lb = [0., -Inf],
    )
    n_rho_root_sol = NLS.solve(
        n_rho_problem,
        abstol=tol,
        reltol=tol,
    )
    T_N_0 = n_rho_root_sol.u[1]
    # TODO [07.08.26] This initial condition makes little sense. Ask Torsten.
    # T_N_0 = (30. * rho0 / (pi^2 * (2*dof_fac_N + dof_fac_A)))^(1. / 4.)

    println("T_N_0 = ", T_N_0)

    x_N_0 = pan.N1.m / T_N_0
    xi_N_0 = n_rho_root_sol.u[2]

    u0 = [y_n_0, y_rho_0, x_N_0, xi_N_0]
    du0 = [0., 0., 0., 0.]
    return u0, du0
end


# Anton: RHS of Boltzmann equation
function dae_func!(
        out,
        du,
        u,
        pan::Pandemolator{T},
        log_x;
        collision_table::Union{Nothing, CollisionTable}=nothing,
    ) where {T<:Real}
    x = exp(log_x)
    T_nu = pan.N1.m / x
    H = pan.H_interp_T_nu(T_nu)
    dT_nu_dt = pan.dT_nu_dt_interp_T_nu(T_nu)
    ent = pan.ent_interp_T_nu(T_nu)
    sf = pan.sf_interp_T_nu(T_nu)
    yield = u[1]
    n = yield * ent
    rho = u[2] / sf^4
    x_N = u[3]
    T_N = pan.N1.m / x_N
    xi_N = u[4]

    coll_n, coll_rho = isnothing(collision_table) ?
        collision_terms(pan, T_nu, T_N, xi_N) :
        collision_terms(collision_table, T_nu, T_N, xi_N)

    # TODO: [13.08.26] Double-check the signs.
    der_Y_n = - (T_nu / dT_nu_dt) * coll_n / ent
    der_Y_rho = -  sf^4 * (T_nu / dT_nu_dt) * (H * rho_3P(pan, T_N, xi_N) + coll_rho)

    n_anal = num_dens(pan, T_N, xi_N)
    rho_anal = energy_dens(pan, T_N, xi_N)

    out[1] = du[1] - der_Y_n
    out[2] = du[2] - der_Y_rho
    out[3] = n - n_anal
    out[4] = rho - rho_anal
end

function dae_func!(
        out,
        du,
        u,
        params::Tuple{Pandemolator{T}, CollisionTable},
        log_x
    ) where {T<:Real}
    pan, collision_table = params
    dae_func!(out, du, u, pan, log_x; collision_table=collision_table)
end


function rho_3P(pan, T_N, xi_N)
    # Prepared for splitting N masses
    return (
        rho_3P_diff(pan.N1, T_N, xi_N)
        + rho_3P_diff(pan.N2, T_N, xi_N)
        + rho_3P_diff(pan.A, T_N, pan.fac_n_A * xi_N)
    )
end


function num_dens(pan, T_N, xi_N; debug=false)
    # if T_N <= 0
    #     println("Error: T_N <= 0 in num_dens")
    #     println("T_N = ", T_N)
    #     println("xi_N = ", xi_N)
    #     error()
    # end
    n_N1 = number_density(pan.N1, T_N, xi_N, debug=debug)
    n_N2 = number_density(pan.N2, T_N, xi_N, debug=debug)
    n_A = number_density(pan.A, T_N, pan.fac_n_A * xi_N, debug=debug)
    if isnothing(n_A)
        number_density(pan.A, T_N, pan.fac_n_A * xi_N, debug=true)
    end
    if debug
        println("n_N1 = ", n_N1)
        println("n_N2 = ", n_N2)
        println("n_A = ", n_A)
    end
    return max(
        n_N1 + n_N2 + pan.fac_n_A * n_A,
        1e-300
    )
end

function energy_dens(pan, T_N, xi_N)
    return max(
        (
            energy_density(pan.N1, T_N, xi_N) 
            + energy_density(pan.N2, T_N, xi_N) 
            + energy_density(pan.A, T_N, pan.fac_n_A * xi_N)
        ),
        1e-300
    )
end

# Anton: solve root-equations for n, rho with jacobian factor
# TODO: Fix parameters
function n_rho_root(Txi_N, params)
    pan = params.pan
    n_ic = params.n_ic
    rho_ic = params.rho_ic
    # T_N = exp(max(min(Txi_N[1], 10.), -100.))
    # xi_N = min(
    #     Txi_N[1] + pan.N1.m / T_N,
    #     (1. - 1e-14) * pan.A.m / (pan.fac_n_A * T_N)
    # )
    T_N = Txi_N[1]
    xi_N = Txi_N[2]
    n = num_dens(pan, T_N, xi_N)
    rho = energy_dens(pan, T_N, xi_N)
    if n / n_ic < 0
        println("n/n_ic < 0 in n_rho_root")
        println("n/n_ic = ", n / n_ic)
        println("n = ", n)
        num_dens(pan, T_N, xi_N, debug=true)
        error()
        return [log(1e-100), log(rho/rho_ic)]
    end
    if rho / rho_ic < 0
        println("rho/rho_ic < 0 in n_rho_root")
        println("rho/rho_ic = ", rho / rho_ic)
        return [log(n/n_ic), log(1e-100)]
    end
    return [log(n/n_ic), log(rho/rho_ic)]
end

function n_root(xi_N, params)
    pan = params.pan
    n_ic = params.n_ic
    T_N = params.T_N
    n = num_dens(pan, T_N, xi_N)
    if n / n_ic < 0
        println("n/n_ic < 0 in n_root")
        return log(1e-100)
    end
    return log(n / n_ic)
end

function rho_root(xi_N, params)
    pan = params.pan
    n_ic = params.n_ic
    rho_ic = params.rho_ic
    T_N = params.T_N
    rho = energy_dens(pan, T_N, xi_N)
    if rho / rho_ic < 0
        println("rho/rho_ic < 0 in rho_root")
        return log(1e-100)
    end
    return log(rho / rho_ic)
end



function C_n(pan, T_nu, T_N, xi_N)
    """
    Anton: A lot of processes do not contriubute due to equilibrium or no change in particle number. 

    Collision operator describing particle alpha: 
    Cn[alpha]_{I_r -> F_r} = eps^alpha_r int dPI |M|^2 prod_{i in I_r} f_i * prod_{j in F_r} (1+k_j*f_j) / kappa_r 

    kappa_r : symmetry factor 
    eps^alpha_r : = -1 if alpha in F_r, = 1 if alpha in I_r

    From this, have Cn[alpha in I_r] = -Cn[beta in F_r], so 
    Cn[alpha in I_r] + Cn[beta in F_r] = 0

    Code: C_n_3_12(type=0) = C[3]_{3<->12} = int dPI |M|^2 * [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]

    n = n1 + n2 + 2*nX
    Then for example the processes 
    X -> 12:
    Cn[1]_{X->12} + Cn[2]_{X->12} + 2*Cn[X]_{X->12} = 0
    11 -> 22:
    2*Cn[1]_{11->22} + 2*Cn[2]_{11->22} = 0 
    But for 11 <-> XX, 
    2*Cn[1]_{11<->XX} + 4*Cn[X]_{11<->XX} = 2*Cn[X]_{11<->XX}
    and X <-> 1nu
    C[1]_{X<->1nu} + 2*C[X]_{X<->1nu} = C[X]_{X<->1nu}

    Thus in total, our Boltzmann equation is 
    n + 3Hn = C[X]_{X<->1nu} + 2*C[X]_{11<->XX} + 2*C[X]_{22<->XX}
    """
    if T_nu < pan.A.m / 50
        return 0.
    end
    # th, m_Gamma_h2 do not matter anymore
    # as long as mN1 = mN2, xiN1 = xiN2, TN1 = TN2, do not need CX_XX_22 separately -- just add factor 2 
    # TODO: [01.06.26] Why divide by 4 and multiply by 4 in return statement? Symmetry factor (2*2)
    # TODO: [15.07.26] Double check sign of collision term
    temps_AA_11 = [T_N, T_N, T_N, T_N]
    xis_AA_11 = [
        pan.fac_n_A * xi_N,
        pan.fac_n_A * xi_N,
        xi_N,
        xi_N,
    ]
    C_AA_N1N1 = coll_12_34(
        pan.mp,
        pan.A,
        pan.A,
        pan.N1,
        pan.N1,
        temps_AA_11,
        xis_AA_11,
    ) / 4.
    temps_AA_22 = [T_N, T_N, T_N, T_N]
    xis_AA_22 = [
        pan.fac_n_A * xi_N,
        pan.fac_n_A * xi_N,
        xi_N,
        xi_N,
    ]
    C_AA_N2N2 = coll_12_34(
        pan.mp,
        pan.A,
        pan.A,
        pan.N2,
        pan.N2,
        temps_AA_22,
        xis_AA_22,
    ) / 4.
    temps_A_N2nu = [T_N, T_N, T_nu]
    xis_A_N2nu = [
        pan.fac_n_A * xi_N,
        xi_N,
        0.,
    ]
    C_A_N2nu = coll_3_12(
        pan.mp,
        pan.N2,
        pan.nu,
        pan.A,
        temps_A_N2nu,
        xis_A_N2nu;
        sq_amp_func=coll_A_Nnu_sq_amp,
        energy_type=Val(0),
    )
    # Factor 2 from number change in n_d = n_N1 + n_N2 + 2*n_A
    return - C_A_N2nu - 2. * C_AA_N1N1 - 2. * C_AA_N2N2
end

# rho = rho_N1 + rho_N2 + rho_X
function C_rho(pan, T_nu, T_N, xi_N)
    """
    Anton: Internal processes of the DS does not contribute due to energy conservation. 
    C1_X->12 + C2_X->12 + CX_X->12 
    = int dPi * (2pi)^4 delta(E1+E2+E3) (E1 + E2 - E3)*fX*(1+k1*f1)*(1+k2*f2) = 0 

    Hence the only contributions come from energy transer between the SM and the DS

    Code: C_rho_3_12(type) = int dPI E_type |M|^2 * [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]

    Trick to save calculation: 
    C[1]_{3<->12} + C[3]_{3<->12}
    = int dPI (E3 - E1) |M|^2 [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]
    = int dPI E_2 |M|^2 [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]
    = C_rho_3_12(type=2, ...)
    """
    if T_nu < pan.A.m / 50.
        return 0.
    end

    temps_A_N2nu = [T_N, T_N, T_nu]
    xis_A_N2nu = [
        pan.fac_n_A * xi_N,
        xi_N,
        0.,
    ]
    CA_A_N2nu = coll_3_12(
        pan.mp,
        pan.N2,
        pan.nu,
        pan.A,
        temps_A_N2nu,
        xis_A_N2nu;
        sq_amp_func=coll_A_Nnu_sq_amp,
        energy_type=Val(3)
    )
    CN2_A_N2nu = coll_3_12(
        pan.mp,
        pan.N2,
        pan.nu,
        pan.A,
        temps_A_N2nu,
        xis_A_N2nu;
        sq_amp_func=coll_A_Nnu_sq_amp,
        energy_type=Val(1)
    )
    # Cnu_A_N2nu = coll_3_12(
    #     pan.mp,
    #     pan.N2,
    #     pan.nu,
    #     pan.A,
    #     temps_A_N2nu,
    #     xis_A_N2nu;
    #     sq_amp_func=coll_A_Nnu_sq_amp,
    #     energy_type=Val(2)
    # )
    # TODO: [15.07.26] Double check sign
    # return Cnu_A_N2nu
    return - CA_A_N2nu + CN2_A_N2nu
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
