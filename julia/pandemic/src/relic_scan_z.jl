"""
relic_scan_z.jl

Same as relic_scan.jl, but wired to the z-parameterized solver
(`PandemolatorZ`/`pandemolate_z` in pandemolator_z.jl) instead of the
original T-keyed `Pandemolator`/`pandemolate`. See pandemolator_z.jl's
header for why z; the two solvers were validated to agree in
test/test_pandemolator_z.jl.

Scans the (y, sin^2 2*theta) parameter plane at fixed particle content
(masses/dof of N1, N2, A, nu) and finds the y at which Omega h^2(y, theta)
equals the Planck value (`omega_d0` in constants_functions.jl).

CONVENTIONS
===========
- `y` is the internal `ModelParams.y`, differing from the "y_pyt" plot
  convention by sqrt(N1.dof * A.dof * nu.dof) -- see test/test_pandemolate.jl.
- `sin2_2theta` means sin^2(2*theta).

WHAT'S BUILT ONCE
==================
Same structure as relic_scan.jl: `TimeTempRelation` and `PandemolatorZ`'s
z-space interpolation closures are built once for the whole scan (one
mutable `pan`, only its `mp` field changes per point); `DodelsonWidrow` is
built once per sin2_2theta column, since it depends on theta but not y.

ROBUSTNESS
==========
Same caveats as relic_scan.jl: no fallback integrator, "converged" is a
retcode + plateau heuristic (see `check_plateau`), and the open collision-
term sign/prefactor TODOs are inherited as-is.
"""

include(joinpath(@__DIR__, "pandemolator_z.jl"))
import NonlinearSolve as NLS

const OMEGA_H2_TARGET_Z = omega_d0   # constants_functions.jl, Planck 2018 central value (0.12)

"""
    RelicPointZ

One root-found (or attempted) point of the scan. See `RelicPoint` in
relic_scan.jl.
"""
struct RelicPointZ
    sin2_2theta::Float64
    theta::Float64
    y::Float64
    omega_h2::Float64
    converged::Bool
    plateau_ok::Bool
    retcode::Symbol
    branch::Int
end

"""
    ScanConfigZ

See `ScanConfig` in relic_scan.jl.
"""
Base.@kwdef struct ScanConfigZ
    sin2_2theta_min::Float64
    sin2_2theta_max::Float64
    n_theta::Int = 40
    log10y_min::Float64
    log10y_max::Float64
    n_coarse::Int = 15
    root_xtol::Float64 = 1e-3
    plateau_frac::Float64 = 0.05
    plateau_tol::Float64 = 1e-3
    omega_reltol_warn::Float64 = 1e-2
    warm_start_halfwidth::Float64 = 0.5
    # ODE tolerances passed to `pandemolate_z` (`nothing` = solver default).
    ode_reltol::Union{Nothing, Float64} = nothing
    ode_abstol::Union{Nothing, Float64} = nothing
    # Wall-clock limit per solve in seconds, including the initial conditions;
    # a solve that exceeds it counts as failed. See `check_deadline`.
    solve_wall_limit::Float64 = Inf
end

"""
    omega_h2_of(pan, u, z)

Omega h^2 implied by the solver state `u` at z (shared by `final_omega_h2_z`
and `check_plateau_z`).
"""
function omega_h2_of(pan::PandemolatorZ, u, z)
    ln_x_N_f = u[3]
    eta_f = u[4]
    T_N_f = T_N_from_ln_x_N(pan, ln_x_N_f)
    xi_N_f = xi_from_eta(pan, eta_f, ln_x_N_f)

    ent_f = pan.ent_interp_z(z)

    y_N1_f = number_density(pan.N1, T_N_f, xi_N_f) / ent_f
    y_N2_f = number_density(pan.N2, T_N_f, xi_N_f) / ent_f
    y_A_f = number_density(pan.A, T_N_f, pan.fac_n_A * xi_N_f; gap=gap_A_from_eta(eta_f)) / ent_f

    rho_dm0 = (pan.N1.m * y_N1_f + pan.N2.m * y_N2_f + pan.fac_n_A * pan.A.m * y_A_f) * s0
    return rho_dm0 / rho_crit0_h2
end

"""
    check_plateau_z(pan, sol; frac=0.05, tol=1e-3)

Whether the *physical* target quantity, Omega h^2(z) (`omega_h2_of`), has
settled over the last `frac` of the solution, to relative spread `tol`.

Differs from the older Y_n-based check (`exp.(sol[1,:])`, n = n_N1 + n_N2 +
2 n_A): right after pandemic thermalization at large y (or generally whenever
A' is a non-negligible fraction of n right up to x = 100), Y_n and Y_rho can
still visibly move as the N/A' split relaxes towards its late-time value,
while the mass-weighted, physically relevant Omega h^2 is already frozen (A'
contributes little to it once `fac_n_A * m_A * Y_A << m_N1 Y_N1`). The old
check therefore flagged many well-converged points as not-plateaued, which
`theta_point` then had to treat as failed solves -- see run_relic_scan's
"no root" cases at large y. Checking the target quantity directly removes
that false-alarm, and is at least as strict for genuine non-convergence
(Omega h^2 cannot plateau if its ingredients, including any dominant
component, are still evolving)."""
function check_plateau_z(pan::PandemolatorZ, sol; frac::Float64=0.05, tol::Float64=1e-3)
    n_pts = length(sol.t)
    n_tail = min(n_pts, max(3, ceil(Int, frac * n_pts)))
    idx = (n_pts - n_tail + 1):n_pts
    tail = [omega_h2_of(pan, sol.u[i], sol.t[i]) for i in idx]
    mean_tail = sum(tail) / length(tail)
    return (maximum(tail) - minimum(tail)) / abs(mean_tail) < tol
end

"""
    final_omega_h2_z(pan, sol)

As `final_omega_h2` in relic_scan.jl, but `sol.t[end]` IS z directly (no
inversion to T_nu needed before the interpolant lookup, since PandemolatorZ's
interpolants are keyed on z natively).
"""
final_omega_h2_z(pan::PandemolatorZ, sol) = omega_h2_of(pan, sol.u[end], sol.t[end])

"""As `solve_point` in relic_scan.jl, calling `pandemolate_z`."""
function solve_point_z(
        pan::PandemolatorZ{T},
        tT_rel::TimeTempRelation{T},
        dw::DodelsonWidrow{T},
        y::Float64, theta::Float64,
        cfg::ScanConfigZ
    ) where T <: Real
    pan.mp = ModelParams{T}(y, theta)
    pan.deadline = time() + cfg.solve_wall_limit
    try
        sol = pandemolate_z(tT_rel, dw, pan; reltol=cfg.ode_reltol, abstol=cfg.ode_abstol)
        ok_retcode = DE.successful_retcode(sol)
        ok_plateau = ok_retcode && check_plateau_z(pan, sol; frac=cfg.plateau_frac, tol=cfg.plateau_tol)
        omega_h2 = ok_retcode ? final_omega_h2_z(pan, sol) : NaN
        converged = ok_retcode && ok_plateau && isfinite(omega_h2) && omega_h2 > 0
        return (converged ? omega_h2 : NaN), converged, ok_plateau, Symbol(sol.retcode)
    catch e
        if e isa SolveDeadlineExceeded
            @warn "pandemolate_z exceeded the wall limit of $(cfg.solve_wall_limit) s at y=$y, theta=$theta"
            return NaN, false, false, :WallLimit
        end
        @warn "pandemolate_z failed at y=$y, theta=$theta" exception = (e, catch_backtrace())
        return NaN, false, false, :Exception
    finally
        pan.deadline = Inf
    end
end

"""As `find_omega_brackets` in relic_scan.jl."""
function find_omega_brackets_z(residual::Function, log10y_min::Float64, log10y_max::Float64; n_scan::Int=15)
    xs = collect(range(log10y_min, log10y_max, length=n_scan))
    vals = residual.(xs)
    brackets = Tuple{Float64,Float64}[]
    for i in 1:(n_scan - 1)
        a, b = vals[i], vals[i+1]
        if isfinite(a) && isfinite(b) && sign(a) != sign(b)
            push!(brackets, (xs[i], xs[i+1]))
        end
    end
    return brackets
end

function _bracket_root_z(residual::Function, lo::Float64, hi::Float64, xtol::Float64)
    prob = NLS.IntervalNonlinearProblem((u, p) -> residual(u), (lo, hi))
    return NLS.solve(prob, NLS.Brent(); abstol=xtol)
end

"""As `scan_theta_column` in relic_scan.jl."""
function scan_theta_column_z(
        theta::Float64, sin2_2theta::Float64,
        pan::PandemolatorZ{Float64}, tT_rel::TimeTempRelation{Float64},
        cfg::ScanConfigZ;
        warm_start_log10y::Union{Nothing,Vector{Float64}}=nothing,
    )
    dw = DodelsonWidrow{Float64}(pan.N1.m, theta, tT_rel)

    function residual(log10y::Float64)
        y = 10.0^log10y
        omega_h2, converged, _, _ = solve_point_z(pan, tT_rel, dw, y, theta, cfg)
        return (!converged || !isfinite(omega_h2) || omega_h2 <= 0) ? NaN : log10(omega_h2) - log10(OMEGA_H2_TARGET_Z)
    end

    roots = Float64[]
    if warm_start_log10y !== nothing
        for x0 in warm_start_log10y
            lo, hi = x0 - cfg.warm_start_halfwidth, x0 + cfg.warm_start_halfwidth
            flo, fhi = residual(lo), residual(hi)
            if isfinite(flo) && isfinite(fhi) && sign(flo) != sign(fhi)
                sol = _bracket_root_z(residual, lo, hi, cfg.root_xtol)
                DE.successful_retcode(sol) && push!(roots, sol.u)
            end
        end
    end

    if isempty(roots)
        for (lo, hi) in find_omega_brackets_z(residual, cfg.log10y_min, cfg.log10y_max; n_scan=cfg.n_coarse)
            sol = _bracket_root_z(residual, lo, hi, cfg.root_xtol)
            if DE.successful_retcode(sol)
                push!(roots, sol.u)
            else
                @warn "root-find failed in bracket (log10y = $lo, $hi) at sin2_2theta=$sin2_2theta"
            end
        end
    end
    sort!(roots)

    points = RelicPointZ[]
    for (branch, log10y) in enumerate(roots)
        y = 10.0^log10y
        omega_h2, converged, plateau_ok, retcode = solve_point_z(pan, tT_rel, dw, y, theta, cfg)
        if converged
            delta = abs(omega_h2 - OMEGA_H2_TARGET_Z) / OMEGA_H2_TARGET_Z
            if delta > cfg.omega_reltol_warn
                @warn "Root at sin2_2theta=$sin2_2theta branch $branch converged to Omega h^2=$omega_h2 (target $OMEGA_H2_TARGET_Z, relative diff $delta)"
            end
        else
            @warn "Root at sin2_2theta=$sin2_2theta, y=$y did not pass the convergence/plateau checks (retcode=$retcode)"
        end
        push!(points, RelicPointZ(sin2_2theta, theta, y, omega_h2, converged, plateau_ok, retcode, branch))
    end
    println("Done with m_N = $(pan.N1.m), sin2_2theta = $sin2_2theta, found $(length(points)) roots: $(join([p.y for p in points], ", "))")
    return points, roots
end

"""As `run_scan` in relic_scan.jl, building a `PandemolatorZ`."""
function run_scan_z(
        N1::Particle{Float64}, N2::Particle{Float64}, A::Particle{Float64},
        nu::Particle{Float64}, cfg::ScanConfigZ; verbose::Bool=false
    )
    tT_rel = TimeTempRelation{Float64}()

    theta0 = asin(sqrt(cfg.sin2_2theta_min)) / 2.0
    mp0 = ModelParams{Float64}(10.0^cfg.log10y_min, theta0)
    pan = PandemolatorZ{Float64}(mp0, N1, N2, A, nu, tT_rel, verbose)

    sin2_2theta_grid = exp10.(range(log10(cfg.sin2_2theta_min), log10(cfg.sin2_2theta_max), length=cfg.n_theta))

    all_points = RelicPointZ[]
    warm_x = nothing
    for sin2_2theta in sin2_2theta_grid
        theta = asin(sqrt(sin2_2theta)) / 2.0
        points, roots = scan_theta_column_z(
            theta, sin2_2theta,
            pan,
            tT_rel,
            cfg;
            warm_start_log10y=warm_x
        )
        append!(all_points, points)
        if !isempty(roots)
            warm_x = roots
        end
    end
    return all_points
end

"""As `save_results` in relic_scan.jl."""
function save_results_z(points::Vector{RelicPointZ}, path::String)
    open(path, "w") do io
        println(io, "sin2_2theta,theta,y,omega_h2,converged,plateau_ok,retcode,branch")
        for p in points
            println(io, "$(p.sin2_2theta),$(p.theta),$(p.y),$(p.omega_h2),$(p.converged),$(p.plateau_ok),$(p.retcode),$(p.branch)")
        end
    end
    return nothing
end
