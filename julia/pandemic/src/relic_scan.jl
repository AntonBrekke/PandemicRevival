"""
relic_scan.jl

Scans the (y, sin^2 2*theta) parameter plane at fixed particle content
(masses/dof of N1, N2, A, nu) and finds the y at which Omega h^2(y, theta)
equals the Planck value (`omega_d0` in constants_functions.jl), i.e. the
contour of correct relic abundance -- a Fig.-3-style plot from Brekke,
Bringmann, Melkild & Schmidt-Hoberg. Wired directly into the existing
`pandemolate` DAE solver in pandemolator.jl.

CONVENTIONS
===========
- `y` throughout this file is the internal `ModelParams.y` that the solver
  itself consumes. This differs from the "y_pyt" convention used in the
  existing Python plots/scripts by a factor sqrt(N1.dof * A.dof * nu.dof)
  -- see the comment in test/test_pandemolate.jl about a degrees-of-freedom
  factor missing in the Python collision term. `RelicPoint` reports both,
  see `y_pyt_from_y`.
- `theta` is the active-sterile mixing angle; `sin2_2theta` means sin^2(2*theta)
  (matches the `th`/`sin2_2th` naming already used in the test files).

SCAN DIRECTION AND WHAT IS BUILT ONCE
======================================
The outer loop is over sin2_2theta (theta); for each theta we root-find y.
This mirrors the physics: the Dodelson-Widrow initial condition
(`DodelsonWidrow`) depends on theta but NOT on y, so for a fixed theta
"column" it is built exactly ONCE and reused unchanged for every y tried
while root-finding that column.

Two more things do not depend on (y, theta) at all and are also built
exactly ONCE for the *entire* scan, not per column:
  - `TimeTempRelation` (the background T_SM/T_nu/H/entropy cosmology) --
    this alone dominates the cost of building a `Pandemolator` from scratch.
  - `Pandemolator`'s T_nu-space interpolation closures, which are only a
    function of `TimeTempRelation`. Since `Pandemolator` is a mutable
    struct, one instance is built up front and then only its `mp`
    (ModelParams) field is mutated per point, instead of reconstructing the
    whole object (and rebuilding those interpolations) at every one of the
    hundreds/thousands of scan points. `DodelsonWidrow` is not a
    `Pandemolator` field at all -- it is threaded through as a plain
    argument to `pandemolate`/`solve_point`, built once per theta column.

ROBUSTNESS
==========
`pandemolate` is a stiff DAE solve with several open TODOs in the collision
terms upstream (see the code review notes) and no fallback algorithm if the
chosen Rodas4P integrator struggles at some corner of parameter space. A
scan must not let one bad point kill the whole run, so `solve_point` always
catches exceptions and reports failure through its return values rather
than propagating.

The solver returns a raw DifferentialEquations.jl solution, not a
convenient (converged, plateau_reached, ...) info struct, so "did this
point converge" is assessed here via:
  1. `DE.successful_retcode(sol)` (or the underlying ODE integration failed)
  2. `check_plateau`: the last `plateau_frac` fraction of the integrated
     x-range must show a comoving yield (Y_n) that has stopped moving to
     within `plateau_tol` (relative) -- a proxy for "freeze-out was
     actually reached inside the integrated range", not just "the ODE
     solver terminated".
This is a first-pass heuristic, not a validated diagnostic -- tighten
`plateau_frac`/`plateau_tol` in `ScanConfig` as needed.

THREADING
=========
Single-threaded as written: `run_scan` mutates one shared `Pandemolator`
across the whole scan, which is the whole point of the performance
optimisation above but is NOT thread-safe. If this needs to go parallel
across theta columns later, give each thread/task its own `Pandemolator`
(built once each, same idea as here) and its own `DodelsonWidrow`, and
pre-warm the lazily-initialised interpolation caches in
constants_functions.jl / densities.jl (`_gstar_cache`, `_dens_cache`,
`_dw_cache` -- all `Ref`s populated on first use) with one single-threaded
solve before spawning threads, to avoid a first-use race.

OPEN PHYSICS CAVEATS (not addressed here -- see code review notes)
====================================================================
The collision-term code carries several unresolved sign/prefactor TODOs
(pandemolator.jl: "why this factor?", "double check sign of collision
term", coll_12_34.jl / coll_3_12.jl: several "check sign" / "check
prefactors"). This scan is infrastructure built on top of the physics as
it currently stands; it will silently inherit whatever those turn out to
be.
"""

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "time_temp_relation.jl"))
include(joinpath(@__DIR__, "dodelson_widrow.jl"))
include(joinpath(@__DIR__, "pandemolator.jl"))

const OMEGA_H2_TARGET = omega_d0   # constants_functions.jl, Planck 2018 central value (0.12)

"""
    y_pyt_from_y(y, N1, A, nu)

Converts the internal `ModelParams.y` to the convention used in the
existing Python plots/scripts, `y_pyt = y * sqrt(N1.dof * A.dof * nu.dof)`.
Assumes `N1.dof == N2.dof` (as in every benchmark point in the repo so far).
"""
y_pyt_from_y(y, N1::Particle, A::Particle, nu::Particle) = y * sqrt(N1.dof * A.dof * nu.dof)

"""
    y_from_y_pyt(y_pyt, N1, A, nu)

Inverse of `y_pyt_from_y`.
"""
y_from_y_pyt(y_pyt, N1::Particle, A::Particle, nu::Particle) = y_pyt / sqrt(N1.dof * A.dof * nu.dof)

"""
    RelicPoint

One root-found (or attempted) point of the scan: the y (internal
convention) at which Omega h^2(y, theta) == OMEGA_H2_TARGET for this theta,
plus enough diagnostics to tell a trustworthy root from a bad one.
"""
struct RelicPoint
    sin2_2theta::Float64
    theta::Float64
    y::Float64
    y_pyt::Float64
    omega_h2::Float64
    converged::Bool     # retcode success AND plateau reached AND omega_h2 finite & positive
    plateau_ok::Bool
    retcode::Symbol
    branch::Int         # which root along this theta column this is (1, 2, ... sorted by y)
end

"""
    ScanConfig

`sin2_2theta_{min,max}`/`n_theta`: the (log-spaced) theta grid.
`log10y_{min,max}`: bracket search range for y (internal convention) at
each theta -- there is no universally "right" default here, it depends on
the mass panel being scanned; size it from a benchmark point you already
trust (e.g. via `y_from_y_pyt`).
"""
Base.@kwdef struct ScanConfig
    sin2_2theta_min::Float64
    sin2_2theta_max::Float64
    n_theta::Int = 40
    log10y_min::Float64
    log10y_max::Float64
    n_coarse::Int = 15                  # resolution of the coarse bracket scan
    root_xtol::Float64 = 1e-3           # tolerance in log10(y) for the Brent root-find
    plateau_frac::Float64 = 0.05        # fraction of the tail of the x-grid checked for a flat yield
    plateau_tol::Float64 = 1e-3         # relative variation allowed in that tail
    omega_reltol_warn::Float64 = 1e-2   # just triggers a @warn if a "converged" root misses target by more
    warm_start_halfwidth::Float64 = 0.5 # in log10(y); continuation between neighbouring theta columns
end

"""
    check_plateau(sol; frac, tol) -> Bool

Heuristic freeze-out check: compares the spread of the comoving yield
Y_n = exp.(sol[1,:]) over the last `frac` fraction of saved points (at
least 3) against `tol`, relative to their mean. See module docstring.
"""
function check_plateau(sol; frac::Float64=0.05, tol::Float64=1e-3)
    n_pts = length(sol.t)
    n_tail = min(n_pts, max(3, ceil(Int, frac * n_pts)))
    tail = exp.(sol[1, end-n_tail+1:end])
    mean_tail = sum(tail) / length(tail)
    return (maximum(tail) - minimum(tail)) / abs(mean_tail) < tol
end

"""
    final_omega_h2(pan, sol)

Reads the relic abundance off the end of a `pandemolate` solution directly
(cheap: just the final state), rather than via `transform_sol` (which
recomputes the whole trajectory and is unnecessary for a scan that only
needs the frozen-out endpoint).

Omega h^2 = (m_N1*Y_N1 + m_N2*Y_N2 + fac_n_A*m_A*Y_A) * s0 / rho_crit0_h2,
i.e. the actual mass-weighted comoving yield today -- NOT `m_N1 * Y_n`,
which would silently be wrong whenever m_N1 != m_N2.
"""
function final_omega_h2(pan::Pandemolator, sol)
    u_f = sol.u[end]
    ln_x_N_f = u_f[3]
    eta_f = u_f[4]
    T_N_f = T_N_from_ln_x_N(pan, ln_x_N_f)
    xi_N_f = xi_from_eta(pan, eta_f, ln_x_N_f)

    T_nu_f = pan.N1.m / exp(sol.t[end])
    ent_f = pan.ent_interp_T_nu(T_nu_f)

    y_N1_f = number_density(pan.N1, T_N_f, xi_N_f) / ent_f
    y_N2_f = number_density(pan.N2, T_N_f, xi_N_f) / ent_f
    y_A_f = number_density(pan.A, T_N_f, pan.fac_n_A * xi_N_f) / ent_f

    rho_dm0 = (pan.N1.m * y_N1_f + pan.N2.m * y_N2_f + pan.fac_n_A * pan.A.m * y_A_f) * s0
    return rho_dm0 / rho_crit0_h2
end

"""
    solve_point(pan, tT_rel, dw, y, theta, cfg) -> (omega_h2, converged, plateau_ok, retcode)

Solves the DAE at one (y, theta) point, mutating `pan.mp` in place and
passing `dw` straight through to `pandemolate` (`tT_rel` and `pan`'s
interpolation closures were already built once, outside the scan -- see
module docstring). Never throws: a solver failure
is reported through the return flags (`omega_h2 = NaN`, `converged =
false`) so one bad point does not kill the whole scan.
"""
function solve_point(
        pan::Pandemolator{T},
        tT_rel::TimeTempRelation{T},
        dw::DodelsonWidrow{T},
        y::Float64, theta::Float64,
        cfg::ScanConfig
    ) where T <: Real
    pan.mp = ModelParams{T}(y, theta)
    try
        sol = pandemolate(tT_rel, dw, pan)
        ok_retcode = DE.successful_retcode(sol)
        ok_plateau = ok_retcode && check_plateau(sol; frac=cfg.plateau_frac, tol=cfg.plateau_tol)
        omega_h2 = ok_retcode ? final_omega_h2(pan, sol) : NaN
        converged = ok_retcode && ok_plateau && isfinite(omega_h2) && omega_h2 > 0
        return (converged ? omega_h2 : NaN), converged, ok_plateau, Symbol(sol.retcode)
    catch e
        @warn "pandemolate failed at y=$y, theta=$theta" exception = (e, catch_backtrace())
        return NaN, false, false, :Exception
    end
end

"""
    find_omega_brackets(residual, log10y_min, log10y_max; n_scan) -> Vector{Tuple{Float64,Float64}}

Coarse scan for ALL sign changes of `residual` over [log10y_min,
log10y_max], so a resonant / non-monotonic Omega h^2(y) at fixed theta does
not silently lose roots to a single bracket.
"""
function find_omega_brackets(residual::Function, log10y_min::Float64, log10y_max::Float64; n_scan::Int=15)
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

function _bracket_root(residual::Function, lo::Float64, hi::Float64, xtol::Float64)
    prob = NLS.IntervalNonlinearProblem((u, p) -> residual(u), (lo, hi))
    return NLS.solve(prob, NLS.Brent(); abstol=xtol)
end

"""
    scan_theta_column(theta, sin2_2theta, pan, tT_rel, cfg; warm_start_log10y=nothing)
        -> (Vector{RelicPoint}, Vector{Float64})

Fixed theta: builds the Dodelson-Widrow initial condition once, then
root-finds every y (internal convention) at which Omega h^2(y, theta) hits
`OMEGA_H2_TARGET`. Tries cheap warm-started brackets around
`warm_start_log10y` (roots from a neighbouring theta) first; falls back to
a full coarse bracket scan over `[cfg.log10y_min, cfg.log10y_max]` if that
finds nothing. Returns both the assessed `RelicPoint`s and the raw
log10(y) roots (for continuation into the next column). The particle
content (N1, N2, A, nu) is read off `pan`, which already carries it.
"""
function scan_theta_column(
        theta::Float64, sin2_2theta::Float64,
        pan::Pandemolator{Float64}, tT_rel::TimeTempRelation{Float64},
        cfg::ScanConfig;
        warm_start_log10y::Union{Nothing,Vector{Float64}}=nothing,
    )
    dw = DodelsonWidrow{Float64}(pan.N1.m, theta, tT_rel)

    function residual(log10y::Float64)
        y = 10.0^log10y
        omega_h2, converged, _, _ = solve_point(pan, tT_rel, dw, y, theta, cfg)
        return (!converged || !isfinite(omega_h2) || omega_h2 <= 0) ? NaN : log10(omega_h2) - log10(OMEGA_H2_TARGET)
    end

    roots = Float64[]
    if warm_start_log10y !== nothing
        for x0 in warm_start_log10y
            lo, hi = x0 - cfg.warm_start_halfwidth, x0 + cfg.warm_start_halfwidth
            flo, fhi = residual(lo), residual(hi)
            if isfinite(flo) && isfinite(fhi) && sign(flo) != sign(fhi)
                sol = _bracket_root(residual, lo, hi, cfg.root_xtol)
                DE.successful_retcode(sol) && push!(roots, sol.u)
            end
        end
    end

    if isempty(roots)
        for (lo, hi) in find_omega_brackets(residual, cfg.log10y_min, cfg.log10y_max; n_scan=cfg.n_coarse)
            sol = _bracket_root(residual, lo, hi, cfg.root_xtol)
            if DE.successful_retcode(sol)
                push!(roots, sol.u)
            else
                @warn "root-find failed in bracket (log10y = $lo, $hi) at sin2_2theta=$sin2_2theta"
            end
        end
    end
    sort!(roots)

    points = RelicPoint[]
    for (branch, log10y) in enumerate(roots)
        y = 10.0^log10y
        omega_h2, converged, plateau_ok, retcode = solve_point(pan, tT_rel, dw, y, theta, cfg)
        if converged
            Δ = abs(omega_h2 - OMEGA_H2_TARGET) / OMEGA_H2_TARGET
            if Δ > cfg.omega_reltol_warn
                @warn "Root at sin2_2theta=$sin2_2theta branch $branch converged to Omega h^2=$omega_h2 (target $OMEGA_H2_TARGET, relative diff $Δ)"
            end
        else
            @warn "Root at sin2_2theta=$sin2_2theta, y=$y did not pass the convergence/plateau checks (retcode=$retcode)"
        end
        push!(points, RelicPoint(sin2_2theta, theta, y, y_pyt_from_y(y, pan.N1, pan.A, pan.nu), omega_h2, converged, plateau_ok, retcode, branch))
    end
    return points, roots
end

"""
    run_scan(N1, N2, A, nu, cfg::ScanConfig; verbose=false) -> Vector{RelicPoint}

Scans sin2_2theta over `cfg.n_theta` log-spaced steps in
`[sin2_2theta_min, sin2_2theta_max]`; for each, root-finds every y giving
Omega h^2(y, theta) == OMEGA_H2_TARGET. See the module docstring for what
gets built once vs. per-column vs. per-point, and for the (lack of)
thread-safety.
"""
function run_scan(
        N1::Particle{Float64}, N2::Particle{Float64}, A::Particle{Float64},
        nu::Particle{Float64}, cfg::ScanConfig; verbose::Bool=false
    )
    tT_rel = TimeTempRelation{Float64}()

    theta0 = asin(sqrt(cfg.sin2_2theta_min)) / 2.0
    mp0 = ModelParams{Float64}(10.0^cfg.log10y_min, theta0)
    pan = Pandemolator{Float64}(mp0, N1, N2, A, nu, tT_rel, verbose)

    sin2_2theta_grid = exp10.(range(log10(cfg.sin2_2theta_min), log10(cfg.sin2_2theta_max), length=cfg.n_theta))

    all_points = RelicPoint[]
    warm_x = nothing
    for sin2_2theta in sin2_2theta_grid
        theta = asin(sqrt(sin2_2theta)) / 2.0
        points, roots = scan_theta_column(theta, sin2_2theta, pan, tT_rel, cfg; warm_start_log10y=warm_x)
        append!(all_points, points)
        if !isempty(roots)
            warm_x = roots
        end
    end
    return all_points
end

"""
    save_results(points, path)

Writes the scan result to a CSV for downstream plotting.
"""
function save_results(points::Vector{RelicPoint}, path::String)
    open(path, "w") do io
        println(io, "sin2_2theta,theta,y,y_pyt,omega_h2,converged,plateau_ok,retcode,branch")
        for p in points
            println(io, "$(p.sin2_2theta),$(p.theta),$(p.y),$(p.y_pyt),$(p.omega_h2),$(p.converged),$(p.plateau_ok),$(p.retcode),$(p.branch)")
        end
    end
    return nothing
end
