"""
relic_scan_theta_z.jl

Relic scan over (m_N, y): for each dark-matter mass m_N (= m_N1 = m_N2,
m_A = 2.5 m_N) and coupling y, root-find the mixing sin^2(2 theta) for which
Omega h^2 = `omega_d0`, using the z-parameterized solver. This is the
complement of relic_scan_z.jl / relic_scan_mN_z.jl, which root-find y at fixed
theta; with theta as the unknown, contours of constant y in the
(m_N, sin^2 2theta) plane come out directly.

ROOTS AND BRANCHES
==================
At fixed (m_N, y), Omega h^2 can cross the target more than once in theta
(production-limited "freeze-in" branch and thermalized "freeze-out" branch).
By default (`first_root_only`) only the freeze-in root is searched for: the
first crossing where Omega h^2 rises through the target with increasing theta.
With `first_root_only = false`, every sign change on the coarse grid is refined
with Brent's method, and each root is stored with the sign of dOmega/dtheta.

FREEZE-IN SEARCH
================
1. Start: a predicted root from the couplings already scanned at this mass
   (linear in log10 y from the last two roots, or the last root with slope
   `default_slope` if there is only one). Step away from it, in the direction
   in which the residual log10(Omega h^2 / omega_d0) points, with growing steps
   until the residual changes sign upwards. Without a prediction, or if this
   fails, the coarse grid is evaluated in increasing theta (at the looser
   `ode_reltol_coarse`) until the first upward crossing.
2. Refinement: Illinois (modified regula falsi) on the residual, which is close
   to linear in log10 sin^2(2 theta) near the root, until |Omega h^2 / omega_d0
   - 1| < `omega_rtol` or the bracket is narrower than `root_xtol`. The last
   solve is the reported point, so no extra solve is needed.

Masses are independent, so the driver (run_relic_scan_theta_z.jl) runs one
process per mass; within one mass, y is scanned in increasing order.
"""

include(joinpath(@__DIR__, "relic_scan_z.jl"))

struct RelicPointThetaZ
    m_N::Float64
    y::Float64
    sin2_2theta::Float64
    omega_h2::Float64
    converged::Bool
    plateau_ok::Bool
    retcode::Symbol
    branch::Int
    slope::Int      # sign of dOmega/dsin^2(2theta) at the root (+1 or -1)
    n_failed::Int   # failed solves in this (m_N, y) search
end

Base.@kwdef struct ScanConfigThetaZ
    log10s_min::Float64 = -19.0
    log10s_max::Float64 = -6.0
    n_coarse::Int = 14
    # In decades of sin^2(2 theta); 2e-3 is ~0.5% in sin^2(2 theta).
    root_xtol::Float64 = 2e-3
    warm_start_halfwidth::Float64 = 0.75
    plateau_frac::Float64 = 0.05
    plateau_tol::Float64 = 1e-3
    omega_reltol_warn::Float64 = 2e-2
    # ODE tolerance for root refinement and the reported point. The default (1e-3)
    # gives O(10%) noise in Omega h^2 on the steep freeze-in branch, and 1e-4
    # still has occasional outliers of a few percent.
    ode_reltol::Float64 = 1e-4
    # ODE tolerance for the coarse grid, where only the sign matters.
    ode_reltol_coarse::Float64 = 1e-3
    # Wall-clock limit per solve in seconds. Normal solves take up to ~150 s at
    # ode_reltol = 1e-5; at large mixing single solves can otherwise take hours.
    solve_wall_limit::Float64 = 600.0
    # Only look for the freeze-in root (first upward crossing in theta).
    first_root_only::Bool = true
    # Freeze-in search: accept the root once Omega h^2 is within omega_rtol of the target.
    omega_rtol::Float64 = 1e-2
    # Stepping from the predicted root: first step (decades), growth factor, max steps.
    march_step::Float64 = 0.25
    march_growth::Float64 = 1.6
    max_march::Int = 8
    # d log10(sin^2 2theta) / d log10(y) for predictions from a single previous
    # root (about -1.7 at 10 keV and -2.1 at 2 keV).
    default_slope::Float64 = -1.9
    max_refine::Int = 12
end

_theta_of(log10s) = asin(sqrt(10.0^log10s)) / 2

# Fallback integration lengths (x = m_N/T_nu) tried, in order, when the
# default x_end = 100 (DodelsonWidrow's default) genuinely has not let Omega
# h^2 settle by the end of the solve -- see the comment in `theta_point`.
# Each step is markedly more expensive (measured ~1.3-5x per step for a
# typical point), so it is only paid when needed.
const PLATEAU_XENDS = (300.0, 1000.0)

"""
    theta_point(pan, tT_rel, y, log10s, cfg, counts; reltol=cfg.ode_reltol)

Solves at (y, sin^2 2theta = 10^log10s). Returns (residual, omega_h2,
converged, plateau_ok, retcode) with residual = log10(Omega h^2 / omega_d0),
NaN if the solve failed. `counts.solves` and `counts.failed` are incremented
once per ODE solve actually performed (including any retries below).

If the solve succeeds (retcode) but `check_plateau_z` reports Omega h^2 has
not yet settled by x = 100, this most often used to be a false alarm from the
old Y_n-based plateau check (fixed directly in `check_plateau_z`); on the
rare points where Omega h^2 genuinely has not settled by x = 100 (dark sector
thermalizes/relaxes late), the solve is retried with a longer integration
(`PLATEAU_XENDS`) before being counted as failed. Without this, a single such
point returns NaN and can silently remove an entire sign change from the
coarse grid or the predicted-root march (both require two adjacent *finite*
residuals to see a crossing), producing a spurious "no root".
"""
function theta_point(pan, tT_rel, y, log10s, cfg::ScanConfigThetaZ, counts; reltol::Float64=cfg.ode_reltol)
    theta = _theta_of(log10s)
    dw = DodelsonWidrow{Float64}(pan.N1.m, theta, tT_rel)
    cfg_z = ScanConfigZ(
        sin2_2theta_min=10.0^log10s, sin2_2theta_max=10.0^log10s, n_theta=1,
        log10y_min=log10(y), log10y_max=log10(y),
        plateau_frac=cfg.plateau_frac, plateau_tol=cfg.plateau_tol,
        ode_reltol=reltol, solve_wall_limit=cfg.solve_wall_limit,
    )
    counts.solves[] += 1
    omega_h2, converged, plateau_ok, retcode = solve_point_z(pan, tT_rel, dw, y, theta, cfg_z)
    if retcode == :Success && !plateau_ok
        for x_end in PLATEAU_XENDS
            dw_x = DodelsonWidrow{Float64}(pan.N1.m, theta, tT_rel; x_end=x_end)
            counts.solves[] += 1
            omega_h2, converged, plateau_ok, retcode = solve_point_z(pan, tT_rel, dw_x, y, theta, cfg_z)
            (retcode == :Success && plateau_ok) && break
        end
    end
    ok = converged && isfinite(omega_h2) && omega_h2 > 0
    ok || (counts.failed[] += 1)
    return (ok ? log10(omega_h2) - log10(OMEGA_H2_TARGET_Z) : NaN, omega_h2, converged, plateau_ok, retcode)
end

"""
From `x0`, step in the direction in which the residual points (up in theta if
Omega h^2 is too small) with growing steps, until the residual changes sign
upwards. Returns (lo, p_lo, hi, p_hi) with p = `theta_point` results, or nothing.
"""
function march_to_bracket(point, x0, cfg::ScanConfigThetaZ)
    xa = clamp(x0, cfg.log10s_min, cfg.log10s_max)
    pa = point(xa)
    isfinite(pa[1]) || return nothing
    dir = pa[1] < 0 ? 1.0 : -1.0
    step = cfg.march_step
    for _ in 1:cfg.max_march
        xb = clamp(xa + dir * step, cfg.log10s_min, cfg.log10s_max)
        xb == xa && return nothing
        pb = point(xb)
        isfinite(pb[1]) || return nothing
        if sign(pb[1]) != sign(pa[1])
            lo, plo, hi, phi = dir > 0 ? (xa, pa, xb, pb) : (xb, pb, xa, pa)
            # Only an upward crossing is the freeze-in root.
            return (plo[1] < 0 < phi[1]) ? (lo, plo, hi, phi) : nothing
        end
        xa, pa = xb, pb
        step *= cfg.march_growth
    end
    return nothing
end

"""
Coarse grid in increasing theta (at `ode_reltol_coarse`) up to the first upward
crossing; its end points are re-evaluated at `ode_reltol` and widened (at most
twice) if the refined residuals do not bracket the root. Returns
(lo, p_lo, hi, p_hi) or nothing.
"""
function coarse_bracket(point, point_coarse, cfg::ScanConfigThetaZ)
    xs = collect(range(cfg.log10s_min, cfg.log10s_max, length=cfg.n_coarse))
    prev_x, prev_f = NaN, NaN
    bracket = nothing
    for x in xs
        f = point_coarse(x)[1]
        # Already above the target at the smallest theta: the freeze-in root lies
        # below this range, and going up in theta would only visit slow points.
        (x == xs[1] && isfinite(f) && f > 0) && return nothing
        if isfinite(prev_f) && isfinite(f) && prev_f < 0 < f
            bracket = (prev_x, x)
            break
        end
        prev_x, prev_f = x, f
    end
    bracket === nothing && return nothing
    lo, hi = bracket
    width = hi - lo
    plo, phi = point(lo), point(hi)
    for _ in 1:2
        (plo[1] < 0 < phi[1]) && break
        if !(plo[1] < 0)
            lo = max(cfg.log10s_min, lo - width); plo = point(lo)
        end
        if !(phi[1] > 0)
            hi = min(cfg.log10s_max, hi + width); phi = point(hi)
        end
    end
    return (plo[1] < 0 < phi[1]) ? (lo, plo, hi, phi) : nothing
end

"""
Illinois (modified regula falsi) refinement of an upward bracket. Returns
(x, p) for the last solve, which is within `omega_rtol` of the target unless
the bracket became narrower than `root_xtol` first.
"""
function refine_root(point, bracket, cfg::ScanConfigThetaZ)
    lo, plo, hi, phi = bracket
    target_close(p) = abs(10.0^p[1] - 1) < cfg.omega_rtol
    target_close(plo) && return (lo, plo)
    target_close(phi) && return (hi, phi)
    flo, fhi = plo[1], phi[1]
    best = abs(flo) < abs(fhi) ? (lo, plo) : (hi, phi)
    side = 0
    for _ in 1:cfg.max_refine
        x = (lo * fhi - hi * flo) / (fhi - flo)
        p = point(x)
        isfinite(p[1]) || return best
        best = (x, p)
        target_close(p) && return best
        if p[1] > 0
            hi, fhi = x, p[1]
            side == 1 && (flo /= 2)
            side = 1
        else
            lo, flo = x, p[1]
            side == -1 && (fhi /= 2)
            side = -1
        end
        (hi - lo) < cfg.root_xtol && return best
    end
    return best
end

"""
    scan_theta_for_y(pan, tT_rel, y, cfg; guess=nothing, warm_start=nothing)

Roots in log10(sin^2 2theta) of Omega h^2 = target at fixed (m_N, y). With
`first_root_only`, only the freeze-in root, starting from `guess` (a predicted
log10 sin^2 2theta) if given. Returns (points, roots, n_failed, n_solves).
"""
function scan_theta_for_y(
        pan,
        tT_rel,
        y::Float64,
        cfg::ScanConfigThetaZ;
        guess::Union{Nothing, Float64}=nothing,
        warm_start::Union{Nothing, Vector{Float64}}=nothing
    )
    counts = (solves=Ref(0), failed=Ref(0))
    point(x) = theta_point(pan, tT_rel, y, x, cfg, counts)
    point_coarse(x) = theta_point(pan, tT_rel, y, x, cfg, counts; reltol=cfg.ode_reltol_coarse)

    if cfg.first_root_only
        bracket = guess === nothing ? nothing : march_to_bracket(point, guess, cfg)
        bracket === nothing && (bracket = coarse_bracket(point, point_coarse, cfg))
        bracket === nothing && return RelicPointThetaZ[], Float64[], counts.failed[], counts.solves[]
        x, p = refine_root(point, bracket, cfg)
        _, omega_h2, converged, plateau_ok, retcode = p
        if converged && abs(omega_h2 / OMEGA_H2_TARGET_Z - 1) > cfg.omega_reltol_warn
            @warn "m_N=$(pan.N1.m) y=$y: root at sin2_2theta=$(10.0^x) has Omega h^2=$omega_h2"
        end
        point_out = RelicPointThetaZ(pan.N1.m, y, 10.0^x, omega_h2, converged, plateau_ok, retcode, 1, 1, counts.failed[])
        return [point_out], [x], counts.failed[], counts.solves[]
    end

    # All roots: coarse grid (or windows around `warm_start`), then Brent.
    res(x) = point(x)[1]
    res_coarse(x) = point_coarse(x)[1]
    brackets = Tuple{Float64, Float64, Int}[]
    if warm_start !== nothing && !isempty(warm_start)
        for x0 in sort(warm_start)
            xs = collect(range(max(cfg.log10s_min, x0 - cfg.warm_start_halfwidth),
                               min(cfg.log10s_max, x0 + cfg.warm_start_halfwidth), length=4))
            vals = res_coarse.(xs)
            for i in 1:length(xs)-1
                a, b = vals[i], vals[i+1]
                (isfinite(a) && isfinite(b) && sign(a) != sign(b)) && push!(brackets, (xs[i], xs[i+1], b > a ? 1 : -1))
            end
        end
    end
    if isempty(brackets)
        xs = collect(range(cfg.log10s_min, cfg.log10s_max, length=cfg.n_coarse))
        vals = res_coarse.(xs)
        for i in 1:length(xs)-1
            a, b = vals[i], vals[i+1]
            (isfinite(a) && isfinite(b) && sign(a) != sign(b)) && push!(brackets, (xs[i], xs[i+1], b > a ? 1 : -1))
        end
    end

    points = RelicPointThetaZ[]
    roots = Float64[]
    for (branch, (lo, hi, slope)) in enumerate(sort(brackets))
        # The bracket was found at the coarse tolerance; widen it (at most twice)
        # if the refined residuals no longer have opposite signs at its ends.
        width = hi - lo
        f_lo, f_hi = res(lo), res(hi)
        for _ in 1:2
            (isfinite(f_lo) && isfinite(f_hi) && sign(f_lo) != sign(f_hi)) && break
            if !isfinite(f_lo) || sign(f_lo) == slope
                lo = max(cfg.log10s_min, lo - width); f_lo = res(lo)
            end
            if !isfinite(f_hi) || sign(f_hi) != slope
                hi = min(cfg.log10s_max, hi + width); f_hi = res(hi)
            end
        end
        (isfinite(f_lo) && isfinite(f_hi) && sign(f_lo) != sign(f_hi)) || continue
        sol = _bracket_root_z(res, lo, hi, cfg.root_xtol)
        DE.successful_retcode(sol) || continue
        x = sol.u
        _, omega_h2, converged, plateau_ok, retcode = point(x)
        if converged && abs(omega_h2 / OMEGA_H2_TARGET_Z - 1) > cfg.omega_reltol_warn
            @warn "m_N=$(pan.N1.m) y=$y: root at sin2_2theta=$(10.0^x) has Omega h^2=$omega_h2"
        end
        push!(points, RelicPointThetaZ(pan.N1.m, y, 10.0^x, omega_h2, converged, plateau_ok,
            retcode, branch, slope, counts.failed[]))
        push!(roots, x)
    end
    return points, roots, counts.failed[], counts.solves[]
end

"""
Predicted log10 sin^2(2 theta) of the freeze-in root at log10 y, from the
roots `history` = [(log10 y, log10 sin^2 2theta), ...] already found at this
mass; nothing if there are none.
"""
function predict_root(history, log10y, cfg::ScanConfigThetaZ)
    isempty(history) && return nothing
    (y1, s1) = history[end]
    if length(history) == 1
        return s1 + cfg.default_slope * (log10y - y1)
    end
    (y0, s0) = history[end-1]
    slope = y1 == y0 ? cfg.default_slope : (s1 - s0) / (y1 - y0)
    return s1 + slope * (log10y - y1)
end

"""
    run_scan_theta_z(m_N, ys, cfg; tT_rel=TimeTempRelation{Float64}(), verbose=false, io=stdout, on_point=nothing)

Scans the couplings `ys` (in the given order) at one mass m_N [GeV]. With
`first_root_only`, each search starts from a root predicted from the previous
ones; otherwise from windows around the previous roots. If given,
`on_point(points, n_solves, t)` is called after each coupling with its
`RelicPointThetaZ`s (a single `:NoRoot` point if none was found), the number of
solves and the wall time in seconds.
"""
function run_scan_theta_z(
        m_N::Float64,
        ys,
        cfg::ScanConfigThetaZ;
        tT_rel=TimeTempRelation{Float64}(),
        verbose::Bool=false,
        io=stdout,
        on_point=nothing
    )
    N1 = Particle{Float64}(m_N, 1, dof=2)
    N2 = Particle{Float64}(m_N, 1, dof=2)
    A = Particle{Float64}(2.5 * m_N, -1, dof=3)
    nu = Particle{Float64}(0.0, 1, dof=2)
    pan = PandemolatorZ{Float64}(
        ModelParams{Float64}(ys[1], _theta_of(-12.0)),
        N1, N2, A, nu,
        tT_rel,
        verbose
    )

    all_points = RelicPointThetaZ[]
    warm = nothing
    history = Tuple{Float64, Float64}[]
    for y in ys
        guess = cfg.first_root_only ? predict_root(history, log10(y), cfg) : nothing
        t = @elapsed points, roots, n_failed, n_solves = scan_theta_for_y(pan, tT_rel, y, cfg; guess=guess, warm_start=warm)
        println(io, "m_N = $m_N, y = $y: $(length(points)) root(s) at sin2_2theta = $(round.(10 .^ roots, sigdigits=4)), " *
            "guess = $(guess === nothing ? "none" : round(10.0^guess, sigdigits=3)), $n_solves solves, $n_failed failed, $(round(t, digits=1)) s")
        flush(io)
        if isempty(points)
            points = [RelicPointThetaZ(m_N, y, NaN, NaN, false, false, :NoRoot, 0, 0, n_failed)]
        else
            warm = roots
            cfg.first_root_only && push!(history, (log10(y), roots[1]))
        end
        append!(all_points, points)
        on_point === nothing || on_point(points, n_solves, t)
    end
    return all_points
end

function save_results_theta_z(points::Vector{RelicPointThetaZ}, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "m_N,y,sin2_2theta,omega_h2,converged,plateau_ok,retcode,branch,slope,n_failed")
        for p in points
            println(io, "$(p.m_N),$(p.y),$(p.sin2_2theta),$(p.omega_h2),$(p.converged),$(p.plateau_ok),$(p.retcode),$(p.branch),$(p.slope),$(p.n_failed)")
        end
    end
    return nothing
end
