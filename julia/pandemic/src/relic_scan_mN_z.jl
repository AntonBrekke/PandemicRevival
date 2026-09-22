"""
relic_scan_mN_z.jl

Extends relic_scan_z.jl with m_N (= m_N1 = m_N2) as a third scanned axis,
alongside sin2_2theta. At each (m_N, sin2_2theta) point, root-finds y so
that Omega h^2 = 0.12, exactly as relic_scan_z.jl does at fixed m_N -- this
file adds an outer loop over m_N around the SAME theta-column machinery
(`scan_theta_column_z`), reused unchanged.

MASS PANEL
==========
m_A = 2.5*m_N and m_N2 = m_N1 = m_N throughout (same ratios as every
benchmark/scan so far) -- varying m_N rescales the whole dark-sector mass
spectrum uniformly; it does not introduce mass splitting.

WHAT'S BUILT ONCE VS. PER m_N VS. PER (m_N, theta)
====================================================
- `TimeTempRelation` (background SM/neutrino cosmology): built ONCE for the
  entire scan. It has no dependence on any dark-sector mass at all.
- `PandemolatorZ`'s z-space interpolation closures (t_interp_z,
  dx_dt_interp_z, ent_interp_z, H_interp_z): these DO depend on m_N, since
  z = log(m_N1/T_nu) and dx/dt = d(m_N1/T)/dt both scale with m_N1. So,
  unlike relic_scan_z.jl's single scan (which builds one `PandemolatorZ` for
  the whole run and only ever mutates `pan.mp`), this scan must rebuild
  `PandemolatorZ` once per m_N value.
- `DodelsonWidrow`: built once per (m_N, theta) column, exactly as in
  relic_scan_z.jl -- it depends on theta but not y.

CONTINUATION
============
Warm-starting (`warm_start_log10y`) runs across the theta grid within one
m_N slice, exactly as in relic_scan_z.jl, but is reset at the start of each
new m_N rather than carried over. A snake-order traversal reusing the
previous m_N's brackets would likely speed this up further; not done here.

THREADING
=========
The m_N loop (`run_scan_mN_z(...; threaded=true)`) parallelizes safely,
UNLIKE relic_scan_z.jl's (y, theta)-only scan, which mutates one shared
`Pandemolator` across the whole run and is explicitly not thread-safe.
Here, every m_N iteration builds its own `PandemolatorZ`/particles/
`DodelsonWidrow` from scratch (`_scan_one_mN`) and never touches another
iteration's state -- the only object shared across iterations is `tT_rel`,
which is read-only after construction. Two things still needed fixing
before this was actually safe, not just embarrassingly parallel in
principle:
  - the lazily-initialised caches in constants_functions.jl/densities.jl
    (`_gstar_cache`, `_dens_cache`, `_dw_cache`, each a `Ref` populated on
    first use) would otherwise race if two threads hit them simultaneously
    on their first solve -- `_warm_up_caches` runs one throwaway solve
    single-threaded before the loop to force that population;
  - accumulating results via a shared `append!` across threads is a data
    race -- each thread instead writes to its own index of a preallocated
    `results::Vector{Vector{RelicPointMNZ}}`, concatenated after the loop.
"""

include(joinpath(@__DIR__, "relic_scan_z.jl"))

"""
    RelicPointMNZ

As `RelicPointZ`, with `m_N` (= m_N1 = m_N2) as an additional axis.
"""
struct RelicPointMNZ
    m_N::Float64
    sin2_2theta::Float64
    theta::Float64
    y::Float64
    omega_h2::Float64
    converged::Bool
    plateau_ok::Bool
    retcode::Symbol
    branch::Int
end

RelicPointMNZ(m_N::Float64, p::RelicPointZ) = RelicPointMNZ(
    m_N, p.sin2_2theta, p.theta, p.y, p.omega_h2,
    p.converged, p.plateau_ok, p.retcode, p.branch,
)

"""
    ScanConfigMNZ

As `ScanConfigZ`, with an additional (log-spaced) m_N grid: `m_N_min`,
`m_N_max`, `n_mN`. Every other field has the same meaning as `ScanConfigZ`
-- in particular `log10y_min`/`log10y_max` is a SINGLE bracket used at every
m_N, so it needs to be wide enough to bracket the root across the whole
mass range scanned, not just at one mass (see the module docstring in
relic_scan_z.jl on how to size it from a trusted benchmark point).
"""
Base.@kwdef struct ScanConfigMNZ
    m_N_min::Float64
    m_N_max::Float64
    n_mN::Int = 20
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
end

_to_scan_config_z(cfg::ScanConfigMNZ) = ScanConfigZ(
    sin2_2theta_min=cfg.sin2_2theta_min, sin2_2theta_max=cfg.sin2_2theta_max, n_theta=cfg.n_theta,
    log10y_min=cfg.log10y_min, log10y_max=cfg.log10y_max, n_coarse=cfg.n_coarse,
    root_xtol=cfg.root_xtol, plateau_frac=cfg.plateau_frac, plateau_tol=cfg.plateau_tol,
    omega_reltol_warn=cfg.omega_reltol_warn, warm_start_halfwidth=cfg.warm_start_halfwidth,
)

"""
    _warm_up_caches(tT_rel, m_N, cfg)

Runs one throwaway solve to force the lazily-initialised caches in
constants_functions.jl/densities.jl (`_gstar_cache`, `_dens_cache`,
`_dw_cache` -- all `Ref`s populated on first use) to populate, and to
JIT-compile the solve path, before `run_scan_mN_z(...; threaded=true)`
spawns threads. Without this, the first solve on each thread would race to
populate the same `Ref` concurrently. Exceptions are swallowed: a failed
warm-up solve still exercises (and populates) the caches, and only the
actual scan's results matter, not this one.
"""
function _warm_up_caches(tT_rel::TimeTempRelation{Float64}, m_N::Float64, cfg::ScanConfigMNZ)
    N1 = Particle{Float64}(m_N, 1, dof=2)
    N2 = Particle{Float64}(m_N, 1, dof=2)
    A = Particle{Float64}(2.5 * m_N, -1, dof=3)
    nu = Particle{Float64}(0.0, 1, dof=2)
    theta = asin(sqrt(cfg.sin2_2theta_min)) / 2.0
    y = 10.0^(0.5 * (cfg.log10y_min + cfg.log10y_max))
    pan = PandemolatorZ{Float64}(ModelParams{Float64}(y, theta), N1, N2, A, nu, tT_rel, false)
    dw = DodelsonWidrow{Float64}(m_N, theta, tT_rel)
    try
        pandemolate_z(tT_rel, dw, pan)
    catch
    end
    return nothing
end

function _scan_one_mN(m_N::Float64, tT_rel::TimeTempRelation{Float64},
                       sin2_2theta_grid, cfg::ScanConfigMNZ, cfg_z::ScanConfigZ,
                       verbose::Bool)
    N1 = Particle{Float64}(m_N, 1, dof=2)
    N2 = Particle{Float64}(m_N, 1, dof=2)
    A = Particle{Float64}(2.5 * m_N, -1, dof=3)
    nu = Particle{Float64}(0.0, 1, dof=2)

    theta0 = asin(sqrt(cfg.sin2_2theta_min)) / 2.0
    mp0 = ModelParams{Float64}(10.0^cfg.log10y_min, theta0)
    pan = PandemolatorZ{Float64}(mp0, N1, N2, A, nu, tT_rel, verbose)

    mN_points = RelicPointMNZ[]
    warm_x = nothing
    for sin2_2theta in sin2_2theta_grid
        theta = asin(sqrt(sin2_2theta)) / 2.0
        points, roots = scan_theta_column_z(theta, sin2_2theta, pan, tT_rel, cfg_z; warm_start_log10y=warm_x)
        append!(mN_points, RelicPointMNZ.(m_N, points))
        if !isempty(roots)
            warm_x = roots
        end
    end
    return mN_points
end

"""
    run_scan_mN_z(cfg::ScanConfigMNZ; verbose=false, threaded=false) -> Vector{RelicPointMNZ}

Scans m_N over `cfg.n_mN` log-spaced steps in [m_N_min, m_N_max]; for each,
runs the same (theta, y) contour search as `run_scan_z` (relic_scan_z.jl),
against a freshly-built `PandemolatorZ` (its z-interpolants depend on m_N,
see module docstring).

`threaded=true` parallelizes the m_N loop with `Threads.@threads` (needs
`julia -t N`/`JULIA_NUM_THREADS` > 1 to do anything). This is safe because
each m_N iteration builds its own `PandemolatorZ`/particles/`DodelsonWidrow`
from scratch and never touches another iteration's -- unlike `run_scan_z`
(relic_scan_z.jl), which mutates a single shared `Pandemolator` across the
whole scan and is explicitly NOT thread-safe for that reason. The only
shared, read-only object here is `tT_rel`, built once before the loop.
Each thread accumulates into its own results vector (indexed by m_N, not a
shared `append!`), concatenated only after the loop -- and the lazily-
initialised caches are pre-warmed first (see `_warm_up_caches`) so no two
threads race to populate the same cache on their first solve.
"""
function run_scan_mN_z(cfg::ScanConfigMNZ; verbose::Bool=false, threaded::Bool=false)
    tT_rel = TimeTempRelation{Float64}()   # independent of m_N -- built once
    cfg_z = _to_scan_config_z(cfg)

    m_N_grid = exp10.(range(log10(cfg.m_N_min), log10(cfg.m_N_max), length=cfg.n_mN))
    sin2_2theta_grid = exp10.(range(log10(cfg.sin2_2theta_min), log10(cfg.sin2_2theta_max), length=cfg.n_theta))

    if threaded
        _warm_up_caches(tT_rel, m_N_grid[1], cfg)
    end

    results = Vector{Vector{RelicPointMNZ}}(undef, length(m_N_grid))
    if threaded
        Threads.@threads for i in eachindex(m_N_grid)
            results[i] = _scan_one_mN(m_N_grid[i], tT_rel, sin2_2theta_grid, cfg, cfg_z, verbose)
        end
    else
        for i in eachindex(m_N_grid)
            results[i] = _scan_one_mN(m_N_grid[i], tT_rel, sin2_2theta_grid, cfg, cfg_z, verbose)
        end
    end
    return reduce(vcat, results)
end

"""
    save_results_mN_z(points, path)

Writes the scan result to a CSV for downstream plotting.
"""
function save_results_mN_z(points::Vector{RelicPointMNZ}, path::String)
    open(path, "w") do io
        println(io, "m_N,sin2_2theta,theta,y,omega_h2,converged,plateau_ok,retcode,branch")
        for p in points
            println(io, "$(p.m_N),$(p.sin2_2theta),$(p.theta),$(p.y),$(p.omega_h2),$(p.converged),$(p.plateau_ok),$(p.retcode),$(p.branch)")
        end
    end
    return nothing
end
