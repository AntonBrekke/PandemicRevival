"""
portal_tables.jl

Tables of the quantum-statistics (FD/BE, Pauli-blocking) correction to the portal 2 <-> 2 rates, from
the exact integrals of coll_portal.jl, for the fluid. They live on (ln x_nu, ln r, gap): x_nu = m_N/T_nu,
r = T_N/T_nu, gap = (m_N - mu)/T_N of both N (equal masses, mu_N1 = mu_N2 as in the fluid).

The (x_nu, r) dependence stays with the MB rate tables (rate_tables.jl: fine grid, cubic splines); only the
change that the N statistics make is tabulated. For each process in PORTAL_EXACT and each of forward /
backward x number / energy (Rf, Rr, Qf, Qr), at a node:
- `full` = exact rate (coll_portal) at the gap; `dil` = the same from the dilute limit (gap 20, where
  f_N ~ 1e-9) scaled by e^(-n (gap - 20)), n = 1 forward, 2 backward: what MB N give at this gap (the
  neutrino is FD in both);
- E = Z + |full - Z| of the dilute rate (Z its non-resonant part, = full for N1 nu -> N2 N2; the
  resonant part of N2 nu -> N1 N2 is a narrow-width finite part of either sign): a positive envelope,
  summed over both directions;
- delta = [(full_fwd - full_bwd) - (dil_fwd - dil_bwd)]/(E_fwd + E_bwd) for the number (R) and the
  energy (Q): the change of the NET rate. It is exactly 0 in the dilute limit, and at chemical
  equilibrium with the neutrinos (T_N = T_nu, balanced mu), where the exact integrals obey detailed
  balance pointwise: the net change varies linearly away from it, which linear interpolation keeps, so
  the fluid's fixed points stay put. (Separate forward and backward corrections, each good to ~1%,
  moved them by up to 0.1 in mu/T.)
The fluid adds delta x (MB table envelope of both directions, Z_tab (1 + |ratio_tab - 1|)) to the MB net
rate (`portal_exact_delta`, `portal_rates`): in the dilute limit the MB tables unchanged.
(Tabulating the exact rates themselves on a coarser 3D grid failed: the resonant rate changes sign and
varies exponentially in (x_nu, r), and linear interpolation missed it by tens of percent and more.)
Points where the rates underflow (cold states, negligible rates) carry delta = 0. Interpolation is
linear; beyond the gap grid delta is clamped at its ends (~0 at the dilute end).
The tables are built at the reference parameters of the MB tables and rescale with them (rate_norms).
"""

isdefined(@__MODULE__, :coll_portal) || include(joinpath(@__DIR__, "coll_portal.jl"))

const PORTAL_EXACT = (:N1nu_N2N2, :N2nu_N1N2)
# grid: every other x_nu and every r node of the MB tables (r includes T_N = T_nu, where the net correction
# vanishes at mu = 0: on the grid, detailed balance holds). Every fourth x_nu and every other r node left
# the fixed point of the fluid off by up to 0.08 in mu/T and missed the correction at cold, nearly
# degenerate low-x_nu states (m_A' = 50) by more than its size.
const PX_LNXNU = LNXNU_GRID[1:2:end]
const PX_LNR = LNR_GRID
const PX_GAP = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 15.0]
const PX_GAP_DILUTE = 20.0

"""Particles (1, 2, 3, 4) of a portal process at the parameters p: N fermions with 2 dof, a massless neutrino."""
function portal_parts(name, p)
    N1 = Particle{Float64}(p.m1, 1; dof=2)
    N2 = Particle{Float64}(p.m2, 1; dof=2)
    nu = Particle{Float64}(0.0, 1; dof=2)
    return name === :N1nu_N2N2 ? (N1, nu, N2, N2) : name === :N2nu_N1N2 ? (N2, nu, N1, N2) :
           error("no exact portal process $name")
end

"""1/2 for identical final particles (as the MB tables)."""
portal_sym(name) = name === :N1nu_N2N2 ? 0.5 : 1.0

"""(full, Z) of (Rf, Rr, Qf, Qr) from coll_portal at (T_N, T_nu, gap), with the tables' symmetry factor."""
function portal_parts_rates(name, p, T_N, T_nu, gap)
    res = coll_portal(_tree(name), p, portal_parts(name, p), (T_N, T_nu, T_N, T_N), (gap, 0.0, gap, gap))
    sym = portal_sym(name)
    full = (res.fwd[1], res.bwd[1], res.fwd[2], res.bwd[2]) .* sym
    Z = haskey(res, :fwdZ) ? (res.fwdZ[1], res.bwdZ[1], res.fwdZ[2], res.bwdZ[2]) .* sym : full
    return full, Z
end

const _NFUG = (1, 2, 1, 2)          # initial N of forward (1) and backward (2) reactions

"""(delta_R, delta_Q) of the net rates at one gap, given the dilute (full, Z) at the same (x_nu, r)."""
function portal_delta(full, dil_full, dil_Z, gap)
    sc = ntuple(i -> exp(-_NFUG[i] * (gap - PX_GAP_DILUTE)), 4)
    d = dil_full .* sc
    z = dil_Z .* sc
    E = ntuple(i -> z[i] + abs(d[i] - z[i]), 4)
    function one(f, b)          # (forward, backward) indices
        env = E[f] + E[b]
        ex = full[f] - full[b]
        (isfinite(env) && env > 0 && isfinite(ex)) || return 0.0
        return (ex - (d[f] - d[b])) / env
    end
    return (one(1, 2), one(3, 4))
end

"""
    build_portal_tables(p_ref, mass_ratios; lnxnu, lnr, gaps, verbose)

The delta tables of every process in PORTAL_EXACT at the reference parameters `p_ref` (those of the MB
tables), threaded over (x_nu, r) nodes: Dict(name => NamedTuple of 3D arrays d_R, d_Q).
"""
function build_portal_tables(p_ref, mass_ratios; lnxnu=PX_LNXNU, lnr=PX_LNR, gaps=PX_GAP, verbose=true)
    p_ref.m2 == p_ref.m1 || error("exact portal tables: equal masses only")
    m = p_ref.m1
    out = Dict{Symbol,Any}()
    for name in PORTAL_EXACT
        t0 = time()
        idx = vec(collect(Iterators.product(eachindex(lnxnu), eachindex(lnr))))
        vals = zeros(length(lnxnu), length(lnr), length(gaps), 2)
        done = Threads.Atomic{Int}(0)
        Threads.@threads :dynamic for (i, j) in idx
            T_nu = m * exp(-lnxnu[i]); T_N = exp(lnr[j]) * T_nu
            dil_full, dil_Z = portal_parts_rates(name, p_ref, T_N, T_nu, PX_GAP_DILUTE)
            for k in eachindex(gaps)
                full, _ = portal_parts_rates(name, p_ref, T_N, T_nu, gaps[k])
                vals[i, j, k, :] .= portal_delta(full, dil_full, dil_Z, gaps[k])
            end
            n = Threads.atomic_add!(done, 1) + 1
            verbose && n % 200 == 0 && (println("  $name: $n/$(length(idx)), $(round((time() - t0) / 60; digits=1)) min"); flush(stdout))
        end
        out[name] = (d_R=vals[:, :, :, 1], d_Q=vals[:, :, :, 2])
        verbose && (println(rpad(name, 12), " ", round((time() - t0) / 60; digits=1), " min"); flush(stdout))
    end
    return (p_ref=p_ref, mass_ratios=mass_ratios, lnxnu=collect(lnxnu), lnr=collect(lnr), gaps=collect(gaps),
            tables=out)
end

"""File of the exact portal tables of a mass-ratio set."""
portal_table_path(dir; mA_over_mN=3.0, mphi_over_mN=100.0) =
    joinpath(dir, "portal_exact_mA$(mA_over_mN)_mphi$(mphi_over_mN).jls")

"""Interpolants of loaded portal delta tables (name => (d_R, d_Q) interpolants)."""
struct PortalExact
    mass_ratios::Tuple{Float64,Float64}
    gap_range::Tuple{Float64,Float64}
    itps::Dict{Symbol,Any}
end

function PortalExact(tabs)
    grid = (tabs.lnxnu, tabs.lnr, tabs.gaps)
    li(v) = extrapolate(interpolate(grid, v, Gridded(Linear())), Flat())
    itps = Dict{Symbol,Any}(name => map(li, v) for (name, v) in tabs.tables)
    return PortalExact(tabs.mass_ratios, extrema(tabs.gaps), itps)
end

"""(delta_R, delta_Q) of the net rates of process `name` at (ln x_nu, ln r, gap) (see the header)."""
function portal_exact_delta(px::PortalExact, name, lxnu, lr, gap)
    t = px.itps[name]
    g = clamp(gap, px.gap_range...)
    return (t.d_R(lxnu, lr, g), t.d_Q(lxnu, lr, g))
end

"""
Net (R, Q) = forward - backward of process `name` at (ln x_nu, ln r) for N at `gap`, from the MB tables at
the reference couplings (fugacities e^-gap per initial N), corrected by the delta tables `px` if given.
"""
function portal_rates(itp::RateInterpolants, px, name, lxnu, lr, gap)
    t = itp.portal[name]
    lnZ = (t.lnRf(lxnu, lr), t.lnRr(lxnu, lr), t.lnQf(lxnu, lr), t.lnQr(lxnu, lr))
    ratio = (t.ratio_Rf(lxnu, lr), t.ratio_Rr(lxnu, lr), t.ratio_Qf(lxnu, lr), t.ratio_Qr(lxnu, lr))
    Z = ntuple(i -> exp(lnZ[i] - _NFUG[i] * gap), 4)
    mb = Z .* ratio
    net = (mb[1] - mb[2], mb[3] - mb[4])
    px === nothing && return net
    d = portal_exact_delta(px, name, lxnu, lr, gap)
    env = ntuple(i -> Z[i] * (1 + abs(ratio[i] - 1)), 4)
    return (net[1] + d[1] * (env[1] + env[2]), net[2] + d[2] * (env[3] + env[4]))
end
