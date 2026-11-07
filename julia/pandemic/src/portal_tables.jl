"""
portal_tables.jl

Tables of the portal 2 <-> 2 rates of N1 nu -> N2 N2 and N2 nu -> N1 N2 (PORTAL_EXACT) from the exact
integrals of coll_portal.jl (FD/BE for all particles: Pauli blocking of the final N), for the fluid. They
live on (ln x_nu, ln r, gap): x_nu = m_N/T_nu, r = T_N/T_nu, gap = (m_N - mu)/T_N of both N (equal
masses, mu_N1 = mu_N2 as in the fluid).

Stored, for each of forward / backward x number / energy (Rf, Rr, Qf, Qr), is the ratio y = full/Z
of the exact rate to the non-resonant part Z of the dilute rate (MB N, FD nu; Z = the whole rate for
N1 nu -> N2 N2), taken at gap 20 (f_N ~ 1e-9) and scaled by its fugacities e^(-n (gap - 20)), n = 1
forward, 2 backward. At the last gap node (70) y is the dilute ratio, the quantity the MB tables store
on their fine 2D grid; the fluid uses only the Pauli change Delta = y(gap) - y(70):
    rate = Z x (ratio_MB + Delta),   Z, ratio_MB from the MB tables
(`portal_rates`, fluid.jl table_collision). The (x_nu, r) structure of the resonant ratio (sign changes
near T_N ~ T_nu, x_nu ~ 2 - 5) stays with the fine splines: y itself on this coarser grid missed it by
up to 30% of Z. Delta = 0 in the dilute limit (gap >= 15), so the MB rates are recovered there exactly.
Detailed balance: at T_N = T_nu the exact rates obey full_bwd = e^(mu/T) full_fwd pointwise, and so do
Z_bwd / Z_fwd; so y_bwd = y_fwd, and Delta_bwd = Delta_fwd, at every node with r = 1 (on the grid): the
fluid's fixed point at equilibrium with the neutrinos is that of the MB tables (balanced to ~1e-6).
Domain: gap >= 0. The fluid never reaches degenerate N (gap < 0) in any run so far (benchmarks at
m_A' = 2.5, 3 and 50 up to g = 1.33: gap >= 0.0). The correction matters where the portal 2 <-> 2 seed
n~ while the dark sector is cold and nearly degenerate (x ~ 1e-3 - 1e-2, gap ~ 0.4 - 2, T_N/T_nu ~ 1e-3
- 1e-2): the forward rates drop by ~13% there, Omega h^2 by 10% at m_A' = 3, 10.57 keV, g = 0.01
(0.1205 -> 0.1083 at the MB root). Below gap 0
Delta is clamped at gap 0. (No occupation factor at rest is divided out: with T_N << T_nu the final N
carry the neutrino's energy and are barely blocked, f (1 - f)^2 = 1/8 at gap 0 against a true
suppression of ~0.8, and the ratio would vary steeply in the gap.)
(Earlier forms failed: the net rate over Z_fwd + Z_bwd varies like a tanh of the steep ln(Z_bwd/Z_fwd)
and is not interpolable on the 3D grid; additive corrections normalised by Z + |resonant part| spike
where the resonant part changes sign; separate forward/backward corrections with different envelopes
broke the fixed point.)
Beyond gap 15 the nodes come from the dilute rates scaled analytically.
Points where Z underflows (cold states, negligible rates) carry y = 0. Interpolation is linear; beyond
the gap grid y is clamped at its ends.
The tables are built at the reference parameters of the MB tables and rescale with them (rate_norms).
"""

isdefined(@__MODULE__, :coll_portal) || include(joinpath(@__DIR__, "coll_portal.jl"))

const PORTAL_EXACT = (:N1nu_N2N2, :N2nu_N1N2)
# grid: every fourth x_nu and every r node of the MB tables (r includes T_N = T_nu: on the grid, detailed
# balance holds exactly). With the Pauli change of the ratio tabulated (the (x_nu, r) structure stays in
# the MB splines), every fourth x_nu, every other r and gap node gave errors <= 1.6% of the rate scale.
const PX_LNXNU = LNXNU_GRID[1:4:end]
const PX_LNR = LNR_GRID
const PX_GAP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 11.0, 15.0]
const PX_GAP_DILUTE = 20.0
const PX_GAP_FAR = [20.0, 30.0, 45.0, 70.0]     # from the dilute rates, scaled

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

"""(y_Rf, y_Rr, y_Qf, y_Qr) at one gap: exact rates over Z, Z from the dilute rates scaled."""
function portal_y(full, dil_Z, gap)
    return ntuple(4) do i
        z = dil_Z[i] * exp(-_NFUG[i] * (gap - PX_GAP_DILUTE))
        (isfinite(z) && z > 0 && isfinite(full[i])) ? full[i] / z : 0.0
    end
end

"""
    build_portal_tables(p_ref, mass_ratios; lnxnu, lnr, gaps, verbose)

The delta tables of every process in PORTAL_EXACT at the reference parameters `p_ref` (those of the MB
tables), threaded over (x_nu, r) nodes: Dict(name => NamedTuple of 3D arrays y_Rf, y_Rr, y_Qf, y_Qr); the gap grid is
`gaps` followed by PX_GAP_FAR (dilute rates scaled).
"""
function build_portal_tables(p_ref, mass_ratios; lnxnu=PX_LNXNU, lnr=PX_LNR, gaps=PX_GAP, verbose=true)
    p_ref.m2 == p_ref.m1 || error("exact portal tables: equal masses only")
    m = p_ref.m1
    out = Dict{Symbol,Any}()
    for name in PORTAL_EXACT
        t0 = time()
        idx = vec(collect(Iterators.product(eachindex(lnxnu), eachindex(lnr))))
        allg = [gaps; PX_GAP_FAR]
        vals = zeros(length(lnxnu), length(lnr), length(allg), 4)
        done = Threads.Atomic{Int}(0)
        Threads.@threads :dynamic for (i, j) in idx
            T_nu = m * exp(-lnxnu[i]); T_N = exp(lnr[j]) * T_nu
            dil_full, dil_Z = portal_parts_rates(name, p_ref, T_N, T_nu, PX_GAP_DILUTE)
            for k in eachindex(allg)
                g = allg[k]
                full = k <= length(gaps) ? portal_parts_rates(name, p_ref, T_N, T_nu, g)[1] :
                       dil_full .* ntuple(q -> exp(-_NFUG[q] * (g - PX_GAP_DILUTE)), 4)
                vals[i, j, k, :] .= portal_y(full, dil_Z, g)
            end
            n = Threads.atomic_add!(done, 1) + 1
            verbose && n % 200 == 0 && (println("  $name: $n/$(length(idx)), $(round((time() - t0) / 60; digits=1)) min"); flush(stdout))
        end
        out[name] = (y_Rf=vals[:, :, :, 1], y_Rr=vals[:, :, :, 2], y_Qf=vals[:, :, :, 3], y_Qr=vals[:, :, :, 4])
        verbose && (println(rpad(name, 12), " ", round((time() - t0) / 60; digits=1), " min"); flush(stdout))
    end
    return (p_ref=p_ref, mass_ratios=mass_ratios, lnxnu=collect(lnxnu), lnr=collect(lnr), gaps=[gaps; PX_GAP_FAR],
            tables=out)
end

"""File of the exact portal tables of a mass-ratio set."""
portal_table_path(dir; mA_over_mN=3.0, mphi_over_mN=100.0) =
    joinpath(dir, "portal_exact_mA$(mA_over_mN)_mphi$(mphi_over_mN).jls")

"""Interpolants of loaded portal tables (name => (y_Rf, y_Rr, y_Qf, y_Qr) interpolants)."""
struct PortalExact
    mass_ratios::Tuple{Float64,Float64}
    gap_range::Tuple{Float64,Float64}
    itps::Dict{Symbol,Any}
end

function PortalExact(tabs)
    k0 = findfirst(>=(0.0), tabs.gaps)          # domain gap >= 0 (tables built with negative gaps too)
    gaps = tabs.gaps[k0:end]
    grid = (tabs.lnxnu, tabs.lnr, gaps)
    li(v) = extrapolate(interpolate(grid, v[:, :, k0:end] .- v[:, :, end:end], Gridded(Linear())), Flat())
    itps = Dict{Symbol,Any}(name => map(li, v) for (name, v) in tabs.tables)
    return PortalExact(tabs.mass_ratios, extrema(gaps), itps)
end

"""Pauli changes (Delta_Rf, Delta_Rr, Delta_Qf, Delta_Qr) of the ratios of process `name` at (ln x_nu, ln r, gap)."""
function portal_exact_y(px::PortalExact, name, lxnu, lr, gap)
    t = px.itps[name]
    g = clamp(gap, px.gap_range...)
    return (t.y_Rf(lxnu, lr, g), t.y_Rr(lxnu, lr, g), t.y_Qf(lxnu, lr, g), t.y_Qr(lxnu, lr, g))
end

"""
Net (R, Q) = forward - backward of process `name` at (ln x_nu, ln r) for N at `gap`, at the reference
couplings (fugacities e^-gap per initial N): with `px` Z x (ratio + Delta) of the MB tables and the Pauli
changes, else the MB tables' rates.
"""
function portal_rates(itp::RateInterpolants, px, name, lxnu, lr, gap)
    t = itp.portal[name]
    lnZ = (t.lnRf(lxnu, lr), t.lnRr(lxnu, lr), t.lnQf(lxnu, lr), t.lnQr(lxnu, lr))
    Z = ntuple(i -> exp(lnZ[i] - _NFUG[i] * gap), 4)
    y = (t.ratio_Rf(lxnu, lr), t.ratio_Rr(lxnu, lr), t.ratio_Qf(lxnu, lr), t.ratio_Qr(lxnu, lr))
    px === nothing || (y = y .+ portal_exact_y(px, name, lxnu, lr, gap))
    r = Z .* y
    return (r[1] - r[2], r[3] - r[4])
end
