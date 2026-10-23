"""
network.jl

Species-resolved Boltzmann network for N1, N2, A', phi (plan in
notes/model_limitations.tex, Sec. "Plan"): one dark temperature T_N, a chemical
potential per species, Maxwell-Boltzmann dark species and a Fermi-Dirac
neutrino, with the collision terms of every reaction of rate_tables.jl.

STATE
=====
u = [ln(Y_N1 + Y_N2), ln(Y_N2/Y_N1), ln Y_A', ln Y_phi, ln Y_K] against
z = ln x, x = m_N/T_nu, with Y_i = n_i/s and Y_K = kappa/s^(4/3),
kappa = rho - sum_i m_i n_i the kinetic energy density of the dark sector.
(Total and ratio of N1, N2 rather than each: see `state_densities`.) kappa rather than rho: once the
species are non-relativistic, T_N from rho would be a difference of nearly
equal numbers. For MB species the kinetic energy per particle,
    k_i(T) = T [x_i (K_1(x_i)/K_2(x_i) - 1) + 3],   x_i = m_i/T,
lies between 3T/2 and 3T, which brackets the solve for T_N (`dark_temperature`).

EQUATIONS
=========
    dn_i/dt + 3H n_i = C_i,
    dkappa/dt + 3H (kappa + P) = C_K,       P = sum_i n_i T_N,
so that, with s ∝ a^-3,
    d ln Y_i/dz = (x/xdot) C_i/n_i,
    d ln Y_K/dz = (x/xdot) [H (kappa - 3P) + C_K]/kappa.
C_K is the kinetic energy gained: E_nu - Delta per forward portal reaction
(Delta = sum_F m - m_a, the rest mass created), -Delta per dark-only reaction.

The solver is an implicit Rosenbrock method (Rodas4P) with a finite-difference
Jacobian, as the fluid solver: the decay and conversion rates reach 1e20 H,
and an L-stable method follows the equilibrium they enforce.
"""

import DifferentialEquations as DE
import OrdinaryDiffEqRosenbrock as ODER   # AutoFiniteDiff comes with `using DifferentialEquations` (time_temp_relation.jl)

include(joinpath(@__DIR__, "time_temp_relation.jl"))
include(joinpath(@__DIR__, "dodelson_widrow.jl"))
include(joinpath(@__DIR__, "interpolation_helpers.jl"))
include(joinpath(@__DIR__, "rate_tables.jl"))

const NSPECIES = 4

"""
DW production history: the fraction F(T) of the final DW yield produced above
T, from Fig. 3 of hep-ph/0612182 (computed at 10 keV; rescaled to other masses
with T_dw(m)/T_dw(10 keV), as `O_h2_dw_Tevo`), made monotonic and
interpolated with a monotone cubic (PCHIP) in ln T so that the source term
dF/dz is continuous. Each N1 is produced with the DW mean momentum of today,
`avg_mom_0_dw` (Fig. 8), redshifted back with the entropy scale factor,
p(z) = p_0 (s(z)/s_0)^(1/3), as the fluid solver's initial condition does.
(Not r_q 3.15 T_nu: r_q = <q>/<q_active> compares with active neutrinos at late
times, which the QCD-era entropy release has heated relative to the
free-streaming N1, by (g_s(T)/g_s,late)^(1/3) ~ 1.7.)
"""
struct DWHistory
    lnT::Vector{Float64}       # ascending, rescaled to 10 keV
    F::Vector{Float64}
    dF::Vector{Float64}        # PCHIP slopes dF/dlnT
    scale::Float64             # T_dw(m)/T_dw(10 keV)
    Y_final::Float64           # final DW yield n/s
    p0::Float64                # mean comoving momentum today
end

function DWHistory(m, Y_final)
    data = _load_table(joinpath(_DW_DIR, "0612182_dw_fig_3.dat"); skipstart=2)
    lnT = log.(data[:, 1])
    F = data[:, 2] ./ data[1, 2]
    for i in 2:length(F)            # production only adds: F non-increasing in T
        F[i] = min(F[i], F[i-1])
    end
    # Fritsch-Carlson slopes
    h = diff(lnT)
    del = diff(F) ./ h
    d = zeros(length(F))
    for i in 2:length(F)-1
        if del[i-1] * del[i] > 0
            w1, w2 = 2h[i] + h[i-1], h[i] + 2h[i-1]
            d[i] = (w1 + w2) / (w1 / del[i-1] + w2 / del[i])
        end
    end
    return DWHistory(lnT, F, d, T_d_dw(m) / T_d_dw(1e-5), Y_final, avg_mom_0_dw(m))
end

"""(F, dF/dln T) of the DW history at plasma temperature T."""
function dw_fraction(h::DWHistory, T)
    l = log(T / h.scale)
    l <= h.lnT[1] && return 1.0, 0.0
    l >= h.lnT[end] && return 0.0, 0.0
    i = searchsortedlast(h.lnT, l)
    dx = h.lnT[i+1] - h.lnT[i]
    t = (l - h.lnT[i]) / dx
    h00, h10, h01, h11 = 2t^3 - 3t^2 + 1, t^3 - 2t^2 + t, -2t^3 + 3t^2, t^3 - t^2
    F = h00 * h.F[i] + h10 * dx * h.dF[i] + h01 * h.F[i+1] + h11 * dx * h.dF[i+1]
    dF = ((6t^2 - 6t) * h.F[i] + (3t^2 - 4t + 1) * dx * h.dF[i] + (-6t^2 + 6t) * h.F[i+1] + (3t^2 - 2t) * dx * h.dF[i+1]) / dx
    return F, dF
end

"""Model and background for one network solve."""
struct Network{FH,FS,FX}
    p::Any                         # rate_params(m_N, g, sin^2 2theta; ...)
    itp::RateInterpolants
    norms::Dict{Symbol,Tuple{Float64,Float64}}
    masses::NTuple{4,Float64}      # N1, N2, A', phi
    dofs::NTuple{4,Int}
    H_interp_z::FH                 # background interpolants keyed on z
    ent_interp_z::FS
    dx_dt_interp_z::FX
    dw::Union{Nothing,DWHistory}   # DW production as a source term (nothing: all at T_dw)
    w_phi::Float64                 # N per phi after its decays (`phi_weight`)
    w_phi2::Float64                # of which N2 (`phi_weight`)
end

"""Age of the Universe [s]."""
const T_AGE = 4.35e17

"""
Weight of phi in Omega h^2 and in the number of N after all decays: the number of N per phi
after its decays (phi -> A'A' -> 4 N, phi -> N1 N1, phi -> N1 nu, by branching ratio), or
m_phi/m_N if phi outlives the Universe (it is then dark matter itself). 4 for
m_phi > 2 m_A'; 1 in the light window m_N1 < m_phi < 2 m_N1 (phi -> N1 nu only).
"""
function phi_weight(p)
    w = (width(:phi_AA, p), width(:phi_N1N1, p), width(:phi_N1nu, p))
    tot = sum(w)
    tot * T_AGE / hbar < 1 && return p.mphi / p.m1
    return (4w[1] + 2w[2] + w[3]) / tot
end

"""Number of N2 per phi after its decays (2 per phi -> A'A'; 0 if phi outlives the Universe)."""
function phi_weight_N2(p)
    w = (width(:phi_AA, p), width(:phi_N1N1, p), width(:phi_N1nu, p))
    tot = sum(w)
    tot * T_AGE / hbar < 1 && return 0.0
    return 2w[1] / tot
end

function Network(itp::RateInterpolants, p, tT_rel::TimeTempRelation; dw=nothing)
    m = p.m1
    all(isapprox.((p.mA / m, p.mphi / m), mass_ratios(itp); rtol=1e-12)) ||
        error("parameters have m_A'/m_N, m_phi/m_N = $((p.mA / m, p.mphi / m)) but the tables $(mass_ratios(itp)): use table_params")
    isapprox(splitting(p), table_delta(itp); atol=1e-12) ||
        error("parameters have N1-N2 splitting $(splitting(p)) but the tables $(table_delta(itp)): use table_params")
    z_grid = log.(x_grid_of(m, tT_rel.T_nu_grid))
    masses = ntuple(k -> field_mass(DARK_SPECIES[k], p), 4)
    dofs = ntuple(k -> field_dof(DARK_SPECIES[k]), 4)
    return Network(p, itp, rate_norms(itp, p), masses, dofs,
                   z_interpolation(z_grid, tT_rel.hubble_grid),
                   z_interpolation(z_grid, tT_rel.ent_grid),
                   z_interpolation(z_grid, dx_dt_grid_of(m, tT_rel.T_nu_grid, tT_rel.dT_nu_dt_grid)), dw, phi_weight(p),
                   phi_weight_N2(p))
end

"""
Kinetic energy per particle of an MB species of mass m at temperature T,
T [x (K_1/K_2 - 1) + 3] with x = m/T; for x > 1e3, where x (K_1/K_2 - 1) loses
digits to cancellation, its asymptotic series 3/2 + 15/(8x) - 15/(8x^2) +
135/(128x^3) (error ~1e-12).
"""
function kinetic_per_particle(m, T)
    x = m / T
    x > 1e3 && return T * (1.5 + 15 / (8x) - 15 / (8x^2) + 135 / (128x^3))
    return T * (x * (besselkx_safe(1, x) / besselkx_safe(2, x) - 1) + 3)
end

"""
    dark_temperature(net, n, kappa)

T_N such that sum_i n_i k_i(T_N) = kappa. Bisection in ln T on the bracket
[kappa/(3 n_tot), kappa/(1.5 n_tot)], to 1e-15.
"""
function dark_temperature(net::Network, n, kappa)
    ntot = sum(n)
    f(T) = sum(n[k] * kinetic_per_particle(net.masses[k], T) for k in 1:NSPECIES) - kappa
    lo, hi = log(kappa / (3ntot)), log(kappa / (1.5ntot))
    for _ in 1:60
        mid = (lo + hi) / 2
        f(exp(mid)) > 0 ? (hi = mid) : (lo = mid)
        hi - lo < 1e-15 && break
    end
    return exp((lo + hi) / 2)
end

"""
One reaction at given (T_N, T_nu, n_N1, n_N2) with the A' and phi densities
factored out: its forward (reverse) rate is exp(lnGf + cI . (ln n_A', ln n_phi)) * xf
(resp. F), so C_A' and C_phi are exactly polynomials (degree <= 2) in their own
density (`qss_coefficients`). Q: the E_nu-weighted portal rates (0 for dark-only
reactions); the kinetic energy gained per reaction is E_nu - Delta.
"""
struct UnitRate
    dN::NTuple{4,Int}
    Delta::Float64
    cI::NTuple{2,Int}       # number of (A', phi) in the initial / final state
    cF::NTuple{2,Int}
    lnGf::Float64
    lnGr::Float64
    xf::Float64             # signed ratios of resonance-subtracted portal rates (1 otherwise)
    xr::Float64
    lnQf::Float64
    lnQr::Float64
    xqf::Float64
    xqr::Float64
end

function unit_rates(net::Network, T_N, T_nu, n1, n2)
    itp, p = net.itp, net.p
    m = p.m1
    lxN, lxnu, lr = log(m / T_N), log(m / T_nu), log(T_N / T_nu)
    # ln(n_i / n_scaled_i) = -gap_i, with n_A' = n_phi = 1 (GeV^3)
    lns = ntuple(k -> log(n_scaled_safe(net.masses[k], net.dofs[k], T_N)), 4)
    lnz = (log(n1) - lns[1], log(n2) - lns[2], -lns[3], -lns[4])
    side(fs) = sum(lnz[dark_index(f)] for f in fs)
    cnt(fs) = (count(==(:A), fs), count(==(:phi), fs))
    # beyond the portal grid (x_nu > 1e3) the portal rates are Boltzmann-suppressed to
    # nothing, and the linear extrapolation of their logs would not be: leave them out
    portal_on = lxnu <= itp.lnxnu[2]
    out = UnitRate[]
    for r in REACTIONS
        lnR, lnQ = net.norms[r.name]
        Delta = mass_sum(r.F, p) - mass_sum(r.I, p)
        if haskey(itp.dark, r.name)
            t = itp.dark[r.name]
            push!(out, UnitRate(delta_N(r), Delta, cnt(r.I), cnt(r.F), t.lnRf(lxN) + lnR + side(r.I),
                                t.lnRr(lxN) + lnR + side(r.F), 1.0, 1.0, -Inf, -Inf, 0.0, 0.0))
        elseif portal_on && haskey(itp.portal, r.name)
            t = itp.portal[r.name]
            push!(out, UnitRate(delta_N(r), Delta, cnt(r.I), cnt(r.F),
                                t.lnRf(lxnu, lr) + lnR + side(r.I), t.lnRr(lxnu, lr) + lnR + side(r.F),
                                t.ratio_Rf(lxnu, lr), t.ratio_Rr(lxnu, lr),
                                t.lnQf(lxnu, lr) + lnQ + side(r.I), t.lnQr(lxnu, lr) + lnQ + side(r.F),
                                t.ratio_Qf(lxnu, lr), t.ratio_Qr(lxnu, lr)))
        end            # else kinematically closed at these mass ratios
    end
    return out
end

"""Net forward rate and kinetic energy gained of one reaction at (ln n_A', ln n_phi)."""
function rate_and_kinetic(ur::UnitRate, lnA, lnphi)
    ef = ur.cI[1] * lnA + ur.cI[2] * lnphi
    er = ur.cF[1] * lnA + ur.cF[2] * lnphi
    G = exp(ur.lnGf + ef) * ur.xf - exp(ur.lnGr + er) * ur.xr
    Q = ur.xqf == 0 ? 0.0 : exp(ur.lnQf + ef) * ur.xqf - exp(ur.lnQr + er) * ur.xqr
    return G, Q - ur.Delta * G
end

"""
(C_i for (N1, N2, A', phi), C_K) from unit rates at the actual n_A', n_phi.

With `elimA` = w > 0 (A' quasi-static), w C_A' is added to C_N1 and C_N2 and
w (m_A' - m_N1 - m_N2) C_A' to C_K, reaction by reaction, so that A' <-> N1 N2
drops out of their sums exactly. At the QSS root C_A' = 0 and nothing changes,
but the decay and inverse-decay fluxes are ~1e11 per unit z there and their
cancellation leaves ~1e-5 rounding noise, which ruins the finite-difference
Jacobian. (The returned C[3] is never modified.) `dmA` = m_A' - m_N1 - m_N2.
"""
function assemble_collision(urs, nA, nphi; names=nothing, breakdown=nothing, elimA=0.0, dmA=0.0)
    lnA, lnphi = log(nA), log(nphi)
    C = zeros(NSPECIES)
    CK = 0.0
    Ctot = 0.0                 # C_N1 + C_N2 summed reaction by reaction (conversions drop out exactly)
    Gcf = Gcr = 0.0            # forward and reverse rates of N1 N1 <-> N2 N2
    for (i, ur) in enumerate(urs)
        G, K = rate_and_kinetic(ur, lnA, lnphi)
        dA = ur.dN[3]
        C[1] += (ur.dN[1] + elimA * dA) * G
        C[2] += (ur.dN[2] + elimA * dA) * G
        C[3] += dA * G
        C[4] += ur.dN[4] * G
        Ctot += (ur.dN[1] + ur.dN[2] + 2elimA * dA) * G
        CK += K + elimA * dmA * dA * G
        if ur.dN == (-2, 2, 0, 0)
            Gcf, Gcr = exp(ur.lnGf), exp(ur.lnGr)
        end
        breakdown === nothing || (breakdown[names[i]] = (G=G, K=K))
    end
    return C, CK, Ctot, Gcf, Gcr
end

"""
Coefficients (a0, a1, a2) of C_k = a0 + a1 n_k + a2 n_k^2 for k = 3 (A') or 4
(phi), the other density held at its value: exact for MB species.
"""
function qss_coefficients(urs, k, nA, nphi)
    j = k - 2                          # index into cI / cF
    lnother = j == 1 ? log(nphi) : log(nA)
    a = zeros(3)
    for ur in urs
        d = ur.dN[k]
        d == 0 && continue
        o = 3 - j
        a[ur.cI[j]+1] += d * exp(ur.lnGf + ur.cI[o] * lnother) * ur.xf
        a[ur.cF[j]+1] -= d * exp(ur.lnGr + ur.cF[o] * lnother) * ur.xr
    end
    return a[1], a[2], a[3]
end

"""Positive root of a0 + a1 n + a2 n^2 = 0 (production a0 > 0, losses a1, a2 <= 0),
and the relaxation rate -(dC/dn) there."""
function qss_root(a0, a1, a2)
    a0 <= 0 && return 0.0, -a1
    nq = 2a0 / (-a1 + sqrt(max(a1^2 - 4a0 * a2, 0.0)))
    return nq, -(a1 + 2a2 * nq)
end

"""
    network_collision(net, T_N, T_nu, n; breakdown=nothing)

(C_i for (N1, N2, A', phi), C_K): number and kinetic-energy collision terms
from all reactions (rate_tables.jl), for MB dark densities `n` at T_N. With a
Dict as `breakdown`, the net rate G and kinetic gain K of every reaction.
"""
function network_collision(net::Network, T_N, T_nu, n; breakdown=nothing)
    urs = unit_rates(net, T_N, T_nu, n[1], n[2])
    names = breakdown === nothing ? nothing : [r.name for r in REACTIONS if haskey(net.itp.dark, r.name) || haskey(net.itp.portal, r.name)]
    return assemble_collision(urs, n[3], n[4]; names=names, breakdown=breakdown)
end

"""
Densities (n, kappa) of the state u at z. u[1] = ln((n_N1 + n_N2)/s) and
u[2] = ln(n_N2/n_N1): the conversions N1 N1 <-> N2 N2 (up to 1e12 per unit z
at large g) then cancel exactly in the equation for the total, and only relax
the ratio.
"""
function state_densities(net::Network, u, z)
    s = net.ent_interp_z(z)
    N = exp(u[1]) * s
    r = exp(u[2])
    n1 = N / (1 + r)
    n = (n1, N - n1, exp(u[3]) * s, exp(u[4]) * s)
    return n, exp(u[5]) * s^(4 / 3)
end

"""State vector from densities (inverse of `state_densities`)."""
state_vector(n, kappa, s) = [log((n[1] + n[2]) / s), log(n[2] / n[1]), log(n[3] / s), log(n[4] / s), log(kappa / s^(4 / 3))]

# Quasi-steady state of A' and phi. Their decay rates reach 1e20 H, which makes
# d ln n/dz a difference of huge numbers. Above LAMBDA_HI relaxations per unit z
# the species is taken at the root n_q of its (exactly quadratic) C_k = 0: ALL
# collision terms use n_q, so that the fluxes through it balance (with the state
# value instead, the lag of a capped relaxation, times lambda ~ 1e15, would leak
# into N1 and N2), and u_k relaxes to ln Y_q at the capped rate. Below LAMBDA_LO
# the full equation; blended in log n between.
const LAMBDA_LO = 1e3
const LAMBDA_HI = 1e4

"""(ln n_eff, weight w, ln Y_q) of species k given the unit rates and the other densities."""
function qss_blend(urs, k, nA, nphi, lnn_k, n1, s, xfac)
    a0, a1, a2 = qss_coefficients(urs, k, nA, nphi)
    nq, rate = qss_root(a0, a1, a2)
    lam = xfac * rate
    # floor in logs: a Boltzmann-suppressed phi underflows any linear floor
    lnq = max(log(nq), log(n1) - 600)
    lam > LAMBDA_LO || return lnn_k, 0.0, lnq - log(s), lam
    w = min(1.0, log(lam / LAMBDA_LO) / log(LAMBDA_HI / LAMBDA_LO))
    return (1 - w) * lnn_k + w * lnq, w, lnq - log(s), lam
end

function network_rhs!(du, u, net::Network, z)
    x = exp(z)
    T_nu = net.p.m1 / x
    H = net.H_interp_z(z)
    xfac = x / net.dx_dt_interp_z(z)
    s = net.ent_interp_z(z)
    n, kappa = state_densities(net, u, z)
    T_N = dark_temperature(net, n, kappa)
    urs = unit_rates(net, T_N, T_nu, n[1], n[2])
    lnA_eff, wA, uqA, lamA = qss_blend(urs, 3, n[3], n[4], log(n[3]), n[1], s, xfac)
    lnphi_eff, wphi, uqphi, lamphi = qss_blend(urs, 4, exp(lnA_eff), n[4], log(n[4]), n[1], s, xfac)
    neff = (n[1], n[2], exp(lnA_eff), exp(lnphi_eff))
    C, CK, Ctot, Gcf, Gcr = assemble_collision(urs, neff[3], neff[4]; elimA=wA,
                                               dmA=net.masses[3] - net.masses[1] - net.masses[2])
    du[1] = xfac * Ctot / (n[1] + n[2])
    du[2] = xfac * (C[2] / n[2] - C[1] / n[1])
    du[3] = xfac * C[3] / neff[3]
    du[4] = xfac * C[4] / neff[4]
    for (k, w, uq, lam) in ((3, wA, uqA, lamA), (4, wphi, uqphi, lamphi))
        w > 0 || continue
        du[k] = (1 - w) * du[k] - w * min(lam, LAMBDA_HI) * (u[k] - uq)
    end
    # N2/N1 ratio: relaxed by the conversions to n_N2/n_N1 = n^_N2/n^_N1 (detailed balance
    # of N1 N1 <-> N2 N2 alone; 1 for equal masses); quasi-static like A' when fast
    lamc = xfac * 2 * (Gcf + Gcr) * (1 / n[1] + 1 / n[2])
    if lamc > LAMBDA_LO
        w = min(1.0, log(lamc / LAMBDA_LO) / log(LAMBDA_HI / LAMBDA_LO))
        ueq = log(n_scaled_safe(net.masses[2], net.dofs[2], T_N) / n_scaled_safe(net.masses[1], net.dofs[1], T_N))
        du[2] = (1 - w) * du[2] - w * min(lamc, LAMBDA_HI) * (u[2] - ueq)
    end
    P = sum(neff) * T_N
    du[5] = xfac * (H * (kappa - 3P) + CK) / kappa
    if net.dw !== nothing
        # DW production of N1 (T = T_nu before neutrino decoupling; dln T_nu/dz = -1)
        _, dF = dw_fraction(net.dw, T_nu)
        S = -net.dw.Y_final * dF                     # dY_N1/dz
        if S > 0
            p = net.dw.p0 * (s / s0)^(1 / 3)
            Ekin = sqrt(p^2 + net.masses[1]^2) - net.masses[1]
            du[1] += S * s / (n[1] + n[2])
            du[2] -= S * s / n[1]
            du[5] += S * s * Ekin / kappa
        end
    end
    return nothing
end

"""
    network_initial_state(net, dw, tT_rel; dz_seed=1e-3)

State at the start of DW production as the fluid solver has it (all of the
DW yield at T_dw, `DodelsonWidrow`): N1 with the DW number and energy density.
N2, A', phi (in this order, each produced from the previous ones) start with
what their production makes in the first `dz_seed` of z (at most n_N1/2 for
N2, 1e-2 n_N1 for A' and phi), so
that their initial d ln n/dz is ~1/dz_seed; N1 and N2 are then rescaled so
that the number of N after all decays is the DW number. Zero is not representable in log variables,
and a tiny seed gives rates whose time step is below the resolution of z.
"""
function network_initial_state(net::Network, dw::DodelsonWidrow, tT_rel::TimeTempRelation; dz_seed=1e-3,
                               F_start=2e-3)
    if net.dw === nothing
        z0 = log(net.p.m1 / tT_rel.T_nu_grid[dw.i_ic])
        n1 = dw.n_ic
        rho1 = dw.rho_ic
    else
        # start where the DW history has produced F_start of the yield, with that much N1
        h = net.dw
        i = findlast(>=(F_start), h.F)
        T0 = exp(h.lnT[i]) * h.scale
        z0 = log(net.p.m1 / T0)
        F0, _ = dw_fraction(h, T0)
        n1 = F0 * h.Y_final * net.ent_interp_z(z0)
        p = h.p0 * (net.ent_interp_z(z0) / s0)^(1 / 3)
        rho1 = n1 * sqrt(p^2 + net.masses[1]^2)
    end
    s = net.ent_interp_z(z0)
    x0 = exp(z0)
    xfac = x0 / net.dx_dt_interp_z(z0)
    T_nu = net.p.m1 / x0
    n = [n1, 1e-30n1, 1e-30n1, 1e-30n1]
    kappa = rho1 - sum(n[k] * net.masses[k] for k in 1:NSPECIES)
    for k in 2:NSPECIES
        T_N = dark_temperature(net, n, kappa)
        C, _ = network_collision(net, T_N, T_nu, n)
        # production over dz_seed (a production rate does not depend on the tiny n_k),
        # at most the equilibrium share n1/2 for N2 and 1e-2 n1 for A', phi (they are
        # taken out of the DW population below: larger seeds can exhaust it at large g)
        C[k] > 0 && (n[k] = max(n[k], min(k == 2 ? n1 / 2 : 1e-2n1, C[k] * xfac * dz_seed)))
    end
    # the seeds come out of the DW population: keep n~ = n_N1 + n_N2 + 2 n_A' + w_phi n_phi
    # (the number of N after all decays) equal to the DW number
    f = (n1 - 2n[3] - net.w_phi * n[4]) / (n[1] + n[2])
    n[1] *= f
    n[2] *= f
    kappa = rho1 - sum(n[k] * net.masses[k] for k in 1:NSPECIES)
    return state_vector(n, kappa, s), z0
end

"""
    solve_network(net, dw, tT_rel; x_end=100, reltol=1e-6, abstol=1e-8, fmax_stop=0.5, dz_seed)

Integrates the network from the DW initial state to x = x_end. Stops with
retcode Terminated if the dark sector becomes degenerate (largest MB
occupation number > `fmax_stop`): the network does not apply there.
`dz_seed` (`network_initial_state`) defaults to 1e-3 for equal masses and to 1e-9 with an
N1-N2 splitting: the phi N1 N1 coupling g delta m_N/m_A' then makes a large phi seed, whose
decays light the A'A' -> phi phi -> 4 A' loop at the first point (CLAUDE.md, network seed).
"""
function solve_network(net::Network, dw::DodelsonWidrow, tT_rel::TimeTempRelation;
                       x_end=100.0, reltol=1e-6, abstol=1e-8, fmax_stop=0.5, dz_seed=nothing, kw...)
    dz_seed = something(dz_seed, table_delta(net.itp) > 0 ? 1e-9 : 1e-3)
    u0, z0 = network_initial_state(net, dw, tT_rel; dz_seed=dz_seed)
    prob = DE.ODEProblem(network_rhs!, u0, (z0, log(x_end)), net)
    # MB validity: stop (retcode Terminated) once a dark occupation number exceeds
    # fmax_stop; beyond it Pauli blocking / Bose enhancement matter and the MB rates,
    # which grow with the fugacities, make the system ever stiffer
    degenerate = DE.DiscreteCallback((u, t, integ) -> maximum(network_diagnostics(net, u, t).fmax) > fmax_stop,
                                     DE.terminate!; save_positions=(false, true))
    # reject steps that leave the representable range (exp of the state overflowing)
    bad(u, p, t) = any(x -> !isfinite(x) || x > 700, u)
    return DE.solve(prob, ODER.Rodas4P(autodiff=AutoFiniteDiff()); reltol=reltol, abstol=abstol,
                    isoutofdomain=bad, callback=degenerate, kw...)
end

"""Omega h^2 of the state u at z, counting each A' as the N1 N2 pair it decays into
and each phi with its weight `w_phi` (4 for phi -> A'A'), of which `w_phi2` are N2. With
an N1-N2 splitting every N2 carries m_N2 = m_N1 + dm."""
function network_omega_h2(net::Network, u, z)
    s = net.ent_interp_z(z)
    # u[1] = ln(Y_N1 + Y_N2), u[2] = ln(Y_N2/Y_N1)
    m1, dm = net.masses[1], net.masses[2] - net.masses[1]
    Y2 = exp(u[1]) / (1 + exp(-u[2]))
    return (m1 * (exp(u[1]) + 2exp(u[3]) + net.w_phi * exp(u[4])) +
            dm * (Y2 + exp(u[3]) + net.w_phi2 * exp(u[4]))) * s0 / rho_crit0_h2
end

"""Diagnostics along a solution: T_N, T_nu, largest MB occupation number per species, Omega h^2."""
function network_diagnostics(net::Network, u, z)
    n, kappa = state_densities(net, u, z)
    T_N = dark_temperature(net, n, kappa)
    # largest occupation e^(-gap) = n / n_scaled(T_N) per dof-summed density
    fmax = ntuple(k -> n[k] / n_scaled_safe(net.masses[k], net.dofs[k], T_N), NSPECIES)
    return (z=z, x=exp(z), T_N=T_N, T_nu=net.p.m1 * exp(-z), fmax=fmax, omega_h2=network_omega_h2(net, u, z))
end
