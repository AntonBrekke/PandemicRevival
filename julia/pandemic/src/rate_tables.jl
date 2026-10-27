"""
rate_tables.jl

Tabulated reaction rates for the species-resolved network (plan in
notes/model_limitations.tex, Sec. "Plan"): N1, N2, A', phi Maxwell-Boltzmann
(MB) at one dark temperature T_N, each with its own chemical potential; the
neutrino Fermi-Dirac at T_nu with mu = 0, including Pauli blocking when it is
in the final state.

REACTIONS AND COLLISION TERMS
=============================
Every reaction r is written in a forward direction with dark initial set I
and dark final set F (`REACTIONS`). With MB dark species the chemical
potentials factor out exactly, and the reaction rates (reactions per volume
and time) are
    Gamma_fwd = Rf * prod_{i in I} e^(-gap_i),   Gamma_rev = Rr * prod_{j in F} e^(-gap_j),
gap = (m - mu)/T_N, where Rf (Rr) is the forward (reverse) reaction rate at
mu = 0 times e^(sum_I m / T_N) (e^(sum_F m / T_N)), so that nothing under- or
overflows. Then dn_i/dt|_r = dN_i (Gamma_fwd - Gamma_rev). Rf and Rr count
reactions, not ordered pairs: half the rate density of thermal_rates.jl for
two identical initial particles.

- Dark-only reactions (kind :dark2, :dark12): detailed balance at the common
  T_N gives Rr = Rf e^((sum_F m - sum_I m)/T_N). Only the exothermic direction
  is integrated; a function of x_N = m_N/T_N.
- Portal reactions (:portal2, :portal12; nu in the initial state forward, in
  the final state in reverse). Both directions are integrals over the same
  a + nu phase space (a the dark initial particle): forward with nu weight
  f_nu(E_nu), reverse with e^(-(E_a + E_nu - sum_F m)/T_N) [1 - f_nu(E_nu)]
  (energy conservation moves the final-state Boltzmann factors onto the a + nu
  side). The Gondolo-Gelmini reduction of thermal_rates.jl holds for any such
  weights. Functions of (x_nu = m_N/T_nu, r = T_N/T_nu). The energy the dark
  sector gains per forward reaction is E_nu, so the energy-transfer rates Qf,
  Qr are the same integrals weighted with E_nu. Where an s-channel pole lies
  inside the integration range, the resonance-subtracted rate can be negative
  and is stored as a positive reference times a signed ratio (`portal_point`).

NORMALISATION
=============
All masses are fixed multiples of m_N (`mA_over_mN`, `mphi_over_mN`), and each
squared amplitude is a single monomial in the couplings (the portal coupling
y_+ theta_0 = g theta_1 m_D/m_A' ties the theta_0 and theta_1 diagrams). The
tables are computed once at reference parameters `p_ref` and rescaled:
    R(p) = R_ref (m_N/m_ref)^4 c_r(p)/c_r(p_ref),    Q: (m_N/m_ref)^5,
with c_r the squared amplitude at a fixed kinematic point in units of m_N
(`coupling_factor`; test/test_rate_tables.jl checks that it factorises).
Resonant s-channels are real-intermediate-state subtracted (the resonance is a
species); at the reference couplings the widths are narrow, and the
narrow-width limit of the subtracted rate does not depend on the width.
"""

include(joinpath(@__DIR__, "thermal_rates.jl"))
import Serialization

const DARK_SPECIES = (:N1, :N2, :A, :phi)

"""
Scaled Bessel function e^x K_nu(x) for nu = 1, 2: SpecialFunctions below x = 1e3,
the asymptotic series above (relative error < 1e-12; SpecialFunctions throws
beyond x ~ 1e8, reached at m/T_N = 1e8).
"""
function besselkx_safe(nu::Int, x)
    x < 1e3 && return SF.besselkx(nu, x)
    mu = 4nu^2
    c1 = (mu - 1) / 8
    c2 = (mu - 1) * (mu - 9) / 128
    c3 = (mu - 1) * (mu - 9) * (mu - 25) / 3072
    return sqrt(pi / (2x)) * (1 + c1 / x + c2 / x^2 + c3 / x^3)
end

"""MB number density without e^(-gap) (as thermal_rates' n_scaled, safe for m/T >> 1)."""
n_scaled_safe(m, g, T) = g * m^2 * T * besselkx_safe(2, m / T) / (2pi^2)
dark_index(f) = f === :N1 ? 1 : f === :N2 ? 2 : f === :A ? 3 : f === :phi ? 4 : error("species $f")

"""A reaction of the network: forward direction, dark initial and final sets
and the TREE_PROCESSES entry it comes from."""
struct Reaction
    name::Symbol
    kind::Symbol          # :dark2, :dark12, :portal2, :portal12
    proc::Symbol          # name in TREE_PROCESSES
    I::Vector{Symbol}     # dark initial state (forward)
    F::Vector{Symbol}     # dark final state
end

"""Change of (N1, N2, A', phi) per forward reaction."""
delta_N(r::Reaction) = ntuple(k -> count(==(DARK_SPECIES[k]), r.F) - count(==(DARK_SPECIES[k]), r.I), 4)

const REACTIONS = [
    # dark sector only, 2 -> 2 (number changing; elastic ones do not enter)
    Reaction(:N1N1_N2N2, :dark2, :N1N1_N2N2, [:N1, :N1], [:N2, :N2]),
    Reaction(:N1N1_AA, :dark2, :N1N1_AA, [:N1, :N1], [:A, :A]),
    Reaction(:N2N2_AA, :dark2, :N2N2_AA, [:N2, :N2], [:A, :A]),
    Reaction(:N1N2_Aphi, :dark2, :N1N2_Aphi, [:N1, :N2], [:A, :phi]),
    Reaction(:AA_phiphi, :dark2, :AA_phiphi, [:A, :A], [:phi, :phi]),
    Reaction(:N1N1_phiphi, :dark2, :N1N1_phiphi, [:N1, :N1], [:phi, :phi]),
    # dark sector only, decays (forward = decay)
    Reaction(:A_N1N2, :dark12, :A_N1N2, [:A], [:N1, :N2]),
    Reaction(:phi_AA, :dark12, :phi_AA, [:phi], [:A, :A]),
    Reaction(:phi_N1N1, :dark12, :phi_N1N1, [:phi], [:N1, :N1]),
    # portal, inverse decays (forward = a nu -> resonance)
    Reaction(:N2nu_A, :portal12, :A_N2nu, [:N2], [:A]),
    Reaction(:N1nu_phi, :portal12, :phi_N1nu, [:N1], [:phi]),
    # portal, 2 -> 2 (forward = a nu -> c d)
    Reaction(:N2nu_N1N2, :portal2, :N2nu_N1N2, [:N2], [:N1, :N2]),
    Reaction(:N1nu_N2N2, :portal2, :N1nu_N2N2, [:N1], [:N2, :N2]),
    Reaction(:N1nu_AA, :portal2, :N1nu_AA, [:N1], [:A, :A]),
    Reaction(:N2nu_Aphi, :portal2, :N2nu_Aphi, [:N2], [:A, :phi]),
    Reaction(:N1nu_phiphi, :portal2, :N1nu_phiphi, [:N1], [:phi, :phi]),
]

reaction(name) = only(filter(r -> r.name === name, REACTIONS))
_tree(name) = only(filter(q -> q.name === name, TREE_PROCESSES))
mass_sum(fs, p) = sum(field_mass(f, p) for f in fs)
symmetry(fs) = length(fs) == 2 && fs[1] === fs[2] ? 2.0 : 1.0

# --------------------------------------------------------------------------- nu-weighted W
# Gauss-Legendre rule of the fixed-node quadratures below
const GL_NODES, GL_WEIGHTS = QuadGK.gauss(24)

"""
Tail integral Phi(x) = ∫_x^∞ dE E^k w(E) of a nu weight times E^k, for
    dir = :fwd   w = f_nu = 1/(e^(E/T_nu) + 1)          (nu initial)
    dir = :rev   w = e^(-E/T_N) [1 - f_nu]              (nu final)
    dir = :mb    w = e^(-E/T_nu)                        (MB nu, for tests)
tabulated as ln Phi + x/T_w (slowly varying) against ln(x/T_w), T_w the
temperature of the exponential fall-off (T_N for :rev, else T_nu), and
interpolated with a natural cubic spline, from 1e-10 T_w to 1e8 T_w (with the shifts of `W_nu` the dominant
E_nu can lie far out in the tail). Below the grid Phi is constant to 1e-10,
above it the incomplete gamma function of the asymptotic weight continues it. A plain concrete struct on purpose: closure- or
Interpolations-typed fields send type inference of the nested quadratures
into a near-endless loop.
"""
struct NuTail
    dir::Symbol
    k::Int
    T_N::Float64
    T_nu::Float64
    T_w::Float64
    lnu_min::Float64
    h::Float64                 # grid step in ln(x/T_w)
    lnPsi::Vector{Float64}     # ln Phi + x/T_w, slowly varying: this is what is interpolated
    d2::Vector{Float64}        # spline second derivatives
end

"""Log of the weight of `t` times E^k at E = x."""
function log_weight(t::NuTail, x)
    lk = t.k == 0 ? 0.0 : t.k * log(x)
    t.dir === :fwd && return lk - x / t.T_nu - log1p(exp(-x / t.T_nu))
    t.dir === :rev && return lk - x / t.T_N - log1p(exp(-x / t.T_nu))
    return lk - x / t.T_nu
end

"""Second derivatives of the natural cubic spline through y on a uniform grid of step h."""
function natural_spline_d2(y, h)
    n = length(y)
    d2 = zeros(n)
    c = zeros(n)
    # tridiagonal system d2[i-1] + 4 d2[i] + d2[i+1] = 6 (y[i+1] - 2y[i] + y[i-1])/h^2, d2[1] = d2[n] = 0
    for i in 2:n-1
        rhs = 6 * (y[i+1] - 2y[i] + y[i-1]) / h^2
        piv = 4 - (i > 2 ? c[i-1] : 0.0)
        c[i] = 1 / piv
        d2[i] = (rhs - (i > 2 ? d2[i-1] : 0.0)) / piv
    end
    for i in n-2:-1:2
        d2[i] -= c[i] * d2[i+1]
    end
    return d2
end

Base.@constprop :none function NuTail(dir::Symbol, k::Int, T_N, T_nu; lnu_min=log(1e-10), lnu_max=log(1e8), n=1400)
    dir in (:fwd, :rev, :mb) || error("nu weight $dir")
    k in (0, 1) || error("moment $k")
    T_w = dir === :rev ? T_N : T_nu
    h = (lnu_max - lnu_min) / (n - 1)
    t0 = NuTail(dir, k, T_N, T_nu, T_w, lnu_min, h, Float64[], Float64[])
    xs = [T_w * exp(lnu_min + (i - 1) * h) for i in 1:n]
    # Psi(x) = e^(x/T_w) Phi(x) does not underflow; backward recursion
    # Psi_i = e^(-(x_{i+1} - x_i)/T_w) Psi_{i+1} + ∫_{x_i}^{x_{i+1}} w(y) e^(x_i/T_w) dy
    # (the integrand falls by e^-80 within 80 T_w: clip there, and use fixed
    # Gauss-Legendre panels no wider than 4 T_w)
    function seg(x0, x1)
        b = min(x1, x0 + 80T_w)
        npan = max(1, ceil(Int, (b - x0) / (4T_w)))
        dx = (b - x0) / npan
        acc = 0.0
        for p in 0:npan-1
            a = x0 + p * dx
            for j in eachindex(GL_NODES)
                y = a + dx * (1 + GL_NODES[j]) / 2
                acc += GL_WEIGHTS[j] * dx / 2 * exp(log_weight(t0, y) + x0 / T_w)
            end
        end
        return acc
    end
    Psi = similar(xs)
    Psi[end] = seg(xs[end], Inf)
    for i in n-1:-1:1
        Psi[i] = exp(-(xs[i+1] - xs[i]) / T_w) * Psi[i+1] + seg(xs[i], xs[i+1])
    end
    lnPsi = log.(Psi)
    return NuTail(dir, k, T_N, T_nu, T_w, lnu_min, h, lnPsi, natural_spline_d2(lnPsi, h))
end

function ln_tail(t::NuTail, x)
    x <= 0 && return t.lnPsi[1]
    n = length(t.lnPsi)
    u = (log(x / t.T_w) - t.lnu_min) / t.h          # fractional grid index from 0
    u <= 0 && return t.lnPsi[1] - x / t.T_w
    if u >= n - 1
        # incomplete gamma: ∫_x^∞ E^k e^(-E/T) ∝ e^(-x/T) (1, x/T + 1) for k = (0, 1), times the
        # slowly varying 1/(1 + e^(-x/T_nu)) of the FD weights (for :rev with T_N << T_nu)
        x_max = t.T_w * exp(t.lnu_min + (n - 1) * t.h)
        poly(y) = t.k == 0 ? 1.0 : y + 1
        lg(y) = t.dir === :mb ? 0.0 : -log1p(exp(-y / t.T_nu))
        return t.lnPsi[n] - x / t.T_w + log(poly(x / t.T_w) / poly(x_max / t.T_w)) + lg(x) - lg(x_max)
    end
    i = floor(Int, u) + 1
    b = u - (i - 1)
    a = 1 - b
    y, d2, h = t.lnPsi, t.d2, t.h
    return a * y[i] + b * y[i+1] + ((a^3 - a) * d2[i] + (b^3 - b) * d2[i+1]) * h^2 / 6 - x / t.T_w
end

"""
Energy of the dark particle (mass m) that minimises w + E_nu^min(E_a)/T_w,
with w = (E_a - m)/T_N and E_nu^min = A/(2(E_a + p_a)), A = s - m^2: the
saddle of the a + nu integrand. From p_a (E_a + p_a) = A T_N/(2 T_w).
"""
function saddle_E(A, m, T_N, T_w)
    K = A * T_N / (2T_w)
    u = sqrt(m^2 + 2K)            # E + p at the saddle
    return (u + m^2 / u) / 2
end

# panel layout of `W_nu`
const W_PANEL_FRACS = (0.1, 0.3, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0)
const W_PANEL_OFFSETS = (1.0, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0)

"""
    W_nu(s, m, T_N, tail; c)

∫ dE_a e^(-(E_a - m)/T_N) ∫ dE_nu E_nu^k w(E_nu) over the kinematic range of a
(mass m) + massless nu at fixed s, times e^c, with the nu weight and k of
`tail` (a `NuTail`). `c` keeps the integrand <= O(1): the saddle value at
threshold for :fwd and :mb (`fwd_shift`), (sum_F m - m)/T_N for :rev.

The quadrature is a fixed Gauss-Legendre rule on panels whose edges move
continuously with the saddle point, in v = sqrt(w) (the E_nu range opens like
sqrt(w)), so that W_nu is a smooth function of s: the real-intermediate-state
subtraction in `rate_density_scaled` cancels a 1/(s - M^2)^2 pole between
W at s0 ± d and fails on the step-like noise of adaptive quadrature.
"""
Base.@constprop :none function W_nu(s, m, T_N, tail::NuTail; c=0.0)
    A = s - m^2
    A > 0 || return 0.0
    function inner(w)
        Ea = m + T_N * w
        pa = sqrt(T_N * w * (2m + T_N * w))
        lo = A / (2 * (Ea + pa))
        hi = A * (Ea + pa) / (2m^2)
        hi > lo || return 0.0
        lnlo = ln_tail(tail, lo)
        return exp(-w + c + lnlo) * -expm1(ln_tail(tail, hi) - lnlo)
    end
    ws = (saddle_E(A, m, T_N, tail.T_w) - m) / T_N
    edges = sort!([0.0; [ws * f for f in W_PANEL_FRACS]; [ws + o for o in W_PANEL_OFFSETS]])
    total = 0.0
    for i in 1:length(edges)-1
        va, vb = sqrt(edges[i]), sqrt(edges[i+1])
        vb > va || continue
        hw, mid = (vb - va) / 2, (vb + va) / 2
        for j in eachindex(GL_NODES)
            v = mid + hw * GL_NODES[j]
            total += GL_WEIGHTS[j] * hw * 2v * inner(v^2)       # dw = 2v dv
        end
    end
    # dE_a = T_N dw
    return T_N * total
end

"""Largest value of -(w + E_nu^min/T_nu) over the a + nu phase space at s = s_lo:
the shift that keeps the forward integrand O(1)."""
function fwd_shift(s_lo, m, T_N, T_nu)
    A = s_lo - m^2
    E = saddle_E(A, m, T_N, T_nu)
    p = sqrt(max(E^2 - m^2, 0.0))
    return (E - m) / T_N + A / (2 * (E + p) * T_nu)
end

# --------------------------------------------------------------------------- s integral
"""Sum of quadgk over consecutive intervals of `pts` (no splatting: see `W_nu`).
`atol` is per interval; give one whenever the integrand changes sign, or a
piece that integrates to ~0 never meets `rtol`."""
function quad_pieces(f::F, pts; rtol, atol=0.0, maxevals=10^5) where {F}
    total = 0.0
    for i in 1:length(pts)-1
        pts[i+1] > pts[i] || continue
        total += QuadGK.quadgk(f, pts[i], pts[i+1]; rtol=rtol, atol=atol, maxevals=maxevals)[1]
    end
    return total
end

"""
    nw_rate_density(tab, m, Tm, W; rtol)

Rate density (times the scalings carried by W) of a + nu -> c + d, a of mass m
and nu massless, as rate_density_scaled with propagator=:ris but in the
narrow-width limit, computed directly: with sigma = X|D|^2 + 2Y Re D + Z,
the RIS-subtracted |D|^2 tends to the Hadamard finite part of 1/(s - M^2)^2
and Re D to the principal value of 1/(s - M^2), neither of which depends on
the width. The direct RIS formula cancels terms of relative size (M/Gamma)^2
near the pole and does not converge for the narrow widths of the reference
couplings. `Ts` (a tuple of temperatures) sets the breakpoints. With `zonly`, only the non-resonant
part Z (|M|^2 of the t- and u-channels, positive).
"""
Base.@constprop :none function nw_rate_density(tab::SigmaTable, m, Ts, W::F; rtol=1e-6, zonly=false) where {F}
    s_lo = max(tab.s_th, m^2)
    rs_lo = sqrt(s_lo)
    s0 = tab.M^2
    part(k, s) = s <= tab.s_th ? 0.0 : tab.interp[k](log(s - tab.s_th))
    flux(s) = (s - m^2) / 2
    # zero-width cross section, valid away from the pole
    sig(s) = (zonly || tab.M == 0) ? part(3, s) : (d = s - s0; part(1, s) / d^2 + 2part(2, s) / d + part(3, s))
    Gq(q) = (rs = rs_lo + q; s = rs^2; 2rs * sig(s) * flux(s) * W(s))
    q_max = sqrt(tab.s_max) - rs_lo
    # breakpoints on the scales of both temperatures: for T_N << T_nu the reverse
    # integrand lives within ~T_N of threshold, and quadgk misses it otherwise
    qk = sort!([T * f for T in Ts for f in (0.01, 0.1, 1.0, 5.0, 20.0, 80.0, 250.0)])
    qpts(lo, hi) = sort!([lo; filter(x -> lo < x < hi, qk); hi])
    if zonly || !(tab.M > 0 && s0 > s_lo)
        return quad_pieces(Gq, qpts(0.0, q_max); rtol=rtol) / (16pi^4)
    end
    L = min(s0 - s_lo, 0.5 * s0)
    below = quad_pieces(Gq, qpts(0.0, sqrt(s0 - L) - rs_lo); rtol=rtol)
    above = quad_pieces(Gq, qpts(sqrt(s0 + L) - rs_lo, q_max); rtol=rtol)
    h0 = part(1, s0) * flux(s0) * W(s0)
    # FP ∫_{-L}^{L} h/d^2 = ∫_0^L [h(s0+d) + h(s0-d) - 2h0]/d^2 - 2h0/L, h = X F W; PV of
    # 2k/d (k = Y F W) paired the same way; z = Z F W regular. W once per s.
    # Below delta the second difference loses its digits to rounding; it is constant there.
    delta = 1e-4 * L
    function fx(d)
        sp, sm = s0 + d, s0 - d
        wp, wm = flux(sp) * W(sp), flux(sm) * W(sm)
        return (part(1, sp) * wp + part(1, sm) * wm - 2h0) / d^2 + 2 * (part(2, sp) * wp - part(2, sm) * wm) / d +
               part(3, sp) * wp + part(3, sm) * wm
    end
    dk = [abs((rs_lo + q)^2 - s0) for q in qk]      # the same breakpoints, in d = |s - s0|
    dpts = sort!(unique!([delta; filter(d -> delta < d < L, [dk; [L * 10.0^-j for j in 3:-1:1]]); L]))
    # fx changes sign: an absolute tolerance on the scale of the whole integral
    scale = abs(2h0 / L) + abs(below) + abs(above) + abs(fx(L)) * L
    J = fx(delta) * delta + quad_pieces(fx, dpts; rtol=rtol, atol=rtol * scale / length(dpts)) - 2h0 / L
    return (below + J + above) / (16pi^4)
end

# --------------------------------------------------------------------------- one grid point
"""
    portal_point(r, p, tab, T_N, T_nu)

The portal reaction `r` at (T_N, T_nu), for the parameters `p`
(unnormalised): (lnref, ratio), each a 4-tuple over (Rf, Rr, Qf, Qr), with
rate = exp(lnref) * ratio. Once the on-shell resonance is subtracted (it is a
species of the network), the rest of a process with an s-channel pole inside
the integration range (N2 nu -> N1 N2, N1 nu -> A'A') can be negative: lnref
is then its non-resonant part Z (positive) and ratio the full subtracted rate
over it. Otherwise ratio = 1. `tab` is the SigmaTable of a 2 -> 2 process
(unused for inverse decays).
"""
Base.@constprop :none function portal_point(r::Reaction, p, tab, T_N, T_nu)
    q = _tree(r.proc)
    a = only(r.I)
    ma = field_mass(a, p)
    Delta = mass_sum(r.F, p) - ma
    tails = (NuTail(:fwd, 0, T_N, T_nu), NuTail(:rev, 0, T_N, T_nu),
             NuTail(:fwd, 1, T_N, T_nu), NuTail(:rev, 1, T_N, T_nu))
    cr = Delta / T_N
    if r.kind === :portal12
        s0 = field_mass(only(r.F), p)^2
        # inverse decay a + nu -> R: sigma = pi sum|M|^2 delta(s - M^2)/sqrt(lambda), as rate_inverse_decay
        pref = (pi / 2) * q.msq(p) / (16pi^4)
        cf = fwd_shift(s0, ma, T_N, T_nu)
        shifts = (cf, cr, cf, cr)
        lnref = ntuple(i -> log(pref * W_nu(s0, ma, T_N, tails[i]; c=shifts[i])) - (isodd(i) ? cf : 0.0), 4)
        return (lnref, (1.0, 1.0, 1.0, 1.0))
    end
    cf = fwd_shift(max(tab.s_th, ma^2), ma, T_N, T_nu)
    shifts = (cf, cr, cf, cr)
    resonant = tab.M > 0 && tab.M^2 > max(tab.s_th, ma^2)
    R(i, zonly) = nw_rate_density(tab, ma, (T_N, T_nu), s -> W_nu(s, ma, T_N, tails[i]; c=shifts[i]); zonly=zonly)
    full = ntuple(i -> R(i, false), 4)
    ref = resonant ? ntuple(i -> R(i, true), 4) : full
    lnref = ntuple(i -> log(ref[i]) - (isodd(i) ? cf : 0.0), 4)
    return (lnref, ntuple(i -> full[i] / ref[i], 4))
end

"""
    gg_rate_scaled(tab, ma, mb, T)

Gondolo-Gelmini rate density of a + b -> c + d, both MB at T with mu = 0
(ordered pairs for a = b), times e^((ma + mb)/T):
    R = T/(32 pi^4) ∫ ds sigma(s) lambda(s, ma^2, mb^2)/sqrt(s) K_1(sqrt(s)/T),
sigma summed over all spins, zero width (the dark-only processes have no pole
inside their range). One stable integral in q = sqrt(s) - sqrt(s_lo), with
K_1 scaled; thermal_rates.jl's two-dimensional W loses p_a to cancellation for
T << m and then never converges.
"""
Base.@constprop :none function gg_rate_scaled(tab::SigmaTable, ma, mb, T; rtol=1e-7)
    s_lo = max(tab.s_th, (ma + mb)^2)
    rs_lo = sqrt(s_lo)
    s0 = tab.M^2
    part(k, s) = s <= tab.s_th ? 0.0 : tab.interp[k](log(s - tab.s_th))
    sig(s) = tab.M == 0 ? part(3, s) : (d = s - s0; part(1, s) / d^2 + 2part(2, s) / d + part(3, s))
    function Gq(q)
        rs = rs_lo + q
        s = rs^2
        # K_1(rs/T) e^((ma + mb)/T) = besselkx(1, rs/T) e^(-(rs - ma - mb)/T)
        return 2rs * sig(s) * kallen(s, ma^2, mb^2) / rs * besselkx_safe(1, rs / T) * exp(-(rs - ma - mb) / T)
    end
    q_max = sqrt(tab.s_max) - rs_lo
    pts = sort!([0.0; filter(x -> 0 < x < q_max, [T * f for f in (0.1, 1.0, 5.0, 20.0, 80.0, 250.0)]); q_max])
    return T / (32pi^4) * quad_pieces(Gq, pts; rtol=rtol)
end

"""
    dark_point(r, p, tab, T_N)

(lnRf, lnRr) of the dark-only reaction `r` at T_N. `tab` is the SigmaTable of
the exothermic direction of a 2 -> 2 process (unused for decays).
"""
Base.@constprop :none function dark_point(r::Reaction, p, tab, T_N)
    mI, mF = mass_sum(r.I, p), mass_sum(r.F, p)
    if r.kind === :dark12
        R = only(r.I)
        M = field_mass(R, p)
        width(r.proc, p) > 0 || return (-Inf, -Inf)
        # decays per volume and time at mu = 0, times e^(M/T_N)
        # Gamma <m/E> = Gamma K_1/K_2 (rate_decay, safe for M/T >> 1)
        lnRf = log(n_scaled_safe(M, field_dof(R), T_N) * width(r.proc, p) *
                   besselkx_safe(1, M / T_N) / besselkx_safe(2, M / T_N))
        return (lnRf, lnRf + (mF - mI) / T_N)
    end
    exo_fwd = mF <= mI
    ini = exo_fwd ? r.I : r.F
    ma, mb = field_mass(ini[1], p), field_mass(ini[2], p)
    lnR_exo = log(gg_rate_scaled(tab, ma, mb, T_N) / symmetry(ini))
    return exo_fwd ? (lnR_exo, lnR_exo + (mF - mI) / T_N) : (lnR_exo + (mI - mF) / T_N, lnR_exo)
end

"""SigmaTable used by `dark_point` / `portal_point` for reaction `r`, valid up to T_max."""
function reaction_sigma_table(r::Reaction, p; T_max, n=800)
    r.kind in (:dark12, :portal12) && return nothing
    q = _tree(r.proc)
    if r.kind === :dark2 && mass_sum(r.F, p) > mass_sum(r.I, p)
        q = reversed_process(q)
    end
    return SigmaTable(q, p; T_max=T_max, n=n)
end

# --------------------------------------------------------------------------- normalisation
"""
Squared amplitude of the process behind `r` at a fixed kinematic point, in
units of m_N (dimensionless): the coupling factor c_r of the header. For a
resonant process the first nonzero of (Z, X/m^4, Y/m^2) is used.
"""
function coupling_factor(r::Reaction, p; s_fac=1.7, t_frac=0.37)
    q = _tree(r.proc)
    m = p.m1
    length(q.initial) == 1 && return q.msq(p) / m^2
    m1, m2 = field_mass(q.initial[1], p), field_mass(q.initial[2], p)
    m3, m4 = field_mass(q.final[1], p), field_mass(q.final[2], p)
    s = s_fac^2 * max((m1 + m2)^2, (m3 + m4)^2)
    tlo, thi = _t_range(s, m1, m2, m3, m4)
    t = tlo + t_frac * (thi - tlo)
    q.xyz === nothing && return q.msq(s, t, p)
    X, Y, Z = q.xyz(s, t, p)
    # X = (s - M^2)^2 |M_s|^2 and Y carry mass^4 and mass^2
    return Z != 0 ? Z : X != 0 ? X / m^4 : Y / m^2
end

"""ln of the factor that turns a reference rate (Q if `energy`) into one at `p`."""
ln_norm(r::Reaction, p, p_ref; energy=false) =
    (energy ? 5 : 4) * log(p.m1 / p_ref.m1) + log(coupling_factor(r, p) / coupling_factor(r, p_ref))

# --------------------------------------------------------------------------- tables
"""Grids: dark-only tables in ln x_N, portal tables in (ln x_nu, ln r), r = T_N/T_nu."""
const LNXN_GRID = range(log(1e-6), log(1e8), length=141)
const LNXNU_GRID = range(log(3e-6), log(1e3), length=87)
const LNR_GRID = range(log(1e-5), log(10.0), length=31)

"""Reference parameters of the tables: 10 keV, g = 1e-3, sin^2 2theta = 1e-14 (narrow widths),
N1-N2 splitting `delta` (rate_params)."""
reference_params(; mA_over_mN=2.5, mphi_over_mN=100.0, delta=0.0) =
    rate_params(1e-5, 1e-3, 1e-14; mA_over_mN=mA_over_mN, mphi_over_mN=mphi_over_mN, delta=delta)

"""
    build_rate_tables(; mA_over_mN, mphi_over_mN, reactions, lnxN, lnxnu, lnr, verbose)

All tables at the reference parameters, threaded over grid points. Returns a
NamedTuple (p_ref, grids, dark::Dict(name => (lnRf, lnRr)), portal::Dict(name
=> (lnRf, lnRr, lnQf, lnQr, ratio_Rf, ratio_Rr, ratio_Qf, ratio_Qr))) ready for
`save_rate_tables`; a portal rate is exp(ln...) * ratio_... (`portal_point`).
"""
function build_rate_tables(; mA_over_mN=2.5, mphi_over_mN=100.0, delta=0.0, reactions=REACTIONS,
                           lnxN=LNXN_GRID, lnxnu=LNXNU_GRID, lnr=LNR_GRID, verbose=true)
    p = reference_params(; mA_over_mN=mA_over_mN, mphi_over_mN=mphi_over_mN, delta=delta)
    m = p.m1
    T_max = m * exp(-min(first(lnxN), first(lnxnu) - last(lnr)))
    dark = Dict{Symbol,Any}()
    portal = Dict{Symbol,Any}()
    for r in reactions
        t0 = time()
        tab = reaction_sigma_table(r, p; T_max=T_max)
        if r.kind in (:dark2, :dark12)
            out = Matrix{Float64}(undef, length(lnxN), 2)
            Threads.@threads for i in eachindex(lnxN)
                out[i, :] .= dark_point(r, p, tab, m * exp(-lnxN[i]))
            end
            dark[r.name] = (lnRf=out[:, 1], lnRr=out[:, 2])
        else
            idx = vec(collect(Iterators.product(eachindex(lnxnu), eachindex(lnr))))
            vals = Array{Float64}(undef, length(lnxnu), length(lnr), 8)
            Threads.@threads for (i, j) in idx
                T_nu = m * exp(-lnxnu[i])
                lnref, ratio = portal_point(r, p, tab, exp(lnr[j]) * T_nu, T_nu)
                vals[i, j, :] .= (lnref..., ratio...)
            end
            portal[r.name] = (lnRf=vals[:, :, 1], lnRr=vals[:, :, 2], lnQf=vals[:, :, 3], lnQr=vals[:, :, 4],
                              ratio_Rf=vals[:, :, 5], ratio_Rr=vals[:, :, 6], ratio_Qf=vals[:, :, 7], ratio_Qr=vals[:, :, 8])
        end
        verbose && (println(rpad(r.name, 14), " ", round(time() - t0; digits=1), " s"); flush(stdout))
    end
    return (p_ref=p, mA_over_mN=mA_over_mN, mphi_over_mN=mphi_over_mN, delta=delta,
            lnxN=collect(lnxN), lnxnu=collect(lnxnu), lnr=collect(lnr), dark=dark, portal=portal)
end

"""File of a table set; tables with an N1-N2 splitting carry `_delta<delta>`."""
rate_table_path(dir; mA_over_mN=2.5, mphi_over_mN=100.0, delta=0.0) =
    joinpath(dir, "rate_tables_mA$(mA_over_mN)_mphi$(mphi_over_mN)" * (delta > 0 ? "_delta$(delta)" : "") * ".jls")

save_rate_tables(path, tabs) = (mkpath(dirname(path)); Serialization.serialize(path, tabs); path)
load_rate_tables(path) = Serialization.deserialize(path)

# --------------------------------------------------------------------------- evaluation
"""
Interpolants of a loaded table set (cubic splines in ln x_N, resp. (ln x_nu,
ln r)); outside the grids they extrapolate linearly in the logs, which is only
safe a little beyond the edges.
"""
struct RateInterpolants
    p_ref::Any
    lnxN::Tuple{Float64,Float64}
    lnxnu::Tuple{Float64,Float64}
    lnr::Tuple{Float64,Float64}
    dark::Dict{Symbol,Any}
    portal::Dict{Symbol,Any}
    delta::Float64
end

function RateInterpolants(tabs)
    rN = range(first(tabs.lnxN), last(tabs.lnxN), length=length(tabs.lnxN))
    rnu = range(first(tabs.lnxnu), last(tabs.lnxnu), length=length(tabs.lnxnu))
    rr = range(first(tabs.lnr), last(tabs.lnr), length=length(tabs.lnr))
    itp1(v) = extrapolate(scale(interpolate(v, BSpline(Cubic(Line(OnGrid())))), rN), Line())
    itp2(v) = extrapolate(scale(interpolate(v, BSpline(Cubic(Line(OnGrid())))), rnu, rr), Line())
    dark = Dict{Symbol,Any}(k => map(itp1, v) for (k, v) in tabs.dark if all(isfinite, v.lnRf))
    portal = Dict{Symbol,Any}(k => map(itp2, v) for (k, v) in tabs.portal)
    return RateInterpolants(tabs.p_ref, extrema(rN), extrema(rnu), extrema(rr), dark, portal,
                            haskey(tabs, :delta) ? tabs.delta : 0.0)
end


"""(m_A'/m_N, m_phi/m_N) of a table set."""
mass_ratios(itp::RateInterpolants) = (itp.p_ref.mA / itp.p_ref.m1, itp.p_ref.mphi / itp.p_ref.m1)

"""N1-N2 splitting delta of a table set (0 for tables built before it existed)."""
table_delta(itp::RateInterpolants) = itp.delta

"""rate_params at (m_N, g, sin^2 2theta) with the mass ratios and splitting of the table set `itp`;
m_nu [GeV] (couplings at equal masses, e.g. the phi N1 N1 Yukawa ∝ g sin^2 theta_1/m_nu) only for
equal-mass tables, where a splitting does not fix it."""
function table_params(itp::RateInterpolants, m_N, g, sin2_2theta; m_nu=M_NU)
    rA, rphi = mass_ratios(itp)
    d = table_delta(itp)
    return rate_params(m_N, g, sin2_2theta; mA_over_mN=rA, mphi_over_mN=rphi, delta=d,
                       m_nu=(d > 0 ? nothing : m_nu))
end

"""ln of the rate and energy normalisation of every reaction at parameters `p` (see `ln_norm`)."""
rate_norms(itp::RateInterpolants, p) =
    Dict(r.name => (ln_norm(r, p, itp.p_ref), ln_norm(r, p, itp.p_ref; energy=true)) for r in REACTIONS)

"""
    dark_collision(itp, norms, p, T_N, T_nu, n)

(dn/dt for (N1, N2, A', phi), drho/dt of the dark sector) from all reactions,
for MB dark species with number densities `n` (4-tuple) at T_N and the
neutrino at T_nu, parameters `p` and `norms = rate_norms(itp, p)`.
"""
function dark_collision(itp::RateInterpolants, norms, p, T_N, T_nu, n)
    m = p.m1
    lxN, lxnu, lr = log(m / T_N), log(m / T_nu), log(T_N / T_nu)
    # ln(n_i / n_scaled_i) = -gap_i
    mg(f) = -log(n_scaled_safe(field_mass(f, p), field_dof(f), T_N))
    lnz = ntuple(k -> log(n[k]) + mg(DARK_SPECIES[k]), 4)
    side(fs) = sum(lnz[dark_index(f)] for f in fs)
    dn = zeros(4)
    drho = 0.0
    for r in REACTIONS
        lnR, lnQ = norms[r.name]
        if haskey(itp.dark, r.name)
            t = itp.dark[r.name]
            G = exp(t.lnRf(lxN) + lnR + side(r.I)) - exp(t.lnRr(lxN) + lnR + side(r.F))
        elseif haskey(itp.portal, r.name)
            t = itp.portal[r.name]
            G = exp(t.lnRf(lxnu, lr) + lnR + side(r.I)) * t.ratio_Rf(lxnu, lr) -
                exp(t.lnRr(lxnu, lr) + lnR + side(r.F)) * t.ratio_Rr(lxnu, lr)
            drho += exp(t.lnQf(lxnu, lr) + lnQ + side(r.I)) * t.ratio_Qf(lxnu, lr) -
                    exp(t.lnQr(lxnu, lr) + lnQ + side(r.F)) * t.ratio_Qr(lxnu, lr)
        else
            continue      # kinematically closed at these mass ratios
        end
        dN = delta_N(r)
        for k in 1:4
            dn[k] += dN[k] * G
        end
    end
    return dn, drho
end
