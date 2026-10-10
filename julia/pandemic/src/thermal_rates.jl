"""
thermal_rates.jl

Thermally averaged rates of every tree-level process of the dark sector
(notes/tree_amplitudes.tex; amplitudes in tree_amplitudes.jl), for comparison
with the Hubble rate along a solved history.

STATISTICS AND NORMALISATION
============================
Maxwell-Boltzmann distributions f = exp(-(E - mu)/T), each species with its own
temperature (the dark sector at T_N, nu at T_nu): Pauli blocking and Bose
enhancement are neglected, which is an O(1) effect at most and only while the
species are relativistic. Every species is described by `Species(m, g, T, gap)`
with gap = (m - mu)/T; e^(-gap) is the fugacity times e^(-m/T), so nothing
under- or overflows at m/T >> 1.

For a + b -> c + d, the rate density (reactions per volume and time; for
identical a = b every ordered pair is counted, i.e. this is n_a^2 <sigma v>,
twice the number of reactions) is
    R = 1/(16 pi^4) ∫ ds sigma(s) F(s) W(s),     F = sqrt(lambda(s, m_a^2, m_b^2))/2,
    W(s) = ∫ dE_a e^(-(E_a - mu_a)/T_a) ∫_{E_b-}^{E_b+} dE_b e^(-(E_b - mu_b)/T_b),
where sigma is summed over all initial and final spins (times 1/2 for identical
final states) and E_b± = [(s - m_a^2 - m_b^2) E_a ± p_a sqrt(lambda)]/(2 m_a^2) are
the limits of E_b at fixed E_a and s (a is the massive one of the pair). The
inner integral is elementary; W is a single quadrature. For a common
temperature this is the Gondolo-Gelmini formula (checked in
test/test_thermal_rates.jl). The rate per particle of species a is
Gamma_a = R/n_a (for identical initial particles this is n_a <sigma v>, the
number of reactions a given particle takes part in).

Inverse decays a + b -> r use sigma = pi sum|M|^2 delta(s - m_r^2)/sqrt(lambda).
Decays use Gamma <m/E> = Gamma_rest K_1/K_2.

RESONANCES
==========
Processes with an s-channel A' or phi pole inside the integration range are
integrated as X |D|^2 + 2 Y Re D + Z (tree_amplitudes.jl), with the narrow
Breit-Wigner either in full (`propagator=:bw`: the on-shell part, i.e. inverse
decay and decay of the resonance, included) or real-intermediate-state
subtracted (`propagator=:ris`: |D|^2 -> ((s-M^2)^2 - M^2 Gamma^2)/((s-M^2)^2 + M^2 Gamma^2)^2,
the off-shell remainder, to be added to the 1 -> 2 rates; the pole is
subtracted analytically, but this is slow at high temperature), or without the
s-channel at all (`propagator=:nonres`: Z only, the t/u-channel part).
"""

import QuadGK
import SpecialFunctions as SF
using Interpolations

isdefined(@__MODULE__, :TreeAmplitudes) || include(joinpath(@__DIR__, "tree_amplitudes.jl"))
using .TreeAmplitudes: TREE_PROCESSES, msq_bw

kallen(a, b, c) = a^2 + b^2 + c^2 - 2a * b - 2a * c - 2b * c

# --------------------------------------------------------------------------- model
"""
    rate_params(m_N, g, sin2_2theta; m_nu=1e-10, m_0=1e3, mphi_over_mN=100.0, mA_over_mN=2.5)

Couplings and masses of tree_amplitudes.jl for a relic point (masses in GeV).
sin(theta_1) from sin^2(2 theta); m_alpha = sqrt(m_nu m_0), sin(theta_0) =
m_alpha/m_0, m_M = sin(theta_1) m_0 m_D/(sqrt2 m_alpha), mu = m_M/m_0,
y_+ = sqrt2 m_M/v_phi with v_phi = m_A'/g (draft, benchmark table).
m_N1 = m_N2 = m_N, as in the solver.
"""
function rate_params(m_N, g, sin2_2theta; m_nu=1e-10, m_0=1e3, mphi_over_mN=100.0, mA_over_mN=2.5)
    th1 = sin(asin(sqrt(sin2_2theta)) / 2)
    m_alpha = sqrt(m_nu * m_0)
    th0 = m_alpha / m_0
    m_M = th1 * m_0 * m_N / (sqrt(2) * m_alpha)
    mA = mA_over_mN * m_N
    y = sqrt(2) * m_M / (mA / g)
    return (g=g, y=y, th0=th0, th1=th1, mu=m_M / m_0, m1=m_N, m2=m_N, mA=mA,
            mphi=mphi_over_mN * m_N, m0=m_0)
end

field_mass(f, p) = f === :N1 ? p.m1 : f === :N2 ? p.m2 : f === :A ? p.mA :
                   f === :phi ? p.mphi : f === :N0 ? p.m0 : f === :nu ? 0.0 : error("field $f")
field_dof(f) = f === :A ? 3 : f === :phi ? 1 : 2

"""A species in the bath: mass, internal dof, temperature, gap = (m - mu)/T (for
m = 0: -mu/T)."""
struct Species
    m::Float64
    g::Int
    T::Float64
    gap::Float64
end

"""n e^(gap): the Maxwell-Boltzmann number density without e^(-gap)."""
n_scaled(a::Species) = a.m == 0 ? a.g * a.T^3 / pi^2 :
                       a.g * a.m^2 * a.T * SF.besselkx(2, a.m / a.T) / (2pi^2)

"""Maxwell-Boltzmann number density."""
n_MB(a::Species) = n_scaled(a) * exp(-a.gap)

# --------------------------------------------------------------------------- widths
function _proc(name)
    i = findfirst(q -> q.name === name, TREE_PROCESSES)
    i === nothing && error("no process $name")
    return TREE_PROCESSES[i]
end

"""Rest-frame width of a 1 -> 2 process (averaged over the initial dof)."""
function width(name, p)
    q = _proc(name)
    M = field_mass(q.initial[1], p)
    m3, m4 = field_mass(q.final[1], p), field_mass(q.final[2], p)
    M > m3 + m4 || return 0.0
    sym = q.final[1] === q.final[2] ? 0.5 : 1.0
    return sym * sqrt(kallen(M^2, m3^2, m4^2)) * q.msq(p) / (16pi * M^3 * field_dof(q.initial[1]))
end

"""Total width of A' or phi from the 1 -> 2 processes of tree_amplitudes.jl."""
total_width(f, p) = sum((width(q.name, p) for q in TREE_PROCESSES
                         if q.id > 100 && q.initial == (f,)); init=0.0)

# --------------------------------------------------------------------------- cross sections
"""
    SigmaTable(q, p; T_max)

sigma(s) of the 2 -> 2 process `q` (an entry of TREE_PROCESSES, or a crossed /
reversed process built by `crossed_process`), summed over all spins, with 1/2
for identical final states, split as (Xt, Yt, Zt) = ∫ dt (X, Y, Z)/(16 pi lambda)
(non-resonant: only Zt). Tabulated in log(s - s_th) up to (sqrt(s_th) + 300 T_max)^2.
"""
struct SigmaTable{I}
    s_th::Float64
    s_max::Float64
    M::Float64       # s-channel resonance mass (0: none)
    Gamma::Float64
    pole_in_range::Bool
    interp::I
end

function _t_range(s, m1, m2, m3, m4)
    rs = sqrt(s)
    E1 = (s + m1^2 - m2^2) / (2rs)
    E3 = (s + m3^2 - m4^2) / (2rs)
    p1 = sqrt(max(kallen(s, m1^2, m2^2), 0.0)) / (2rs)
    p3 = sqrt(max(kallen(s, m3^2, m4^2), 0.0)) / (2rs)
    base = m1^2 + m3^2 - 2 * E1 * E3   # not 2E1: Julia reads that as the literal 20.0
    return base - 2p1 * p3, base + 2p1 * p3
end

function _t_breakpoints(tlo, thi, p)
    L = thi - tlo
    k = min(p.m1, p.mA)^2 / 10
    ks = Float64[]
    while k < L / 2
        push!(ks, k)
        k *= 10
    end
    return sort!(unique!([tlo; tlo .+ ks; thi .- ks; thi]))
end

function sigma_parts(q, s, p)
    m1, m2 = field_mass(q.initial[1], p), field_mass(q.initial[2], p)
    m3, m4 = field_mass(q.final[1], p), field_mass(q.final[2], p)
    lam = kallen(s, m1^2, m2^2)
    (lam > 0 && s > (m3 + m4)^2) || return (0.0, 0.0, 0.0)
    tlo, thi = _t_range(s, m1, m2, m3, m4)
    sym = q.final[1] === q.final[2] ? 0.5 : 1.0
    c = sym / (16pi * lam)
    # t- and u-channel peaks sit at the two ends of [tlo, thi], with widths ~ m^2:
    # geometric breakpoints towards both ends, or QuadGK misses one at s >> m^2
    pts = _t_breakpoints(tlo, thi, p)
    # Far above threshold the amplitudes with external A' cancel like
    # (s/m_A^2)^2 between diagrams (gauge cancellation), beyond double
    # precision: evaluate them in 256-bit BigFloat there.
    hp = s > 1e3 * max((m1 + m2)^2, (m3 + m4)^2)
    pb = hp ? map(x -> BigFloat(x; precision=256), p) : p
    ev(f) = hp ? (t -> Float64.(f(BigFloat(s; precision=256), BigFloat(t; precision=256), pb))) :
                 (t -> f(s, t, p))
    kw = (rtol=1e-6, maxevals=10^5)
    if q.xyz === nothing
        g = ev(q.msq)
        return (0.0, 0.0, c * QuadGK.quadgk(g, pts...; kw...)[1])
    end
    g = ev(q.xyz)
    v = QuadGK.quadgk(t -> collect(g(t)), pts...; kw...)[1]
    return (c * v[1], c * v[2], c * v[3])
end

function SigmaTable(q, p; T_max, n=500)
    m1, m2 = field_mass(q.initial[1], p), field_mass(q.initial[2], p)
    m3, m4 = field_mass(q.final[1], p), field_mass(q.final[2], p)
    s_th = max((m1 + m2)^2, (m3 + m4)^2)
    s_max = (sqrt(s_th) + 300 * T_max)^2
    us = range(log(1e-14 * s_th), log(s_max - s_th), length=n)
    vals = [sigma_parts(q, s_th + exp(u), p) for u in us]
    itp = [linear_interpolation(us, [v[k] for v in vals]; extrapolation_bc=Flat()) for k in 1:3]
    res = q.resonance === nothing ? nothing : q.resonance
    M = res === nothing ? 0.0 : field_mass(res, p)
    G = res === nothing ? 0.0 : total_width(res, p)
    # breakpoints are only needed if the pole lies inside the integration range;
    # outside it the Breit-Wigner equals the plain propagator
    return SigmaTable(s_th, s_max, M, G, M^2 > s_th, itp)
end

function sigma_eval(tab::SigmaTable, s, propagator)
    s <= tab.s_th && return 0.0
    u = log(s - tab.s_th)
    X, Y, Z = tab.interp[1](u), tab.interp[2](u), tab.interp[3](u)
    (tab.M == 0 || propagator === :nonres) && return Z
    d = s - tab.M^2
    MG2 = (tab.M * tab.Gamma)^2
    D2 = propagator === :bw ? 1 / (d^2 + MG2) : (d^2 - MG2) / (d^2 + MG2)^2
    return X * D2 + 2Y * d / (d^2 + MG2) + Z
end

"""
    crossed_process(name; initial, final, msq)

A process not in TREE_PROCESSES (a crossing or the reverse of one), for
SigmaTable. `msq(s, t, p)` in the kinematics of the new process.
"""
crossed_process(name; initial, final, msq) =
    (id=0, name=name, initial=initial, final=final, resonance=nothing, msq=msq, xyz=nothing)

"""The reverse c + d -> a + b of a 2 -> 2 process (same sum|M|^2 as a function of s and
t = (p_a - p_c)^2 = (p_c - p_a)^2)."""
reversed_process(q) = (id=-q.id, name=Symbol(q.name, :_rev), initial=q.final, final=q.initial,
                       resonance=q.resonance, msq=q.msq, xyz=q.xyz)

# --------------------------------------------------------------------------- thermal integrals
"""
W(s) e^(gap_a + gap_b), with a the massive one: ∫ dE_a e^(-(E_a - m_a)/T_a)
∫ dE_b e^(-(E_b - m_b)/T_b) over the kinematic range at fixed s.
"""
function W_scaled(s, a::Species, b::Species)
    if a.m == 0
        a, b = b, a
    end
    a.m > 0 || error("two massless initial particles")
    lam = kallen(s, a.m^2, b.m^2)
    lam > 0 || return 0.0
    A = s - a.m^2 - b.m^2
    B = sqrt(lam)
    inner(w) = begin
        Ea = a.m + a.T * w
        pa = sqrt(max(Ea^2 - a.m^2, 0.0))
        # E_b- from the product of the two roots, E_b+ E_b- = (A^2 + 4 m_b^2 p_a^2)/(4 m_a^2):
        # (A E_a - p_a B)/(2 m_a^2) cancels catastrophically for E_a >> m_a
        Ebm = (A^2 + 4 * b.m^2 * pa^2) / (2 * (A * Ea + pa * B))
        lo = max(Ebm - b.m, 0.0)
        width = pa * B / a.m^2      # E_b+ - E_b-
        # e^(-lo/T_b) - e^(-hi/T_b), without cancellation
        a.T * exp(-w) * b.T * exp(-lo / b.T) * -expm1(-width / b.T)
    end
    return QuadGK.quadgk(inner, 0.0, 1.0, 5.0, 20.0, 60.0, Inf; rtol=1e-6)[1]
end

"""
    rate_density_scaled(tab, a, b; propagator=:ris, W=nothing)

R e^(gap_a + gap_b) for a + b -> c + d (see the header). A function `W(s)`
replaces `W_scaled(s, a, b)` when given: any weights that depend only on E_a
and E_b reduce to the same single integral over s (rate_tables.jl uses this
for a Fermi-Dirac neutrino).
"""
function rate_density_scaled(tab::SigmaTable, a::Species, b::Species; propagator::Symbol=:ris, rtol=1e-4,
                             W=nothing)
    # function barrier: a Union-typed W sends type inference of the nested
    # quadrature closures into a near-endless loop
    Ws = W === nothing ? (s -> W_scaled(s, a, b)) : W
    return _rate_density_scaled(tab, a, b, Ws, propagator, rtol)
end

Base.@constprop :none function _rate_density_scaled(tab::SigmaTable, a::Species, b::Species, Ws::F, propagator::Symbol, rtol) where {F}
    s_lo = max(tab.s_th, (a.m + b.m)^2)
    rs_lo = sqrt(s_lo)
    Tm = max(a.T, b.T)
    flux(s) = sqrt(max(kallen(s, a.m^2, b.m^2), 0.0)) / 2
    # integrand per ds, and per dq with q = sqrt(s) - sqrt(s_lo) (ds = 2 sqrt(s) dq)
    G(s) = sigma_eval(tab, s, propagator) * flux(s) * Ws(s)
    Gq(q) = (rs = rs_lo + q; 2rs * G(rs^2))
    q_max = sqrt(tab.s_max) - rs_lo
    function q_integral(q_lo, q_hi)
        q_hi > q_lo || return 0.0
        pts = Float64[q_lo; filter(x -> q_lo < x < q_hi, [Tm, 5Tm, 20Tm, 80Tm, 250Tm]); q_hi]
        return QuadGK.quadgk(Gq, pts...; rtol=rtol)[1]
    end
    s0 = tab.M^2
    if propagator === :nonres || !(tab.pole_in_range && s0 > s_lo)
        return q_integral(0.0, q_max) / (16pi^4)
    end
    aw = tab.M * tab.Gamma
    if propagator === :bw
        # positive Lorentzian: breakpoints on the width scale suffice
        qr, dq = tab.M - rs_lo, tab.Gamma / 2
        pts = sort!(unique!(filter(x -> 0 <= x <= q_max,
              [0.0, Tm, 5Tm, 20Tm, 80Tm, 250Tm, q_max, (qr .+ dq .* (-1e5, -1e3, -30.0, -3.0, -1.0, 0.0, 1.0, 3.0, 30.0, 1e3, 1e5))...])))
        return QuadGK.quadgk(Gq, pts...; rtol=rtol)[1] / (16pi^4)
    end
    # RIS: subtract the singular part analytically in a window |s - s0| < L.
    # With h = X F W (the coefficient of |D|^2) and the even kernel R(d) =
    # (d^2 - aw^2)/(d^2 + aw^2)^2, pair s0 ± d: G(s0+d) + G(s0-d) - 2 h(s0) R(d) is
    # smooth (the Re D part is odd and cancels to O(d)), and ∫_0^L R = -L/(L^2 + aw^2).
    L = min(s0 - s_lo, 0.5 * s0)
    h0 = tab.interp[1](log(s0 - tab.s_th)) * flux(s0) * Ws(s0)
    R(d) = (d^2 - aw^2) / (d^2 + aw^2)^2
    sym(d) = G(s0 + d) + G(s0 - d) - 2h0 * R(d)
    dpts = Float64[0.0; filter(x -> x < L, aw .* 10.0 .^ (0:2:30)); L]
    J = QuadGK.quadgk(sym, dpts...; rtol=rtol, atol=0.0)[1] + 2h0 * (-L / (L^2 + aw^2))
    below = q_integral(0.0, sqrt(s0 - L) - rs_lo)
    above = q_integral(sqrt(s0 + L) - rs_lo, q_max)
    return (below + J + above) / (16pi^4)
end

"""
    rate_2to2(tab, a, b; propagator=:ris)

(Gamma_a, Gamma_b, R): rates per particle of a and of b, and the rate density R
(= n_a n_b <sigma v>; for a = b, n_a^2 <sigma v>).
"""
function rate_2to2(tab::SigmaTable, a::Species, b::Species; propagator::Symbol=:ris)
    Rs = rate_density_scaled(tab, a, b; propagator=propagator)
    return (Rs * exp(-b.gap) / n_scaled(a), Rs * exp(-a.gap) / n_scaled(b), Rs * exp(-a.gap - b.gap))
end

"""
    rate_inverse_decay(name, p, a, b)

Inverse decay a + b -> r of the 1 -> 2 process `name` (r -> a b): (Gamma_a, Gamma_b, R).
"""
function rate_inverse_decay(name, p, a::Species, b::Species)
    q = _proc(name)
    M = field_mass(q.initial[1], p)
    M > a.m + b.m || return (0.0, 0.0, 0.0)
    Rs = (pi / 2) * q.msq(p) * W_scaled(M^2, a, b) / (16pi^4)
    return (Rs * exp(-b.gap) / n_scaled(a), Rs * exp(-a.gap) / n_scaled(b), Rs * exp(-a.gap - b.gap))
end

"""Thermally averaged decay rate Gamma K_1/K_2 of the 1 -> 2 process `name` at temperature T."""
function rate_decay(name, p, T)
    q = _proc(name)
    M = field_mass(q.initial[1], p)
    return width(name, p) * SF.besselkx(1, M / T) / SF.besselkx(2, M / T)
end
