"""
Late decays of the dark states and the cosmological bounds they face (post-processing at the relic
roots, like the Lyman-alpha lengths). Units GeV, times in s, lengths in Mpc.

Decays:
- N1 -> nu gamma (X-rays) and N1 -> 3 nu (invisible), the Majorana rates of a sterile neutrino
  with mixing sin^2 2theta_1. N2 does not mix (exactly, for m_+ = m_-) and is stable: N2 and A'
  are odd under the dark parity of eq. (DM_int), so N2 -> N1 + anything is forbidden.
- A' and phi from the 1 -> 2 widths of tree_amplitudes.jl (`width`); N1 -> phi nu likewise when
  m_phi < m_N.
- phi -> nu nu, not among the tree amplitudes: the phi nu nu Yukawa is d m_nu / d v_phi, which the
  mass matrix of the draft (eq. massmatrix, m_+ = m_-) gives as 2 m_nu delta_m / v_phi
  = 2 g theta_1^2 m_N / m_A', independent of m_nu and m_0 (`y_phi_nunu`; checked against an
  exact diagonalisation in test/test_late_decays.jl). It is the only decay of phi for m_phi < m_N.

Cosmology: flat LCDM after e+e- annihilation (photons, N_eff = 3.044 massless nu, matter,
Lambda), valid for t >~ 10 s. Bounds on a component of fraction f of the (pre-decay) cold matter
that decays to radiation with lifetime tau (`ddm_bound`):
- Delta N_eff of the decays before recombination < 0.3 (Planck 2018 + BAO, 95%);
- tau < t_0: decayed fraction between recombination and today, f (e^-t_rec/tau - e^-t_0/tau)
  < 0.038 (CMB, Poulin, Serpico & Lesgourgues, arXiv:1606.02073; for tau > t_0 their f Gamma
  bound, weaker than the next one, applies instead);
- tau > t_0: f / tau < 1 / 246 Gyr (Planck + lensing + BAO, arXiv:2205.05636; f Gamma is what
  is constrained for Gamma <~ H0).
"""

isdefined(@__MODULE__, :hbar) || include(joinpath(@__DIR__, "constants_functions.jl"))
isdefined(@__MODULE__, :width) || include(joinpath(@__DIR__, "thermal_rates.jl"))
import QuadGK

const G_F = 1.1663787e-5
const ALPHA_EM = 1 / 137.035999
const SEC_PER_GYR = 3.15576e16

# --------------------------------------------------------------------------- decay rates [1/s]
theta1(s2) = sin(asin(sqrt(s2)) / 2)

"""N1 -> nu gamma (Majorana): 1.38e-29/s (sin^2 2theta/1e-7)(m/keV)^5 (Pal & Wolfenstein, x2)."""
gamma_nugamma(m, s2) = 9 * ALPHA_EM * G_F^2 * m^5 * s2 / (1024 * pi^4) / hbar

"""N1 -> 3 nu (Majorana, summed over flavours): G_F^2 m^5 sin^2 theta / (96 pi^3)."""
gamma_3nu(m, s2) = G_F^2 * m^5 * theta1(s2)^2 / (96 * pi^3) / hbar

"""Yukawa of phi nu nu in L = -(1/2) y phi nu nu: d m_nu/d v_phi = 2 g theta_1^2 m_N / m_A'."""
y_phi_nunu(p) = 2 * p.g * p.th1^2 * p.m1 / p.mA

"""phi -> nu nu (massless Majorana nu, identical particles): y^2 m_phi / (16 pi)."""
gamma_phi_nunu(p) = y_phi_nunu(p)^2 * p.mphi / (16pi) / hbar

gamma_tree(name, p) = width(name, p) / hbar
gamma_A(p) = (width(:A_N1N2, p) + width(:A_N2nu, p)) / hbar
gamma_phi(p) = (width(:phi_AA, p) + width(:phi_N1N1, p) + width(:phi_N1nu, p)) / hbar + gamma_phi_nunu(p)

# --------------------------------------------------------------------------- cosmology
const H_LITTLE = 0.674
const H0_GEV = H_LITTLE * 2.1331e-42
const OMEGA_GAMMA = 2.4728e-5 / H_LITTLE^2 * (T0 / (2.7255 * 8.6173324e-14))^4
const NU_PER_GAMMA = 7 / 8 * (4 / 11)^(4 / 3)           # rho of one nu species / rho_gamma
const OMEGA_R = OMEGA_GAMMA * (1 + 3.044 * NU_PER_GAMMA)
const OMEGA_C = omega_d0 / H_LITTLE^2
const OMEGA_M = (omega_d0 + omega_b0) / H_LITTLE^2
const OMEGA_L = 1 - OMEGA_M - OMEGA_R
const A_REC = 1 / 1090.0

hubble(a) = H0_GEV * sqrt(OMEGA_R / a^4 + OMEGA_M / a^3 + OMEGA_L)

"""Cosmic time [s] at scale factor a (radiation-dominated start: t = 1/(2H) for a -> 0)."""
function time_of_a(a)
    a0 = min(a, 1e-8)
    t = 1 / (2 * H0_GEV * sqrt(OMEGA_R) / a0^2)
    a > a0 && (t += QuadGK.quadgk(l -> 1 / hubble(exp(l)), log(a0), log(a); rtol=1e-8)[1])
    return t * hbar
end

"""Scale factor at cosmic time t [s] (bisection in log a)."""
function a_of_time(t)
    lo, hi = log(1e-14), 0.5
    for _ in 1:80
        mid = (lo + hi) / 2
        time_of_a(exp(mid)) < t ? (lo = mid) : (hi = mid)
    end
    return exp((lo + hi) / 2)
end

const T_REC = time_of_a(A_REC)
const T_AGE0 = time_of_a(1.0)

"""m_N / T_nu at cosmic time t (the x of the solvers' Lyman-alpha output)."""
x_nu_at_time(m, t) = m * a_of_time(t) / (cbrt(4 / 11) * T0)

"""
Comoving free-streaming length [Mpc] down to redshift `z_end` (50, as in lyman_alpha.jl) of a
particle born at scale factor a_d with momentum/mass u0, redshifting freely.
"""
function free_stream_born(u0, a_d; z_end=50.0)
    a_e = 1 / (1 + z_end)
    a_d < a_e || return 0.0
    f(l) = (a = exp(l); u = u0 * a_d / a; u / sqrt(1 + u^2) / (a * hubble(a)))
    return QuadGK.quadgk(f, log(a_d), log(a_e); rtol=1e-6)[1] / Mpc
end

"""`free_stream_born` averaged over exponential decay times with lifetime tau [s]."""
function free_stream_decay(u0, tau; z_end=50.0)
    g(s) = exp(-s) * free_stream_born(u0, a_of_time(s * tau); z_end=z_end)
    return QuadGK.quadgk(g, 0.0, 1.0, 5.0, 40.0; rtol=1e-4)[1]
end

"""
Delta N_eff after recombination from a component of fraction f of the cold matter that decays
with lifetime tau, putting a fraction eps of its rest energy into radiation (decays after
recombination are not counted here; see `ddm_bound`).
"""
function delta_neff(f, tau, eps)
    tr = T_REC / tau
    # rho_rad,new / rho_nu,1 = eps f (Omega_c / Omega_nu1) * <a at decay>, the decay-weighted mean
    I = QuadGK.quadgk(s -> exp(-s) * a_of_time(s * tau), 0.0, min(tr, 1.0), min(tr, 5.0), min(tr, 40.0); rtol=1e-5)[1]
    return eps * f * OMEGA_C / (OMEGA_GAMMA * NU_PER_GAMMA) * I
end

"""
Status of a decaying component (fraction f, lifetime tau [s], fraction eps of the rest energy to
radiation): :allowed, :neff, :cmb (decayed fraction after recombination) or :late (tau > t_0).
"""
function ddm_bound(f, tau, eps)
    delta_neff(f, tau, eps) > 0.3 && return :neff
    # the CMB and late-time bounds are on the decayed mass, scaled here by the radiated share
    tau < T_AGE0 && eps * f * (exp(-T_REC / tau) - exp(-T_AGE0 / tau)) > 0.038 && return :cmb
    tau > T_AGE0 && eps * f * 246 * SEC_PER_GYR / tau > 1 && return :late
    return :allowed
end
