"""
kinetic_decoupling.jl

Kinetic decoupling of the dark sector and the Lyman-alpha lengths that
depend on it (free-streaming length after T_kd, sound horizon before it),
with the physics of the old Python code (code/sterile_caller.py, `C_therm_kd`
and the lambda_fs / r_sound computation), see src/lyman_alpha.jl.

RATE
====
As in the old code (non-relativistic branch, `C_dd_dd_gon_gel`), the
momentum-exchange rate density is

    C_kd = 2 * (1/4) * T/(32 pi^4) ∫ ds sigma(s) (s - 4 m^2) sqrt(s) K_1(sqrt(s)/T) e^(2 xi),

with sigma = ∫ dt sum|M|^2 / (64 pi s p_cm^2) summed over initial and final
dof (no symmetry factor), and kinetic decoupling at the first time after
T_nu < 0.1 m_A' (where the old search starts) at which C_kd / (H n_N1) < 1
(or < 3 for the "3" variant). The Maxwell-Boltzmann form is used throughout
(the old code switched to quantum statistics for m/T - xi < 4, and to A' <->
N1 N2 for T_N > m_A', neither of which occurs after T_nu < 0.1 m_A').

AMPLITUDE
=========
Default (`amp=:fixed`): N1 N1 -> N2 N2 for m_N1 = m_N2, which for degenerate
masses acts as elastic scattering, with the corrected amplitude, eq. (A6fixed)
of notes/amplitudes_majorana.tex (in pandemic_neutrinos/, outside this repo):
4 x the t- and u-channel terms of the draft's (A6) and the opposite sign on
its interference term. At threshold it reproduces the draft's
sigma_NR = g^4 m^2 / (8 pi m_A^4) (eq. NNconversionNR, with the spin average
1/4 and the identical-particle factor 1/2). `amp=:old` is a port of the old
code's `vector_mediator.M2_gen` with all four masses equal; it is the same
amplitude, agreeing with :fixed to rounding. `amp=:A6` is (A6) as printed in
the draft, kept for comparison only: the cross section is 3/4 of :fixed at
threshold, dips to ~0.11 of it near s = 20 m^2 and tends to 1/4 at large s.

CHANNELS
========
`channels=:all` (default) adds N1 N2 -> N1 N2 (s-channel A' and t/u exchange;
notes/tree_amplitudes.tex, process 2) to the momentum-exchange rate, with
n_N1 = n_N2 as in the solver:
    C_kd = T e^(2 xi)/(32 pi^4) ∫ ds [sigma_11/2 + sigma_12] (s - 4m^2) sqrt(s) K_1,
i.e. Gamma_kd = C_kd/n_N1 = n_N1 <sigma v>_11->22 + n_N2 <sigma v>_12->12, with
sigma_ij summed over all dof (the 1/2 is the identical-particle factor of
11 -> 22). The A' propagator of the s-channel is a Breit-Wigner with the
A' -> N1 N2 width: its pole at s = m_A'^2 lies inside the thermal integral, and
the on-shell part (inverse decay N1 N2 -> A' and decay back) is genuine
momentum exchange, since A' is not a separate species in this
post-processing. `channels=:conversion` is N1 N1 -> N2 N2 alone, the rate
used before 2026-10-03.

The old code's `sigma_gen_new` returned 1e3 times the cross section (the 1e-3
rescaling of its integrand had been commented out), so its C_kd was 1e3 times
too large; `sigma_scale=1e3` together with `amp=:old` reproduces that.
"""

include(joinpath(@__DIR__, "lyman_alpha.jl"))
include(joinpath(@__DIR__, "tree_amplitudes.jl"))
using .TreeAmplitudes: msq_A_N1N2, xyz_N1N2_N1N2, msq_bw

# Lyman-alpha limit on the sound horizon in Mpc (code/sterile_res/plotter.py).
const LYA_R_S_MAX_MPC = 0.34

"""
    sq_amp_NN_A6(s, t, m, m_A, g)

sum |M|^2 of N1 N1 -> N2 N2 for m_N1 = m_N2 = m, eq. (A6) of the draft as
printed, which is wrong (see the header); for comparison only.
"""
function sq_amp_NN_A6(s, t, m, m_A, g)
    m2, mA2 = m * m, m_A * m_A
    u = 4 * m2 - s - t
    return 4 * g^4 * (
        ((s - 2 * m2)^2 + s * t + t^2 / 2) / (t - mA2)^2
        + ((s - 2 * m2)^2 + s * u + u^2 / 2) / (u - mA2)^2
        - ((s - 4 * m2)^2 - 4 * m2^2) / ((t - mA2) * (u - mA2))
    )
end

"""
    sq_amp_NN_fixed(s, t, m, m_A, g)

sum |M|^2 of N1 N1 -> N2 N2 for m_N1 = m_N2 = m, the corrected (A6): eq.
(A6fixed) of notes/amplitudes_majorana.tex.
"""
function sq_amp_NN_fixed(s, t, m, m_A, g)
    m2, mA2 = m * m, m_A * m_A
    u = 4 * m2 - s - t
    return 16 * g^4 * (
        ((s - 2 * m2)^2 + s * t + t^2 / 2) / (t - mA2)^2
        + ((s - 2 * m2)^2 + s * u + u^2 / 2) / (u - mA2)^2
        + ((s - 4 * m2)^2 - 4 * m2^2) / ((t - mA2) * (u - mA2))
    )
end

"""
    sq_amp_NN_old(s, t, m, m_A, g)

sum |M|^2 of the old code's `vector_mediator.M2_gen(s, t, m, m, m, m, g^4,
m_A^2, 0)` (t- and u-channel A' exchange and their interference), evaluated
in units of m (|M|^2 is dimensionless), where the expression is better
conditioned. Converted mechanically from the Python source.
"""
function sq_amp_NN_old(s, t, m, m_A, g)
    return g^4 * _M2_gen_old(s / m^2, t / m^2, 1.0, 1.0, 1.0, 1.0, (m_A / m)^2)
end

function _M2_gen_old(s, t, m1, m2, m3, m4, m_X2)
    m12 = m1*m1; m13 = m1*m12; m14 = m12*m12; m15 = m12*m13
    m22 = m2*m2; m23 = m2*m22; m24 = m22*m22; m25 = m22*m23
    m32 = m3*m3; m33 = m3*m32; m34 = m32*m32; m35 = m32*m33
    m42 = m4*m4; m43 = m4*m42; m44 = m42*m42; m45 = m42*m43
    m_X = sqrt(m_X2)
    m_X4 = m_X2*m_X2
    u = m12 + m22 + m32 + m42 - s - t
    s2 = s*s
    t2 = t*t
    t_prop = 1. / ((t - m_X2)*(t - m_X2))
    u_prop = 1. / ((u - m_X2)*(u - m_X2))
    tt = 1/m_X4*4*((m2-m4)*(m23+m4*m22-(m42+t)*m2+m4*(-m42-4*m_X2+t))*m14+((m2-m4)^2*t2-(2*m_X4+4*m4*(m4-m2)*m_X2+(m2-m4)^2*((m2+m4)^2-2*m32))*t+2*(-m32*m24+(2*(m4-m_X)*(m4+m_X)*m32+m_X2*(-2*m42+m_X2+2*s))*m22-2*m4*m_X2*(m_X2-2*m32)*m2-m32*(m44+2*m_X2*m42-2*m_X4)+m_X2*(m42*(2*m42+m_X2)-2*(m42+m_X2)*s)))*m12+2*m3*(-2*(m22-4*m4*m2+m42)*m_X4-2*(m22-m42)^2*m_X2-(m2-m4)^2*t2+(2*m_X4+2*(m2-m4)^2*m_X2+(m22-m42)^2)*t)*m1+m34*m44+2*m32*m42*m_X4+4*m_X4*s2+(2*m_X4+m32*m42)*t2-4*m32*m_X4*s-4*m42*m_X4*s+4*m32*m42*m_X2*s+m24*m32*(m32+4*m_X2-t)-(m32+m42)*(2*m_X4+m32*m42)*t+4*m_X4*s*t+2*m2*m4*(m32-t)*(m32*(t-2*m_X2)-2*m_X4)-m22*((2*m42-4*m_X2+t)*m34+(-2*m_X4+4*(s+t)*m_X2-t2+m42*(4*m_X2-2*t))*m32+2*m_X4*(-2*m42+2*s+t))) * t_prop
    uu = -1/m_X4*4*(2*(m2-m3)^2*m4*m15-(m2-m3)^2*(m22+m32+4*m42-s-t)*m14+2*m4*(m24-4*m3*m23+2*(3*m32+m42-m_X2-s-t)*m22+4*m3*(-m32-m42+m_X2+s+t)*m2+m34-2*m_X4+2*m32*(m42-m_X2-s-t))*m13+(2*m3*m25+(-4*m32-2*m42+s+t)*m24+4*m3*(m32+3*m42-m_X2-s-t)*m23-(4*m34+(20*m42-8*m_X2-6*(s+t))*m32+4*m44+2*m_X4+s2+t2+4*m_X2*s+2*s*t-2*m42*(4*m_X2+3*(s+t)))*m22+2*m3*(m34+(6*m42-2*(m_X2+s+t))*m32+4*m44+(s+t)*(2*m_X2+s+t)-2*m42*(4*m_X2+3*(s+t)))*m2+m34*(-2*m42+s+t)+2*m_X4*(-2*m42+s+t)-m32*(4*m44-2*(4*m_X2+3*(s+t))*m42+2*m_X4+(s+t)^2+4*m_X2*t))*m12+2*m4*(-2*m3*m25+(4*m32+m42-s-t)*m24+4*m3*(-m32-m42+m_X2+s+t)*m23+(4*m34+(6*m42-8*m_X2-6*(s+t))*m32+(m42-s-t)*(m42-2*m_X2-s-t))*m22-2*m3*(m34+2*(m42-m_X2-s-t)*m32+m44+4*m_X4+s2+t2+2*m_X2*s+2*(m_X2+s)*t-2*m42*(m_X2+s+t))*m2+(m34+(m42-2*m_X2-s-t)*m32-2*m_X4)*(m42-s-t))*m1-m34*m44-2*m32*m42*m_X4+2*m25*m3*m42-2*m_X4*s2-m32*m42*s2-2*m_X4*t2-m32*m42*t2+m32*m44*s+2*m32*m_X4*s+2*m42*m_X4*s+m34*m42*s-4*m32*m42*m_X2*s+m32*m44*t+2*m32*m_X4*t+2*m42*m_X4*t+m34*m42*t-2*m32*m42*s*t+m24*m42*(-4*m32-m42+s+t)+2*m23*m3*(m44+2*m32*m42-2*(m_X2+s+t)*m42-2*m_X4)+2*m2*m3*(m32-s-t)*(m44+m32*m42-(2*m_X2+s+t)*m42-2*m_X4)+m22*(-4*m42*m34+(-2*m44+(8*m_X2+6*(s+t))*m42-4*m_X4)*m32+m44*(s+t)+2*m_X4*(s+t)-m42*(2*m_X4+4*t*m_X2+(s+t)^2))) * u_prop
    tu = 1/m_X4*4*((m2-m3)*(m2-m4)*m4*m15+(-m24+m3*m23+m43*m2+2*m4*m_X2*m2-m3*m43+2*m42*m_X2-4*m3*m4*m_X2+(m2-m3)*(m2-m4)*t)*m14+(m4*m24+((m3-m4)^2-4*m_X2-s)*m23-(m33+m4*m32+(m42-2*m_X2-s)*m3+m43+2*m4*t)*m22+(2*m4*m33+(m42-2*m_X2)*m32+2*m4*(m42-2*m_X2+t)*m3-(m42+2*m_X2)*(m42-s-2*t))*m2-m33*(m42-2*m_X2)-m32*(m43-2*m4*m_X2)+2*m4*m_X2*(m42+2*m_X2-t)+m3*(m44+(4*m_X2-s-2*t)*m42-2*m_X2*(s+t)))*m13+(m3*m25+t*m24-(m33+m4*m32+m42*m3+2*t*m3+m43-2*m4*m_X2-m4*s)*m23+(m4*m33+(2*m42-6*m_X2+t)*m32+m4*(m42-12*m_X2-s+2*t)*m3-t2+4*m_X2*s-s*t+m42*(t-6*m_X2))*m22+(-((m42-2*m_X2)*m33)-m4*(m42+2*(t-7*m_X2))*m32+(2*(5*m_X2-t)*m42-2*m_X2*s+t*(s+t))*m3+m4*(m44+(4*m_X2-s-2*t)*m42-4*m_X4+t*(s+t)-2*m_X2*(3*s+2*t)))*m2+m33*(m43-2*m4*m_X2)+2*m_X2*((m42+2*m_X2)*(m42-s)-m42*t)+m32*((t-6*m_X2)*m42+2*m_X2*(2*m_X2+t))-m3*m4*(m44+(2*m_X2-s-2*t)*m42-4*m_X4+t*(s+t)-2*m_X2*(3*s+2*t)))*m12-(m3*(m3+m4)*m25+((m3+m4)*t-m3*(m32+2*m_X2))*m24+(m34-2*m4*m33-(m42-2*m_X2+s+2*t)*m32-2*m4*(m42-2*m_X2+t)*m3+2*m_X2*(m42-s-2*t))*m23-(m35-(m42-4*m_X2+s+2*t)*m33-m4*(m42+2*(t-5*m_X2))*m32+(-4*m_X4-2*(3*s+2*t)*m_X2+2*m42*(7*m_X2-t)+t*(s+t))*m3+m4*(2*(m42-s)*m_X2+t2+s*t))*m22+(m4*m35+4*m_X2*m34+2*m43*m33+4*m4*m_X2*m33-4*m_X4*m32+12*m42*m_X2*m32+m42*s*m32-6*m_X2*s*m32+m45*m3-16*m4*m_X4*m3+4*m43*m_X2*m3-4*m4*m_X2*s*m3-4*m42*m_X4+4*m44*m_X2+2*m_X2*s2+((m3+m4)^2+4*m_X2)*t2+4*m_X4*s-6*m42*m_X2*s-((m3+m4)^2+4*m_X2)*(m32+m42-s)*t)*m2+m34*m4*(t-m42)+m33*(2*m_X2*(s+t)-m42*(2*m_X2+s))+2*m4*m_X2*((t-2*m_X2)*m42-t2+2*m_X2*(s+t))-m32*m4*(m44-2*t*m42-2*m_X2*s+t*(s+t))+m3*((t-2*m_X2)*m44+(4*m_X4+(6*s+4*t)*m_X2-t*(s+t))*m42-2*m_X2*(2*t*m_X2+(s+t)^2)))*m1-m34*m44-4*m33*m43*m_X2+4*m_X4*s2-2*m3*m4*m_X2*s2-m3*m4*(4*m_X2+m3*m4)*t2+m25*m32*m4-4*m32*m_X4*s-4*m42*m_X4*s-4*m3*m4*m_X4*s-m33*m43*s+2*m3*m43*m_X2*s+4*m32*m42*m_X2*s+2*m33*m4*m_X2*s+m3*m4*(4*m_X2+m3*m4)*(m32+m42-s)*t+m24*m3*(-m4*m32+2*m_X2*m3+m4*(t-4*m_X2))+m23*(m4*m34-(m42-2*m_X2)*m33-m4*(m42-4*m_X2+s+2*t)*m32+2*m_X2*(m42+2*m_X2-t)*m3+2*m4*m_X2*(m42-s-t))+m22*(-m4*m35+2*m_X2*m34+m4*(m42-2*m_X2+s+2*t)*m33+((t-6*m_X2)*m42+2*m_X2*(2*m_X2-s-t))*m32-m4*(-4*m_X4+2*m42*m_X2-2*(3*s+2*t)*m_X2+t*(s+t))*m3+4*m_X4*(m42-s)+2*m42*m_X2*t)+m2*(m42*m35+m4*(2*m_X2-t)*m34+(m44+4*m_X4-2*(m42+m_X2)*t)*m33+m4*(-4*m_X4-2*(3*s+2*t)*m_X2+m42*(2*m_X2+s)+t*(s+t))*m32+(-t*m44+(t*(s+t)-2*m_X2*s)*m42+2*m_X2*t2-4*m_X4*(s+t))*m3+2*m4*m_X2*(-((s+t)*m42)+(s+t)^2+2*m_X2*t))) * t_prop*(t-m_X2)*u_prop*(u-m_X2)
    return tt + uu + tu
end

"""
    sigma_NN(s, m, m_A, g; amp=:fixed)

∫ dt sum|M|^2 / (64 pi s p_cm^2) for N N -> N N with all masses m: the cross
section summed over initial and final dof, without a symmetry factor (as
`vector_mediator.sigma_gen_new`, without its factor 1e3).
"""
function sigma_NN(s, m, m_A, g; amp::Symbol=:fixed)
    p2 = s / 4 - m * m
    p2 > 0 || return 0.0
    f = amp === :fixed ? sq_amp_NN_fixed : amp === :A6 ? sq_amp_NN_A6 :
        amp === :old ? sq_amp_NN_old : error("unknown amplitude $amp")
    val = QuadGK.quadgk(t -> f(s, t, m, m_A, g), -4 * p2, 0.0; rtol=1e-8)[1]
    return val / (64 * pi * s * p2)
end

_kd_params(m, m_A, g) = (g=g, y=0.0, th0=0.0, th1=0.0, mu=0.0, m1=m, m2=m, mA=m_A, mphi=1.0, m0=1.0)

"""
    width_A(m, m_A, g)

Width of A' -> N1 N2 for m_N1 = m_N2 = m (A' -> N2 nu is O(sin^2 theta_1) and
dropped): sqrt(lambda) sum|M|^2 / (48 pi m_A^3), averaged over the 3 polarisations.
"""
width_A(m, m_A, g) = sqrt(m_A^2 * (m_A^2 - 4m^2)) * msq_A_N1N2(_kd_params(m, m_A, g)) / (48pi * m_A^3)

"""
    sigma_NN12(s, m, m_A, g, Gamma_A)

∫ dt sum|M|^2 / (64 pi s p_cm^2) for N1 N2 -> N1 N2 with m_N1 = m_N2 = m, summed
over all dof, Breit-Wigner A' of width Gamma_A in the s-channel.
"""
function sigma_NN12(s, m, m_A, g, Gamma_A)
    p2 = s / 4 - m * m
    p2 > 0 || return 0.0
    p = _kd_params(m, m_A, g)
    val = QuadGK.quadgk(t -> msq_bw(xyz_N1N2_N1N2(s, t, p), s, m_A, Gamma_A), -4 * p2, 0.0; rtol=1e-8)[1]
    return val / (64 * pi * s * p2)
end

"""
e^x K_1(x). Asymptotic series for x > 1e4 (relative error < 1e-13), where
the Amos routine behind `besselkx` refuses the argument.
"""
_besselk1x(x) = x < 1e4 ? SF.besselkx(1, x) : sqrt(pi / (2x)) * (1 + 3 / (8x) - 15 / (128x^2) + 315 / (3072x^3))

"""
    C_kd(T, gap, m, m_A, g; amp=:fixed, sigma_scale=1.0, channels=:all)

Momentum-exchange rate density at dark temperature T and gap = (m - mu)/T
(see CHANNELS in the header). Integrated in w = (sqrt(s) - 2m)/T, with
e^(2 xi) K_1(sqrt(s)/T) = e^(-w - 2 gap) besselkx(1, sqrt(s)/T); for
`channels=:all` with breakpoints around the A' pole, w = (m_A - 2m)/T, on the
scale of its width.
"""
function C_kd(T, gap, m, m_A, g; amp::Symbol=:fixed, sigma_scale=1.0, channels::Symbol=:all)
    channels in (:all, :conversion) || error("unknown channels $channels")
    Gamma_A = channels === :all ? width_A(m, m_A, g) : 0.0
    sig(s) = sigma_NN(s, m, m_A, g; amp=amp) / 2 +
             (channels === :all ? sigma_NN12(s, m, m_A, g, Gamma_A) : 0.0)
    integrand(w) = begin
        rs = 2 * m + w * T
        s = rs * rs
        # ds = 2 sqrt(s) T dw
        2 * rs * T * sig(s) * (s - 4 * m * m) * rs * _besselk1x(rs / T) * exp(-w)
    end
    pts = [0.0, 1.0, 5.0, 20.0, 80.0]
    if channels === :all
        w_res, dw = (m_A - 2m) / T, Gamma_A / (2T)
        append!(pts, filter(>(0), w_res .+ dw .* (-1e4, -1e2, -10.0, -1.0, 0.0, 1.0, 10.0, 1e2, 1e4)))
        sort!(unique!(pts))
    end
    val = QuadGK.quadgk(integrand, pts...; rtol=1e-6)[1]
    return sigma_scale * T / (32 * pi^4) * exp(-2 * gap) * val
end

"""
    DarkHistory(pan, sol)

(T_N, gap_N) of N as a function of z = log(m_N/T_nu): from the solution up to
its end (x = 100), then that of a non-relativistic gas with conserved number
in kinetic equilibrium, T_N ∝ a^-2 at fixed gap (corrections O(T_N/m_N), which
is < 1e-3 there; the old code integrated them to O((T/m)^6)).
"""
struct DarkHistory{P}
    pan::P
    sol::Any    # ODESolution kept untyped, see `free_streaming_length`
    z_f::Float64
    T_f::Float64
    gap_f::Float64
    a_f::Float64
end

function DarkHistory(pan, sol)
    T_f, gap_f = N_temp_gap(pan, sol.u[end])
    return DarkHistory{typeof(pan)}(pan, sol, sol.t[end], T_f, gap_f, scale_factor_z(pan, sol.t[end]))
end

function temp_gap(h::DarkHistory, z)
    z <= h.z_f && return N_temp_gap(h.pan, h.sol(z)::Vector{Float64})
    return h.T_f * (h.a_f / scale_factor_z(h.pan, z))^2, h.gap_f
end

"""Gamma_kd / (crit H) with Gamma_kd = C_kd / n_N1, at z."""
function kd_ratio(h::DarkHistory, z; crit=1.0, amp::Symbol=:fixed, sigma_scale=1.0, channels::Symbol=:all)
    pan = h.pan
    T, gap = temp_gap(h, z)
    C = C_kd(T, gap, pan.N1.m, pan.A.m, pan.mp.y; amp=amp, sigma_scale=sigma_scale, channels=channels)
    return C / (crit * pan.H_interp_z(z) * number_density(pan.N1, T, 0.0; gap=gap))
end

"""
    find_kd(h, z_lo, z_hi; crit=1.0, dz=0.05, kw...)

First z in [z_lo, z_hi] at which Gamma_kd < crit H (as the forward search of
the old code), refined by bisection to 1e-4 in z. Returns z_lo if the dark
sector is already decoupled there and z_hi if it never decouples.
"""
function find_kd(h::DarkHistory, z_lo, z_hi; crit=1.0, dz=0.05, kw...)
    r(z) = kd_ratio(h, z; crit=crit, kw...)
    r(z_lo) < 1 && return z_lo
    z_prev = z_lo
    while z_prev < z_hi
        z = min(z_prev + dz, z_hi)
        if r(z) < 1
            a, b = z_prev, z
            while b - a > 1e-4
                c = (a + b) / 2
                r(c) < 1 ? (b = c) : (a = c)
            end
            return b
        end
        z_prev = z
    end
    return z_hi
end

"""Pressure of particle `p` at (temp, gap); n T in the Maxwell-Boltzmann limit."""
function pressure(p::Particle, temp, gap)
    if p.k == 0 || gap >= m_T_r_MB
        return number_density(p, temp, 0.0; gap=gap) * temp
    end
    return thermal_moment(p, temp, gap, (E, mom) -> mom^3 / 3)
end

"""(P, rho) of the whole dark sector (N1, N2, A') in the solver state u."""
function dark_P_rho(pan, u)
    T_N, gap_N = N_temp_gap(pan, u)
    gap_A = gap_A_from_eta(u[4])
    xi_N = pan.N1.m / T_N - gap_N
    P = pressure(pan.N1, T_N, gap_N) + pressure(pan.N2, T_N, gap_N) + pressure(pan.A, T_N, gap_A)
    return P, energy_dens(pan, T_N, xi_N; gap_A=gap_A)
end

"""
    sound_horizon(h, z_hi; n_grid=2000) -> comoving r_s in GeV^-1

∫ c_s / a dt from the start of the solution to z_hi, with c_s^2 = dP/drho of
the dark sector, set to 0 unless dP/dt < 0, drho/dt < 0 and dP/drho < 0.3
(as `c_sound_grid` in the old code). During the solution P and rho are taken
on a uniform grid in z and differentiated by central differences (of their
logarithms); after it c_s^2 = 5T/(3m + 7.5T), the old code's
non-relativistic continuation (P = n T, rho = n(m + 3T/2)).
"""
function sound_horizon(h::DarkHistory, z_hi; n_grid=2000)
    pan, sol = h.pan, h.sol
    z_i = sol.t[1]
    z_hi > z_i || return 0.0
    z_s = min(z_hi, h.z_f)
    zs = range(z_i, z_s, length=n_grid)
    PR = [dark_P_rho(pan, sol(z)::Vector{Float64}) for z in zs]
    lP = log.(first.(PR))
    lR = log.(last.(PR))
    r = 0.0
    f_prev = 0.0
    for i in eachindex(zs)
        j0, j1 = max(i - 1, 1), min(i + 1, n_grid)
        dlP = lP[j1] - lP[j0]
        dlR = lR[j1] - lR[j0]
        cs2 = (PR[i][1] * dlP) / (PR[i][2] * dlR)
        ok = dlP < 0 && dlR < 0 && cs2 < 0.3
        f = ok ? sqrt(cs2) / scale_factor_z(pan, zs[i]) * dt_dz(pan, zs[i]) : 0.0
        i > 1 && (r += (f + f_prev) / 2 * step(zs))
        f_prev = f
    end
    if z_hi > h.z_f
        cs(z) = begin
            T, _ = temp_gap(h, z)
            sqrt(5T / (3 * pan.N1.m + 7.5T))
        end
        r += comoving_distance(pan, cs, h.z_f, z_hi; rtol=1e-6)
    end
    return r
end

"""
    lyman_alpha_lengths(pan, tT_rel, sol; amp=:fixed, sigma_scale=1.0, channels=:all, z_red_end=LYA_Z_RED_END)

Lyman-alpha lengths of the old code for the solved point `sol`, for kinetic
decoupling at Gamma_kd = H (and, with suffix 3, = 3H). NamedTuple with, in Mpc,
`lambda_fs` (free streaming from T_kd to z_red_end, spectrum frozen at T_kd)
and `r_s` (sound horizon up to T_kd, not capped at z_red_end, as in the old
code), and `x_kd` = m_N/T_nu at decoupling (Inf if the dark sector is still
coupled at the end of the time grid).
"""
function lyman_alpha_lengths(pan::PandemolatorZ, tT_rel::TimeTempRelation, sol;
        amp::Symbol=:fixed, sigma_scale=1.0, channels::Symbol=:all, z_red_end=LYA_Z_RED_END)
    h = DarkHistory(pan, sol)
    m = pan.N1.m
    z_lo = max(log(m / (0.1 * pan.A.m)), sol.t[1])
    z_hi = log(m / tT_rel.T_nu_grid[end])
    z_end = z_at_redshift(m, tT_rel, z_red_end)
    res = map((1.0, 3.0)) do crit
        z_kd = find_kd(h, z_lo, z_hi; crit=crit, amp=amp, sigma_scale=sigma_scale, channels=channels)
        lam = 0.0
        if z_kd < z_end
            T_kd, gap_kd = temp_gap(h, z_kd)
            a_kd = scale_factor_z(pan, z_kd)
            v(z) = mean_speed(pan.N1, T_kd, gap_kd; r=a_kd / scale_factor_z(pan, z))
            lam = comoving_distance(pan, v, z_kd, z_end; rtol=1e-6)
        end
        (x_kd=z_kd >= z_hi ? Inf : exp(z_kd), lambda_fs=lam / Mpc, r_s=sound_horizon(h, z_kd) / Mpc)
    end
    return (x_kd=res[1].x_kd, lambda_fs=res[1].lambda_fs, r_s=res[1].r_s,
            x_kd3=res[2].x_kd, lambda_fs3=res[2].lambda_fs, r_s3=res[2].r_s)
end

# --------------------------------------------------------------------------
# N2/N1 after N1 N1 <-> N2 N2 freeze-out, with the mass splitting the solver
# ignores (draft, Eqs. (Nratio)-(finalNratio))

"""m_nu used for the N1-N2 splitting delta_m = sin^2(theta_1) m_N/m_nu (BP1 value)."""
const M_NU_SPLIT = 1e-10   # GeV

"""
    n2_fraction(h, delta; z_hi)

N2/(N1 + N2) from N1 N1 <-> N2 N2 with m_N2 = (1 + delta) m_N1, integrated from
the start of the solution to z_hi:
    d f2/dt = n [(1 - f2)^2 <sigma v>_11->22 - f2^2 <sigma v>_22->11],
n = n_N1 + n_N2 (the solver's, then conserved, ∝ a^-3), T_N from the solution
and then ∝ a^-2. Non-relativistic cross sections (draft, eq. NNconversionNR):
sigma_22->11 v = sigma_0 sqrt(v^2 + 8 delta), sigma_0 = g^4 m^2/(8 pi m_A^4), with
v the relative velocity, Maxwell-Boltzmann averaged at T_N (reduced mass m/2);
<sigma v>_11->22 = e^(-2 delta m/T) <sigma v>_22->11 (detailed balance), so that
f2/(1 - f2) -> e^(-delta m/T) in equilibrium. Integrated in the log-odds
ell = log(f2/(1 - f2)), starting in equilibrium; implicit (the rate is >> H early).

Returns (f2, x_d_fo, delta_x_d_fo, delta_x_d_end): the final fraction; m/T_N
and delta m/T_N where n <sigma v>_22->11 drops below H (Inf if never, NaN if
it never exceeds H); and
delta m/T_N at the end of the solution (x = 100), where m_N1 = m_N2 stops
being assumed.
"""
function n2_fraction(h::DarkHistory, delta; z_hi)
    pan = h.pan
    m, m_A, g = pan.N1.m, pan.A.m, pan.mp.y
    sigma0 = g^4 * m^2 / (8pi * m_A^4)
    n_end = 2 * number_density(pan.N1, h.T_f, 0.0; gap=h.gap_f)
    function n_T(z)
        T, gap = temp_gap(h, z)
        n = z <= h.z_f ? 2 * number_density(pan.N1, T, 0.0; gap=gap) :
            n_end * (h.a_f / scale_factor_z(pan, z))^3
        return n, T
    end
    # <sqrt(v^2 + 8 delta)> over the relative-velocity Maxwellian, y = v sqrt(m/(4T))
    function sv22(T)
        c = 4T / m
        val = QuadGK.quadgk(y -> y^2 * exp(-y^2) * sqrt(c * y^2 + 8delta), 0.0, Inf; rtol=1e-8)[1]
        return sigma0 * 4 / sqrt(pi) * val
    end
    function rhs(ell, _, z)
        n, T = n_T(z)
        s22 = sv22(T)
        s11 = exp(-2 * delta * m / T) * s22
        f2 = 1 / (1 + exp(-ell))
        df2 = dt_dz(pan, z) * n * ((1 - f2)^2 * s11 - f2^2 * s22)
        return df2 / (f2 * (1 - f2))
    end
    z0 = h.sol.t[1]
    _, T0 = n_T(z0)
    prob = DE.ODEProblem(rhs, -delta * m / T0, (z0, z_hi))
    sol = DE.solve(prob, ODER.Rodas4P(autodiff=AutoFiniteDiff()); reltol=1e-7, abstol=1e-9)
    f2 = 1 / (1 + exp(-sol.u[end]))
    # freeze-out: last z at which n <sigma v>_22->11 > H, on a grid
    zs = range(z0, z_hi, length=400)
    ratio = [(nT = n_T(z); nT[1] * sv22(nT[2]) / pan.H_interp_z(z)) for z in zs]
    i = findlast(>(1), ratio)
    x_d_fo = i === nothing ? NaN : (i == length(zs) ? Inf : m / n_T(zs[i])[2])
    return (f2=f2, x_d_fo=x_d_fo, delta_x_d_fo=delta * x_d_fo, delta_x_d_end=delta * m / h.T_f)
end
