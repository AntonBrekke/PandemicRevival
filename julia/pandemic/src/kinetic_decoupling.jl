"""
kinetic_decoupling.jl

The momentum-exchange rate of the dark matter that sets its kinetic decoupling
(lyman_alpha.jl), as in Bringmann et al. (arXiv:2206.10630) and the old Python code
(code/sterile_caller.py, `C_therm_kd`):

    Gamma_kd = C_kd / n_N1 = n_N1 <sigma v>_11->22 + n_N2 <sigma v>_12->12,
    C_kd = T/(32 pi^4) ∫ ds [e^(-2 gap1) sigma_11/2 + e^(-gap1 - gap2) sigma_12] (s - 4m^2) sqrt(s) K_1(sqrt(s)/T) e^(2m/T),

Maxwell-Boltzmann, with gap_i = (m - mu_i)/T and sigma_ij summed over all dof (the 1/2 is
the identical-particle factor of N1 N1 -> N2 N2, which for degenerate masses acts as
elastic scattering).

- N1 N1 -> N2 N2: t- and u-channel A', the corrected eq. (A6) of the draft
  (notes/amplitudes_majorana.tex; equal to the old code's `vector_mediator.M2_gen`). At
  threshold sigma/8 = g^4 m^2/(8 pi m_A^4), the draft's sigma_NR.
- N1 N2 -> N1 N2 (notes/tree_amplitudes.tex, process 2): s-channel A' with a
  Breit-Wigner of the A' -> N1 N2 width, and t/u exchange. The pole at s = m_A'^2 lies
  inside the thermal integral; its on-shell part (inverse decay N1 N2 -> A' and decay
  back) is genuine momentum exchange. Near threshold sigma_12 = 32.5 sigma_11.

`CkdTable` tabulates the cross sections once per (m, g); `C_kd(tab, T, gap; gap2)` is
then a fixed-grid quadrature (test/test_ckd_table.jl checks it against nested adaptive
quadrature to 2e-3).
"""

import QuadGK
import SpecialFunctions as SF

isdefined(@__MODULE__, :TreeAmplitudes) || include(joinpath(@__DIR__, "tree_amplitudes.jl"))
using .TreeAmplitudes: msq_A_N1N2, xyz_N1N2_N1N2, msq_bw

"""
    sq_amp_NN(s, t, m, m_A, g)

sum |M|^2 of N1 N1 -> N2 N2 for m_N1 = m_N2 = m: eq. (A6fixed) of
notes/amplitudes_majorana.tex.
"""
function sq_amp_NN(s, t, m, m_A, g)
    m2, mA2 = m * m, m_A * m_A
    u = 4 * m2 - s - t
    return 16 * g^4 * (
        ((s - 2 * m2)^2 + s * t + t^2 / 2) / (t - mA2)^2
        + ((s - 2 * m2)^2 + s * u + u^2 / 2) / (u - mA2)^2
        + ((s - 4 * m2)^2 - 4 * m2^2) / ((t - mA2) * (u - mA2))
    )
end

"""∫ dt sum|M|^2 / (64 pi s p_cm^2) of N1 N1 -> N2 N2, all masses m: summed over all dof, no symmetry factor."""
function sigma_NN(s, m, m_A, g)
    p2 = s / 4 - m * m
    p2 > 0 || return 0.0
    val = QuadGK.quadgk(t -> sq_amp_NN(s, t, m, m_A, g), -4 * p2, 0.0; rtol=1e-8)[1]
    return val / (64 * pi * s * p2)
end

_kd_params(m, m_A, g) = (g=g, y=0.0, th0=0.0, th1=0.0, mu=0.0, m1=m, m2=m, mA=m_A, mphi=1.0, m0=1.0)

"""
Width of A' -> N1 N2 for m_N1 = m_N2 = m (A' -> N2 nu is O(sin^2 theta_1) and dropped):
sqrt(lambda) sum|M|^2 / (48 pi m_A^3), averaged over the 3 polarisations.
"""
width_A(m, m_A, g) = sqrt(m_A^2 * (m_A^2 - 4m^2)) * msq_A_N1N2(_kd_params(m, m_A, g)) / (48pi * m_A^3)

"""∫ dt sum|M|^2 / (64 pi s p_cm^2) of N1 N2 -> N1 N2, m_N1 = m_N2 = m, all dof, Breit-Wigner A' of width Gamma_A."""
function sigma_NN12(s, m, m_A, g, Gamma_A)
    p2 = s / 4 - m * m
    p2 > 0 || return 0.0
    p = _kd_params(m, m_A, g)
    val = QuadGK.quadgk(t -> msq_bw(xyz_N1N2_N1N2(s, t, p), s, m_A, Gamma_A), -4 * p2, 0.0; rtol=1e-8)[1]
    return val / (64 * pi * s * p2)
end

"""
e^x K_1(x). Asymptotic series for x > 1e4 (relative error < 1e-13), where the Amos
routine behind `besselkx` refuses the argument.
"""
_besselk1x(x) = x < 1e4 ? SF.besselkx(1, x) : sqrt(pi / (2x)) * (1 + 3 / (8x) - 15 / (128x^2) + 315 / (3072x^3))

"""
    CkdTable(m, m_A, g; per_decade=60, narrow_du=1e-9)

sigma_11/2 and sigma_12 tabulated once per (m, m_A, g) on a fixed grid in
u = sqrt(s) - 2m: log-spaced from 1e-18 m to 1e7 m, plus a cluster around the A' pole
u = m_A - 2m from L_in = 1e-3 of its half-width du = Gamma_A/2 out to 0.1 m (a trapezoid
from the log grid straight to a few widths would weight the tail by an interval of up to
~1e10 widths at small g).

Below g ~ 1e-6 the width is too small for a grid in u (du < narrow_du m: the offsets
would be lost to rounding of u ~ m). The pole is then the narrow-width delta function of
area 2 du atan(L_in/du) times the peak value (the Lorentzian within +-L_in = 1e3 du; the
peak is evaluated at s = m_A^2 exactly), the cluster starts at L_in, and the interval
across the pole is left out of the trapezoid.
"""
struct CkdTable
    m::Float64
    u::Vector{Float64}
    w11::Vector{Float64}   # 2 s sigma_11/2 (s - 4m^2): the integrand without the thermal factor
    w12::Vector{Float64}   # the same for sigma_12
    u_res::Float64
    res_area::Float64      # narrow-width pole of w12 (0 if the pole is on the grid)
end

function CkdTable(m, m_A, g; per_decade=60, narrow_du=1e-9)
    Gamma_A = width_A(m, m_A, g)
    u = collect(m .* 10 .^ range(-18, 7, length=25 * per_decade + 1))
    u_res, du = m_A - 2m, Gamma_A / 2
    narrow = du < narrow_du * m
    L_in = narrow ? 1e3 * du : 1e-3 * du
    n_off = round(Int, per_decade * log10(0.1m / L_in)) + 1
    offs = 10 .^ range(log10(L_in), log10(0.1m), length=n_off)
    append!(u, filter(>(0), narrow ? [u_res .- offs; u_res .+ offs] : [u_res; u_res .- offs; u_res .+ offs]))
    sort!(unique!(u))
    narrow && filter!(x -> abs(x - u_res) >= L_in * (1 - 1e-6), u)
    s = @. (2m + u)^2
    w11 = [2 * si * sigma_NN(si, m, m_A, g) / 2 * (si - 4m^2) for si in s]
    w12 = [2 * si * sigma_NN12(si, m, m_A, g, Gamma_A) * (si - 4m^2) for si in s]
    res_area = 0.0
    if narrow
        sA = m_A^2
        res_area = 2 * sA * sigma_NN12(sA, m, m_A, g, Gamma_A) * (sA - 4m^2) * 2du * atan(L_in / du)
    end
    return CkdTable(m, u, w11, w12, u_res, res_area)
end

"""
    C_kd(tab, T, gap; gap2=gap)

Momentum-exchange rate density at dark temperature T, with gap = (m - mu_N1)/T and gap2
that of N2 (gap2 = Inf: N1 N1 -> N2 N2 alone).
"""
function C_kd(tab::CkdTable, T, gap; gap2=gap)
    m, u = tab.m, tab.u
    w12 = exp(gap - gap2)
    thermal(ui) = _besselk1x((2m + ui) / T) * exp(-ui / T)
    acc, f_prev = 0.0, 0.0
    skip = tab.res_area > 0
    for i in eachindex(u)
        f = (tab.w11[i] + w12 * tab.w12[i]) * thermal(u[i])
        if i > 1 && !(skip && u[i-1] < tab.u_res < u[i])
            acc += (f + f_prev) / 2 * (u[i] - u[i-1])
        end
        f_prev = f
        u[i] > 200 * T && break   # e^(-u/T) < e^-200 from here on
    end
    acc += w12 * tab.res_area * thermal(tab.u_res)
    return T / (32 * pi^4) * exp(-2 * gap) * acc
end
