"""
DM self-interaction cross section per mass, sigma/m, for the constraint
sigma/m < 1 cm^2/g in plot/relic_scan_theta.py. It depends only on (m_N, g, m_A'),
not on the mixing or the dark-sector history.

VARIANTS
========
"mix" (default): the DM is a mixture of N1 and N2, with N2 fraction f2 (from
    N1 N1 <-> N2 N2 freeze-out, column f_N2 of lyman_alpha.csv) and splitting
    m_N2 - m_N1 = delta m (column delta_m). The mean self-scattering cross
    section per particle is
        sigma = f1^2 sigma_11 + 2 f1 f2 sigma_12 + f2^2 sigma_22,
    with sigma_11 = N1 N1 -> N2 N2 (below), endothermic: x sqrt(1 - 8 delta/v^2),
    zero below threshold; sigma_22 = N2 N2 -> N1 N1, x sqrt(1 + 8 delta/v^2);
    and sigma_12 = N1 N2 -> N1 N2 (s-channel A' and exchange; process 2 of
    notes/tree_amplitudes.tex), at low velocity
        sigma_12 = g^4 m^2 (16 m^4 - 20 m^2 m_A^2 + 7 m_A^4) / (4 pi m_A^4 (m_A^2 - 4 m^2)^2),
    65 x sigma_11 at m_A = 2.5 m. All three are isotropic at low velocity, so
    also the transfer cross sections. Until 2026-10-03 only sigma_11 was used.
"fixed": N1 N1 -> N2 N2 with the corrected amplitude, eq. (A6fixed)
    of notes/amplitudes_majorana.tex (in pandemic_neutrinos/, outside this
    repo): 4 x the t- and u-channel terms of the draft's (A6) and the opposite
    sign on its interference term. For m_N1 = m_N2 it acts as elastic
    scattering, which the draft names as the dominant self-interaction.
    Spin-averaged (1/4) and with the factor 1/2 for identical final-state
    particles, at non-relativistic relative velocity `v_rel` (default 1e-3):
    sigma = g^4 m^2 / (8 pi m_A^4) (1 + O(v^2)), the draft's eq. NNconversionNR.
    The amplitude is isotropic at threshold, so this is also the transfer
    cross section sigma_T = ∫ dsigma (1 - cos theta) that the draft quotes.
"A6": the same with (A6) as printed in the draft, which is wrong; 3/4 of
    "fixed" at threshold. For comparison only.
"old": code/sterile_res/plotter_3.py verbatim,
    sigma/m = g^4 m (4m^4 - 2m^2 m_A^2 + m_A^4) / (pi m_A^4 (m_A^2 - 4m^2)^2).
    Not traceable to an amplitude in the draft or the code: for m_A >> m it is
    the spin-summed (not averaged, no identical-particle factor) N1 N1 -> N2 N2
    cross section, 8 x the draft's sigma_NR; the (m_A^2 - 4m^2)^-2 factor (an
    s-channel A' at threshold, 6.0 x at m_A = 2.5 m) cannot come from
    N1 N1 -> N2 N2, which has no s-channel.

No variant accounts for the DM being ~50/50 N1/N2 for m_N1 = m_N2 (only
same-species pairs convert, and N1 N2 -> N1 N2 is not in the draft).
"""

import numpy as np
from scipy.integrate import quad

# 1 cm^2/g in GeV^-3 (conv_cm2_g in src/constants_functions.jl and code/constants_functions.py)
GEV3_PER_CM2_G = 4.57821356e3


def sq_amp_fixed(s, t, m, m_A, g):
    """sum |M|^2 of N1 N1 -> N2 N2 for m_N1 = m_N2 = m, the corrected (A6):
    eq. (A6fixed) of notes/amplitudes_majorana.tex."""
    m2, mA2 = m * m, m_A * m_A
    u = 4 * m2 - s - t
    return 16 * g**4 * (((s - 2 * m2)**2 + s * t + t**2 / 2) / (t - mA2)**2
                        + ((s - 2 * m2)**2 + s * u + u**2 / 2) / (u - mA2)**2
                        + ((s - 4 * m2)**2 - 4 * m2**2) / ((t - mA2) * (u - mA2)))


def sq_amp_A6(s, t, m, m_A, g):
    """sum |M|^2 of N1 N1 -> N2 N2 for m_N1 = m_N2 = m, eq. (A6) of the draft
    as printed (wrong, see the module docstring)."""
    m2, mA2 = m * m, m_A * m_A
    u = 4 * m2 - s - t
    return 4 * g**4 * (((s - 2 * m2)**2 + s * t + t**2 / 2) / (t - mA2)**2
                       + ((s - 2 * m2)**2 + s * u + u**2 / 2) / (u - mA2)**2
                       - ((s - 4 * m2)**2 - 4 * m2**2) / ((t - mA2) * (u - mA2)))


def sigma_NN(m, m_A, g, v_rel=1e-3, amp=sq_amp_fixed):
    """Spin-averaged N1 N1 -> N2 N2 cross section with the identical-particle
    factor 1/2, at relative velocity v_rel (CM momentum p = m v_rel / 2), for
    the amplitude `amp`. Units of m^-2."""
    p2 = (m * v_rel / 2)**2
    s = 4 * (m * m + p2)
    # dsigma/dt = sum|M|^2 / (64 pi s p^2), t in [-4 p^2, 0] (equal masses)
    val = quad(lambda t: amp(s, t, m, m_A, g) / (64 * np.pi * s * p2), -4 * p2, 0.0,
               epsrel=1e-10, epsabs=0.0)[0]
    return val / 4 / 2


def sigma_12(m, m_A, g):
    """Spin-averaged N1 N2 -> N1 N2 cross section at low velocity (m_N1 = m_N2 = m).
    Units of m^-2. Numerically verified in notes/tree_amplitudes."""
    return (g**4 * m**2 * (16 * m**4 - 20 * m**2 * m_A**2 + 7 * m_A**4)
            / (4 * np.pi * m_A**4 * (m_A**2 - 4 * m**2)**2))


def sigma_mix(m, m_A, g, f2, delta, v_rel=1e-3):
    """Mean self-scattering cross section of an N1/N2 mixture with N2 fraction f2
    and splitting delta = (m_N2 - m_N1)/m_N1, at relative velocity v_rel."""
    f1 = 1 - f2
    s0 = sigma_NN(m, m_A, g, v_rel=v_rel)
    r = 8 * delta / v_rel**2
    s11 = s0 * np.sqrt(max(1 - r, 0.0))
    s22 = s0 * np.sqrt(1 + r)
    return f1 * f1 * s11 + 2 * f1 * f2 * sigma_12(m, m_A, g) + f2 * f2 * s22


def sigma_over_m(m, g, m_A_over_m, variant="mix", f2=None, delta=None, v_rel=1e-3):
    """sigma/m in cm^2/g for DM mass m [GeV], gauge coupling g and m_A'/m_N;
    variant "mix" needs the N2 fraction f2 and the splitting delta."""
    m_A = m_A_over_m * m
    if variant == "mix":
        if f2 is None or delta is None:
            raise ValueError('variant "mix" needs f2 and delta')
        s_m = sigma_mix(m, m_A, g, f2, delta, v_rel=v_rel) / m
    elif variant == "fixed":
        s_m = sigma_NN(m, m_A, g) / m
    elif variant == "A6":
        s_m = sigma_NN(m, m_A, g, amp=sq_amp_A6) / m
    elif variant == "old":
        s_m = g**4 * m * (4 * m**4 - 2 * m**2 * m_A**2 + m_A**4) / (m_A**4 * np.pi * (m_A**2 - 4 * m**2)**2)
    else:
        raise ValueError(f"unknown self-interaction variant {variant}")
    return s_m / GEV3_PER_CM2_G
