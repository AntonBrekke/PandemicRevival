"""
DM self-interaction cross section per mass, sigma/m, for the constraint
sigma/m < 1 cm^2/g in plot/relic_scan_theta.py. It depends only on (m_N, g, m_A'),
not on the mixing or the dark-sector history.

VARIANTS
========
"A6" (default): N1 N1 -> N2 N2 from the draft's amplitude (A6), which for
    m_N1 = m_N2 acts as elastic scattering and which the draft names as the
    dominant self-interaction. Spin-averaged (1/4) and with the factor 1/2 for
    identical final-state particles, at non-relativistic relative velocity
    `v_rel` (default 1e-3): sigma = 3 g^4 m^2 / (32 pi m_A^4) (1 + O(v^2)).
    The amplitude is isotropic at threshold, so this is also the transfer
    cross section sigma_T = ∫ dsigma (1 - cos theta) that the draft quotes.
    NB: this is 3/4 of the draft's eq. NNconversionNR, g^4 m^2/(8 pi m_A^4);
    that discrepancy between (A6) and NNconversionNR is still open.
"old": code/sterile_res/plotter_3.py verbatim,
    sigma/m = g^4 m (4m^4 - 2m^2 m_A^2 + m_A^4) / (pi m_A^4 (m_A^2 - 4m^2)^2).
    Not traceable to an amplitude in the draft or the code: for m_A >> m it is
    the spin-summed (not averaged, no identical-particle factor) N1 N1 -> N2 N2
    cross section, 8 x the draft's sigma_NR; the (m_A^2 - 4m^2)^-2 factor (an
    s-channel A' at threshold, 6.0 x at m_A = 2.5 m) cannot come from
    N1 N1 -> N2 N2, which has no s-channel.

Neither variant accounts for the DM being ~50/50 N1/N2 for m_N1 = m_N2 (only
same-species pairs convert, and N1 N2 -> N1 N2 is not in the draft).
"""

import numpy as np
from scipy.integrate import quad

# 1 cm^2/g in GeV^-3 (conv_cm2_g in src/constants_functions.jl and code/constants_functions.py)
GEV3_PER_CM2_G = 4.57821356e3


def sq_amp_A6(s, t, m, m_A, g):
    """sum |M|^2 of N1 N1 -> N2 N2 for m_N1 = m_N2 = m, eq. (A6) of the draft."""
    m2, mA2 = m * m, m_A * m_A
    u = 4 * m2 - s - t
    return 4 * g**4 * (((s - 2 * m2)**2 + s * t + t**2 / 2) / (t - mA2)**2
                       + ((s - 2 * m2)**2 + s * u + u**2 / 2) / (u - mA2)**2
                       - ((s - 4 * m2)**2 - 4 * m2**2) / ((t - mA2) * (u - mA2)))


def sigma_A6(m, m_A, g, v_rel=1e-3):
    """Spin-averaged N1 N1 -> N2 N2 cross section with the identical-particle
    factor 1/2, at relative velocity v_rel (CM momentum p = m v_rel / 2). Units of m^-2."""
    p2 = (m * v_rel / 2)**2
    s = 4 * (m * m + p2)
    # dsigma/dt = sum|M|^2 / (64 pi s p^2), t in [-4 p^2, 0] (equal masses)
    val = quad(lambda t: sq_amp_A6(s, t, m, m_A, g) / (64 * np.pi * s * p2), -4 * p2, 0.0,
               epsrel=1e-10, epsabs=0.0)[0]
    return val / 4 / 2


def sigma_over_m(m, g, m_A_over_m, variant="A6"):
    """sigma/m in cm^2/g for DM mass m [GeV], gauge coupling g and m_A'/m_N."""
    m_A = m_A_over_m * m
    if variant == "A6":
        s_m = sigma_A6(m, m_A, g) / m
    elif variant == "old":
        s_m = g**4 * m * (4 * m**4 - 2 * m**2 * m_A**2 + m_A**4) / (m_A**4 * np.pi * (m_A**2 - 4 * m**2)**2)
    else:
        raise ValueError(f"unknown self-interaction variant {variant}")
    return s_m / GEV3_PER_CM2_G
