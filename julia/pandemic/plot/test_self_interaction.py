#! /usr/bin/env python3
"""Checks of plot/self_interaction.py. Run: conda run -n pandemic python plot/test_self_interaction.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_interaction import sigma_12, sigma_mix, sigma_NN, sigma_over_m, sq_amp_fixed


def close(a, b, rtol):
    assert abs(a / b - 1) < rtol, (a, b, a / b)


# corrected (A6) at threshold: sum|M|^2 = 64 g^4 m^4 / m_A^4 -> sigma = g^4 m^2 / (8 pi m_A^4),
# the draft's eq. NNconversionNR
for m, r, g in [(1e-5, 2.5, 1e-3), (3e-6, 2.5, 0.1), (1e-4, 10.0, 1e-2)]:
    close(sigma_NN(m, r * m, g, v_rel=1e-4), g**4 * m**2 / (8 * np.pi * (r * m)**4), 1e-6)

# massless limit, averaged over initial spins: Moller, 2 g^4 [(s^2+u^2)/t^2 + (s^2+t^2)/u^2 + 2 s^2/(t u)]
s_, t_, g = 3.0, -1.2, 0.7
u_ = -s_ - t_
moller = 2 * g**4 * ((s_**2 + u_**2) / t_**2 + (s_**2 + t_**2) / u_**2 + 2 * s_**2 / (t_ * u_))
close(sq_amp_fixed(s_, t_, 0.0, 0.0, g) / 4, moller, 1e-12)

# sigma/m ∝ g^4 / m^3 at fixed m_A/m
close(sigma_over_m(2e-5, 2e-3, 2.5, 0.5, 0.0) / sigma_over_m(1e-5, 1e-3, 2.5, 0.5, 0.0), 16 / 8, 1e-6)

# N1 N2 -> N1 N2 at m_A = 2.5 m: 64.963 x sigma_11 (numerically, from explicit
# spinors, in notes/tree_amplitudes); a 50/50 mixture without splitting: 32.98 x
m, g = 1e-5, 1e-3
close(sigma_12(m, 2.5 * m, g) / sigma_NN(m, 2.5 * m, g, v_rel=1e-5), 64.9630, 1e-4)
close(sigma_mix(m, 2.5 * m, g, 0.5, 0.0, v_rel=1e-5) / sigma_NN(m, 2.5 * m, g, v_rel=1e-5), 32.9815, 1e-4)
# pure N1 below the threshold of N1 N1 -> N2 N2: no self-scattering at all
assert sigma_mix(m, 2.5 * m, g, 0.0, 1e-6, v_rel=1e-3) == 0.0
# pure N1 without splitting: the conversion cross section
close(sigma_mix(m, 2.5 * m, g, 0.0, 0.0), sigma_NN(m, 2.5 * m, g), 1e-12)
print("all checks passed")
