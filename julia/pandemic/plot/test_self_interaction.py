#! /usr/bin/env python3
"""Checks of plot/self_interaction.py. Run: conda run -n pandemic python plot/test_self_interaction.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_interaction import GEV3_PER_CM2_G, sigma_NN, sigma_over_m, sq_amp_A6, sq_amp_fixed


def close(a, b, rtol):
    assert abs(a / b - 1) < rtol, (a, b, a / b)


# corrected (A6) at threshold: sum|M|^2 = 64 g^4 m^4 / m_A^4 -> sigma = g^4 m^2 / (8 pi m_A^4),
# the draft's eq. NNconversionNR; (A6) as printed gives 3/4 of it
for m, r, g in [(1e-5, 2.5, 1e-3), (3e-6, 2.5, 0.1), (1e-4, 10.0, 1e-2)]:
    close(sigma_NN(m, r * m, g, v_rel=1e-4), g**4 * m**2 / (8 * np.pi * (r * m)**4), 1e-6)
    close(sigma_NN(m, r * m, g, v_rel=1e-4, amp=sq_amp_A6), 3 * g**4 * m**2 / (32 * np.pi * (r * m)**4), 1e-6)

# massless limit, averaged over initial spins: Moller, 2 g^4 [(s^2+u^2)/t^2 + (s^2+t^2)/u^2 + 2 s^2/(t u)]
s_, t_, g = 3.0, -1.2, 0.7
u_ = -s_ - t_
moller = 2 * g**4 * ((s_**2 + u_**2) / t_**2 + (s_**2 + t_**2) / u_**2 + 2 * s_**2 / (t_ * u_))
close(sq_amp_fixed(s_, t_, 0.0, 0.0, g) / 4, moller, 1e-12)

# "old" is plotter_3.py's expression
m, g, mX = 1e-5, 1e-3, 2.5e-5
old = g**4 * m * (4 * m**4 - 2 * m**2 * mX**2 + mX**4) / (mX**4 * np.pi * (mX**2 - 4 * m**2)**2)
close(sigma_over_m(m, g, 2.5, "old"), old / GEV3_PER_CM2_G, 1e-12)

# sigma/m ∝ g^4 / m^3 at fixed m_A/m
close(sigma_over_m(2e-5, 2e-3, 2.5) / sigma_over_m(1e-5, 1e-3, 2.5), 16 / 8, 1e-6)

for v in ("old", "A6"):
    ratio = sigma_over_m(m, g, 2.5, v) / sigma_over_m(m, g, 2.5)
    print(f"{v} / fixed at m_A = 2.5 m: {ratio:.4g} (boundary mass shift x{ratio**(1/3):.3g})")
print("all checks passed")
