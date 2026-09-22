#! /usr/bin/env python3
"""Checks of plot/self_interaction.py. Run: conda run -n pandemic python plot/test_self_interaction.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_interaction import GEV3_PER_CM2_G, sigma_A6, sigma_over_m


def close(a, b, rtol):
    assert abs(a / b - 1) < rtol, (a, b, a / b)


# (A6) at threshold: sum|M|^2 = 48 g^4 m^4 / m_A^4 -> sigma = 3 g^4 m^2 / (32 pi m_A^4)
for m, r, g in [(1e-5, 2.5, 1e-3), (3e-6, 2.5, 0.1), (1e-4, 10.0, 1e-2)]:
    close(sigma_A6(m, r * m, g, v_rel=1e-4), 3 * g**4 * m**2 / (32 * np.pi * (r * m)**4), 1e-6)

# "old" is plotter_3.py's expression
m, g, mX = 1e-5, 1e-3, 2.5e-5
old = g**4 * m * (4 * m**4 - 2 * m**2 * mX**2 + mX**4) / (mX**4 * np.pi * (mX**2 - 4 * m**2)**2)
close(sigma_over_m(m, g, 2.5, "old"), old / GEV3_PER_CM2_G, 1e-12)

# sigma/m ∝ g^4 / m^3 at fixed m_A/m
close(sigma_over_m(2e-5, 2e-3, 2.5) / sigma_over_m(1e-5, 1e-3, 2.5), 16 / 8, 1e-6)

ratio = sigma_over_m(m, g, 2.5, "old") / sigma_over_m(m, g, 2.5, "A6")
print(f"all checks passed; old / A6 at m_A = 2.5 m: {ratio:.4g} (boundary mass shift x{ratio**(1/3):.3g})")
