import time
import numpy as np
import matplotlib.pyplot as plt

import pandemolator as pan
import constants_functions as cf


def test_dw():
    m_d = 1e-5
    sin2_2th = 5e-16
    th = np.arcsin(np.sqrt(sin2_2th)) / 2.

    tT_rel = pan.TimeTempRelation()

    T_d_dw = cf.T_d_dw(m_d) # temperature of maximal d production by Dodelson-Widrow mechanism

    i_ic = np.argmax(tT_rel.T_nu_grid < T_d_dw)      # Anton: Start when T_nu < T_dw
    i_end = np.argmax(tT_rel.T_nu_grid < m_d/2e1)    # Anton: End when T_nu < m_d/20 <--> 20 < m_d/T_nu

    T_ic = tT_rel.T_nu_grid[i_ic]
    T_end = tT_rel.T_nu_grid[i_end]

    sf_ic_norm_0 = (cf.s0/(cf.s_SM_no_nu(tT_rel.T_SM_grid[i_ic]) + cf.s_nu(tT_rel.T_nu_grid[i_ic])))**(1./3.)

    n_ic = cf.n_0_dw(m_d, th) / (sf_ic_norm_0**3.)
    rho_ic = n_ic * cf.avg_mom_0_dw(m_d) / sf_ic_norm_0

    print(f"T_d_dw = {T_d_dw:.3e}")
    print(f"i_ic = {i_ic}")
    print(f"T_ic = {T_ic:.3e}")
    print(f"i_end = {i_end}")
    print(f"T_end = {T_end:.3e}")
    print(f"n_ic = {n_ic:.3e}")
    print(f"rho_ic = {rho_ic:.3e}")


test_dw()