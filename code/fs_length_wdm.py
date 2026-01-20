#! /usr/bin/env python3

import numpy as np
from scipy.integrate import quad
import constants_functions as cf
import utils
import matplotlib.pyplot as plt
from math import exp, log, sqrt

import densities as dens

import pandemolator

Ttrel = pandemolator.TimeTempRelation()
ent_grid = np.array([cf.s_SM_no_nu(T)+cf.s_nu(T_nu) for T, T_nu in zip(Ttrel.T_SM_grid, Ttrel.T_nu_grid)])
sf_norm_today = (cf.s0/ent_grid)**(1./3.)

T_SM_ini = 1e-2
i_ini = np.argmax(Ttrel.T_SM_grid < T_SM_ini)
T_SM_ini = Ttrel.T_SM_grid[i_ini]
i_fs_max = np.argmax(Ttrel.T_SM_grid/cf.T0 - 1. < 15.) # only assume free-streaming until z = 50

def lambda_fs(m_wdm):
    T_wdm_ini = ((2.*cf.pi2*cf.rho_d0/(3.*cf.zeta3*m_wdm))**(1./3.))/sf_norm_today[i_ini]
    sf_norm_ini = (ent_grid[i_ini]/ent_grid)**(1./3.)
    p_star_grid = np.logspace(np.log10(1e-6*T_wdm_ini), np.log10(1e2*T_wdm_ini), 1000)
    f_wdm_grid = 1./(np.exp(p_star_grid/T_wdm_ini) + 1.)
    n_wdm_ini = utils.simp(p_star_grid, f_wdm_grid*(p_star_grid**2.))*2./(2.*cf.pi2)
    integrand_fs_length = np.zeros(Ttrel.t_grid.size)
    for i in range(i_ini, i_fs_max):
        p_cur = p_star_grid/sf_norm_ini[i]
        E_cur = np.sqrt(m_wdm*m_wdm + p_cur*p_cur)
        v = utils.simp(p_star_grid, f_wdm_grid*(p_star_grid**3.)/E_cur)*2./(2.*cf.pi2*n_wdm_ini*sf_norm_ini[i])
        integrand_fs_length[i] = v/sf_norm_today[i]

    plt.semilogx(m_wdm*sf_norm_today/(T_wdm_ini*sf_norm_today[i_ini]), utils.cumsimp(Ttrel.t_grid, integrand_fs_length)/cf.Mpc)
    plt.show()

    l_fs = utils.simp(Ttrel.t_grid, integrand_fs_length) / cf.Mpc
    return l_fs

print(3.3, lambda_fs(3.3e-6))
# print(5.3, lambda_fs(5.3e-6))
# print(1.9, lambda_fs(1.9e-6))
exit(1)
m_wdm_grid = np.logspace(-6, -4, 21)
lambda_fs_grid = np.array([lambda_fs(m_wdm) for m_wdm in m_wdm_grid])

# plt.loglog(m_wdm_grid, lambda_fs_grid)
# plt.show()
