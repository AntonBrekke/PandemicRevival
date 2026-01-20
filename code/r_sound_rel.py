#! /usr/bin/env python3

import numpy as np
from scipy.integrate import quad
import constants_functions as cf
import utils
import matplotlib.pyplot as plt
from math import exp, log, sqrt

import densities as dens

import pandemolator

Ttrel = pandemolator.TimeTempRelation(t_gp_pd = 1000)
ent_grid = np.array([cf.s_SM_no_nu(T)+cf.s_nu(T_nu) for T, T_nu in zip(Ttrel.T_SM_grid, Ttrel.T_nu_grid)])
sf_norm_today = (cf.s0/ent_grid)**(1./3.)

Mcut_const = 1e11*(15**(-4.))
T_kd_const = 1e-7*((5e10/Mcut_const)**(1./3.))
i_max = np.argmax(Ttrel.T_SM_grid < T_kd_const)
print(T_kd_const, Ttrel.T_SM_grid[i_max-1], Ttrel.T_SM_grid[i_max])

print(utils.simp(Ttrel.t_grid[:i_max], 1./(np.sqrt(3.)*sf_norm_today[:i_max])) / cf.Mpc)
plt.loglog(Ttrel.T_SM_grid[:i_max], utils.cumsimp(Ttrel.t_grid[:i_max], 1./(np.sqrt(3.)*sf_norm_today[:i_max])) / cf.Mpc)
plt.show()
exit(1)
