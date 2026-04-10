#! /usr/bin/env python3

import numpy as np
import numba as nb
from scipy.integrate import quad
from scipy.special import kn

from math import exp, log, sqrt, isfinite

import vector_mediator

max_exp_arg = 3e2
rtol_int = 1e-4
fac_res_width = 1e4


def ker_C_dd_dd_gon_gel(log_s, m_d, k_d, T_d, xi_d, vert_el, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub):
    s = exp(log_s)
    if s <= 4.*m_d*m_d:
        return 0.

    sigma = vector_mediator.sigma_gen_new(s, m_d, m_d, m_d, m_d, vert_el, m_d**2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, sub=False)

    sqrt_s = sqrt(s)
    if sqrt_s/T_d < max_exp_arg and 2.*xi_d < 6e2:
        res = s*sigma*(s-4.*m_d*m_d)*sqrt_s*kn(1, sqrt_s/T_d)*exp(2.*xi_d)
    else:
        x = T_d/sqrt_s
        kn_xi = exp(2.*xi_d - 1./x)*(sqrt(0.5*np.pi*x) + 0.375*sqrt(0.5*np.pi*(x**3.)) - (15./128.)*sqrt(0.5*np.pi*(x**5.)))
        res = s*sigma*(s-4.*m_d*m_d)*sqrt_s*kn_xi

    if not isfinite(res):
        return 0.
    return res



def C_dd_dd_gon_gel(m_d, k_d, T_d, xi_d, vert_el, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub=False):
    s_min = 4.*m_d*m_d
    s_max = max((5e2*T_d)**2., 1e2*s_min)

    s_vals = np.sort(np.array([s_min, s_max, m_X2-fac_res_width*sqrt(m_Gamma_X2), m_X2, m_X2+fac_res_width*sqrt(m_Gamma_X2)]))
    s_vals = s_vals[s_vals >= s_min]
    s_vals = s_vals[s_vals <= s_max]

    res = 0.
    for i in range(len(s_vals)-1):
        cur_res, err = quad(ker_C_dd_dd_gon_gel, log(s_vals[i]), log(s_vals[i+1]), args=(m_d, k_d, T_d, xi_d, vert_el, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub), epsabs=0., epsrel=rtol_int, limit=100)
        res += cur_res
    return res*T_d/(32.*(np.pi**4.))

