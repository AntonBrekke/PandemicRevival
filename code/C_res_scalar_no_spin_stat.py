#! /usr/bin/env python3

import numpy as np
import numba as nb
from scipy.integrate import quad
from scipy.special import kn
import scipy

import ctypes
from math import exp, log, sqrt, pi, fabs, atan, asin, tan, isfinite

import densities as dens
import scalar_mediator

max_exp_arg = 3e2
rtol_int = 1e-4
fac_res_width = 1e4

addr = nb.extending.get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1spence")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
spence = functype(addr)

@nb.jit(nopython=True)
def Li2(z):
    return spence(1.-z)

# \int_{E_min}^{E_max} dE f

@nb.jit(nopython=True, cache=True)
def spec_n_int(E_max, E_min, k, T, xi):
    exp_arg_max = xi - E_max/T
    exp_arg_min = xi - E_min/T

    exp_max = exp(min(exp_arg_max, max_exp_arg))
    exp_min = exp(min(exp_arg_min, max_exp_arg))
    if k == 1.:
        return T*log((1.+exp_min)/(1.+exp_max))
    
    elif k == -1.:
        return T*log((1.-exp_max)/(1.-exp_min))
    
    return -T*(exp_max-exp_min)

# \int_{E_min}^{E_max} dE E f

@nb.jit(nopython=True)
def spec_rho_int(E_max, E_min, k, T, xi):
    exp_arg_max = xi - E_max/T
    exp_arg_min = xi - E_min/T

    exp_max = exp(min(exp_arg_max, max_exp_arg))
    exp_min = exp(min(exp_arg_min, max_exp_arg))

    if k == 1.:
        res = T*(-E_max*log(1.+exp_max)+E_min*log(1.+exp_min)+T*(Li2(-exp_max)-Li2(-exp_min)))
        return res
        # return T*(-E_max*log(1.+exp_max)+E_min*log(1.+exp_min)+T*(Li2(-exp_max)-Li2(-exp_min)))
    elif k == -1.:
        res = T*(E_max*log(1.-exp_max)-E_min*log(1.-exp_min)-T*(Li2(exp_max)-Li2(exp_min)))
        return res

    return -T*((E_max+T)*exp_max-(E_min+T)*exp_min)



@nb.jit(nopython=True)
def ker_C_12_3(log_E1, m1, m2, m3, k1, k2, T1, T2, xi1, xi2, type):
    E1 = exp(log_E1)
    p1 = sqrt(max((E1-m1)*(E1+m1), 0.))

    exp_arg_1 = E1/T1 - xi1
    exp_1 = exp(min(-exp_arg_1, max_exp_arg))
    f1 = exp_1/(1. + k1*exp_1)

    sqrt_arg = max(m3**4. - 2.*m3*m3*(m1*m1+m2*m2) + (((m2-m1)*(m2+m1))**2.), 0.)
    sqrt_fac = sqrt(sqrt_arg)

    E2_min = (E1*(m3*m3-m1*m1-m2*m2)-p1*sqrt_fac)/(2.*m1*m1)
    E2_max = (E1*(m3*m3-m1*m1-m2*m2)+p1*sqrt_fac)/(2.*m1*m1)

    if E2_min >= E2_max:
        return 0.

    if type == 0:
        E2_integral = spec_n_int(E2_max, E2_min, k2, T2, xi2)
    else:
        E2_integral = spec_rho_int(E2_max, E2_min, k2, T2, xi2)

    res = E1*f1*E2_integral
    if not isfinite(res):
        return 0.

    return res

# type = 0: C_n, type = 1: C_rho (E2*f1*f2)
def C_12_3(m1, m2, m3, k1, k2, T1, T2, xi1, xi2, M2, type=0):
    E1_min = max(m1, 1e-200)
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*E1_min)

    res, err = quad(ker_C_12_3, log(E1_min), log(E1_max), args=(m1, m2, m3, k1, k2, T1, T2, xi1, xi2, type), epsabs=0., epsrel=rtol_int, limit=100)

    return M2*res/(32.*(pi**3.))

def Gamma_scat(p1, m1, m2, m3, k2, T2, xi2, M2):
    E1 = sqrt(p1*p1+m1*m1)
    m12 = m1*m1
    m22 = m2*m2
    m32 = m3*m3

    if m1 > 0.:
        sqrt_arg = m12*m12 + (((m2-m3)*(m2+m3))**2.) - 2.*m12*(m22+m32)
        if sqrt_arg <= 0.:
            return 0.

        sqrt_fac = sqrt(sqrt_arg)
        E2_min = max(m2, (E1*(m32-m12-m22) - p1*sqrt_fac)/(2.*m12), 1e-200)
        E2_max = (E1*(m32-m12-m22) + p1*sqrt_fac)/(2.*m12)

    else:
        E2_min = max(m2, E1*m22/((m3-m2)*(m3+m2)) + ((m3-m2)*(m3+m2))/(4.*E1))
        E2_max = max((max_exp_arg + xi2)*T2, 1e1*E2_min)

    if E2_max <= E2_min:
        return 0.

    return spec_n_int(E2_max, E2_min, k2, T2, xi2)*M2/(16.*np.pi*p1*E1)


@nb.jit(nopython=True, cache=True)
def ker_C_3_12(log_E, m3, k3, T3, xi3):
    E = exp(log_E)
    p = sqrt((E-m3)*(E+m3)) if E > m3 else 0.

    exp_arg = E/T3 - xi3
    exp_fac = exp(min(-exp_arg, max_exp_arg))
    f3 = exp_fac/(1.+k3*exp_fac)

    return E*p*f3

def C_3_12(m1, m2, m3, k3, T3, xi3, M2):
    sqrt_arg = (m1**4.) - 2.*m1*m1*(m2*m2+m3*m3) + (((m2-m3)*(m2+m3))**2.)
    sqrt_fac = sqrt(sqrt_arg) if sqrt_arg > 0. else 0.

    prefac = M2*sqrt_fac/(4.*((2.*np.pi)**3.)*m3*m3)

    if k3 == 0.:
        x = m3/T3
        if x < 5e2:
            int_fac = exp(xi3)*m3*T3*kn(1, x)
        else:
            int_fac = exp(xi3-x)*m3*T3*sqrt(0.5*np.pi)*(sqrt(1./x) + 0.375*((1./x)**1.5))
    else:
        E3_min = max(m3, 1e-200)
        E3_max = max((max_exp_arg + xi3)*T3, 1e1*E3_min)
        int_fac, err = quad(ker_C_3_12, log(E3_min), log(E3_max), args= (m3, k3, T3, xi3), epsabs=0., epsrel=rtol_int)

    return prefac*int_fac


# phi phi -> d d

@nb.jit(nopython=True, cache=True)
def sigma_pp_dd(s, m_d, m_phi, vert):
    m_d2 = m_d*m_d
    m_phi2 = m_phi*m_phi
    m_phi4 = m_phi2*m_phi2

    if s <= 4.*m_phi2 or s <= 4.*m_d2:
        return 0.

    p3cm = sqrt(0.25*s - m_phi2)
    p1cm = sqrt(0.25*s - m_d2)

    t0 = -((p1cm-p3cm)**2.)
    t1 = -((p1cm+p3cm)**2.)

    int0 = 4*(t0-m_d2) + m_phi4*(1./(m_d2-t0)+1./(m_d2+2.*m_phi2-s-t0))
    int1 = 4*(t1-m_d2) + m_phi4*(1./(m_d2-t1)+1./(m_d2+2.*m_phi2-s-t1))
    log_part = (6.*m_phi4-4.*m_phi2*s+s*s)*log((m_d2+2.*m_phi2-s-t0)*(m_d2-t1)/((m_d2-t0)*(m_d2+2.*m_phi2-s-t1)))/(2.*m_phi2-s)

    return vert*(int0-int1+log_part)/(8.*pi*s*(4.*m_phi2-s))

# d d -> phi phi

@nb.jit(nopython=True, cache=True)
def sigma_dd_pp(s, m_d, m_phi, vert):
    return sigma_pp_dd(s, m_d, m_phi, vert)*(4.*m_phi2-s)/(4.*m_d2-s)



@nb.jit(nopython=True)
def ker_C_pp_dd_E1(log_E1, s, k_phi, T_d, xi_phi, m_phi, type):
    E1 = exp(log_E1)
    p1 = sqrt(max((E1-m_phi)*(E1+m_phi), 0.))

    exp_arg_1 = E1/T_d - xi_phi
    exp_1 = exp(min(-exp_arg_1, max_exp_arg))
    f1 = exp_1/(1. + k_phi*exp_1)

    sqrt_arg = max(s*s - 4.*s*m_phi*m_phi, 0.)
    sqrt_fac = sqrt(sqrt_arg)

    E2_min = (E1*(s-2.*m_phi*m_phi)-p1*sqrt_fac)/(2.*m_phi*m_phi)
    E2_max = (E1*(s-2.*m_phi*m_phi)+p1*sqrt_fac)/(2.*m_phi*m_phi)

    if E2_min >= E2_max:

        return 0.

    if type == 0:
        E2_integral = spec_n_int(E2_max, E2_min, k_phi, T_d, xi_phi)
    else:
        E2_integral = spec_rho_int(E2_max, E2_min, k_phi, T_d, xi_phi)

    res = E1*f1*E2_integral*sqrt_fac
    if not isfinite(res):
        return 0.
    return res


def ker_C_pp_dd_s(log_s, k_phi, T_d, xi_phi, m_d, m_phi, vert, type):
    s = exp(log_s)
    if s <= 4.*m_phi*m_phi or s <= 4.*m_d*m_d:
        return 0.

    sigma = sigma_pp_dd(s, m_d, m_phi, vert)

    E1_min = max(m_phi, 1e-200)
    E1_max = max((max_exp_arg + xi_phi)*T_d, 1e1*m_phi)

    specs_int, err = quad(ker_C_pp_dd_E1, log(E1_min), log(E1_max), args=(s, k_phi, T_d, xi_phi, m_phi, type), epsabs=0., epsrel=rtol_int, limit=100)

    res = s*sigma*specs_int # factor from v_Moeller contained in specs_int
    if not isfinite(res):
        return 0.
    return res

# type = 0: C_n, type = 1: C_rho (E2*f1*f2)

def C_pp_dd(m_d, m_phi, k_phi, T_d, xi_phi, vert, type=0):
    s_min = max(4.*m_d*m_d, 4.*m_phi*m_phi)
    s_max = (5e2*T_d)**2.

    if s_max <= s_min:
        return 0.
    
    res, err = quad(ker_C_pp_dd_s, log(s_min), log(s_max), args=(k_phi, T_d, xi_phi, m_d, m_phi, vert, type), epsabs=0., epsrel=rtol_int, limit=100)

    return res/(32.*(np.pi**4.))


@nb.jit(nopython=True)
def ker_C_dd_pp_E1(log_E1, s, k_d, T_d, xi_d, m_d, type):
    E1 = exp(log_E1)
    p1 = sqrt(max((E1-m_d)*(E1+m_d), 0.))

    exp_arg_1 = E1/T_d - xi_d
    exp_1 = exp(min(-exp_arg_1, max_exp_arg))
    f1 = exp_1/(1. + k_d*exp_1)

    sqrt_arg = max(s*s - 4.*s*m_d*m_d, 0.)
    sqrt_fac = sqrt(sqrt_arg)

    E2_min = (E1*(s-2.*m_d*m_d)-p1*sqrt_fac)/(2.*m_d*m_d)
    E2_max = (E1*(s-2.*m_d*m_d)+p1*sqrt_fac)/(2.*m_d*m_d)

    if E2_min >= E2_max:
        return 0.
    
    if type == 0:
        E2_integral = spec_n_int(E2_max, E2_min, k_d, T_d, xi_d)
    else:
        E2_integral = spec_rho_int(E2_max, E2_min, k_d, T_d, xi_d)

    res = E1*f1*E2_integral*sqrt_fac
    if not isfinite(res):
        return 0.
    return E1*f1*E2_integral*sqrt_fac



def ker_C_dd_pp_s(log_s, k_d, T_d, xi_d, m_d, m_phi, vert, type):
    s = exp(log_s)
    if s <= 4.*m_phi*m_phi or s <= 4.*m_d*m_d:
        return 0.
    
    sigma = sigma_pp_dd(s, m_d, m_phi, vert)

    E1_min = max(m_phi, 1e-200)
    E1_max = max((max_exp_arg + xi_d)*T_d, 1e1*E1_min)

    specs_int, err = quad(ker_C_dd_pp_E1, log(E1_min), log(E1_max), args=(s, k_d, T_d, xi_d, m_d, type), epsabs=0., epsrel=rtol_int, limit=100)

    res = s*sigma*specs_int # factor from v_Moeller contained in specs_int
    if not isfinite(res):
        return 0.

    return res



# type = 0: C_n, type = 1: C_rho (E2*f1*f2)

def C_dd_pp(m_d, m_phi, k_d, T_d, xi_d, vert, type=0):
    s_min = max(4.*m_d*m_d, 4.*m_phi*m_phi)
    s_max = (5e2*T_d)**2.

    if s_max <= s_min:

        return 0.
    
    res, err = quad(ker_C_dd_pp_s, log(s_min), log(s_max), args=(k_d, T_d, xi_d, m_d, m_phi, vert, type), epsabs=0., epsrel=rtol_int, limit=100)
    return res/(32.*(np.pi**4.))


@nb.jit(nopython=True)
def ker_C_12_34_E1(log_E1, s, m1, m2, k1, k2, T1, T2, xi1, xi2, type):
    E1 = exp(log_E1)
    p1 = sqrt(max((E1-m1)*(E1+m1), 0.))

    exp_arg_1 = E1/T1 - xi1
    exp_1 = exp(min(-exp_arg_1, max_exp_arg))
    f1 = exp_1/(1. + k1*exp_1)

    sqrt_arg = max(s*s - 2.*s*(m1*m1+m2*m2) + (((m2-m1)*(m2+m1))**2.), 0.)
    sqrt_fac = sqrt(sqrt_arg)

    E2_min = (E1*(s-m1*m1-m2*m2)-p1*sqrt_fac)/(2.*m1*m1)
    E2_max = (E1*(s-m1*m1-m2*m2)+p1*sqrt_fac)/(2.*m1*m1)

    if E2_min >= E2_max:
        return 0.

    if type == 0.:
        E2_integral = spec_n_int(E2_max, E2_min, k2, T2, xi2)
    else:
        E2_integral = spec_rho_int(E2_max, E2_min, k2, T2, xi2)

    res = E1*f1*E2_integral*sqrt_fac

    if not isfinite(res):
        return 0.
    return res



def ker_C_12_34_s(log_s, m1, m2, m3, m4, k1, k2, T1, T2, xi1, xi2, vert, m_phi2, m_Gamma_phi2, type, res_sub):
    s = exp(log_s)

    if s <= (m1+m2)**2. or s <= (m3+m4)**2.:
        return 0.

    sigma = scalar_mediator.sigma_gen(s, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub=res_sub)

    E1_min = max(m1, 1e-200)
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*E1_min)

    specs_int, err = quad(ker_C_12_34_E1, log(E1_min), log(E1_max), args=(s, m1, m2, k1, k2, T1, T2, xi1, xi2, type), epsabs=0., epsrel=rtol_int, limit=100)

    res = s*sigma*specs_int # factor from v_Moeller contained in specs_int
    if not isfinite(res):
        return 0.
    return res



def C_12_34(m1, m2, m3, m4, k1, k2, T1, T2, xi1, xi2, vert, m_phi2, m_Gamma_phi2, type=0, res_sub=True):
    s_min = max((m1+m2)**2., (m3+m4)**2.)
    s_max = max((5e2*max(T1, T2))**2., 1e2*s_min)

    s_vals = np.sort(np.array([s_min, s_max, m_phi2-fac_res_width*sqrt(m_Gamma_phi2), m_phi2, m_phi2+fac_res_width*sqrt(m_Gamma_phi2)]))
    s_vals = s_vals[s_vals >= s_min]
    s_vals = s_vals[s_vals <= s_max]

    res = 0.
    for i in range(len(s_vals)-1):
        cur_res, err = quad(ker_C_12_34_s, log(s_vals[i]), log(s_vals[i+1]), args=(m1, m2, m3, m4, k1, k2, T1, T2, xi1, xi2, vert, m_phi2, m_Gamma_phi2, type, res_sub), epsabs=0., epsrel=rtol_int, limit=100)

        res += cur_res
    return res/(32.*(np.pi**4.))



def ker_C_dd_dd_gon_gel(log_s, m_d, k_d, T_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, res_sub):
    s = exp(log_s)
    if s <= 4.*m_d*m_d:
        return 0.

    if res_sub:
        sigma = scalar_mediator.sigma_gen(s, m_d, m_d, m_d, m_d, vert_el, m_phi2, m_Gamma_phi2, sub=res_sub)
    else:
        sigma = 2.*scalar_mediator.sigma_el(s, m_d*m_d, vert_el, m_phi2, m_Gamma_phi2)

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



def C_dd_dd_gon_gel(m_d, k_d, T_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, res_sub=False):
    s_min = 4.*m_d*m_d
    s_max = max((5e2*T_d)**2., 1e2*s_min)

    s_vals = np.sort(np.array([s_min, s_max, m_phi2-fac_res_width*sqrt(m_Gamma_phi2), m_phi2, m_phi2+fac_res_width*sqrt(m_Gamma_phi2)]))
    s_vals = s_vals[s_vals >= s_min]
    s_vals = s_vals[s_vals <= s_max]

    res = 0.
    for i in range(len(s_vals)-1):
        cur_res, err = quad(ker_C_dd_dd_gon_gel, log(s_vals[i]), log(s_vals[i+1]), args=(m_d, k_d, T_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, res_sub), epsabs=0., epsrel=rtol_int, limit=100)
        res += cur_res
    return res*T_d/(32.*(np.pi**4.))



if __name__ == '__main__':
    from math import asin, cos, sin
    import C_res_scalar
    import matplotlib.pyplot as plt
    import time

    m_d = 1e-5
    m_a = 0.
    m_phi = 2.5e-5
    sin2_2th = 1e-12
    y = 2e-4

    k_d = 1.
    k_a = 1.
    k_phi = -1.
    dof_d = 2.
    dof_phi = 1.

    m_d2 = m_d*m_d
    m_a2 = m_a*m_a
    m_phi2 = m_phi*m_phi

    th = 0.5*asin(sqrt(sin2_2th))
    c_th = cos(th)
    s_th = sin(th)
    y2 = y*y

    Gamma_phi = scalar_mediator.Gamma_phi(y, th, m_phi, m_d)
    m_Gamma_phi2 = m_phi2*Gamma_phi*Gamma_phi

    T_nu = 0.44188097340593474
    T_d = 0.6510550394714374
    xi_d = -8.551301127056323/T_d
    xi_phi = 2.*xi_d

    vert_el = y2*y2*(c_th**8.)
    vert_tr = y2*y2*(c_th**6.)*(s_th**2.)
    M2_dd = 2.*y2*(c_th**4.)*(m_phi2 - 4.*m_d2)
    M2_aa = 2.*y2*(s_th**4.)*(m_phi2 - 4.*m_a2)
    M2_da = 2.*y2*(s_th**2.)*(c_th**2.)*(m_phi2 - ((m_a + m_d)**2.))

    print("d a -> phi")
    print('n')
    start = time.time()
    res = C_12_3(m_d, m_a, m_phi, 1., 1., T_d, T_nu, xi_d, 0., M2_da, type=0)
    end = time.time()
    print(res, end-start)

    start = time.time()
    res = C_res_scalar.C_n_3_12(m_d, m_a, m_phi, 1., 1., -1., T_d, T_nu, T_d, xi_d, 0., xi_phi, M2_da, type=1)
    end = time.time()
    print(res, end-start)

    print('rho')
    start = time.time()
    res = C_12_3(m_d, m_a, m_phi, 1., 1., T_d, T_nu, xi_d, 0., M2_da, type=1)
    end = time.time()
    print(res, end-start)

    start = time.time()
    res = C_res_scalar.C_rho_3_12(2, m_d, m_a, m_phi, 1., 1., -1., T_d, T_nu, T_d, xi_d, 0., xi_phi, M2_da)
    end = time.time()
    print(res, end-start)

    print("d a -> d d")
    print('n')
    start = time.time()
    res = C_12_34(m_d, m_a, m_d, m_d, 1., 1., T_d, T_nu, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, type=0)
    end = time.time()
    print(res, end-start)

    start = time.time()
    res = C_res_scalar.C_34_12(0., 0., 1., m_d, m_a, m_d, m_d, 1., 1., 1., 1., T_d, T_nu, T_d, T_d, xi_d, 0., xi_d, xi_d, vert_tr, m_phi2, m_Gamma_phi2)
    end = time.time()
    print(res, end-start)

    print('rho')
    start = time.time()
    res = C_12_34(m_d, m_a, m_d, m_d, 1., 1., T_d, T_nu, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, type=1)
    end = time.time()
    print(res, end-start)

    start = time.time()
    res = C_res_scalar.C_34_12(2, 0., 1., m_d, m_a, m_d, m_d, 1., 1., 1., 1., T_d, T_nu, T_d, T_d, xi_d, 0., xi_d, xi_d, vert_tr, m_phi2, m_Gamma_phi2)
    end = time.time()
    print(res, end-start)

    print("d d -> phi phi")
    start = time.time()
    res = C_dd_pp(m_d, m_phi, 0., T_d, xi_d, vert_el, type=0)
    end = time.time()
    print(res, end-start)

    start = time.time()
    res = C_res_scalar.C_n_pp_dd(m_d, m_phi, 1., -1., T_d, xi_d, xi_phi, vert_el, type=1)
    end = time.time()
    print(res, end-start)

    print("phi phi -> d d")
    start = time.time()
    res = C_pp_dd(m_d, m_phi, 0., T_d, xi_phi, vert_el, type=0)
    end = time.time()
    print(res, end-start)

    start = time.time()
    res = C_res_scalar.C_n_pp_dd(m_d, m_phi, 1., -1., T_d, xi_d, xi_phi, vert_el, type=-1)
    end = time.time()

    print(res, end-start)
    exit(1)

