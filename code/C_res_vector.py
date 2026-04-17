#! /usr/bin/env python3

from scipy.integrate import quad
import numpy as np
import numba as nb
import cmath as cm
from math import exp, log, sqrt, pi, fabs, atan, asin, tan, isfinite
import vegas
from scipy.special import kn
import densities as dens
from scipy.integrate import quad
import vector_mediator
import scalar_mediator

max_exp_arg = 3e2
rtol_int = 1e-4
spin_stat_irr = 1e3
fac_res_width = 1e4
offset = 1.+1e-14

"""
This file contains the collision operators for number density and energy density 
"""

@nb.jit(nopython=True, cache=True)
def ker_C_n_3_12_E2(log_E2, E1, f1, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, type):
    """
    For some reason, matrix element is taken as input. 
    You can find them in 
    * pandemolator.py 
    * sterile_caller.py 
    * sterile_pandemic.py 
    if they need to be changed
    """
    E2 = exp(log_E2)
    E3 = E1 + E2
    exp_arg_2 = E2/T2 - xi2
    exp_arg_3 = E3/T3 - xi3
    f2 = 1./(exp(exp_arg_2) + k2) if exp_arg_2 < max_exp_arg else 0.
    f3 = 1./(exp(exp_arg_3) + k3) if exp_arg_3 < max_exp_arg else 0.

    if type == 0:
        if T1 == T2 and T1 == T3:
            chem_eq_fac = 1.-exp(-xi1-xi2+xi3)
            dist = f1*f2*(1.-k3*f3)*chem_eq_fac
        else:
            dist_1 = f1*f2*(1-k3*f3)
            dist_2 = f3*(1-k1*f1)*(1-k2*f2)
            dist = dist_1 - dist_2
            if fabs(dist) <= 1e-12*max(fabs(dist_1), fabs(dist_2)):
                return 0.
    elif type == -1:
        dist = -f3*(1.-k1*f1)*(1.-k2*f2)
    elif type == 1:
        dist = f1*f2*(1.-k3*f3)

    res = E2*dist           # Anton: Factor E2 comes from substitution v = ln(E2) in integral
    if not isfinite(res):
        return 0.
    return res

def ker_C_n_3_12_E1(log_E1, m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, type):
    E1 = exp(log_E1)
    p1 = sqrt((E1 - m1)*(E1 + m1)) if E1 > m1 else 0.

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
        E2_min = max(m2, E1*m22/((m3-m2)*(m3+m2)) + ((m3-m2)*(m3+m2))/(4.*E1), 1e-200)
        E2_max = max((max_exp_arg + xi2)*T2, 1e1*E2_min)
    if E2_max <= E2_min:
        return 0.

    exp_arg_1 = E1/T1 - xi1
    f1 = 1./(exp(exp_arg_1) + k1) if exp_arg_1 < max_exp_arg else 0.

    res_2, err = quad(ker_C_n_3_12_E2, log(E2_min), log(E2_max), args=(E1, f1, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, type), epsabs=0., epsrel=rtol_int)

    res = E1*res_2          # Anton: Factor E1 comes from substitution u = ln(E1) for integral
    if not isfinite(res):
        return 0.
    return res

# Anton: This is never called ? 
def Gamma_scat(p1, m1, m2, m3, k2, k3, T2, T3, xi2, xi3, M2):
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
        E2_min = max(m2, E1*m22/((m3-m2)*(m3+m2)) + ((m3-m2)*(m3+m2))/(4.*E1), 1e-200)
        E2_max = max((max_exp_arg + xi2)*T2, 1e1*E2_min)
    if E2_max <= E2_min:
        return 0.
    res, err = quad(ker_C_n_3_12_E2, log(E2_min), log(E2_max), args=(E1, 1., 0., k2, k3, E1, T2, T3, 0., xi2, xi3, 1), epsabs=0., epsrel=rtol_int)
    return M2*res/(16.*pi*p1*E1)

# type == -1: only 3 -> 1 2, type == 0: both reactions, type == 1: only 1 2 -> 3
def C_n_3_12(m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, M2, type=0):
    """
    Anton: Momentum 3 has been eliminated in this collision operator. 
    p1_vec = p1(0,0,1), p2_vec = p2(sin(th), 0, cos(th))
    """
    E1_min = max(m1, 1e-200)
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*m1)    # Infinity, could use np.inf in quad 

    res, err = quad(ker_C_n_3_12_E1, log(E1_min), log(E1_max), args=(m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, type), epsabs=0., epsrel=rtol_int)
    
    return M2*res/(32.*(pi**3.))

# E_type * (f1*f2*(1+f3) - f3*(1-f1)*(1-f2))
@nb.jit(nopython=True, cache=True)
def ker_C_rho_3_12_E2(log_E2, type, E1, f1, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3):
    E2 = exp(log_E2)
    E3 = E1 + E2
    exp_arg_2 = E2/T2 - xi2
    exp_arg_3 = E3/T3 - xi3
    f2 = 1./(exp(exp_arg_2) + k2) if exp_arg_2 < max_exp_arg else 0.
    f3 = 1./(exp(exp_arg_3) + k3) if exp_arg_3 < max_exp_arg else 0.

    dist_1 = f1*f2*(1-k3*f3)
    dist_2 = f3*(1-k1*f1)*(1-k2*f2)
    dist = dist_1 - dist_2
    if fabs(dist) <= 1e-12*max(fabs(dist_1), fabs(dist_2)):
        return 0.

    if type == 1.:
        Etype = E1
    elif type == 2.:
        Etype = E2
    else:
        Etype = E3

    res = E2*Etype*dist         # Anton: Factor E2 comes from substitution v = ln(E2) in integral
    if not isfinite(res):
        return 0.
    return res

def ker_C_rho_3_12_E1(log_E1, type, m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3):
    E1 = exp(log_E1)
    p1 = sqrt((E1 - m1)*(E1 + m1)) if E1 > m1 else 0.

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
        E2_min = max(m2, E1*m22/((m3-m2)*(m3+m2)) + ((m3-m2)*(m3+m2))/(4.*E1), 1e-200)
        E2_max = max((max_exp_arg + xi2)*T2, 1e1*E2_min)
        # print(E1/T1, E2_min/T2, E2_max/T2, E1*m22/((m3-m2)*(m3+m2))/T2, ((m3-m2)*(m3+m2))/(4.*E1)/T2)
    if E2_max <= E2_min:
        return 0.

    exp_arg_1 = E1/T1 - xi1
    f1 = 1./(exp(exp_arg_1) + k1) if exp_arg_1 < max_exp_arg else 0.

    res_2, err = quad(ker_C_rho_3_12_E2, log(E2_min), log(E2_max), args=(type, E1, f1, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3), epsabs=0., epsrel=rtol_int)

    res = E1*res_2          # Anton: Factor E1 comes from substitution u = ln(E1) in integral
    if not isfinite(res):
        return 0.
    return res

def C_rho_3_12(type, m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3, M2):
    E1_min = max(m1, 1e-10*T1)
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*m1)

    res, err = quad(ker_C_rho_3_12_E1, log(E1_min), log(E1_max), args=(type, m1, m2, m3, k1, k2, k3, T1, T2, T3, xi1, xi2, xi3), epsabs=0., epsrel=rtol_int)

    return M2*res/(32.*(pi**3.))

# @nb.jit(nopython=True, cache=True)
def ker_C_n_XX_dd_s_t_integral_revival(ct_min, ct_max, ct_p, ct_m, a, s, E1, E3, p1, p3, m_d, m_X, vert):

    """
    Solved for matrix-element and integrated over variable t in collision
    operator using Mathematica, int_Rt |M|^2/sqrt((t-tm)*(t-tp)). 
    These expressions can be found in nu_s_nu_s_to_XX.nb
    """
    n = s.size
    s2 = s*s
    m_d2 = m_d*m_d
    m_d4 = m_d2*m_d2
    m_d6 = m_d2*m_d4
    m_d8 = m_d4*m_d4
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_X6 = m_X2*m_X4
    m_X8 = m_X4*m_X4

    # t_max,min - t_m,p = 2p1*p3*(ct_max - ct_m)
    t_add = m_d2 + m_X2
    t_min = t_add - 2.*E1*(E3 - p1*p3/E1*ct_min)
    t_max = t_add - 2.*E1*(E3 - p1*p3/E1*ct_max)
    t_m = t_add - 2.*E1*(E3 - p1*p3/E1*ct_m)
    t_p = t_add - 2.*E1*(E3 - p1*p3/E1*ct_p)
    
    """
    Anton: 
    x = t - tm, y = t - tp 
    sqrt(x)*sqrt(y) / sqrt(x*y) appears in expressions, makes trouble. 
    This is either 1 for (x>0 or y>0) and -1 for (x<0 and y<0).
    ill-defined for x->0, y->0 wrt. limit to 0+ or 0-.
    x>=0 & y>=0 -> 1
    x>=0 & y<=0 -> 1
    x<=0 & y>=0 -> 1
    x<=0 & y<=0 -> -1
    t_m > t_p always. 
    t_m >= t_max > t_min >= t_p 
    X_max = t_max - t_m, Y_max = t_max - t_p
    X_min = t_min - t_m, Y_min = t_min - t_p
    For all non-zero integration regions of ct_max, ct_min
    Y_max > 0 always => sqrt_fac_tmax = 1.
    Always have X_min < 0, and Y_min >= 0. For = case, 
    as ct_min --> c_p from above, Y_min --> 0+.
    Therefore, sqrt_fac_tmin = 1 as well.
    """    
    
    # in_min_neq = (ct_min != ct_p)
    # in_max_neq = (ct_max != ct_m)
    in_min_neq = ~np.isclose(ct_min, ct_p, rtol=0, atol=1e-12)
    in_max_neq = ~np.isclose(ct_max, ct_m, rtol=0, atol=1e-12)
    in_any_neq = np.logical_or(in_min_neq, in_max_neq)

    SQA = 1

    # print(t_max-t_m)
    # print(t_min-t_p)
    # print(t_min-t_m)

    term1_max = np.zeros(n) + 0j
    term1_min = np.zeros(n) + 0j

    prefac_2 = -1/(SQA*(2*m_X2-s)*(a*(m_d2-t_m)*(m_d2-t_p)+0j)**(3/2))*4*a*(16*m_d8-16*m_d6*(t_m+t_p)-2*m_d4*(12*m_X4+4*m_X2*(-s+t_m+t_p)+s2-2*s*(t_m+t_p)-8*t_m*t_p)+2*m_d2*(-2*m_X6+m_X4*(s+8*(t_m+t_p))-2*m_X2*(s*(t_m+t_p)-4*t_m*t_p)+s*(s*(t_m+t_p)-4*t_m*t_p))+2*m_X6*(t_m+t_p)-m_X4*(s*(t_m+t_p)+8*t_m*t_p)-2*s2*t_m*t_p)

    # 1j due to a**3/2 = ((-1)*(-a))**3/2 = -i * (-a)**3/2
    term2_max = prefac_2 * (-np.log(t_m-t_p+0j))
    term2_min = prefac_2 * (-np.log(t_p-t_m+0j))
    # term2 = term2_max - term2_min
    # term2 = -prefac_2 * np.pi

    prefac_3 = -(4*(-16*m_d8+16*m_d6*(-6*m_X2+3*s+t_m+t_p)-2*m_d4*(84*m_X4-28*m_X2*(3*s+t_m+t_p)+19*s2+14*s*(t_m+t_p)+8*t_m*t_p)+2*m_d2*(-34*m_X6+m_X4*(57*s+16*(t_m+t_p))-2*m_X2*(12*s2+9*s*(t_m+t_p)+4*t_m*t_p)+s*(2*s2+3*s*(t_m+t_p)+4*t_m*t_p))+24*m_X8-2*m_X6*(12*s+7*(t_m+t_p))+m_X4*(14*s2+7*s*(t_m+t_p)+8*t_m*t_p)-4*m_X2*s2*(2*s+t_m+t_p)+2*s2*(s+t_m)*(s+t_p)))/(np.sqrt(a+0j)*SQA*(2*m_X2-s)*(m_d2+2*m_X2-s-t_m+0j)**(3/2)*(m_d2+2*m_X2-s-t_p+0j)**(3/2))

    # -1j due to sqrt(a) = sqrt((-1)*(-a)) = i * sqrt(-a)
    term3_max = prefac_3 * (-np.log(t_m-t_p+0j))
    term3_min = prefac_3 * (np.log(m_d2+2*m_X2-s-t_p+0j) - np.log((t_m-t_p)*(-m_d2-2*m_X2+s+t_p)+0j))
    # term3 = prefac_3 * np.pi

    # log_part = 16/(SQA*np.sqrt(-a)) * np.pi
    log_part = -16/(SQA*np.sqrt(a+0j)) * (np.log(t_m-t_p+0j)-np.log(t_p-t_m+0j))

    if np.any(in_max_neq):
        a_xn = a[in_max_neq]
        t_max_xn = t_max[in_max_neq]
        t_m_xn = t_m[in_max_neq]
        t_p_xn = t_p[in_max_neq]
        s_xn = s[in_max_neq]
        s2_xn = s_xn*s_xn 
        # print('t_max - t_p', t_max_xn-t_p_xn)   # > 0
        # print('t_max - t_m', t_max_xn-t_m_xn)   # < 0

        term1_max[in_max_neq] = (8*(2*m_d2+m_X2)**2*np.sqrt(a_xn*(t_max_xn-t_m_xn)*(t_max_xn-t_p_xn))*(-2*m_d6+m_d4*(-6*m_X2+3*s_xn+2*(t_max_xn+t_m_xn+t_p_xn))-m_d2*(12*m_X4-4*m_X2*(3*s_xn+t_max_xn+t_m_xn+t_p_xn)+3*s2_xn+2*s_xn*(t_max_xn+t_m_xn+t_p_xn)+2*t_p_xn*(t_max_xn+t_m_xn)+2*t_max_xn*t_m_xn)-8*m_X6+4*m_X4*(3*s_xn+t_max_xn+t_m_xn+t_p_xn)-2*m_X2*(3*s2_xn+2*s_xn*(t_max_xn+t_m_xn+t_p_xn)+t_max_xn*(t_m_xn+t_p_xn)+t_m_xn*t_p_xn)+t_m_xn*t_p_xn*(s_xn+2*t_max_xn)+s_xn*(s_xn+t_max_xn)*(s_xn+t_m_xn)+s_xn*t_p_xn*(s_xn+t_max_xn)))/(a_xn*(m_d2-t_max_xn)*(m_d2-t_m_xn)*(m_d2-t_p_xn)*(m_d2+2*m_X2-s_xn-t_max_xn)*(m_d2+2*m_X2-s_xn-t_m_xn)*(m_d2+2*m_X2-s_xn-t_p_xn))

        term2_max[in_max_neq] = prefac_2[in_max_neq] * (np.log(m_d2-t_max_xn+0j)-np.log(m_d2*(2*t_max_xn-t_m_xn-t_p_xn)+2*np.sqrt((m_d2-t_m_xn)*(m_d2-t_p_xn)*(t_max_xn-t_m_xn)*(t_max_xn-t_p_xn)+0j)-t_max_xn*(t_m_xn+t_p_xn)+2*t_m_xn*t_p_xn+0j))

        term3_max[in_max_neq] = prefac_3[in_max_neq] * (np.log(m_d2+2*m_X2-s_xn-t_max_xn+0j)-np.log(2*np.sqrt(t_max_xn-t_m_xn+0j)*np.sqrt(m_d2+2*m_X2-s_xn-t_m_xn+0j)*np.sqrt((t_max_xn-t_p_xn+0j)*(m_d2+2*m_X2-s_xn-t_p_xn))+m_d2*(2*t_max_xn-t_m_xn-t_p_xn)+m_X2*(4*t_max_xn-2*(t_m_xn+t_p_xn))+t_p_xn*(s_xn-t_max_xn+2*t_m_xn)-2*s_xn*t_max_xn+s_xn*t_m_xn-t_max_xn*t_m_xn+0j))

    if np.any(in_min_neq):
        a_mn = a[in_min_neq]
        t_min_mn = t_min[in_min_neq]
        t_m_mn = t_m[in_min_neq]
        t_p_mn = t_p[in_min_neq]
        s_mn = s[in_min_neq]
        s2_mn = s_mn*s_mn 
        # print('t_min - t_p', t_min_mn-t_p_mn)   # > 0
        # print('t_min - t_m', t_min_mn-t_m_mn)   # < 0

        term1_min[in_min_neq] = (8*(2*m_d2+m_X2)**2*np.sqrt(a_mn*(t_min_mn-t_m_mn)*(t_min_mn-t_p_mn))*(-2*m_d6+m_d4*(-6*m_X2+3*s_mn+2*(t_min_mn+t_m_mn+t_p_mn))-m_d2*(12*m_X4-4*m_X2*(3*s_mn+t_min_mn+t_m_mn+t_p_mn)+3*s2_mn+2*s_mn*(t_min_mn+t_m_mn+t_p_mn)+2*t_p_mn*(t_min_mn+t_m_mn)+2*t_min_mn*t_m_mn)-8*m_X6+4*m_X4*(3*s_mn+t_min_mn+t_m_mn+t_p_mn)-2*m_X2*(3*s2_mn+2*s_mn*(t_min_mn+t_m_mn+t_p_mn)+t_min_mn*(t_m_mn+t_p_mn)+t_m_mn*t_p_mn)+t_m_mn*t_p_mn*(s_mn+2*t_min_mn)+s_mn*(s_mn+t_min_mn)*(s_mn+t_m_mn)+s_mn*t_p_mn*(s_mn+t_min_mn)))/(a_mn*(m_d2-t_min_mn)*(m_d2-t_m_mn)*(m_d2-t_p_mn)*(m_d2+2*m_X2-s_mn-t_min_mn)*(m_d2+2*m_X2-s_mn-t_m_mn)*(m_d2+2*m_X2-s_mn-t_p_mn))

        term2_min[in_min_neq] = prefac_2[in_min_neq] * (np.log(m_d2-t_min_mn+0j)-np.log(m_d2*(2*t_min_mn-t_m_mn-t_p_mn)+2*np.sqrt((m_d2-t_m_mn)*(m_d2-t_p_mn)*(t_min_mn-t_m_mn)*(t_min_mn-t_p_mn)+0j)-t_min_mn*(t_m_mn+t_p_mn)+2*t_m_mn*t_p_mn+0j))

        term3_min[in_min_neq] = prefac_3[in_min_neq] * (np.log(m_d2+2*m_X2-s_mn-t_min_mn+0j)-np.log(2*np.sqrt(t_min_mn-t_m_mn+0j)*np.sqrt(m_d2+2*m_X2-s_mn-t_m_mn+0j)*np.sqrt((t_min_mn-t_p_mn+0j)*(m_d2+2*m_X2-s_mn-t_p_mn))+m_d2*(2*t_min_mn-t_m_mn-t_p_mn)+m_X2*(4*t_min_mn-2*(t_m_mn+t_p_mn))+t_p_mn*(s_mn-t_min_mn+2*t_m_mn)-2*s_mn*t_min_mn+s_mn*t_m_mn-t_min_mn*t_m_mn+0j))

    if np.any(in_any_neq):
        log_part[in_any_neq] = ( -((16*np.log(2*np.sqrt((t_max[in_any_neq]-t_m[in_any_neq])*(t_max[in_any_neq]-t_p[in_any_neq])+0j)+2*t_max[in_any_neq]-t_m[in_any_neq]-t_p[in_any_neq]+0j))/(np.sqrt(a[in_any_neq]+0j)*SQA)) + ((16*np.log(2*np.sqrt((t_min[in_any_neq]-t_m[in_any_neq])*(t_min[in_any_neq]-t_p[in_any_neq])+0j)+2*t_min[in_any_neq]-t_m[in_any_neq]-t_p[in_any_neq]+0j))/(np.sqrt(a[in_any_neq]+0j)*SQA)) ).real

    int_tmax = term1_max + term2_max + term3_max
    int_tmin = term1_min + term2_min + term3_min

    return vert*(int_tmax - int_tmin + log_part).real

# @nb.jit(nopython=True, cache=True)
def ker_C_n_XX_dd_s(s, E1, E2, E3, p1, p3, m_d, m_X, m_h, s12_min, s12_max, s34_min, s34_max, vert, th, m_Gamma_h2):
    p12 = p1*p1
    p32 = p3*p3
    # Anton: a,b,c definition in a*cos^2 + b*cos + c = 0 in integrand
    a = np.fmin(-4.*p32*((E1+E2)*(E1+E2) - s), -1e-200)
    b = 2.*(p3/p1)*(s-2.*E1*(E1+E2))*(s-2.*E3*(E1+E2))
    sqrt_arg = 4.*(p32/p12)*(s-s12_min)*(s-s12_max)*(s-s34_min)*(s-s34_max)
    sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 0.))

    # Anton: ct_p, ct_m solutions of a*cos^2 + b*cos + c = 0 for cos. ct_m > ct_p
    ct_p = (-b + sqrt_fac)/(2.*a)
    ct_m = (-b - sqrt_fac)/(2.*a)
    # Anton: R_theta integration region {-1 <= cos <= 1 | ct_p <= cos <= ct_m}
    ct_min = np.fmin(np.fmax(-1., ct_p), 1.)
    ct_max = np.fmax(np.fmin(1., ct_m), ct_min)
    in_res = (ct_max > ct_min)

    # Anton: return zero for integral if it is not inside defined region.
    # x = [0,0,0], x[[True,False,True]] = [1,2,3] => x = [1,0,3]
    t_int = np.zeros(s.size)
    # print(in_res.size)
    # if np.count_nonzero(in_res) == 0: 
    #     return t_int
    # t_int[in_res] = ker_C_n_XX_dd_s_t_integral_new_3(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_d, m_X, vert)
    # t_int[in_res] = ker_C_n_XX_dd_s_t_integral_Higgs_new(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_d, m_X, m_h, vert, th, m_Gamma_h2)

    t_int[in_res] = ker_C_n_XX_dd_s_t_integral_revival(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_d, m_X, vert)

    return t_int

# 3 4 -> 1 2 <=> X X -> d d
# @nb.jit(nopython=True, cache=True)
def ker_C_n_XX_dd(x, m_d, m_X, m_h, k_d, k_X, T_d, xi_d, xi_X, vert, th, m_Gamma_h2):
    """
    Anton: Seems like E1 <--> E3, E2 <--> E4 compared to article.
    Set up for production of X. 
    """
    log_E3_min = log(m_X*offset)
    log_E3_max = log(max((max_exp_arg + xi_X)*T_d, 1e1*m_X))
    E3 = np.exp(np.fmin(log_E3_min * (1.-x[:,0]) + log_E3_max * x[:,0], 6e2))

    E4_min = np.fmax(2.*m_d-E3, m_X*offset)
    log_E4_min = np.log(E4_min)
    log_E4_max = np.log(np.fmax(1e1*E4_min, (max_exp_arg + xi_X)*T_d))
    E4 = np.exp(np.fmin(log_E4_min * (1.-x[:,1]) + log_E4_max * x[:,1], 6e2))

    log_E1_min = np.log(m_d*offset)
    log_E1_max = np.log(np.fmax(E3+E4-m_d, m_d*offset))
    E1 = np.exp(np.fmin(log_E1_min * (1.-x[:,2]) + log_E1_max * x[:,2], 6e2))
    E2 = E3 + E4 - E1

    exp_arg_1 = E1/T_d - xi_d
    exp_arg_2 = E2/T_d - xi_d
    exp_arg_3 = E3/T_d - xi_X
    exp_arg_4 = E4/T_d - xi_X
    exp_1 = np.exp(np.fmin(-exp_arg_1, max_exp_arg))
    exp_2 = np.exp(np.fmin(-exp_arg_2, max_exp_arg))
    exp_3 = np.exp(np.fmin(-exp_arg_3, max_exp_arg))
    exp_4 = np.exp(np.fmin(-exp_arg_4, max_exp_arg))
    f1 = exp_1/(1. + k_d*exp_1)
    f2 = exp_2/(1. + k_d*exp_2)
    f3 = exp_3/(1. + k_X*exp_3)
    f4 = exp_4/(1. + k_X*exp_4)
    # Anton: Assumed 1,2 in final state
    dist = f3*f4*(1.-k_d*f1)*(1.-k_d*f2)
    # dist = f1*f2*(1.-k_phi*f3)*(1.-k_phi*f4)

    # Anton: Three-momentum p^2 = E^2 - m^2 = (E - m)*(E + m)
    p1 = np.sqrt(np.fmax((E1-m_d)*(E1+m_d), 1e-200))
    p2 = np.sqrt(np.fmax((E2-m_d)*(E2+m_d), 1e-200))
    p3 = np.sqrt(np.fmax((E3-m_X)*(E3+m_X), 1e-200))
    p4 = np.sqrt(np.fmax((E4-m_X)*(E4+m_X), 1e-200))

    s12_min = np.fmax(2.*m_d*m_d+2.*E1*(E2-p1*p2/E1), 4.*m_d*m_d)
    s12_max = 2.*m_d*m_d+2.*E1*E2+2.*p1*p2
    s34_min = np.fmax(2.*m_X*m_X+2.*E3*(E4-p3*p4/E3), 4.*m_X*m_X)
    s34_max = 2.*m_X*m_X+2.*E3*E4+2.*p3*p4
    log_s_min = np.log(np.fmax(np.fmax(s12_min, s34_min), 1e-200))
    log_s_max = np.log(np.fmax(np.fmin(s12_max, s34_max), 1e-200))
    s = np.exp(np.fmin(log_s_min * (1.-x[:,3]) + log_s_max * x[:,3], 6e2))

    ker_s = ker_C_n_XX_dd_s(s, E1, E2, E3, p1, p3, m_d, m_X, m_h, s12_min, s12_max, s34_min, s34_max, vert, th, m_Gamma_h2)
    ker_s[~np.isfinite(ker_s)] = 0.0

    jac = E3*(log_E3_max-log_E3_min)*E4*(log_E4_max-log_E4_min)*E1*(log_E1_max-log_E1_min)*s*(log_s_max-log_s_min)
    res = jac*p3*dist*ker_s
    res[np.logical_not(np.isfinite(res))] = 0.
    return res

# type == -1: only X X -> d d, type == 0: both reactions, type == 1: only d d -> X X, type == 2: (X X -> d d, d d -> X X)
def C_n_XX_dd(m_d, m_X, m_h, k_d, k_X, T_d, xi_d, xi_X, vert, th, m_Gamma_h2, type=0):
    """
    Anton: 
    Collision operator C[X]_XX_dd for X, C[X]_XX_dd = -C[d]_XX_dd.
    1,2 = d,d, 3,4 = X,X
    
    dist = f3*f4*f1t*f2t
    type = -1, XX --> dd
    -f3*f4*f1t*f2t = -1*dist
    --> chem_eq_fac = -1

    type = 0, XX <--> dd
    f1*f2*f3t*f4t - f1t*f2t*f3*f4 = (exp(xi1 + xi2 - xi3 - x4) - 1)*dist
    --> chem_eq_fac = (exp(2*(xi_d - xi_X)) - 1)

    type = 1, dd --> XX
    f1*f2*f3t*f4t = exp(xi1 + xi - xi3 - xi4)*dist
    --> chem_eq_fac = exp(2*(xi_d - xi_X))

    and same procedure in the spin_stat_irr case. 
    """
    if m_X/T_d - xi_X > spin_stat_irr: # spin-statistics irrelevant here
        th_avg_s_v = th_avg_sigma_v_XX_dd(T_d, m_d, m_X, vert)
        if th_avg_s_v <= 0.:
            if type == 2:
                return np.array([0., 0.])
            return 0.
        if type == 0:
            chem_eq_fac = exp(2.*xi_d) - exp(2.*xi_X)
        elif type == -1:
            chem_eq_fac = -exp(2.*xi_X)
        elif type == 1:
            chem_eq_fac = exp(2.*xi_d)
        elif type == 2:
            return np.array([-exp(2.*xi_X), exp(2.*xi_d)])*th_avg_s_v
        return chem_eq_fac*th_avg_s_v

    # Anton: Factors to get correct 'dist' in ker_C_n_XX_dd for different types
    if type == 0:
        # (e^(2*xi_d-2*xi_X)-1)*fX*fX*fdt*fdt = fd*fd*fXt*fXt - fX*fX*fdt*fdt)
        # Gives C[X]_XX_dd for dd <--> XX
        chem_eq_fac = exp(2.*(xi_d-xi_X)) - 1.
    elif type == -1:
        chem_eq_fac = -1.       # Anton: -1 since C_n_XX_dd describes X, and X is in initial state 
    elif type == 1:
        chem_eq_fac = exp(2.*(xi_d-xi_X))

    # Send arrays in batches
    @vegas.batchintegrand
    def kernel(x):
        return ker_C_n_XX_dd(x, m_d, m_X, m_h, k_d, k_X, T_d, xi_d, xi_X, vert, th, m_Gamma_h2)

    """
    Anton: Order of integration in analytic expression: E1, E2, E3, s. 
    Implementation reads the order: E3, E4, E1, where E2 has been eliminated instead of E4.  
    Seems like a change of variables has been done, see inside ker_C_n_XX_dd function. 
    Seemingly, 

    x_i = ln(E_i / E_i_min) / ln(E_i_max / E_i_min) where E_i_min/max is lower/upper integration bound of E_i. 
    s' = ln(s / s_min) / ln(s_max / s_min) where s_min/max is lower/upper integration bound of s.

    Then {x_i, s' in [0, 1]}, and 
    jacobian = E1*(log_E1_max - log_E1_min)*E2*(log_E2_max - log_E2_min)*E3*(log_E3_max - log_E3_min)*s*(log_s_max - log_s_min)
    """

    # Anton: Monte-Carlo integration of the 4 integrals from 0 to 1 
    integ = vegas.Integrator(4 * [[0., 1.]])
    result = integ(kernel, nitn=10, neval=2e5)
    # print(result.summary())
    # if result.mean != 0.:
    #     print("Vegas error pp dd: ", result.sdev/fabs(result.mean), result.mean, result.Q)
    # print("pp dd", result.mean*chem_eq_fac/(256.*(pi**6.)), (exp(2.*xi_d)-exp(2.*xi_X))*th_avg_sigma_v_XX_dd(T_d, m_d, m_phi, vert))

    if type == 2:
        return np.array([-1., exp(2.*(xi_d-xi_X))])*result.mean/(256.*(pi**6.))
    
    return result.mean*chem_eq_fac/(256.*(pi**6.))


# Anton: NOT UPDATED NOR CALLED 
@nb.jit(nopython=True, cache=True)
def M2_XX_dd(s, t, m_d2, vert, m_X2):
    m_d4 = m_d2*m_d2
    m_d8 = m_d4*m_d4
    m_X4 = m_X2*m_X2

    t2 = t*t
    t3 = t*t2
    u = 2*m_d2 + 2*m_X2 - s - t
    u2 = u*u
    u3 = u*u2

    M2 = -(8*(6*m_d8-m_d4*(6*m_X4-4*m_X2*(t+u)+3*t2+14*t*u+3*u2)+m_d2*(6*m_X4*(t+u)-16*m_X2*t*u+t3+7*t2*u+7*t*u2+u3)+m_X4*(t2-8*t*u+u2)+4*m_X2*t*u*(t+u)-t*u*(t2+u2)))/((m_d2-t)**2*(m_d2-u)**2)

    return vert*M2 

# @nb.jit(nopython=True, cache=True)
def ker_C_34_12_s_t_integral(s, ct, a, ct_m, ct_p, E1, E3, p1, p3, m1, m2, m3, m4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub=False):
    m12 = m1*m1
    m32 = m3*m3
    t = m12 + m32 - 2.*E1*(E3 - p1*p3/E1*ct)
    # t_add = m12 + m32
    # t_m = t_add - 2.*E1*(E3 - p1*p3/E1*ct_m)
    # t_p = t_add - 2.*E1*(E3 - p1*p3/E1*ct_p)

    integrand = vector_mediator.M2_gen(s, t, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub=False) * 1/(np.sqrt(a*(ct - ct_m)*(ct - ct_p)))
    # print('integrand', integrand)
    # print()
    return integrand

# @nb.jit(nopython=True, cache=True)
def ker_C_34_12_s(s, E1, E2, E3, a, ct, p1, p2, p3, ct_min, ct_max, ct_m, ct_p, m1, m2, m3, m4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub):
    # t = (p1-p3)^2 = m1^2 + m3^2 - 2p1p3 = m1^2 + m3^2 - 2(E1E3 - p1p3ct)
    #   = (E1-E3, p1-p3) = E1^2 + E3^2 - 2E1*E3 - p1^2 - p3^2 + 2p1*p3*ct
    #   = (E1-p1)*(E1+p1) + (E3-p3)*(E3+p3) - 2*(E1*E3 - p1*p3*ct)
    # t = m1**2 + m3**2 - 2.*E1*(E3 - p1*p3/E1*ct)
    
    # d = t - m_X2
    # print("t min/max:", t.min(), t.max())
    # print("|t-mX2| min:", np.min(np.abs(d)))
    # print("fraction with |t-mX2| < 1e-6*mX2:", np.mean(np.abs(d) < 1e-6*m_X2))
    # print("fraction with |t| < mX2:", np.mean(np.abs(t) < m_X2))
    
    # Anton: New
    t_int = ker_C_34_12_s_t_integral(s, ct, a, ct_m, ct_p, E1, E3, p1, p3, m1, m2, m3, m4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub)

    # print('t_int', t_int)
    # print()
    return t_int

# @nb.jit(nopython=True, cache=True)
def ker_C_34_12(x, log_s_min, log_s_max, type, nFW, nBW, m1, m2, m3, m4, k1, k2, k3, k4, T1, T2, T3, T4, xi1, xi2, xi3, xi4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub, thermal_width):
    m12 = m1*m1
    m22 = m2*m2
    m32 = m3*m3
    m42 = m4*m4

    # Anton: Integration order is switched, now it is s, E1, E2, E3, t
    s = np.exp(np.fmin(log_s_min * (1.-x[:,0]) + log_s_max * x[:,0], 6e2))

    E1_min = m1
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*m1)
    log_E1_min = log(E1_min)
    log_E1_max = log(E1_max)
    E1 = np.exp(np.fmin(log_E1_min * (1.-x[:,1]) + log_E1_max * x[:,1], 6e2))
    # if E1 <= m1:
    #     return 0. # otherwise problems in computation (division by p1)
    p1 = np.sqrt(np.fmax((E1-m1)*(E1+m1), 1e-200))

    sqrt_fac_2 = np.sqrt(np.fmax(s*s-2.*(m12+m22)*s+((m1+m2)*(m1-m2))**2., 0.))
    E2_min = np.fmax((E1*(s-m12-m22)-p1*sqrt_fac_2)/(2.*m12), max(m2*offset, 1e-200))
    E2_max = (E1*(s-m12-m22)+p1*sqrt_fac_2)/(2.*m12)
    log_E2_min = np.log(E2_min)
    log_E2_max = np.log(E2_max)
    E2 = np.exp(log_E2_min * (1.-x[:,2]) + log_E2_max * x[:,2])
    p2 = np.sqrt(np.fmax((E2-m2)*(E2+m2), 1e-200))

    E12 = E1+E2
    E122 = E12*E12
    sqrt_fac_3 = np.sqrt(np.fmax((E122-s)*(s*s-2.*(m32+m42)*s+((m3+m4)*(m3-m4))**2.), 0.))
    E3_min = np.fmax((E12*(s+m32-m42)-sqrt_fac_3)/(2.*s), max(m3*offset, 1e-200))
    E3_max = (E12*(s+m32-m42)+sqrt_fac_3)/(2.*s)
    log_E3_min = np.log(E3_min)
    log_E3_max = np.log(E3_max)
    E3 = np.exp(np.fmin(log_E3_min * (1.-x[:,3]) + log_E3_max * x[:,3], 6e2))
    p3 = np.sqrt(np.fmax((E3-m3)*(E3+m3), 1e-200))

    E4 = E1 + E2 - E3
    p4 = np.sqrt(np.fmax((E4-m4)*(E4+m4), 1e-200))

    s12_min = m12 + m22 + 2.*E1*(E2 - p1*p2/E1)
    s12_max = m12 + m22 + 2.*E1*(E2 + p1*p2/E1)
    s34_min = m32 + m42 + 2.*E3*(E4 - p3*p4/E3)
    s34_max = m32 + m42 + 2.*E3*(E4 + p3*p4/E3)

    p12 = p1*p1
    p32 = p3*p3
    a = np.fmin(-4.*p32*((E1+E2)*(E1+E2) - s), -1e-200)
    b = 2.*(p3/p1)*(s-2.*E1*(E1+E2)+(m1-m2)*(m1+m2))*(s-2.*E3*(E1+E2)+(m3-m4)*(m3+m4))
    sqrt_arg = 4.*(p32/p12)*(s-s12_min)*(s-s12_max)*(s-s34_min)*(s-s34_max)
    sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 0.))
    ct_p = (-b+sqrt_fac)/(2.*a)
    ct_m = (-b-sqrt_fac)/(2.*a)
    ct_min = np.fmin(np.fmax(-1., ct_p), 1.)
    ct_max = np.fmax(np.fmin(1., ct_m), ct_min)
    
    ct = ct_min + (ct_max - ct_min)*x[:,4]

    exp_arg_1 = E1/T1 - xi1
    exp_arg_2 = E2/T2 - xi2
    exp_arg_3 = E3/T3 - xi3
    exp_arg_4 = E4/T4 - xi4
    exp_1 = np.exp(np.fmin(-exp_arg_1, max_exp_arg))
    exp_2 = np.exp(np.fmin(-exp_arg_2, max_exp_arg))
    exp_3 = np.exp(np.fmin(-exp_arg_3, max_exp_arg))
    exp_4 = np.exp(np.fmin(-exp_arg_4, max_exp_arg))
    f1 = exp_1/(1. + k1*exp_1)
    f2 = exp_2/(1. + k2*exp_2)
    f3 = exp_3/(1. + k3*exp_3)
    f4 = exp_4/(1. + k4*exp_4)
    dist_FW = nFW*f3*f4*(1.-k1*f1)*(1.-k2*f2)
    dist_BW = nBW*f1*f2*(1.-k3*f3)*(1.-k4*f4)

    if type == 0.:
        Etype = np.ones(E1.size)
    elif type == 1.:
        Etype = E1
    elif type == 2.:
        Etype = E2
    elif type == 3.:
        Etype = E3
    elif type == 4.:
        Etype = E4
    else:
        Etype = E12
    dist = Etype*(dist_FW+dist_BW)

    if thermal_width:
        m_X = sqrt(m_X2)
        sqrt_arg = (m_X2-4.*m3*m3)*((E1+E2)**2.-m_X2)
        sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 1e-200))
        E3p = 0.5*(E12+sqrt_fac/m_X)
        E3m = 0.5*(E12-sqrt_fac/m_X)
        exp_3p_xi = np.exp(np.fmin(xi3-E3p/T3, max_exp_arg))
        exp_3m_xi = np.exp(np.fmin(xi3-E3m/T3, max_exp_arg))
        E3_integral = sqrt_fac/(T3*m_X) + np.log((1.+exp_3p_xi)/(1.+exp_3m_xi))
        m_Gamma_X_T = sqrt(m_Gamma_X2)*(1.+m_X*T3*np.log((1.+exp_3p_xi)/(1.+exp_3m_xi))/sqrt_fac)
        m_Gamma_X_T2 = m_Gamma_X_T*m_Gamma_X_T

        m_h = sqrt(m_h2)
        sqrt_arg = (m_h2-4.*m3*m3)*((E1+E2)**2.-m_h2)
        sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 1e-200))
        E3p = 0.5*(E12+sqrt_fac/m_h)
        E3m = 0.5*(E12-sqrt_fac/m_h)
        exp_3p_xi = np.exp(np.fmin(xi3-E3p/T3, max_exp_arg))
        exp_3m_xi = np.exp(np.fmin(xi3-E3m/T3, max_exp_arg))
        E3_integral = sqrt_fac/(T3*m_h) + np.log((1.+exp_3p_xi)/(1.+exp_3m_xi))
        m_Gamma_h_T = sqrt(m_Gamma_X2)*(1.+m_h*T3*np.log((1.+exp_3p_xi)/(1.+exp_3m_xi))/sqrt_fac)
        m_Gamma_h_T2 = m_Gamma_h_T*m_Gamma_h_T
    else:
        m_Gamma_h_T2 = m_Gamma_h2*np.ones(s.size)
        m_Gamma_X_T2 = m_Gamma_X2*np.ones(s.size)

    ker_s = ker_C_34_12_s(s, E1, E2, E3, a, ct, p1, p2, p3, ct_min, ct_max, ct_m, ct_p, m1, m2, m3, m4, vert, m_d2, m_X2, m_h2, m_Gamma_X_T2, m_Gamma_h_T2, res_sub)

    jac = E1*(log_E1_max-log_E1_min)*E2*(log_E2_max-log_E2_min)*E3*(log_E3_max-log_E3_min)*s*(log_s_max-log_s_min)*(ct_max-ct_min)
    res = jac*p3*dist*ker_s
    res[np.logical_not(np.isfinite(res))] = 0.
    # print('res', res)
    # print()
    return res

# 3 4 -> 1 2 (all neutrinos); nFW (nBW): # of particle occurence in final-initial state for forward 3 4 -> 1 2 (backward 1 2 -> 3 4) reaction
# type indicates if for n (0) or rho (1 for E1, 2 for E2, 3 for E3, 4 for E4, 12 for E1+E2 = E3+E4)
# note that when using thermal width it is assumed that m3 = m4 = md, T3 = T4 = Td, xi3 = xi4 = xid, xi_phi = 2 xi_d
# and 3, 4 are fermions, phi is boson
# Anton: Added Monte-Carlo integral for t-integration
def C_34_12(type, nFW, nBW, m1, m2, m3, m4, k1, k2, k3, k4, T1, T2, T3, T4, xi1, xi2, xi3, xi4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub=False, thermal_width=True):
    # Anton: Integration order is now s, E1, E2, E3, t
    s_min = max((m1+m2)**2., (m3+m4)**2.)*offset # to prevent accuracy problems
    E1_max = max((max_exp_arg + xi1)*T1, 1e1*m1)
    p1_max = sqrt((E1_max-m1)*(E1_max+m1))
    E2_max = max((max_exp_arg + xi2)*T2, 1e1*m2)
    p2_max = sqrt((E2_max-m2)*(E2_max+m2))
    E3_max = max((max_exp_arg + xi3)*T3, 1e1*m3)
    p3_max = sqrt((E3_max-m3)*(E3_max+m3))
    E4_max = max((max_exp_arg + xi4)*T4, 1e1*m4)
    p4_max = sqrt((E4_max-m4)*(E4_max+m4))
    s12_max = m1*m1+m2*m2+2.*E1_max*E2_max+2.*p1_max*p2_max
    s34_max = m3*m3+m4*m4+2.*E3_max*E4_max+2.*p3_max*p4_max
    s_max = max(s12_max, s34_max)
    # Resonance splitting
    s_vals = np.sort(np.array([s_min, s_max, m_X2-fac_res_width*sqrt(m_Gamma_X2), m_X2, m_X2+fac_res_width*sqrt(m_Gamma_X2)]))
    s_vals = s_vals[s_vals >= s_min]
    s_vals = s_vals[s_vals <= s_max]

    np.seterr(divide='ignore')
    res = 0.
    integ = vegas.Integrator(5 * [[0., 1.]])
    for i in range(len(s_vals)-1):
        log_s_lo = log(s_vals[i])
        log_s_hi = log(s_vals[i+1])
        @vegas.batchintegrand
        def kernel(x):
            return ker_C_34_12(x, log_s_lo, log_s_hi, type, nFW, nBW, m1, m2, m3, m4, k1, k2, k3, k4, T1, T2, T3, T4, xi1, xi2, xi3, xi4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub, thermal_width)
        result = integ(kernel, nitn=10, neval=1e4)
        # print(result.summary())
        # if result.mean != 0.:
        #     print("Vegas error 34 12: ", result.sdev/fabs(result.mean), result.mean/(256.*(pi**6.)), result.Q)
        res += result.mean
        # print(result.mean, result.sdev, result.sdev / abs(result.mean), result.Q)
        # print('res2', res)
        # print()

    # @vegas.batchintegrand
    # def kernel(x):
    #     return ker_C_34_12(x, log(s_min), log(s_max), type, nFW, nBW, m1, m2, m3, m4, k1, k2, k3, k4, T1, T2, T3, T4, xi1, xi2, xi3, xi4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub, thermal_width)
    # integ = vegas.Integrator(5 * [[0., 1.]])
    # result = integ(kernel, nitn=10, neval=1e5)
    # res = result.mean
    np.seterr(divide='warn')
    # print("34 12:", res/(256.*(pi**6.)), (th_avg_sigma_v_33_11(m3, m4, m1, T1, vert, m_phi2, m_Gamma_phi2)*(nBW*exp(xi1+xi2))))

    # print('res_tot', res)
    # print()
    return res/(256.*(pi**6.))


# @nb.jit(nopython=True, cache=True)
def ker_C_n_11_22_s_t_integral_revival(ct_min, ct_max, ct_p, ct_m, a, s, E1, E3, p1, p3, m_d, m_X, vert):

    """
    Solved for matrix-element and integrated over variable t in collision
    operator using Mathematica, int_Rt |M|^2/sqrt((t-tm)*(t-tp)). 
    These expressions can be found in nu_s_nu_s_to_XX.nb
    """
    n = s.size
    s2 = s*s
    s3 = s2*s
    m_d2 = m_d*m_d
    m_d4 = m_d2*m_d2
    m_d6 = m_d2*m_d4
    m_d8 = m_d4*m_d4
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_X6 = m_X2*m_X4
    m_X8 = m_X4*m_X4

    # t_max,min - t_m,p = 2p1*p3*(ct_max - ct_m)
    t_add = m_d2 + m_X2
    t_min = t_add - 2.*E1*(E3 - p1*p3/E1*ct_min)
    t_max = t_add - 2.*E1*(E3 - p1*p3/E1*ct_max)
    t_m = t_add - 2.*E1*(E3 - p1*p3/E1*ct_m)
    t_p = t_add - 2.*E1*(E3 - p1*p3/E1*ct_p)
    
    """
    Anton: 
    x = t - tm, y = t - tp 
    sqrt(x)*sqrt(y) / sqrt(x*y) appears in expressions, makes trouble. 
    This is either 1 for (x>0 or y>0) and -1 for (x<0 and y<0).
    ill-defined for x->0, y->0 wrt. limit to 0+ or 0-.
    x>=0 & y>=0 -> 1
    x>=0 & y<=0 -> 1
    x<=0 & y>=0 -> 1
    x<=0 & y<=0 -> -1
    t_m > t_p always. 
    t_m >= t_max > t_min >= t_p 
    X_max = t_max - t_m, Y_max = t_max - t_p
    X_min = t_min - t_m, Y_min = t_min - t_p
    For all non-zero integration regions of ct_max, ct_min
    Y_max > 0 always => sqrt_fac_tmax = 1.
    Always have X_min < 0, and Y_min >= 0. For = case, 
    as ct_min --> c_p from above, Y_min --> 0+.
    Therefore, sqrt_fac_tmin = 1 as well.
    """    
    
    # in_min_neq = (ct_min != ct_p)
    # in_max_neq = (ct_max != ct_m)
    in_min_neq = ~np.isclose(ct_min, ct_p, rtol=0, atol=1e-12)
    in_max_neq = ~np.isclose(ct_max, ct_m, rtol=0, atol=1e-12)
    in_any_neq = np.logical_or(in_min_neq, in_max_neq)

    SQA = 1

    # print(t_max-t_m)
    # print(t_min-t_p)
    # print(t_min-t_m)

    term1_max = np.zeros(n) + 0j
    term1_min = np.zeros(n) + 0j

    prefac_2 = 1/(SQA*(-4*m_d2+2*m_X2+s)*(a*(m_X2-t_m)*(m_X2-t_p)+0j)**(3/2))*4*a*(32*m_d6*(2*m_X2-t_m-t_p)-8*m_d4*(10*m_X4+2*m_X2*(5*s-4*(t_m+t_p))-5*s*(t_m+t_p)+6*t_m*t_p)-4*m_d2*(2*m_X6-m_X4*(16*s+3*(t_m+t_p))+2*m_X2*(-4*s2+5*s*(t_m+t_p)+2*t_m*t_p)+4*s*(s*(t_m+t_p)-t_m*t_p))+4*m_X8+2*m_X6*(s-3*(t_m+t_p))-m_X4*(12*s2+7*s*(t_m+t_p)-8*t_m*t_p)+2*m_X2*s*(-2*s2+3*s*(t_m+t_p)+6*t_m*t_p)+2*s3*(t_m+t_p))

    # 1j due to a**3/2 = ((-1)*(-a))**3/2 = -i * (-a)**3/2
    term2_max = prefac_2 * (-np.log(t_m-t_p+0j))
    term2_min = prefac_2 * (-np.log(t_p-t_m+0j))
    # term2 = term2_max - term2_min
    # term2 = -prefac_2 * np.pi

    prefac_3 = -1/(np.sqrt(a+0j)*SQA*(4*m_d2-2*m_X2-s)*(-4*m_d2+m_X2+s+t_m+0j)**(3/2)*(-4*m_d2+m_X2+s+t_p+0j)**(3/2))*4*(1024*m_d8-32*m_d6*(10*m_X2+32*s+7*(t_m+t_p))+8*m_d4*(-18*m_X4+26*m_X2*s+48*s2+19*s*(t_m+t_p)+6*t_m*t_p)+4*m_d2*(14*m_X6+m_X4*(20*s+11*(t_m+t_p))+m_X2*(-12*s2+6*s*(t_m+t_p)+4*t_m*t_p)-4*s*(2*s+t_m)*(2*s+t_p))-4*m_X8-2*m_X6*(7*s+3*(t_m+t_p))-m_X4*(10*s2+15*s*(t_m+t_p)+8*t_m*t_p)+2*m_X2*s*(2*s2-3*s*(t_m+t_p)-6*t_m*t_p)+2*s3*(2*s+t_m+t_p)) 

    # -1j due to sqrt(a) = sqrt((-1)*(-a)) = i * sqrt(-a)
    term3_max = -prefac_3 * (np.log(4*m_d2-m_X2-s-t_m+0j)-np.log(-4*m_d2+m_X2+s+t_m+0j)+np.log(t_m-t_p))
    term3_min = -prefac_3 * np.log(t_m-t_p) 
    # term3 = prefac_3 * np.pi

    # log_part = 16/(SQA*np.sqrt(-a)) * np.pi
    log_part = 16/(SQA*np.sqrt(a+0j)) * (np.log(t_m-t_p+0j)-np.log(t_p-t_m+0j))

    if np.any(in_max_neq):
        a_xn = a[in_max_neq]
        t_max_xn = t_max[in_max_neq]
        t_m_xn = t_m[in_max_neq]
        t_p_xn = t_p[in_max_neq]
        s_xn = s[in_max_neq]
        s2_xn = s_xn*s_xn 
        # print('t_max - t_p', t_max_xn-t_p_xn)   # > 0
        # print('t_max - t_m', t_max_xn-t_m_xn)   # < 0

        term1_max[in_max_neq] = (8*(8*m_d4-8*m_d2*s_xn+m_X4+2*s_xn*(m_X2+s_xn))*np.sqrt(a_xn*(t_max_xn-t_m_xn)*(t_max_xn-t_p_xn))*(-64*m_d6+16*m_d4*(3*m_X2+3*s_xn+t_max_xn+t_m_xn+t_p_xn)-4*m_d2*(3*m_X4+2*m_X2*(3*s_xn+t_max_xn+t_m_xn+t_p_xn)+3*s2_xn+2*s_xn*(t_max_xn+t_m_xn+t_p_xn)+t_p_xn*(t_max_xn+t_m_xn)+t_max_xn*t_m_xn)+m_X4*(3*s_xn+2*(t_max_xn+t_m_xn+t_p_xn))+m_X2*s_xn*(3*s_xn+2*(t_max_xn+t_m_xn+t_p_xn))+t_m_xn*t_p_xn*(s_xn+2*t_max_xn)+s_xn*(s_xn+t_max_xn)*(s_xn+t_m_xn)+s_xn*t_p_xn*(s_xn+t_max_xn)))/(a_xn*(m_X2-t_max_xn)*(m_X2-t_m_xn)*(m_X2-t_p_xn)*(-4*m_d2+m_X2+s_xn+t_max_xn)*(-4*m_d2+m_X2+s_xn+t_m_xn)*(-4*m_d2+m_X2+s_xn+t_p_xn))

        term2_max[in_max_neq] = prefac_2[in_max_neq] * (np.log(m_X2-t_max_xn+0j)-np.log(m_X2*(2*t_max_xn-t_m_xn-t_p_xn)+2*np.sqrt((m_X2-t_m_xn)*(m_X2-t_p_xn)*(t_max_xn-t_m_xn)*(t_max_xn-t_p_xn)+0j)-t_max_xn*(t_m_xn+t_p_xn)+2*t_m_xn*t_p_xn+0j))

        term3_max[in_max_neq] = prefac_3[in_max_neq] * (np.log(-4*m_d2+m_X2+s_xn+t_max_xn+0j)-np.log(2*np.sqrt(t_max_xn-t_m_xn+0j)*np.sqrt(-4*m_d2+m_X2+s_xn+t_p_xn+0j)*np.sqrt((t_max_xn-t_p_xn)*(-4*m_d2+m_X2+s_xn+t_m_xn)+0j)+m_d2*(8*t_max_xn-4*(t_m_xn+t_p_xn))+m_X2*(-2*t_max_xn+t_m_xn+t_p_xn)-t_max_xn*(2*s_xn+t_m_xn+t_p_xn)+s_xn*t_m_xn+s_xn*t_p_xn+2*t_m_xn*t_p_xn+0j))

    if np.any(in_min_neq):
        a_mn = a[in_min_neq]
        t_min_mn = t_min[in_min_neq]
        t_m_mn = t_m[in_min_neq]
        t_p_mn = t_p[in_min_neq]
        s_mn = s[in_min_neq]
        s2_mn = s_mn*s_mn 
        # print('t_min - t_p', t_min_mn-t_p_mn)   # > 0
        # print('t_min - t_m', t_min_mn-t_m_mn)   # < 0

        term1_min[in_min_neq] = (8*(8*m_d4-8*m_d2*s_mn+m_X4+2*s_mn*(m_X2+s_mn))*np.sqrt(a_mn*(t_min_mn-t_m_mn)*(t_min_mn-t_p_mn))*(-64*m_d6+16*m_d4*(3*m_X2+3*s_mn+t_min_mn+t_m_mn+t_p_mn)-4*m_d2*(3*m_X4+2*m_X2*(3*s_mn+t_min_mn+t_m_mn+t_p_mn)+3*s2_mn+2*s_mn*(t_min_mn+t_m_mn+t_p_mn)+t_p_mn*(t_min_mn+t_m_mn)+t_min_mn*t_m_mn)+m_X4*(3*s_mn+2*(t_min_mn+t_m_mn+t_p_mn))+m_X2*s_mn*(3*s_mn+2*(t_min_mn+t_m_mn+t_p_mn))+t_m_mn*t_p_mn*(s_mn+2*t_min_mn)+s_mn*(s_mn+t_min_mn)*(s_mn+t_m_mn)+s_mn*t_p_mn*(s_mn+t_min_mn)))/(a_mn*(m_X2-t_min_mn)*(m_X2-t_m_mn)*(m_X2-t_p_mn)*(-4*m_d2+m_X2+s_mn+t_min_mn)*(-4*m_d2+m_X2+s_mn+t_m_mn)*(-4*m_d2+m_X2+s_mn+t_p_mn))

        term2_min[in_min_neq] = prefac_2[in_min_neq] * (np.log(m_X2-t_min_mn+0j)-np.log(m_X2*(2*t_min_mn-t_m_mn-t_p_mn)+2*np.sqrt((m_X2-t_m_mn)*(m_X2-t_p_mn)*(t_min_mn-t_m_mn)*(t_min_mn-t_p_mn)+0j)-t_min_mn*(t_m_mn+t_p_mn)+2*t_m_mn*t_p_mn+0j))

        term3_min[in_min_neq] = prefac_3[in_min_neq] * (np.log(-4*m_d2+m_X2+s_mn+t_min_mn+0j)-np.log(2*np.sqrt(t_min_mn-t_m_mn+0j)*np.sqrt(-4*m_d2+m_X2+s_mn+t_p_mn+0j)*np.sqrt((t_min_mn-t_p_mn)*(-4*m_d2+m_X2+s_mn+t_m_mn)+0j)+m_d2*(8*t_min_mn-4*(t_m_mn+t_p_mn))+m_X2*(-2*t_min_mn+t_m_mn+t_p_mn)-t_min_mn*(2*s_mn+t_m_mn+t_p_mn)+s_mn*t_m_mn+s_mn*t_p_mn+2*t_m_mn*t_p_mn+0j))

    if np.any(in_any_neq):
        log_part[in_any_neq] = ( (16*np.log(2*np.sqrt((t_max[in_any_neq]-t_m[in_any_neq])*(t_max[in_any_neq]-t_p[in_any_neq]))+2*t_max[in_any_neq]-t_m[in_any_neq]-t_p[in_any_neq]))/(np.sqrt(a[in_any_neq]+0j)*SQA) - (16*np.log(2*np.sqrt((t_min[in_any_neq]-t_m[in_any_neq])*(t_min[in_any_neq]-t_p[in_any_neq]))+2*t_min[in_any_neq]-t_m[in_any_neq]-t_p[in_any_neq]))/(np.sqrt(a[in_any_neq]+0j)*SQA)).real

    int_tmax = term1_max + term2_max + term3_max
    int_tmin = term1_min + term2_min + term3_min

    return vert*(int_tmax - int_tmin + log_part).real

# @nb.jit(nopython=True, cache=True)
def ker_C_n_11_22_s(s, E1, E2, E3, p1, p3, m_d, m_X, s12_min, s12_max, s34_min, s34_max, vert, th):
    p12 = p1*p1
    p32 = p3*p3
    # Anton: a,b,c definition in a*cos^2 + b*cos + c = 0 in integrand
    a = np.fmin(-4.*p32*((E1+E2)*(E1+E2) - s), -1e-200)
    b = 2.*(p3/p1)*(s-2.*E1*(E1+E2))*(s-2.*E3*(E1+E2))
    sqrt_arg = 4.*(p32/p12)*(s-s12_min)*(s-s12_max)*(s-s34_min)*(s-s34_max)
    sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 0.))

    # Anton: ct_p, ct_m solutions of a*cos^2 + b*cos + c = 0 for cos. ct_m > ct_p
    ct_p = (-b + sqrt_fac)/(2.*a)
    ct_m = (-b - sqrt_fac)/(2.*a)
    # Anton: R_theta integration region {-1 <= cos <= 1 | ct_p <= cos <= ct_m}
    ct_min = np.fmin(np.fmax(-1., ct_p), 1.)
    ct_max = np.fmax(np.fmin(1., ct_m), ct_min)
    in_res = (ct_max > ct_min)

    # Anton: return zero for integral if it is not inside defined region.
    # x = [0,0,0], x[[True,False,True]] = [1,2,3] => x = [1,0,3]
    t_int = np.zeros(s.size)
    # print(in_res.size)
    # if np.count_nonzero(in_res) == 0: 
    #     return t_int
    # t_int[in_res] = ker_C_n_XX_dd_s_t_integral_new_3(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_d, m_X, vert)
    # t_int[in_res] = ker_C_n_XX_dd_s_t_integral_Higgs_new(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_d, m_X, m_h, vert, th, m_Gamma_h2)

    t_int[in_res] = ker_C_n_11_22_s_t_integral_revival(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_d, m_X, vert)

    return t_int

# 3 4 -> 1 2 <=> X X -> d d
# @nb.jit(nopython=True, cache=True)
def ker_C_n_11_22(x, m_d, m_X, k_d, T_d, xi_d, vert, th):
    """
    Anton: Seems like E1 <--> E3, E2 <--> E4 compared to article.
    Set up for production of X. 
    """
    log_E3_min = log(m_d*offset)
    log_E3_max = log(max((max_exp_arg + xi_d)*T_d, 1e1*m_d))
    E3 = np.exp(np.fmin(log_E3_min * (1.-x[:,0]) + log_E3_max * x[:,0], 6e2))

    E4_min = np.fmax(2.*m_d-E3, m_d*offset)
    log_E4_min = np.log(E4_min)
    log_E4_max = np.log(np.fmax(1e1*E4_min, (max_exp_arg + xi_d)*T_d))
    E4 = np.exp(np.fmin(log_E4_min * (1.-x[:,1]) + log_E4_max * x[:,1], 6e2))

    log_E1_min = np.log(m_d*offset)
    log_E1_max = np.log(np.fmax(E3+E4-m_d, m_d*offset))
    E1 = np.exp(np.fmin(log_E1_min * (1.-x[:,2]) + log_E1_max * x[:,2], 6e2))
    E2 = E3 + E4 - E1

    exp_arg_1 = E1/T_d - xi_d
    exp_arg_2 = E2/T_d - xi_d
    exp_arg_3 = E3/T_d - xi_d
    exp_arg_4 = E4/T_d - xi_d
    exp_1 = np.exp(np.fmin(-exp_arg_1, max_exp_arg))
    exp_2 = np.exp(np.fmin(-exp_arg_2, max_exp_arg))
    exp_3 = np.exp(np.fmin(-exp_arg_3, max_exp_arg))
    exp_4 = np.exp(np.fmin(-exp_arg_4, max_exp_arg))
    f1 = exp_1/(1. + k_d*exp_1)
    f2 = exp_2/(1. + k_d*exp_2)
    f3 = exp_3/(1. + k_d*exp_3)
    f4 = exp_4/(1. + k_d*exp_4)
    # Anton: Assumed 1,2 in final state
    dist = f3*f4*(1.-k_d*f1)*(1.-k_d*f2)
    # dist = f1*f2*(1.-k_phi*f3)*(1.-k_phi*f4)

    # Anton: Three-momentum p^2 = E^2 - m^2 = (E - m)*(E + m)
    p1 = np.sqrt(np.fmax((E1-m_d)*(E1+m_d), 1e-200))
    p2 = np.sqrt(np.fmax((E2-m_d)*(E2+m_d), 1e-200))
    p3 = np.sqrt(np.fmax((E3-m_d)*(E3+m_d), 1e-200))
    p4 = np.sqrt(np.fmax((E4-m_d)*(E4+m_d), 1e-200))

    s12_min = np.fmax(2.*m_d*m_d+2.*E1*(E2-p1*p2/E1), 4.*m_d*m_d)
    s12_max = 2.*m_d*m_d+2.*E1*E2+2.*p1*p2
    s34_min = np.fmax(2.*m_d*m_d+2.*E3*(E4-p3*p4/E3), 4.*m_d*m_d)
    s34_max = 2.*m_d*m_d+2.*E3*E4+2.*p3*p4
    log_s_min = np.log(np.fmax(np.fmax(s12_min, s34_min), 1e-200))
    log_s_max = np.log(np.fmax(np.fmin(s12_max, s34_max), 1e-200))
    s = np.exp(np.fmin(log_s_min * (1.-x[:,3]) + log_s_max * x[:,3], 6e2))

    ker_s = ker_C_n_11_22_s(s, E1, E2, E3, p1, p3, m_d, m_X, s12_min, s12_max, s34_min, s34_max, vert, th)
    ker_s[~np.isfinite(ker_s)] = 0.0

    jac = E3*(log_E3_max-log_E3_min)*E4*(log_E4_max-log_E4_min)*E1*(log_E1_max-log_E1_min)*s*(log_s_max-log_s_min)
    res = jac*p3*dist*ker_s
    res[np.logical_not(np.isfinite(res))] = 0.
    return res

# type == -1: only X X -> d d, type == 0: both reactions, type == 1: only d d -> X X, type == 2: (X X -> d d, d d -> X X)
def C_n_11_22(m_d, m_X, k_d, T_d, xi_d, vert, th, type=0):
    # if m_X/T_d - xi_X > spin_stat_irr: # spin-statistics irrelevant here
    #     th_avg_s_v = th_avg_sigma_v_XX_dd(T_d, m_d, m_X, vert)
    #     if th_avg_s_v <= 0.:
    #         if type == 2:
    #             return np.array([0., 0.])
    #         return 0.
    #     if type == 0:
    #         chem_eq_fac = exp(2.*xi_d) - exp(2.*xi_X)
    #     elif type == -1:
    #         chem_eq_fac = -exp(2.*xi_X)
    #     elif type == 1:
    #         chem_eq_fac = exp(2.*xi_d)
    #     elif type == 2:
    #         return np.array([-exp(2.*xi_X), exp(2.*xi_d)])*th_avg_s_v
    #     return chem_eq_fac*th_avg_s_v

    # Anton: Factors to get correct 'dist' in ker_C_n_XX_dd for different types
    if type == 0:
        # (e^(2*xi_d-2*xi_X)-1)*fX*fX*fdt*fdt = fd*fd*fXt*fXt - fX*fX*fdt*fdt)
        # Gives C[X]_XX_dd for dd <--> XX
        chem_eq_fac = 0.
    elif type == -1:
        chem_eq_fac = -1.       # Anton: -1 since C_n_XX_dd describes X, and X is in initial state 
    elif type == 1:
        chem_eq_fac = 1.

    # Send arrays in batches
    @vegas.batchintegrand
    def kernel(x):
        return ker_C_n_11_22(x, m_d, m_X, k_d, T_d, xi_d, vert, th)

    """
    Anton: Order of integration in analytic expression: E1, E2, E3, s. 
    Implementation reads the order: E3, E4, E1, where E2 has been eliminated instead of E4.  
    Seems like a change of variables has been done, see inside ker_C_n_XX_dd function. 
    Seemingly, 

    x_i = ln(E_i / E_i_min) / ln(E_i_max / E_i_min) where E_i_min/max is lower/upper integration bound of E_i. 
    s' = ln(s / s_min) / ln(s_max / s_min) where s_min/max is lower/upper integration bound of s.

    Then {x_i, s' in [0, 1]}, and 
    jacobian = E1*(log_E1_max - log_E1_min)*E2*(log_E2_max - log_E2_min)*E3*(log_E3_max - log_E3_min)*s*(log_s_max - log_s_min)
    """

    # Anton: Monte-Carlo integration of the 4 integrals from 0 to 1 
    integ = vegas.Integrator(4 * [[0., 1.]])
    result = integ(kernel, nitn=10, neval=3e5)
    # print(result.summary())
    # if result.mean != 0.:
    #     print("Vegas error pp dd: ", result.sdev/fabs(result.mean), result.mean, result.Q)
    # print("pp dd", result.mean*chem_eq_fac/(256.*(pi**6.)), (exp(2.*xi_d)-exp(2.*xi_X))*th_avg_sigma_v_XX_dd(T_d, m_d, m_phi, vert))

    if type == 2:
        return np.array([-1., 1.])*result.mean/(256.*(pi**6.))
    
    return result.mean*chem_eq_fac/(256.*(pi**6.))


# @nb.jit(nopython=True, cache=True)
def ker_C_n_XX_pp_s_t(ct_min, ct_max, ct_p, ct_m, a, s, E1, E3, p1, p3, m_d, m_X, vert):

    """
    Solved for matrix-element and integrated over variable t in collision
    operator using Mathematica, int_Rt |M|^2/sqrt((t-tm)*(t-tp)). 
    These expressions can be found in nu_s_nu_s_to_XX.nb
    """
    n = s.size
    s2 = s*s
    m_d2 = m_d*m_d
    m_d4 = m_d2*m_d2
    m_d6 = m_d2*m_d4
    m_d8 = m_d4*m_d4
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_X6 = m_X2*m_X4
    m_X8 = m_X4*m_X4

    # t_max,min - t_m,p = 2p1*p3*(ct_max - ct_m)
    t_add = m_d2 + m_X2
    t_min = t_add - 2.*E1*(E3 - p1*p3/E1*ct_min)
    t_max = t_add - 2.*E1*(E3 - p1*p3/E1*ct_max)
    t_m = t_add - 2.*E1*(E3 - p1*p3/E1*ct_m)
    t_p = t_add - 2.*E1*(E3 - p1*p3/E1*ct_p)
    
    """
    Anton: 
    x = t - tm, y = t - tp 
    sqrt(x)*sqrt(y) / sqrt(x*y) appears in expressions, makes trouble. 
    This is either 1 for (x>0 or y>0) and -1 for (x<0 and y<0).
    ill-defined for x->0, y->0 wrt. limit to 0+ or 0-.
    x>=0 & y>=0 -> 1
    x>=0 & y<=0 -> 1
    x<=0 & y>=0 -> 1
    x<=0 & y<=0 -> -1
    t_m > t_p always. 
    t_m >= t_max > t_min >= t_p 
    X_max = t_max - t_m, Y_max = t_max - t_p
    X_min = t_min - t_m, Y_min = t_min - t_p
    For all non-zero integration regions of ct_max, ct_min
    Y_max > 0 always => sqrt_fac_tmax = 1.
    Always have X_min < 0, and Y_min >= 0. For = case, 
    as ct_min --> c_p from above, Y_min --> 0+.
    Therefore, sqrt_fac_tmin = 1 as well.
    """    
    
    M2 = vert*(8 + (s-2*m_X2)**2/m_X4)
    # integral 1/sqrt(c*(x-a)*(x-b)) given c < 0 and a < x < b is 
    # = 1/sqrt(-a) * arcsinh((2x-a-b) / (b-a)) 
    # c = a, a = t_p, b = t_m

    int_max = M2 * 1/np.sqrt(-a) * np.arcsinh((2*t_max-t_p-t_m)/(t_m-t_p))
    int_min = M2 * 1/np.sqrt(-a) * np.arcsinh((2*t_min-t_p-t_m)/(t_m-t_p))

    return (int_max - int_min).real

# @nb.jit(nopython=True, cache=True)
def ker_C_n_XX_pp_s(s, E1, E2, E3, p1, p3, m_phi, m_X, m_h, s12_min, s12_max, s34_min, s34_max, vert, th, m_Gamma_h2):
    p12 = p1*p1
    p32 = p3*p3
    # Anton: a,b,c definition in a*cos^2 + b*cos + c = 0 in integrand
    a = np.fmin(-4.*p32*((E1+E2)*(E1+E2) - s), -1e-200)
    b = 2.*(p3/p1)*(s-2.*E1*(E1+E2))*(s-2.*E3*(E1+E2))
    sqrt_arg = 4.*(p32/p12)*(s-s12_min)*(s-s12_max)*(s-s34_min)*(s-s34_max)
    sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 0.))

    # Anton: ct_p, ct_m solutions of a*cos^2 + b*cos + c = 0 for cos. ct_m > ct_p
    ct_p = (-b + sqrt_fac)/(2.*a)
    ct_m = (-b - sqrt_fac)/(2.*a)
    # Anton: R_theta integration region {-1 <= cos <= 1 | ct_p <= cos <= ct_m}
    ct_min = np.fmin(np.fmax(-1., ct_p), 1.)
    ct_max = np.fmax(np.fmin(1., ct_m), ct_min)
    in_res = (ct_max > ct_min)

    # Anton: return zero for integral if it is not inside defined region.
    # x = [0,0,0], x[[True,False,True]] = [1,2,3] => x = [1,0,3]
    t_int = np.zeros(s.size)
    # print(in_res.size)
    # if np.count_nonzero(in_res) == 0: 
    #     return t_int
    # t_int[in_res] = ker_C_n_XX_dd_s_t_integral_new_3(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_phi, m_X, vert)
    # t_int[in_res] = ker_C_n_XX_dd_s_t_integral_Higgs_new(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_phi, m_X, m_h, vert, th, m_Gamma_h2)

    t_int[in_res] = ker_C_n_XX_pp_s_t(ct_min[in_res], ct_max[in_res], ct_p[in_res], ct_m[in_res], a[in_res], s[in_res], E1[in_res], E3[in_res], p1[in_res], p3[in_res], m_phi, m_X, vert)

    return t_int

# 3 4 -> 1 2 <=> X X -> d d
# @nb.jit(nopython=True, cache=True)
def ker_C_n_XX_pp(x, m_phi, m_X, m_h, k_phi, k_X, T_d, xi_phi, xi_X, vert, th, m_Gamma_h2):
    """
    Anton: Seems like E1 <--> E3, E2 <--> E4 compared to article.
    Set up for production of X. 
    """
    log_E3_min = log(m_X*offset)
    log_E3_max = log(max((max_exp_arg + xi_X)*T_d, 1e1*m_X))
    E3 = np.exp(np.fmin(log_E3_min * (1.-x[:,0]) + log_E3_max * x[:,0], 6e2))

    E4_min = np.fmax(2.*m_phi-E3, m_X*offset)
    log_E4_min = np.log(E4_min)
    log_E4_max = np.log(np.fmax(1e1*E4_min, (max_exp_arg + xi_X)*T_d))
    E4 = np.exp(np.fmin(log_E4_min * (1.-x[:,1]) + log_E4_max * x[:,1], 6e2))

    log_E1_min = np.log(m_phi*offset)
    log_E1_max = np.log(np.fmax(E3+E4-m_phi, m_phi*offset))
    E1 = np.exp(np.fmin(log_E1_min * (1.-x[:,2]) + log_E1_max * x[:,2], 6e2))
    E2 = E3 + E4 - E1

    exp_arg_1 = E1/T_d - xi_phi
    exp_arg_2 = E2/T_d - xi_phi
    exp_arg_3 = E3/T_d - xi_X
    exp_arg_4 = E4/T_d - xi_X
    exp_1 = np.exp(np.fmin(-exp_arg_1, max_exp_arg))
    exp_2 = np.exp(np.fmin(-exp_arg_2, max_exp_arg))
    exp_3 = np.exp(np.fmin(-exp_arg_3, max_exp_arg))
    exp_4 = np.exp(np.fmin(-exp_arg_4, max_exp_arg))
    f1 = exp_1/(1. + k_phi*exp_1)
    f2 = exp_2/(1. + k_phi*exp_2)
    f3 = exp_3/(1. + k_X*exp_3)
    f4 = exp_4/(1. + k_X*exp_4)
    # Anton: Assumed 1,2 in final state
    dist = f3*f4*(1.-k_phi*f1)*(1.-k_phi*f2)
    # dist = f1*f2*(1.-k_phi*f3)*(1.-k_phi*f4)

    # Anton: Three-momentum p^2 = E^2 - m^2 = (E - m)*(E + m)
    p1 = np.sqrt(np.fmax((E1-m_phi)*(E1+m_phi), 1e-200))
    p2 = np.sqrt(np.fmax((E2-m_phi)*(E2+m_phi), 1e-200))
    p3 = np.sqrt(np.fmax((E3-m_X)*(E3+m_X), 1e-200))
    p4 = np.sqrt(np.fmax((E4-m_X)*(E4+m_X), 1e-200))

    s12_min = np.fmax(2.*m_phi*m_phi+2.*E1*(E2-p1*p2/E1), 4.*m_phi*m_phi)
    s12_max = 2.*m_phi*m_phi+2.*E1*E2+2.*p1*p2
    s34_min = np.fmax(2.*m_X*m_X+2.*E3*(E4-p3*p4/E3), 4.*m_X*m_X)
    s34_max = 2.*m_X*m_X+2.*E3*E4+2.*p3*p4
    log_s_min = np.log(np.fmax(np.fmax(s12_min, s34_min), 1e-200))
    log_s_max = np.log(np.fmax(np.fmin(s12_max, s34_max), 1e-200))
    s = np.exp(np.fmin(log_s_min * (1.-x[:,3]) + log_s_max * x[:,3], 6e2))

    ker_s = ker_C_n_XX_pp_s(s, E1, E2, E3, p1, p3, m_phi, m_X, m_h, s12_min, s12_max, s34_min, s34_max, vert, th, m_Gamma_h2)
    ker_s[~np.isfinite(ker_s)] = 0.0

    jac = E3*(log_E3_max-log_E3_min)*E4*(log_E4_max-log_E4_min)*E1*(log_E1_max-log_E1_min)*s*(log_s_max-log_s_min)
    res = jac*p3*dist*ker_s
    res[np.logical_not(np.isfinite(res))] = 0.
    return res

# type == -1: only X X -> d d, type == 0: both reactions, type == 1: only d d -> X X, type == 2: (X X -> d d, d d -> X X)
def C_n_XX_pp(m_phi, m_X, m_h, k_phi, k_X, T_d, xi_phi, xi_X, vert, th, m_Gamma_h2, type=0):
    """
    Anton: 
    Collision operator C[X]_XX_pp for X, C[X]_XX_pp = -C[phi]_XX_pp.
    1,2 = phi,phi, 3,4 = X,X
    
    dist = f3*f4*f1t*f2t
    type = -1, XX --> pp
    -f3*f4*f1t*f2t = -1*dist
    --> chem_eq_fac = -1

    type = 0, XX <--> dd
    f1*f2*f3t*f4t - f1t*f2t*f3*f4 = (exp(xi1 + xi2 - xi3 - x4) - 1)*dist
    --> chem_eq_fac = (exp(2*(xi_d - xi_X)) - 1)

    type = 1, dd --> XX
    f1*f2*f3t*f4t = exp(xi1 + xi - xi3 - xi4)*dist
    --> chem_eq_fac = exp(2*(xi_p - xi_X))

    and same procedure in the spin_stat_irr case. 
    """
    if m_X/T_d - xi_X > spin_stat_irr: # spin-statistics irrelevant here
        th_avg_s_v = th_avg_sigma_v_XX_dd(T_d, m_d, m_X, vert)
        if th_avg_s_v <= 0.:
            if type == 2:
                return np.array([0., 0.])
            return 0.
        if type == 0:
            chem_eq_fac = exp(2.*xi_d) - exp(2.*xi_X)
        elif type == -1:
            chem_eq_fac = -exp(2.*xi_X)
        elif type == 1:
            chem_eq_fac = exp(2.*xi_d)
        elif type == 2:
            return np.array([-exp(2.*xi_X), exp(2.*xi_d)])*th_avg_s_v
        return chem_eq_fac*th_avg_s_v

    # Anton: Factors to get correct 'dist' in ker_C_n_XX_dd for different types
    if type == 0:
        # (e^(2*xi_d-2*xi_X)-1)*fX*fX*fdt*fdt = fd*fd*fXt*fXt - fX*fX*fdt*fdt)
        # Gives C[X]_XX_dd for dd <--> XX
        chem_eq_fac = exp(2.*(xi_phi-xi_X)) - 1.
    elif type == -1:
        chem_eq_fac = -1.       # Anton: -1 since C_n_XX_pp describes X, and X is in initial state 
    elif type == 1:
        chem_eq_fac = exp(2.*(xi_phi-xi_X))

    # Send arrays in batches
    @vegas.batchintegrand
    def kernel(x):
        return ker_C_n_XX_pp(x, m_d, m_X, m_h, k_phi, k_X, T_d, xi_phi, xi_X, vert, th, m_Gamma_h2)

    """
    Anton: Order of integration in analytic expression: E1, E2, E3, s. 
    Implementation reads the order: E3, E4, E1, where E2 has been eliminated instead of E4.  
    Seems like a change of variables has been done, see inside ker_C_n_XX_dd function. 
    Seemingly, 

    x_i = ln(E_i / E_i_min) / ln(E_i_max / E_i_min) where E_i_min/max is lower/upper integration bound of E_i. 
    s' = ln(s / s_min) / ln(s_max / s_min) where s_min/max is lower/upper integration bound of s.

    Then {x_i, s' in [0, 1]}, and 
    jacobian = E1*(log_E1_max - log_E1_min)*E2*(log_E2_max - log_E2_min)*E3*(log_E3_max - log_E3_min)*s*(log_s_max - log_s_min)
    """

    # Anton: Monte-Carlo integration of the 4 integrals from 0 to 1 
    integ = vegas.Integrator(4 * [[0., 1.]])
    result = integ(kernel, nitn=10, neval=1e4)
    # print(result.summary())
    # if result.mean != 0.:
    #     print("Vegas error pp dd: ", result.sdev/fabs(result.mean), result.mean, result.Q)
    # print("pp dd", result.mean*chem_eq_fac/(256.*(pi**6.)), (exp(2.*xi_d)-exp(2.*xi_X))*th_avg_sigma_v_XX_dd(T_d, m_d, m_phi, vert))

    if type == 2:
        return np.array([-1., exp(2.*(xi_phi-xi_X))])*result.mean/(256.*(pi**6.))
    
    return result.mean*chem_eq_fac/(256.*(pi**6.))


#############################################################################
###############    CROSS-SECTIONS + THERMAL AVERAGES   ######################
#############################################################################

# X X -> d d
# Anton: NOT UPDATED
@nb.jit(nopython=True, cache=True)
def sigma_XX_dd(s, m_d, m_X, vert):
    """
    Anton: Since sigma ~ int d(cos(theta)) |M|^2 for 2 to 2 process, we must integrate |M|^2 analytically. 
    Switch integration to t = m_d^2 + m_phi^2 - 2E1*E3 + 2p1*p3*cos(theta), d(cos(theta)) = 1/(2*p1*p3)dt
    Since sigma is Lorentz invariant, calculate in CM-frame
    t = (p1-p3)^2 = (E1cm - E3cm)^2 - (p1cm - p3cm)^2
      = (E1cm - E3cm)^2 - (p1cm^2 + p3cm^2 - 2*p1cm*p3cm*cos(theta))
    This gives upper and lower bounds (cos(theta)=1, cos(theta)=-1)
    t_upper = (E1cm - E3cm)^2 - (p1cm - p3cm)^2 = (E1cm-E3cm + (p1cm-p3cm))*(E1cm-E3cm - (p1cm-p3cm))
    t_lower = (E1cm - E3cm)^2 - (p1cm + p3cm)^2 = (E1cm-E3cm + (p1cm+p3cm))*(E1cm-E3cm - (p1cm+p3cm))
    s = (p1/3 + p2/4)^2 = (E1/3cm + E2/4cm)^2 
    sqrt(s) = E1/3cm + E2/4cm
    Trick: E2/4^2 = E1/3^2 - m1/3^2 + m2/4^2
    => (sqrt(s) - E1/3cm)^2 = E1/3cm^2 - m1/3^2 + m2/4^2
    => E1/3cm = (s + m1/3^2 - m2/4^2) / (2*sqrt(s))
    which would also give momentum 
    p1/3cm = sqrt(E1/3cm^2 - m1/3^2) = 1/(2*sqrt(s))*sqrt([s - (m1/3 + m2/4)^2]^2 - 4*m1/3^2*m2/4^2)
    for integration bounds. 
    Two heavysides - one from integration of phase-space H(E_cm - m3 - m4), one from demanding p1/2cm positive: 
    H(1/(4*s)*{[s - (m1 + m2)]^2 - 4*m1^2*m2^2}) = H([s - (m1 + m2)^2]^2 - 4*m1^2*m2^2)
    = H(s - m1 - m2 - 2*m1*m2) = H(s - (m1 + m2)^2) = H(E_cm - m1 - m2)
    Cross-section:
    sigma = H(E_cm - m3 - m4)*H(E_cm - m1 - m2)/(64*pi*p1cm^2) 
          * int_{t_lower}^{t_upper} dt |M|^2
    Note: This function can be vectorized, but is not needed. 
          Use np.vectorize(sigma_XX_dd)(s, m_d, m_X, vert) instead if array output is wanted.
    """
    m_d2 = m_d*m_d
    m_X2 = m_X*m_X
    m_d4 = m_d2*m_d2
    m_X4 = m_X2*m_X2

    # Anton: Heavyside-functions
    if s < 4*m_d**2 or s < 4*m_X**2:
        return 0. 

    s2 = s*s

    # Anton: Three-momenta in CM-frame 
    p1cm = np.sqrt(0.25*s - m_d2)
    p3cm = np.sqrt(0.25*s - m_X2)

    # Anton: Upper and lower integration bound 
    # Anton: Add imaginary unit to avoid trouble with log etc
    t_upper = -(p1cm - p3cm)**2 + 0j
    t_lower = -(p1cm + p3cm)**2 + 0j

    # Anton: t-integrated squared matrix elements
    # Anton: imaginary parts from upper - lower will cancel
    
    int_t_M2_upper = 8*vert*((2*m_d2+m_X2)**2/(-m_d2-2*m_X2+s+t_upper) + (2*m_d2+m_X2)**2/(t_upper-m_d2) + ((-8*m_d4+4*m_d2*(s-2*m_X2)+4*m_X4+s2)*(np.log(t_upper-m_d2)-np.log(-m_d2-2*m_X2+s+t_upper)))/(2*m_X2-s) - 2*t_upper)

    int_t_M2_lower = 8*vert*((2*m_d2+m_X2)**2/(-m_d2-2*m_X2+s+t_lower) + (2*m_d2+m_X2)**2/(t_lower-m_d2) + ((-8*m_d4+4*m_d2*(s-2*m_X2)+4*m_X4+s2)*(np.log(t_lower-m_d2)-np.log(-m_d2-2*m_X2+s+t_lower)))/(2*m_X2-s) - 2*t_lower)

    sigma = ((int_t_M2_upper - int_t_M2_lower).real / (64.*np.pi*s*p1cm*p1cm))
    # Anton: divide by symmetry factor 2 for identical particles in phase space integral
    return sigma / 2

# X X -> d d
# Anton: Removed longitudinal component by hand 
@nb.jit(nopython=True, cache=True)
def sigma_XX_dd_new(s, m_d, m_X, vert):
    """
    Anton: Since sigma ~ int d(cos(theta)) |M|^2 for 2 to 2 process, we must integrate |M|^2 analytically. 
    Switch integration to t = m_d^2 + m_phi^2 - 2E1*E3 + 2p1*p3*cos(theta), d(cos(theta)) = 1/(2*p1*p3)dt
    Since sigma is Lorentz invariant, calculate in CM-frame
    t = (p1-p3)^2 = (E1cm - E3cm)^2 - (p1cm - p3cm)^2
      = (E1cm - E3cm)^2 - (p1cm^2 + p3cm^2 - 2*p1cm*p3cm*cos(theta))
    This gives upper and lower bounds (cos(theta)=1, cos(theta)=-1)
    t_upper = (E1cm - E3cm)^2 - (p1cm - p3cm)^2 = (E1cm-E3cm + (p1cm-p3cm))*(E1cm-E3cm - (p1cm-p3cm))
    t_lower = (E1cm - E3cm)^2 - (p1cm + p3cm)^2 = (E1cm-E3cm + (p1cm+p3cm))*(E1cm-E3cm - (p1cm+p3cm))
    s = (p1/3 + p2/4)^2 = (E1/3cm + E2/4cm)^2 
    sqrt(s) = E1/3cm + E2/4cm
    Trick: E2/4^2 = E1/3^2 - m1/3^2 + m2/4^2
    => (sqrt(s) - E1/3cm)^2 = E1/3cm^2 - m1/3^2 + m2/4^2
    => E1/3cm = (s + m1/3^2 - m2/4^2) / (2*sqrt(s))
    which would also give momentum 
    p1/3cm = sqrt(E1/3cm^2 - m1/3^2) = 1/(2*sqrt(s))*sqrt([s - (m1/3 + m2/4)^2]^2 - 4*m1/3^2*m2/4^2)
    for integration bounds. 
    Two heavysides - one from integration of phase-space H(E_cm - m3 - m4), one from demanding p1/2cm positive: 
    H(1/(4*s)*{[s - (m1 + m2)]^2 - 4*m1^2*m2^2}) = H([s - (m1 + m2)^2]^2 - 4*m1^2*m2^2)
    = H(s - m1 - m2 - 2*m1*m2) = H(s - (m1 + m2)^2) = H(E_cm - m1 - m2)
    Cross-section:
    sigma = H(E_cm - m3 - m4)*H(E_cm - m1 - m2)/(64*pi*p1cm^2) 
          * int_{t_lower}^{t_upper} dt |M|^2
    Note: This function can be vectorized, but is not needed. 
          Use np.vectorize(sigma_XX_dd)(s, m_d, m_X, vert) instead if array output is wanted.
    """
    m_d2 = m_d*m_d
    m_d4 = m_d2*m_d2
    m_d6 = m_d2*m_d4
    m_d8 = m_d4*m_d4
    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_X6 = m_X2*m_X4
    m_X8 = m_X4*m_X4

    m_h = 3*m_X
    m_h2 = m_h*m_h
    m_h4 = m_h2*m_h2

    s2 = s*s
    s3 = s*s2
    # Anton: Heavyside-functions
    if s < 4*m_d**2 or s < 4*m_X**2:
        return 0. 

    # Anton: Three-momenta in CM-frame 
    p1cm = np.sqrt(0.25*s - m_d2)
    p3cm = np.sqrt(0.25*s - m_X2)

    # Anton: Upper and lower integration bound 
    # Anton: Add imaginary unit to avoid trouble with log etc
    t_upper = -(p1cm - p3cm)**2 + 0j
    t_lower = -(p1cm + p3cm)**2 + 0j

    # Anton: Polarization sum set to zero 
    # int_t_M2_upper = 8*vert*((m_X2-6*m_d2)**2/(-m_d2-2*m_X2+s+t_upper)+(m_X2-6*m_d2)**2/(t_upper-m_d2)+((24*m_d4+m_d2*(12*s-40*m_X2)+4*m_X4+s2)*(np.log(t_upper-m_d2)-np.log(-m_d2-2*m_X2+s+t_upper)))/(2*m_X2-s)-2*t_upper)

    # int_t_M2_lower = 8*vert*((m_X2-6*m_d2)**2/(-m_d2-2*m_X2+s+t_lower)+(m_X2-6*m_d2)**2/(t_lower-m_d2)+((24*m_d4+m_d2*(12*s-40*m_X2)+4*m_X4+s2)*(np.log(t_lower-m_d2)-np.log(-m_d2-2*m_X2+s+t_lower)))/(2*m_X2-s)-2*t_lower)

    # Longitudinal removed by hand  
    int_t_M2_upper = 8*vert*((m_X2-4*m_d2)**2/(-m_d2-2*m_X2+s+t_upper)-(m_X2-4*m_d2)**2/(m_d2-t_upper)+((4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*(np.log(m_d2-t_upper)-np.log(m_d2+2*m_X2-s-t_upper)))/(2*m_X6-m_X4*s)-2*t_upper)

    int_t_M2_lower = 8*vert*((m_X2-4*m_d2)**2/(-m_d2-2*m_X2+s+t_lower)-(m_X2-4*m_d2)**2/(m_d2-t_lower)+((4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*(np.log(m_d2-t_lower)-np.log(m_d2+2*m_X2-s-t_lower)))/(2*m_X6-m_X4*s)-2*t_lower)

    # With longitudinal 
    # int_t_M2_upper = 8*vert*((m_X2-4*m_d2)**2/(-m_d2-2*m_X2+s+t_upper)-(m_X2-4*m_d2)**2/(m_d2-t_upper)+(4*m_d2*t_upper*(s-4*m_X2))/m_X4+((4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*(np.log(m_d2-t_upper)-np.log(m_d2+2*m_X2-s-t_upper)))/(2*m_X6-m_X4*s)-2*t_upper)

    # int_t_M2_lower = 8*vert*((m_X2-4*m_d2)**2/(-m_d2-2*m_X2+s+t_lower)-(m_X2-4*m_d2)**2/(m_d2-t_lower)+(4*m_d2*t_lower*(s-4*m_X2))/m_X4+((4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*(np.log(m_d2-t_lower)-np.log(m_d2+2*m_X2-s-t_lower)))/(2*m_X6-m_X4*s)-2*t_lower)

    sigma = ((int_t_M2_upper - int_t_M2_lower).real / (64.*np.pi*s*p1cm*p1cm))
    # Anton: divide by symmetry factor 2 for identical particles in phase space integral
    return sigma / 2

# Anton: Added Higgs instead of removing longitudinal by hand
@nb.jit(nopython=True, cache=True)
def sigma_XX_dd_Higgs(s, m_d, m_X, m_h, vert, th, m_Gamma_h2):
    """
    Anton: Since sigma ~ int d(cos(theta)) |M|^2 for 2 to 2 process, we must integrate |M|^2 analytically. 
    Switch integration to t = m_d^2 + m_phi^2 - 2E1*E3 + 2p1*p3*cos(theta), d(cos(theta)) = 1/(2*p1*p3)dt
    Since sigma is Lorentz invariant, calculate in CM-frame
    t = (p1-p3)^2 = (E1cm - E3cm)^2 - (p1cm - p3cm)^2
      = (E1cm - E3cm)^2 - (p1cm^2 + p3cm^2 - 2*p1cm*p3cm*cos(theta))
    This gives upper and lower bounds (cos(theta)=1, cos(theta)=-1)
    t_upper = (E1cm - E3cm)^2 - (p1cm - p3cm)^2 = (E1cm-E3cm + (p1cm-p3cm))*(E1cm-E3cm - (p1cm-p3cm))
    t_lower = (E1cm - E3cm)^2 - (p1cm + p3cm)^2 = (E1cm-E3cm + (p1cm+p3cm))*(E1cm-E3cm - (p1cm+p3cm))
    s = (p1/3 + p2/4)^2 = (E1/3cm + E2/4cm)^2 
    sqrt(s) = E1/3cm + E2/4cm
    Trick: E2/4^2 = E1/3^2 - m1/3^2 + m2/4^2
    => (sqrt(s) - E1/3cm)^2 = E1/3cm^2 - m1/3^2 + m2/4^2
    => E1/3cm = (s + m1/3^2 - m2/4^2) / (2*sqrt(s))
    which would also give momentum 
    p1/3cm = sqrt(E1/3cm^2 - m1/3^2) = 1/(2*sqrt(s))*sqrt([s - (m1/3 + m2/4)^2]^2 - 4*m1/3^2*m2/4^2)
    for integration bounds. 
    Two heavysides - one from integration of phase-space H(E_cm - m3 - m4), one from demanding p1/2cm positive: 
    H(1/(4*s)*{[s - (m1 + m2)]^2 - 4*m1^2*m2^2}) = H([s - (m1 + m2)^2]^2 - 4*m1^2*m2^2)
    = H(s - m1 - m2 - 2*m1*m2) = H(s - (m1 + m2)^2) = H(E_cm - m1 - m2)
    Cross-section:
    sigma = H(E_cm - m3 - m4)*H(E_cm - m1 - m2)/(64*pi*s*p1cm^2) 
          * int_{t_lower}^{t_upper} dt |M|^2
    Note: This function can be vectorized, but is not needed. 
          Use np.vectorize(sigma_XX_dd)(s, m_d, m_X, vert) instead if array output is wanted.
    """
    m_d2 = m_d*m_d
    m_d3 = m_d*m_d2
    m_d4 = m_d2*m_d2

    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_X6 = m_X2*m_X4
    m_X8 = m_X4*m_X4

    gss = np.cos(th)**2
    gss2 = gss*gss

    m_h2 = m_h*m_h
    m_h4 = m_h2*m_h2

    # off-shell propagators in PVS-scheme 
    # Can cause problem with e.g. negativity of cross-section
    hprop = (s-m_h2)/((s-m_h2)**2 + m_Gamma_h2)
    hprop2 = ((s-m_h2)**2-m_Gamma_h2)/((s-m_h2)**2 + m_Gamma_h2)**2

    # hprop = 1j / (s-m_h2)
    # hprop2 = 1 / (s-m_h2)**2

    # off-shell propagators in CUT-scheme with top-hat cut 
    # delta = 1e3*np.sqrt(m_Gamma_h2)
    # x = s-m_h2
    # hprop = (1 -(x < delta)*(x > -delta)) * 1 / ((s-m_h2 + 1j*m_Gamma_h2))
    # hprop2 = hprop*hprop.conjugate()

    s2 = s*s
    # Anton: Heavyside-functions
    if s < 4*m_d**2 or s < 4*m_X**2:
        return 0. 

    # Anton: Three-momenta in CM-frame 
    p1cm = np.sqrt(0.25*s - m_d2)
    p3cm = np.sqrt(0.25*s - m_X2)

    # Anton: Upper and lower integration bound 
    # Anton: Add imaginary unit to avoid trouble with log etc
    t_upper = -(p1cm - p3cm)**2 + 0j
    t_lower = -(p1cm + p3cm)**2 + 0j

    # Added Higgs instead of removing longitudinal by hand 

    # int_t_M2_upper = 8*vert*(-((2*t_upper*(gss2*((m_h2-s)**2+m_Gamma_h2)*(m_d2*(8*m_X2-2*s)+m_X4)-4*gss*m_d2*(m_h2-s)*(4*m_X4-2*m_X2*s+s2)+2*m_d2*(4*m_d2-s)*(12*m_X4-4*m_X2*s+s2)))/(gss2*m_X4*((m_h2-s)**2+m_Gamma_h2)))+((np.log(t_upper-m_d2)-np.log(-m_d2-2*m_X2+s+t_upper))*(gss*((m_h2-s)**2+m_Gamma_h2)*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))-8*m_d2*(m_h2-s)*(s-2*m_X2)*(m_d2*(8*m_X4-4*m_X2*s+s2)-2*m_X6)))/(gss*m_X4*(2*m_X2-s)*((m_h2-s)**2+m_Gamma_h2))+(m_X2-4*m_d2)**2/(-m_d2-2*m_X2+s+t_upper)-(m_X2-4*m_d2)**2/(m_d2-t_upper))

    # int_t_M2_lower = 8*vert*(-((2*t_lower*(gss2*((m_h2-s)**2+m_Gamma_h2)*(m_d2*(8*m_X2-2*s)+m_X4)-4*gss*m_d2*(m_h2-s)*(4*m_X4-2*m_X2*s+s2)+2*m_d2*(4*m_d2-s)*(12*m_X4-4*m_X2*s+s2)))/(gss2*m_X4*((m_h2-s)**2+m_Gamma_h2)))+((np.log(t_lower-m_d2)-np.log(-m_d2-2*m_X2+s+t_lower))*(gss*((m_h2-s)**2+m_Gamma_h2)*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))-8*m_d2*(m_h2-s)*(s-2*m_X2)*(m_d2*(8*m_X4-4*m_X2*s+s2)-2*m_X6)))/(gss*m_X4*(2*m_X2-s)*((m_h2-s)**2+m_Gamma_h2))+(m_X2-4*m_d2)**2/(-m_d2-2*m_X2+s+t_lower)-(m_X2-4*m_d2)**2/(m_d2-t_lower))

    int_t_M2_upper = 8*vert*(-((4*m_d2*t_upper*(gss2*(4*m_X2-s)+2*gss*hprop*(4*m_X4-2*m_X2*s+s2)+hprop2*(4*m_d2-s)*(12*m_X4-4*m_X2*s+s2)))/(gss2*m_X4))+(1/(gss*m_X4*(2*m_X2-s)))*(gss*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*np.log(m_d2-t_upper)-gss*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*np.log(m_d2+2*m_X2-s-t_upper)+8*hprop*m_d2*(2*m_X2-s)*(m_d2*(8*m_X4-4*m_X2*s+s2)-2*m_X6)*(np.log(-m_d2-2*m_X2+s+t_upper)-np.log(t_upper-m_d2)))-(m_X2-4*m_d2)**2/(m_d2+2*m_X2-s-t_upper)-(m_X2-4*m_d2)**2/(m_d2-t_upper)-2*t_upper)

    int_t_M2_lower = 8*vert*(-((4*m_d2*t_lower*(gss2*(4*m_X2-s)+2*gss*hprop*(4*m_X4-2*m_X2*s+s2)+hprop2*(4*m_d2-s)*(12*m_X4-4*m_X2*s+s2)))/(gss2*m_X4))+(1/(gss*m_X4*(2*m_X2-s)))*(gss*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*np.log(m_d2-t_lower)-gss*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*np.log(m_d2+2*m_X2-s-t_lower)+8*hprop*m_d2*(2*m_X2-s)*(m_d2*(8*m_X4-4*m_X2*s+s2)-2*m_X6)*(np.log(-m_d2-2*m_X2+s+t_lower)-np.log(t_lower-m_d2)))-(m_X2-4*m_d2)**2/(m_d2+2*m_X2-s-t_lower)-(m_X2-4*m_d2)**2/(m_d2-t_lower)-2*t_lower)
    
    # int_t_M2_upper = 8*vert*((1/((2*m_X2-s)))*((4*m_d2*(-4*m_X2-3*s)+(4*m_X4+s2))*np.log(m_d2-t_upper)-(4*m_d2*(-4*m_X2-3*s)+(4*m_X4+s2))*np.log(m_d2+2*m_X2-s-t_upper))-(m_X2-4*m_d2)**2/(m_d2+2*m_X2-s-t_upper)-(m_X2-4*m_d2)**2/(m_d2-t_upper)-2*t_upper)

    # int_t_M2_lower = 8*vert*((1/((2*m_X2-s)))*((4*m_d2*(-4*m_X2-3*s)+(4*m_X4+s2))*np.log(m_d2-t_lower)-(4*m_d2*(-4*m_X2-3*s)+(4*m_X4+s2))*np.log(m_d2+2*m_X2-s-t_lower))-(m_X2-4*m_d2)**2/(m_d2+2*m_X2-s-t_lower)-(m_X2-4*m_d2)**2/(m_d2-t_lower)-2*t_lower)

    # Remove longitudinal by hand in addition to Higgs 
    # int_t_M2_upper = -((8*vert*(gss2*((m_X2-6*m_d2)**2/(m_d2+2*m_X2-s-t_upper)+(m_X2-6*m_d2)**2/(m_d2-t_upper)+2*t_upper)+32*gss*hprop*m_d2*t_upper-(gss*(gss*(24*m_d4+m_d2*(12*s-40*m_X2)+4*m_X4+s2)-16*hprop*m_d2*(2*m_X2-s)*(6*m_d2-m_X2-s))*(np.log(t_upper-m_d2)-np.log(-m_d2-2*m_X2+s+t_upper)))/(2*m_X2-s)+64*hprop2*m_d2*t_upper*(4*m_d2-s)))/gss2)

    # int_t_M2_lower = -((8*vert*(gss2*((m_X2-6*m_d2)**2/(m_d2+2*m_X2-s-t_lower)+(m_X2-6*m_d2)**2/(m_d2-t_lower)+2*t_lower)+32*gss*hprop*m_d2*t_lower-(gss*(gss*(24*m_d4+m_d2*(12*s-40*m_X2)+4*m_X4+s2)-16*hprop*m_d2*(2*m_X2-s)*(6*m_d2-m_X2-s))*(np.log(t_lower-m_d2)-np.log(-m_d2-2*m_X2+s+t_lower)))/(2*m_X2-s)+64*hprop2*m_d2*t_lower*(4*m_d2-s)))/gss2)

    # Switch sign on Higgs diagram 
    # int_t_M2_upper = 8*vert*(-((4*m_d2*t_upper*(gss2*(4*m_X2-s)-2*gss*hprop*(4*m_X4-2*m_X2*s+s2)+hprop2*(4*m_d2-s)*(12*m_X4-4*m_X2*s+s2)))/(gss2*m_X4))+(1/(gss*m_X4*(2*m_X2-s)))*(gss*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*np.log(m_d2-t_upper)-gss*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*np.log(m_d2+2*m_X2-s-t_upper)+8*hprop*m_d2*(2*m_X2-s)*(m_d2*(8*m_X4-4*m_X2*s+s2)-2*m_X6)*(np.log(t_upper-m_d2)-np.log(-m_d2-2*m_X2+s+t_upper)))-(m_X2-4*m_d2)**2/(m_d2+2*m_X2-s-t_upper)-(m_X2-4*m_d2)**2/(m_d2-t_upper)-2*t_upper)

    # int_t_M2_lower = 8*vert*(-((4*m_d2*t_lower*(gss2*(4*m_X2-s)-2*gss*hprop*(4*m_X4-2*m_X2*s+s2)+hprop2*(4*m_d2-s)*(12*m_X4-4*m_X2*s+s2)))/(gss2*m_X4))+(1/(gss*m_X4*(2*m_X2-s)))*(gss*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*np.log(m_d2-t_lower)-gss*(4*m_d4*s*(4*m_X2-s)+4*m_d2*m_X2*(-4*m_X4-3*m_X2*s+s2)+m_X4*(4*m_X4+s2))*np.log(m_d2+2*m_X2-s-t_lower)+8*hprop*m_d2*(2*m_X2-s)*(m_d2*(8*m_X4-4*m_X2*s+s2)-2*m_X6)*(np.log(t_lower-m_d2)-np.log(-m_d2-2*m_X2+s+t_lower)))-(m_X2-4*m_d2)**2/(m_d2+2*m_X2-s-t_lower)-(m_X2-4*m_d2)**2/(m_d2-t_lower)-2*t_lower)

    sigma = ((int_t_M2_upper - int_t_M2_lower).real / (64.*np.pi*s*p1cm*p1cm))
    # Anton: divide by symmetry factor 2 for identical particles in phase space integral
    return sigma / 2

# Anton: Added Higgs instead of removing longitudinal by hand
@nb.jit(nopython=True, cache=True)
def sigma_Xh_dd_Higgs(s, m_d, m_X, m_h, vert, th, m_Gamma_X2):
    """
    Anton: Since sigma ~ int d(cos(theta)) |M|^2 for 2 to 2 process, we must integrate |M|^2 analytically. 
    Switch integration to t = m_d^2 + m_phi^2 - 2E1*E3 + 2p1*p3*cos(theta), d(cos(theta)) = 1/(2*p1*p3)dt
    Since sigma is Lorentz invariant, calculate in CM-frame
    t = (p1-p3)^2 = (E1cm - E3cm)^2 - (p1cm - p3cm)^2
      = (E1cm - E3cm)^2 - (p1cm^2 + p3cm^2 - 2*p1cm*p3cm*cos(theta))
    This gives upper and lower bounds (cos(theta)=1, cos(theta)=-1)
    t_upper = (E1cm - E3cm)^2 - (p1cm - p3cm)^2 = (E1cm-E3cm + (p1cm-p3cm))*(E1cm-E3cm - (p1cm-p3cm))
    t_lower = (E1cm - E3cm)^2 - (p1cm + p3cm)^2 = (E1cm-E3cm + (p1cm+p3cm))*(E1cm-E3cm - (p1cm+p3cm))
    s = (p1/3 + p2/4)^2 = (E1/3cm + E2/4cm)^2 
    sqrt(s) = E1/3cm + E2/4cm
    Trick: E2/4^2 = E1/3^2 - m1/3^2 + m2/4^2
    => (sqrt(s) - E1/3cm)^2 = E1/3cm^2 - m1/3^2 + m2/4^2
    => E1/3cm = (s + m1/3^2 - m2/4^2) / (2*sqrt(s))
    which would also give momentum 
    p1/3cm = sqrt(E1/3cm^2 - m1/3^2) = 1/(2*sqrt(s))*sqrt([s - (m1/3 + m2/4)^2]^2 - 4*m1/3^2*m2/4^2)
    for integration bounds. 
    Two heavysides - one from integration of phase-space H(E_cm - m3 - m4), one from demanding p1/2cm positive: 
    H(1/(4*s)*{[s - (m1 + m2)]^2 - 4*m1^2*m2^2}) = H([s - (m1 + m2)^2]^2 - 4*m1^2*m2^2)
    = H(s - m1 - m2 - 2*m1*m2) = H(s - (m1 + m2)^2) = H(E_cm - m1 - m2)
    Cross-section:
    sigma = H(E_cm - m3 - m4)*H(E_cm - m1 - m2)/(64*pi*p1cm^2) 
          * int_{t_lower}^{t_upper} dt |M|^2
    Note: This function can be vectorized, but is not needed. 
          Use np.vectorize(sigma_XX_dd)(s, m_d, m_X, vert) instead if array output is wanted.
    """
    m_d2 = m_d*m_d
    m_d4 = m_d2*m_d2
    m_d6 = m_d2*m_d4

    m_X2 = m_X*m_X
    m_X4 = m_X2*m_X2
    m_X6 = m_X2*m_X4

    gss = np.cos(th)**2
    gss2 = gss*gss

    m_h2 = m_h*m_h
    m_h4 = m_h2*m_h2
    m_h6 = m_h2*m_h4

    # off-shell propagators in PVS-scheme 
    # Can cause problem with e.g. negativity of cross-section
    sprop = (s-m_X2)/((s-m_X2)**2 + m_Gamma_X2)
    sprop2 = ((s-m_X2)**2-m_Gamma_X2)/((s-m_X2)**2 + m_Gamma_X2)**2

    # off-shell propagators in CUT-scheme with top-hat cut 
    # delta = 1e3*np.sqrt(m_Gamma_h2)
    # x = s-m_h2
    # hprop = (1 -(x < delta)*(x > -delta)) * 1 / ((s-m_h2 + 1j*m_Gamma_h2))
    # hprop2 = hprop*hprop.conjugate()

    s2 = s*s
    s3 = s*s2
    # Anton: Heavyside-functions
    if s < 4*m_d**2 or s < (m_X+m_h)**2:
        return 0. 

    """
    E1cm = (s + m1*m1 - m2*m2) / (2*np.sqrt(s))
    E3cm = (s + m3*m3 - m4*m4) / (2*np.sqrt(s))
    p1cm = np.sqrt((E1cm - m1)*(E1cm + m1))
    p3cm = np.sqrt((E3cm - m3)*(E3cm + m3))

    E13diff = (m1*m1 - m2*m2 - m3*m3 + m4*m4) / (2*np.sqrt(s))
    t_upper = (E13diff + (p1cm - p3cm))*(E13diff - (p1cm - p3cm))
    t_lower = (E13diff + (p1cm + p3cm))*(E13diff - (p1cm + p3cm))
    """

    # Anton: Make upper and lower integration bounds 
    E1cm = np.sqrt(s)/2
    E3cm = (s + m_X2 - m_h2) / (2*np.sqrt(s))
    # Anton: Three-momenta in CM-frame 
    p1cm = np.sqrt((E1cm - m_d)*(E1cm + m_d))
    p3cm = np.sqrt((E3cm - m_X)*(E3cm + m_X))

    # Anton: Upper and lower integration bound 
    # Anton: Add imaginary unit to avoid trouble with log etc
    E13diff = (m_h2 - m_X2) / (2*np.sqrt(s))
    t_upper = (E13diff + (p1cm - p3cm))*(E13diff - (p1cm - p3cm)) + 0j
    t_lower = (E13diff + (p1cm + p3cm))*(E13diff - (p1cm + p3cm)) + 0j

    # Added Higgs instead of removing longitudinal by hand 

    int_t_M2_upper = (1/(gss2*m_X4))*8*vert*((2*gss2*m_d2*m_X2*(m_h2-4*m_d2)*(4*m_d2-m_X2))/(m_d2-t_upper)+t_upper*(gss2*m_d2*(-4*m_d2-2*m_X2+s)+4*gss*m_d2*sprop*(m_h2*(m_X2+s)+2*m_d2*m_X2-3*m_X4-s2)-4*sprop2*(m_h4*m_d2*(2*m_X2-s)+m_h2*(m_d2*(-3*m_X4-2*m_X2*s+2*s2)+m_X6)+m_d4*m_X4+m_d2*(11*m_X6-6*m_X4*s+4*m_X2*s2-s3)-2*m_X6*s))+2*m_X2*sprop*t_upper**2*(m_X2*sprop*(m_h2+2*m_d2+m_X2-s)-2*gss*m_d2)+2*gss*m_d2*np.log(t_upper-m_d2)*(gss*m_X2*(m_h2-8*m_d2+m_X2-s)+4*sprop*(m_h4*(-m_d2)+m_h2*(2*m_d2*(m_X2+s)+m_X4-m_X2*s)-m_d2*(9*m_X4-2*m_X2*s+s2)+m_X4*(m_X2+s)))-4/3*m_X4*sprop2*t_upper**3)

    int_t_M2_lower = (1/(gss2*m_X4))*8*vert*((2*gss2*m_d2*m_X2*(m_h2-4*m_d2)*(4*m_d2-m_X2))/(m_d2-t_lower)+t_lower*(gss2*m_d2*(-4*m_d2-2*m_X2+s)+4*gss*m_d2*sprop*(m_h2*(m_X2+s)+2*m_d2*m_X2-3*m_X4-s2)-4*sprop2*(m_h4*m_d2*(2*m_X2-s)+m_h2*(m_d2*(-3*m_X4-2*m_X2*s+2*s2)+m_X6)+m_d4*m_X4+m_d2*(11*m_X6-6*m_X4*s+4*m_X2*s2-s3)-2*m_X6*s))+2*m_X2*sprop*t_lower**2*(m_X2*sprop*(m_h2+2*m_d2+m_X2-s)-2*gss*m_d2)+2*gss*m_d2*np.log(t_lower-m_d2)*(gss*m_X2*(m_h2-8*m_d2+m_X2-s)+4*sprop*(m_h4*(-m_d2)+m_h2*(2*m_d2*(m_X2+s)+m_X4-m_X2*s)-m_d2*(9*m_X4-2*m_X2*s+s2)+m_X4*(m_X2+s)))-4/3*m_X4*sprop2*t_lower**3)

    sigma = ((int_t_M2_upper - int_t_M2_lower).real / (64.*np.pi*s*p1cm*p1cm))
    # Anton: divide by symmetry factor 2 for identical particles in phase space integral
    return sigma

# # # Thermal averages # # # 

def ker_th_avg_sigma_v_XX_dd(log_s, T_d, m_d, m_X, vert):
    s = exp(log_s)
    sqrt_s = sqrt(s)    
    sigma = sigma_XX_dd_Higgs(s, m_d, m_X, vert)
    return s*sigma*(s-4.*m_X*m_X)*sqrt_s*kn(1, sqrt_s/T_d)

# only \int d^3 p3 d^3 p4 sigma v exp(-(E3+E4)/T)/(2 pi)^6
def th_avg_sigma_v_XX_dd(T_d, m_d, m_X, vert):
    s_min = max(4.*m_d*m_d, 4.*m_X*m_X)
    s_max = (5e2*T_d)**2.
    if s_max <= s_min:
        return 0.

    res, err = quad(ker_th_avg_sigma_v_XX_dd, log(s_min), log(s_max), args=(T_d, m_d, m_X, vert), epsabs=0., epsrel=rtol_int)

    return res*T_d/(32.*(pi**4.))
    # return res/(8.*(m_phi**4.)*T_d*(kn(2, m_phi/T_d)**2.))

def ker_th_avg_sigma_v_33_11(log_s, m1, m2, m3, T, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, res_sub):
    s = exp(log_s)
    sqrt_s = sqrt(s)
    sigma = vector_mediator.sigma_gen_new(s, m1, m2, m3, m3, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, sub=False)
    # print(log_s, s*sigma*(s-4.*m1*m1)*sqrt_s*kn(1, sqrt_s/T3))
    return s*sigma*(s-4.*m3*m3)*sqrt_s*kn(1, sqrt_s/T)

# only \int d^3 p3 d^3 p4 sigma v exp(-(E3+E4)/T)/(2 pi)^6
def th_avg_sigma_v_33_11(m1, m2, m3, T, vert, m_X2, m_Gamma_X2, res_sub=True):
    s_min = max((m1+m2)*(m1+m2), 4.*m3*m3)*offset
    s_max = (1e3*T)**2.
    if s_max <= s_min:
        return 0.
    s_vals = np.sort(np.array([s_min, s_max, m_X2-fac_res_width*sqrt(m_Gamma_X2), m_X2, m_X2+fac_res_width*sqrt(m_Gamma_X2)]))
    s_vals = s_vals[s_vals >= s_min]
    s_vals = s_vals[s_vals <= s_max]

    res = 0.
    for i in range(len(s_vals)-1):
        cur_res, err = quad(ker_th_avg_sigma_v_33_11, log(s_vals[i]), log(s_vals[i+1]), args=(m1, m2, m3, T, vert, m_X2, m_Gamma_X2, res_sub), epsabs=0., epsrel=rtol_int, limit=100)
        res += cur_res

    return res*T/(32.*(pi**4.))

# Using Maxwell-Boltzmann approx. 
def ker_th_avg_sigma_v_22_11(log_s, m1, m2, T, vert, m_X):
    s = exp(log_s)
    sqrt_s = sqrt(s)
    sigma = vector_mediator.sigma_22_11(s, m1, m2, m_X, vert)
    # sigma = vert/s
    # print(log_s, s*sigma*(s-4.*m1*m1)*sqrt_s*kn(1, sqrt_s/T3))
    # Note: extra factor s from jacobian in s -> log(s)
    return s*sigma*(s-4*m1*m1)*sqrt_s*kn(1, sqrt_s/T)

def th_avg_sigma_v_22_11(m1, m2, T, vert, m_X, m_Gamma_X2, naive=False):
    s_min = max((m1+m1)**2, (m2+m2)**2)*offset
    s_max = (1e3*T)**2.
    m_X2 = m_X*m_X
    if s_max <= s_min:
        return 0.
    s_vals = np.sort(np.array([s_min, s_max, m_X2-fac_res_width*sqrt(m_Gamma_X2), m_X2, m_X2+fac_res_width*sqrt(m_Gamma_X2)]))
    s_vals = s_vals[s_vals >= s_min]
    s_vals = s_vals[s_vals <= s_max]

    # res = 0.
    # for i in range(len(s_vals)-1):
    #     cur_res, err = quad(ker_th_avg_sigma_v_22_11, log(s_vals[i]), log(s_vals[i+1]), args=(m1, m2, T, vert, m_X), epsabs=0., epsrel=rtol_int, limit=100)
    #     res += cur_res

    res, err = quad(ker_th_avg_sigma_v_22_11, log(s_min), log(s_max), args=(m1, m2, T, vert, m_X), epsabs=0., epsrel=rtol_int, limit=100)
    if naive: 
        return vert/T**2
    return 1/(8*m1**2*m2**2*T*kn(2,m1/T)*kn(2,m2/T))*res

########################################################
if __name__ == '__main__':
    # Anton: Mostly for debugging 
    import matplotlib.pyplot as plt
    import time
    import pandemolator as pandemolator
    import constants_functions as cf

    m_d = 1e-5          # 1e-6*M GeV = M keV, 2e-5 GeV = 20 keV
    m_nu = 0 
    m_N1, m_N2 = m_d, m_d
    m_X = 2.5*m_d       # MeV-scale dark photon
    sin2_2th = 2e-14
    y = 5.57e-4
    th = 0.5*np.arcsin(np.sqrt(sin2_2th))

    vert = y**4
    # vert = 1

    m_d2 = m_d*m_d
    m_X2 = m_X*m_X
    Gamma_X = vector_mediator.Gamma_X_new(y, m_X, m_N1, m_N2, m_nu, sin2_2th)
    m_Gamma_X2 = m_X2*Gamma_X*Gamma_X

    # Anton: Calculate SM neutrino temperature
    Ttrel = pandemolator.TimeTempRelation()
    H = Ttrel.hubble_grid
    T_d_dw = cf.T_d_dw(m_d) # temperature of maximal d production by Dodelson-Widrow mechanism
    i_ic = np.argmax(Ttrel.T_nu_grid < T_d_dw)      # Anton: Start when T_nu < T_dw just occured 
    i_end = np.argmax(Ttrel.T_nu_grid < m_d/2e1)    # Anton: End when T_nu < 20/m_d <--> 20 < m_d/T_nu
    sf_ic_norm_0 = cf.s0/(cf.s_SM_no_nu(Ttrel.T_SM_grid[i_ic]) + cf.s_nu(Ttrel.T_nu_grid[i_ic]))
    T_nu_ic = Ttrel.T_nu_grid[i_ic]
    T_rescaled = T_nu_ic * cf.T_d_dw(1e-5) / cf.T_d_dw(m_d)
    n_ic = cf.n_0_dw(m_d, th) / sf_ic_norm_0      # 2* due to 2 species of neutrinos

    np.seterr(all=None, divide=None, over=None, under=None, invalid=None)

    """
    time1 = time.time()
    print('Timing function')
    th_avg_sigma_v_22_11(m_N1, m_N2, Ttrel.T_nu_grid[100], vert, m_X, m_Gamma_X2)
    time2 = time.time()
    print(f'th_avg_22_11 ran in: {time2-time1:.3f} seconds')

    T_DW_end = lambda md: md*1e4 
    sf_ic_norm_0 = lambda md: (cf.s0/(np.vectorize(cf.s_SM_no_nu)(T_DW_end(md)) + np.vectorize(cf.s_nu)(T_DW_end(md))))**(1./3.)
    A = lambda md, th: (0.1/T_DW_end(md))**3. * cf.n_0_dw(md, th) / (sf_ic_norm_0(md)**3. * th**2.) 

    T_eq = (147.8e9)*(y/0.1)**4*(th**2/1e-15) * 1e-6    # GeV 
    T_eq = (A(m_d, th)*1e9)*(y/0.1)**4*(th**2/1e-15) * 1e-6    # GeV 
    T_eq2 = (0.1)**3*m_X**2/((y/0.1)**4*(th**2/1e-15)*147.8)
    skip = 100
    # n2 = a^-3 / a_ic^-3 * n_ic, s ~ a^-3 => n2 = s / s_ic * n_ic
    n2 = lambda T: n_ic * (T / T_nu_ic)**3*((21/4 + np.vectorize(cf.g_s_before_nu_dec)(T))/(21/4+cf.g_s_before_nu_dec(T_nu_ic)))

    i_eq = np.where(np.min(abs(T_eq - Ttrel.T_nu_grid[::skip])) == abs(T_eq - Ttrel.T_nu_grid[::skip]))
    C_22_11 = np.array([th_avg_sigma_v_22_11(m_N1, m_N2, T, vert, m_X, m_Gamma_X2) for T in Ttrel.T_nu_grid[::skip]])
    C_22_11_naive = np.array([th_avg_sigma_v_22_11(m_N1, m_N2, T, vert, m_X, m_Gamma_X2, naive=True) for T in Ttrel.T_nu_grid[::skip]])
    A = H[::skip][i_eq] / C_22_11[i_eq]
    x = m_d / Ttrel.T_nu_grid

    s = np.vectorize(cf.s_SM_no_nu)(Ttrel.T_SM_grid) + np.vectorize(cf.s_nu)(Ttrel.T_nu_grid)
    nd_dw = cf.O_h2_dw_Tevo(Ttrel.T_nu_grid, m_d, th)*cf.rho_crit0_h2/m_d * s/cf.s0  

    plt.loglog(x, nd_dw)
    plt.loglog(x, n2(Ttrel.T_nu_grid))
    plt.show()

    # plt.loglog(x, Ttrel.T_nu_grid, label='T_nu')
    # plt.loglog(x1_dw, y1_dw, label='n2_dw')
    # plt.loglog(x, n2(Ttrel.T_nu_grid), label='n2')
    # plt.loglog(x, np.ones_like(x)*T_eq, label='T_eq')
    plt.axvline(m_d/T_eq, label='m_d/T_eq', color='k')
    plt.axvline(m_d/T_eq2, label='m_d/T_eq2', color='r')
    # plt.axvline(m_d/T_nu_ic, label='m_d/T_nu_ic')
    plt.loglog(x, 1e6*H, label='H')
    # plt.loglog(x[::skip], 1e6*abs(C_22_11), label='th_avg_22_11')
    plt.loglog(x[::skip], 1e6*abs(nd_dw[::skip]*C_22_11), label='C_22_11')
    plt.loglog(x[::skip], 1e6*abs(nd_dw[::skip]*C_22_11_naive), label='C_22_11_naive')
    plt.xlabel('m/T')
    plt.xlim(1e-5, 2e1)
    plt.ylim(1e-28, 1e-8)
    plt.legend()
    plt.show()
    """

    # load_str = './sterile_test/md_1e-05;mX_2.5e-05;sin22th_5e-16;y_2.522e-03;full_new.dat' 
    # load_str = './sterile_test/md_1e-05;mX_2.5e-05;sin22th_2e-14;y_5.57e-04;full_new.dat'
    load_str = './sterile_test/md_1e-05;mX_2.5e-05;sin22th_4.85e-13;y_1e-04;full_new.dat'

    var_list = load_str.split(';')[:-1]
    m_d, m_X, sin22th, y = [eval(s.split('_')[-1]) for s in var_list]
    th = 0.5*np.arcsin(np.sqrt(sin22th))
    vert = y**4 

    data_evo = np.loadtxt(load_str)
    data_skip_rate = 100

    t_grid = data_evo[::data_skip_rate,0]
    T_SM_grid = data_evo[::data_skip_rate,1]
    T_nu_grid = data_evo[::data_skip_rate,2]
    ent_grid = data_evo[::data_skip_rate, 3]
    H_grid = data_evo[::data_skip_rate,4]
    sf_grid = data_evo[::data_skip_rate,5]
    T_d_grid = data_evo[::data_skip_rate,6]
    xi_d_grid = data_evo[::data_skip_rate,7]
    xi_X_grid = data_evo[::data_skip_rate,8]
    n_d_grid = data_evo[::data_skip_rate,9]
    n_X_grid = data_evo[::data_skip_rate,10]
    n_nu_grid = 2.*0.75*(cf.zeta3/cf.pi2)*(T_nu_grid**3.)

    m_phi = 1.5*m_d
    k_phi = -1
    k_X = -1
    C_XX_pp_both = np.array([2.*C_n_XX_pp(m_phi, m_X, 0, k_phi, k_X, T_d, 0, xi_X, vert, th, 0, type=2) / 4. for T_d, xi_X in zip(T_d_grid, xi_X_grid)])

    C_XX_pp = -C_XX_pp_both[:,0]
    C_pp_XX = C_XX_pp_both[:,1]
    
    Gamma_XX_pp = C_XX_pp / n_d_grid
    Gamma_pp_XX = C_pp_XX / n_d_grid

    x_grid = m_d/T_nu_grid

    fig = plt.figure()
    ax = fig.add_subplot()

    ch = 'crimson' # crimson
    c1 = '#797ef6' # orchid
    c2 = '#1aa7ec' # sky blue
    c3 = '#4adede' # turquoise
    c4 = '#ffa62b' # gold
    c5 = '#1e2f97' # dark blue

    ax.loglog(x_grid, 1e6*H_grid, color=ch, ls='-', zorder=0) #83781B

    ax.loglog(x_grid, 1e6*abs(Gamma_XX_pp), color=c1, ls='-', zorder=-4) #114B5F
    # ax.loglog(x_grid, 1e6*abs(Gamma_pp_XX), color=c2, ls='-', zorder=-4) #458751

    ax.set_xlim(2e-5, 20)
    ax.set_ylim(1e-28, 1e-10)

    x_therm = 1e-3
    ax.fill_betweenx([1e-28, 1e0], 1e-5, x_therm, color='white', alpha=1, zorder=-3)
    ax.axvline(x_therm, ls=':', color='0', zorder=-2)

    dark_therm_x = 10**((np.log10(x_therm) + np.log10(x_grid)[0]) / 2)
    ax.text(dark_therm_x, 8e-21, r'$\mathrm{Dark}$', color='0', horizontalalignment='center')
    ax.text(dark_therm_x, 8e-22, r'$\mathrm{Thermalization}$', color='0', horizontalalignment='center')
    ax.text(dark_therm_x, 8e-23, r'$\rightarrow$', color='0', horizontalalignment='center')

    Hubble_x = 10**((np.log10(x_therm) + np.log10(x_grid)[0]) / 2)
    ax.text(Hubble_x, 1e-15, r"$H$", color=ch, rotation=0, va='top')
    ax.text(1, 1e-15, r"$A^\prime A^\prime \to \phi \phi$", color=c1, rotation=0, ha='left', va='bottom')

    # ax.text(1, 1e-18, r"$\phi \phi \to A^\prime A^\prime$", color=c2, rotation=0, ha='left', va='bottom')

    md_str = f'{m_d:.3e}'.split('e-')
    mX_str = f'{m_X:.3e}'.split('e-')
    mphi_str = f'{m_phi:.3e}'.split('e-')
    sin22th_str = f'{sin22th:.3e}'.split('e-')
    y_str = f'{y:.3e}'.split('e-')
    md_str = md_str[0] + '\cdot 10^{-' + md_str[1].lstrip('0') + '}'
    mX_str = f'{m_X/m_d}m_N'
    mphi_str = f'{m_phi/m_d}m_N'
    y_str = y_str[0] + '\cdot 10^{-' + y_str[1].lstrip('0') + '}'
    sin22th_str = sin22th_str[0] + '\cdot 10^{-' + sin22th_str[1].lstrip('0') + '}'

    fig.suptitle(fr"$m_N={md_str}, m_A={mX_str}, m_\phi={mphi_str}, y={y_str}, \sin^2(2\theta)={sin22th_str}$", fontsize=10)

    plt.show()