#! /usr/bin/env python3
import numpy as np
import mpmath as mpm
import math as math
from scipy.integrate import quad
from scipy.special import kn
import constants_functions as cf

pi = np.pi
pi2 = np.pi**2.
pi3 = pi2*pi

rtol_int = 1e-5
m_T_r_MB = 1e2
m_T_r_nr = 6e2
m_T_r_ur = 1e-3
max_f_arg = 1e2

def n(k, T, m, dof, xi):
    if k == 1. and m/T - xi < m_T_r_MB:
        return n_fermion_xi(T, m, dof, xi)
    elif k == -1. and m/T - xi < m_T_r_MB:
        return n_boson_xi(T, m, dof, xi)
    elif k == 0. or m/T - xi >= m_T_r_MB:
        if m/T > 1e-10 and m/T < m_T_r_nr:
            return dof*math.exp(xi)*kn(2, m/T)*m*m*T/(2.*cf.pi2)
        elif m/T >= m_T_r_nr:
            return dof*math.exp(xi-m/T)*((m*T/(2.*pi))**1.5 + (15./16.)*math.sqrt(m*(T**5.)/(2.*pi3)))
        return dof*math.exp(xi)*(T**3.)/cf.pi2
    return None

def n_der_T(k, T, m, dof, xi):
    if k == 1. and m/T - xi < m_T_r_MB:
        return n_der_T_fermion_xi(T, m, dof, xi)
    elif k == -1. and m/T - xi < m_T_r_MB:
        return n_der_T_boson_xi(T, m, dof, xi)
    elif k == 0. or m/T - xi >= m_T_r_MB:
        return rho(k, T, m, dof, xi)/(T*T)
    return None

def n_der_xi(k, T, m, dof, xi):
    if k == 1. and m/T - xi < m_T_r_MB:
        return n_der_xi_fermion_xi(T, m, dof, xi)
    elif k == -1. and m/T - xi < m_T_r_MB:
        return n_der_xi_boson_xi(T, m, dof, xi)
    elif k == 0. or m/T - xi >= m_T_r_MB:
        return n(k, T, m, dof, xi)
    return None

def rho(k, T, m, dof, xi):
    if k == 1. and m/T - xi < m_T_r_MB:
        return rho_fermion_xi(T, m, dof, xi)
    elif k == -1. and m/T - xi < m_T_r_MB:
        return rho_boson_xi(T, m, dof, xi)
    elif k == 0. or m/T - xi >= m_T_r_MB:
        if m/T > 1e-10 and m/T < m_T_r_nr:
            return dof*math.exp(xi)*(m*kn(1, m/T)+3.*T*kn(2, m/T))*m*m*T/(2.*cf.pi2)
        elif m/T >= m_T_r_nr:
            return (m+1.5*T)*n(k, T, m, dof, xi)
        return dof*3.*math.exp(xi)*(T**4.)/cf.pi2
    return None

def rho_der_T(k, T, m, dof, xi):
    if k == 1. and m/T - xi < m_T_r_MB:
        return rho_der_T_fermion_xi(T, m, dof, xi)
    elif k == -1. and m/T - xi < m_T_r_MB:
        return rho_der_T_boson_xi(T, m, dof, xi)
    elif k == 0. or m/T - xi >= m_T_r_MB:
        if m/T > 1e-10 and m/T < m_T_r_nr:
            return dof*math.exp(xi)*(m*kn(2, m/T)+3.*T*kn(3, m/T))*(m**3.)/(2.*T*cf.pi2)
        elif m/T >= m_T_r_nr:
            return m*n_der_T(k, T, m, dof, xi) + 1.5*n(k, T, m, dof, xi)
        return dof*12.*math.exp(xi)*(T**3.)/cf.pi2
    return None

def rho_der_xi(k, T, m, dof, xi):
    if k == 1. and m/T - xi < m_T_r_MB:
        return rho_der_xi_fermion_xi(T, m, dof, xi)
    elif k == -1. and m/T - xi < m_T_r_MB:
        return rho_der_xi_boson_xi(T, m, dof, xi)
    elif k == 0. or m/T - xi >= m_T_r_MB:
        return rho(k, T, m, dof, xi)
    return None

def P(k, T, m, dof, xi):
    if k == 1. and m/T - xi < m_T_r_MB:
        return P_fermion_xi(T, m, dof, xi)
    elif k == -1. and m/T - xi < m_T_r_MB:
        return P_boson_xi(T, m, dof, xi)
    elif k == 0. or m/T - xi >= m_T_r_MB:
        if m/T > 1e-10 and m/T < m_T_r_nr:
            return dof*math.exp(xi)*kn(2, m/T)*m*m*T*T/(2.*cf.pi2)
        elif m/T >= m_T_r_nr:
            return T*n(k, T, m, dof, xi)
        return dof*math.exp(xi)*(T**4.)/cf.pi2
    return None

def rho_3P_diff(k, T, m, dof, xi):
    if k == 1. and m/T - xi < m_T_r_MB:
        return rho_3P_diff_fermion_xi(T, m, dof, xi)
    elif k == -1. and m/T - xi < m_T_r_MB:
        return rho_3P_diff_boson_xi(T, m, dof, xi)
    elif k == 0. or m/T - xi >= m_T_r_MB:
        if m/T > 1e-10 and m/T < m_T_r_nr:
            return dof*math.exp(xi)*kn(1, m/T)*m*m*m*T/(2.*cf.pi2)
        elif m/T >= m_T_r_nr:
            return (m-1.5*T)*n(k, T, m, dof, xi)
        return dof*math.exp(xi)*m*m*T*T/(2.*cf.pi2)
    return None

def n_fermion_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        return -dof * ((m*T/(2.*pi))**1.5) * np.float64(mpm.re(mpm.polylog(1.5, -np.exp(xi-x))))
    if m/T < m_T_r_ur:
        return -dof * T**3. * np.float64(mpm.re(mpm.polylog(3, -np.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        return E2*math.sqrt((E-m)*(E+m))/(math.exp(E/T - xi) + 1.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.n_fermion(T, m, 1)
    return (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def n_boson_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        return dof * ((m*T/(2.*pi))**1.5) * np.float64(mpm.re(mpm.polylog(1.5, math.exp(xi-x))))
    if m/T < m_T_r_ur:
        return dof * T**3. * np.float64(mpm.re(mpm.polylog(3, math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        return E2*math.sqrt((E-m)*(E+m))/(math.exp(E/T - xi) - 1.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.n_boson(T, m, 1)
    return None if xi*T >= m else (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def n_der_T_fermion_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        sqrt_T = math.sqrt(T)
        return -dof * ((m/(2.*pi))**1.5) * np.float64(mpm.re(mpm.polylog(0.5, -math.exp(xi-x)))*m/sqrt_T + 1.5*mpm.re(mpm.polylog(1.5, -math.exp(xi-x)))*sqrt_T)
    if m/T < m_T_r_ur:
        return -dof * 3. * T**2. * np.float64(mpm.re(mpm.polylog(3, -math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        exp_ET_xi = math.exp(E/T - xi)
        return E2*math.sqrt((E-m)*(E+m))*E*exp_ET_xi /((T*(exp_ET_xi + 1.))**2.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    return (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=0., epsrel=rtol_int)[0]

def n_der_T_boson_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        sqrt_T = math.sqrt(T)
        return dof * ((m/(2.*pi))**1.5) * np.float64(mpm.re(mpm.polylog(0.5, math.exp(xi-x)))*m/sqrt_T + 1.5*mpm.re(mpm.polylog(1.5, math.exp(xi-x)))*sqrt_T)
    if m/T < m_T_r_ur:
        return dof * 3. * T**2. * np.float64(mpm.re(mpm.polylog(3, math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        exp_ET_xi = math.exp(E/T - xi)
        return E2*math.sqrt((E-m)*(E+m))*E*exp_ET_xi /((T*(exp_ET_xi - 1.))**2.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    return None if xi*T >= m else (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=0., epsrel=rtol_int)[0]

def n_der_xi_fermion_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        return -dof * ((m*T/(2.*pi))**1.5) * np.float64(mpm.re(mpm.polylog(0.5, -math.exp(xi-x))))
    if m/T < m_T_r_ur:
        return -dof * T**3. * np.float64(mpm.re(mpm.polylog(2, -math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        exp_ET_xi = math.exp(E/T - xi)
        return E2*math.sqrt((E-m)*(E+m))*exp_ET_xi /(((exp_ET_xi + 1.))**2.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.n_fermion(T, m, xi)
    return (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def n_der_xi_boson_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        return dof * ((m*T/(2.*pi))**1.5) * np.float64(mpm.re(mpm.polylog(0.5, math.exp(xi-x))))
    if m/T < m_T_r_ur:
        return dof * T**3. * np.float64(mpm.re(mpm.polylog(2, math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        exp_ET_xi = math.exp(E/T - xi)
        return E2*math.sqrt((E-m)*(E+m))*exp_ET_xi /(((exp_ET_xi - 1.))**2.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.n_boson(T, m, xi)
    return None if xi*T >= m else (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def rho_fermion_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        return -dof * ((m*T/(2.*pi))**1.5) * np.float64(m*mpm.re(mpm.polylog(1.5, -math.exp(xi-x))) + 1.5*T*mpm.re(mpm.polylog(2.5, -math.exp(xi-x))))
    if m/T < m_T_r_ur:
        return -dof * 3. * T**4. * np.float64(mpm.re(mpm.polylog(4, -math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        return E2*math.sqrt((E-m)*(E+m))*E/(math.exp(E/T - xi) + 1.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.rho_fermion(T, m, 1.)
    return (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def rho_boson_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        return dof * ((m*T/(2.*pi))**1.5) * np.float64(m*mpm.re(mpm.polylog(1.5, math.exp(xi-x))) + 1.5*T*mpm.re(mpm.polylog(2.5, math.exp(xi-x))))
    if m/T < m_T_r_ur:
        return dof * 3. * T**4. * np.float64(mpm.re(mpm.polylog(4, math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        return E2*math.sqrt((E-m)*(E+m))*E/(math.exp(E/T - xi) - 1.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.rho_boson(T, m, 1.)
    return None if xi*T >= m else (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def rho_der_T_fermion_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        sqrt_T = math.sqrt(T)
        return -dof*0.25*((m/(2.*pi))**1.5)*np.float64(4.*m*m*mpm.re(mpm.polylog(0.5, -math.exp(xi-x)))/sqrt_T+3*sqrt_T*(4.*m*mpm.re(mpm.polylog(1.5, -math.exp(xi-x)))+5.*T*mpm.re(mpm.polylog(2.5, -math.exp(xi-x)))))
    if m/T < m_T_r_ur:
        return -dof * 12. * T**3. * np.float64(mpm.re(mpm.polylog(4, -math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        exp_ET_xi = math.exp(E/T - xi)
        return E2*math.sqrt((E-m)*(E+m))*E2*exp_ET_xi /((T*(exp_ET_xi + 1.))**2.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.rho_der_fermion(T, m, 1.)
    return (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def rho_der_T_boson_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        sqrt_T = math.sqrt(T)
        return dof*0.25*((m/(2.*pi))**1.5)*np.float64(4.*m*m*mpm.re(mpm.polylog(0.5, math.exp(xi-x)))/sqrt_T+3*sqrt_T*(4.*m*mpm.re(mpm.polylog(1.5, math.exp(xi-x)))+5.*T*mpm.re(mpm.polylog(2.5, math.exp(xi-x)))))
    if m/T < m_T_r_ur:
        return dof * 12. * T**3. * np.float64(mpm.re(mpm.polylog(4, math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        exp_ET_xi = math.exp(E/T - xi)
        return E2*math.sqrt((E-m)*(E+m))*E2*exp_ET_xi /((T*(exp_ET_xi - 1.))**2.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.rho_der_boson(T, m, 1.)
    return None if xi*T >= m else (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def rho_der_xi_fermion_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        return -dof * ((m*T/(2.*pi))**1.5) * np.float64(m*mpm.re(mpm.polylog(0.5, -math.exp(xi-x))) + 1.5*T*mpm.re(mpm.polylog(1.5, -math.exp(xi-x))))
    if m/T < m_T_r_ur:
        return -dof * 3. * T**4. * np.float64(mpm.re(mpm.polylog(3, -math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        exp_ET_xi = math.exp(E/T - xi)
        return E2*math.sqrt((E-m)*(E+m))*E*exp_ET_xi /(((exp_ET_xi + 1.))**2.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.rho_fermion(T, m, xi)
    return (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def rho_der_xi_boson_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        return dof * ((m*T/(2.*pi))**1.5) * np.float64(m*mpm.re(mpm.polylog(0.5, math.exp(xi-x))) + 1.5*T*mpm.re(mpm.polylog(1.5, math.exp(xi-x))))
    if m/T < m_T_r_ur:
        return dof * 3. * T**4. * np.float64(mpm.re(mpm.polylog(3, math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        exp_ET_xi = math.exp(E/T - xi)
        return E2*math.sqrt((E-m)*(E+m))*E*exp_ET_xi /(((exp_ET_xi - 1.))**2.)
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.rho_boson(T, m, xi)
    return None if xi*T >= m else (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def P_fermion_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        return -dof * T * ((m*T/(2.*pi))**1.5) * np.float64(mpm.re(mpm.polylog(2.5, -math.exp(xi-x))))
    if m/T < m_T_r_ur:
        return -dof * T**4. * np.float64(mpm.re(mpm.polylog(4, -math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        p2 = (E-m)*(E+m)
        return E2*math.sqrt(p2)*p2/(3.*E*(math.exp(E/T - xi) + 1.))
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.P_fermion(T, m, 1.)
    return (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def P_boson_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        x = m/T
        return dof * T * ((m*T/(2.*pi))**1.5) * np.float64(mpm.re(mpm.polylog(2.5, math.exp(xi-x))))
    if m/T < m_T_r_ur:
        return dof * T**4. * np.float64(mpm.re(mpm.polylog(4, math.exp(xi))))/pi2

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        E2 = E*E
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        p2 = (E-m)*(E+m)
        return E2*math.sqrt(p2)*p2/(3.*E*(math.exp(E/T - xi) - 1.))
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.P_boson(T, m, 1.)
    return None if xi*T >= m else (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def rho_3P_diff_fermion_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        return dof*math.exp(xi)*kn(1, m/T)*m*m*m*T/(2.*cf.pi2)
    if m/T < m_T_r_ur:
        return -dof * ((m*T)**2.) * np.float64(mpm.re(mpm.polylog(2, -math.exp(xi))))/(2.*pi2)

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        p2 = (E-m)*(E+m)
        return E*math.sqrt(p2)*m2/((math.exp(E/T - xi) + 1.))
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.rho_3P_diff_fermion(T, m, 1.)
    return (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]

def rho_3P_diff_boson_xi(T, m, dof, xi):
    if m/T-xi > m_T_r_MB:
        return dof*math.exp(xi)*kn(1, m/T)*m*m*m*T/(2.*cf.pi2)
    if m/T < m_T_r_ur:
        return dof * ((m*T)**2.) * np.float64(mpm.re(mpm.polylog(2, math.exp(xi))))/(2.*pi2)

    m2 = m**2.
    def integrand(log_E):
        E = math.exp(log_E)
        if math.fabs(E/T - xi) > max_f_arg or E <= m:
            return 0.
        p2 = (E-m)*(E+m)
        return E*math.sqrt(p2)*m2/((math.exp(E/T - xi) - 1.))
    log_E_min = math.log(1e-6*T) if m == 0. else math.log(m)
    E_max = (max_f_arg + xi)*T
    if E_max <= m:
        E_max = m + max_f_arg*T
    log_E_max = math.log(E_max)

    epsabs = 0. if m/T > 200. else 1e-6*2.*pi2*cf.rho_3P_diff_boson(T, m, 1.)
    return (dof/(2.*pi2))*quad(integrand, log_E_min, log_E_max, epsabs=epsabs, epsrel=rtol_int)[0]
