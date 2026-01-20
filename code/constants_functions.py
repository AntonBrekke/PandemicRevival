#! /usr/bin/env python3

import numpy as np
import math as math
import os
import utils
from scipy.interpolate import interp1d

# Mathematical constants
pi2   = np.pi*np.pi
zeta3 = 1.202056903

temp_nu_dec_sm   = 1.4e-3                       # temperature of nu decoupling in the SM [GeV]
hubble_T5_nu_dec = 1.62211799511e-10            # hubble(temp_nu_dec_sm)/temp_nu_dec_sm**5 [GeV^4]
G                = 6.7086096877e-39             # Newton constant [GeV^(-2)]
hbar             = 6.582119514e-25              # reduced Planck constant [GeV*s]
c_light          = 2.99792458e10                # speed of light [cm/s]
Mpc              = 1.563738357134461e+38        # Mpc in 1/GeV

conv_GeV_cm_3 = 1.3014892628900395e41       # conversion factor from GeV^(3) to cm^(-3)
conv_cm2_g    = 4.57821356e3                # conversion factor from cm^2/g to GeV^{-3}
omega_d0      = 0.12                        # omega_d = Omega_d h^2 from Planck
omega_b0      = 0.02237                     # omega_b = Omega_b h^2 from Planck
rho_crit0_h2  = 1.053672e-5 / conv_GeV_cm_3 # rho_{crit, today} h^-2 in GeV^4
rho_d0        = omega_d0 * rho_crit0_h2     # dark matter density today in GeV^4
rho_b0        = omega_b0 * rho_crit0_h2     # baryon density today in GeV^4
rho_m0        = rho_d0 + rho_b0
T0            = 2.72548*8.6173324e-14       # CMB temperature today [GeV]
s0            = (2.*pi2/45.) * (T0**3.) * 3.9267
                                            # entropy density today [GeV^3]
                                            # factor 3.9267 for instantaneous nu-decoupling in SM at 1.4 MeV (our calculation), 3.93782 with more sophisticated Neff-calculation
                                            # for consistency use the former

package_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = package_dir + '/data/'
g_star_dir = data_dir + 'g_star/'

# g_stars as a function of T in GeV
g_rho_no_nu_grid = np.loadtxt(g_star_dir+'g_rho_no_nu.dat')
g_rho_no_nu = utils.LogInterp(g_rho_no_nu_grid[:,0], g_rho_no_nu_grid[:,1], extrap='b')
g_rho_der_no_nu_grid = np.loadtxt(g_star_dir+'g_rho_der_no_nu.dat')
g_rho_der_no_nu = utils.LogInterp(g_rho_der_no_nu_grid[:,0], g_rho_der_no_nu_grid[:,1], extrap='b')
g_s_no_nu_grid = np.loadtxt(g_star_dir+'g_s_no_nu.dat')
g_s_no_nu = utils.LogInterp(g_s_no_nu_grid[:,0], g_s_no_nu_grid[:,1], extrap='b')
g_P_no_nu_grid = np.loadtxt(g_star_dir+'g_P_no_nu.dat')
g_P_no_nu = utils.LogInterp(g_P_no_nu_grid[:,0], g_P_no_nu_grid[:,1], extrap='b')

g_rho_before_nu_dec = lambda T: g_rho_no_nu(T) + 6.*7./8.
g_rho_der_before_nu_dec = lambda T: g_rho_der_no_nu(T)
g_s_before_nu_dec = lambda T: g_s_no_nu(T) + 6.*7./8.
g_P_before_nu_dec = lambda T: g_P_no_nu(T) + 6.*7./8.

rho_SM_no_nu = lambda T: pi2*g_rho_no_nu(T)*(T**4.)/30.
rho_der_SM_no_nu = lambda T: (4.*g_rho_no_nu(T) + g_rho_der_no_nu(T)*T)*pi2*(T**3.)/30.
P_SM_no_nu = lambda T: pi2*g_P_no_nu(T)*(T**4.)/90.
s_SM_no_nu = lambda T: pi2*g_s_no_nu(T)*(T**3.)*2./45.

rho_SM_before_nu_dec = lambda T: pi2*g_rho_before_nu_dec(T)*(T**4.)/30.
rho_der_SM_before_nu_dec = lambda T: (4.*g_rho_before_nu_dec(T) + g_rho_der_before_nu_dec(T)*T)*pi2*(T**3.)/30.
P_SM_before_nu_dec = lambda T: pi2*g_P_before_nu_dec(T)*(T**4.)/90.
s_SM_before_nu_dec = lambda T: pi2*g_s_before_nu_dec(T)*(T**3.)*2./45.

rho_nu = lambda T: pi2*6.*(7./8.)*(T**4.)/30.
P_nu = lambda T: pi2*6.*(7./8.)*(T**4.)/90.
s_nu = lambda T: pi2*6.*(7./8.)*(T**3.)*2./45.

rho_d = lambda T_SM, T_nu: rho_d0 * (s_SM_no_nu(T_SM) + s_nu(T_nu)) / s0
rho_b = lambda T_SM, T_nu: rho_b0 * (s_SM_no_nu(T_SM) + s_nu(T_nu)) / s0
rho_m = lambda T_SM, T_nu: rho_m0 * (s_SM_no_nu(T_SM) + s_nu(T_nu)) / s0

# Dodelson-Widrow stuff
data_n_dw = np.loadtxt(data_dir + 'dw/0612182_dw_fig_4.dat', skiprows=2)
log_C_e_dw_interp = interp1d(np.log(data_n_dw[:,0]), np.log(data_n_dw[:,1]), bounds_error=False, fill_value=(np.log(data_n_dw[0,1]), np.log(data_n_dw[-1,1])))
C_e_dw = lambda md: np.exp(log_C_e_dw_interp(np.log(md)))
O_h2_dw = lambda md, th: 0.11*C_e_dw(md)*((0.5*np.sin(2.*th)*md*1e10)**2.) # abundance produced by DW mechanism
n_0_dw = lambda md, th: O_h2_dw(md, th)*rho_crit0_h2/md

sf_nu_dec_sm = (s0/s_SM_before_nu_dec(temp_nu_dec_sm))**(1./3.)
data_avg_mom_dw = np.loadtxt(data_dir + 'dw/0612182_dw_fig_8.dat', skiprows=2)
avg_mom_interp_dw = interp1d(np.log(data_avg_mom_dw[:,0]), np.log(data_avg_mom_dw[:,1]), bounds_error=False, fill_value=(np.log(data_avg_mom_dw[0,1]), np.log(data_avg_mom_dw[-1,1])))
avg_mom_0_dw = lambda md: np.exp(avg_mom_interp_dw(np.log(md)))*7.*pi2*pi2*temp_nu_dec_sm*sf_nu_dec_sm/(180.*zeta3)

T_d_dw = lambda md: 0.133*((1e6*md)**1./3.) # temperature of maximal d production by DW mechanism

data_Tevo_dw = np.loadtxt(data_dir + 'dw/0612182_dw_fig_3.dat', skiprows=2)
data_Tevo_dw[:,1] = data_Tevo_dw[:,1]/data_Tevo_dw[-1,1]
Tevo_dw_interp = interp1d(np.log10(data_Tevo_dw[:,0]), data_Tevo_dw[:,1], bounds_error=False, fill_value=(1., 0.))
def O_h2_dw_Tevo(T, md, th):
    O_h2 = O_h2_dw(md, th)
    T_max_prod = T_d_dw(md)
    T_ref = T_d_dw(1e-5)
    T_rescaled = T * T_ref / T_max_prod
    return Tevo_dw_interp(np.log10(T_rescaled)) * O_h2

# # prefactor of FD-spectrum from DW mechanism --> assumes constant prefactor, not really ok
norm_f_d_dw = lambda md, th, dofd: 4.*pi2*s_SM_before_nu_dec(T_d_dw(md))*O_h2_dw(md, th)*rho_crit0_h2/(3.*zeta3*dofd*(T_d_dw(md)**3.)*md*s0)

dens_dir = data_dir + 'densities/'

rho_red_boson = np.loadtxt(dens_dir + 'rho_red_boson.dat')
rho_red_boson_interp = utils.LogInterp(rho_red_boson[:,0], rho_red_boson[:,1])
def rho_boson(T, m, dof, xi=0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > rho_red_boson[-1,0] or xi > 700.:
        return dof*(m + 1.5*T)*math.exp(xi-x)*((m*T/(2.*np.pi))**1.5)
    if x < rho_red_boson[0,0]:
        return dof*math.exp(xi)*pi2*(T**4.)/30.
    return dof*(T**4.)*math.exp(xi)*rho_red_boson_interp(x)

rho_red_fermion = np.loadtxt(dens_dir + 'rho_red_fermion.dat')
rho_red_fermion_interp = utils.LogInterp(rho_red_fermion[:,0], rho_red_fermion[:,1])
def rho_fermion(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > rho_red_fermion[-1,0] or xi > 700.:
        return dof*(m + 1.5*T)*math.exp(xi-x)*((m*T/(2.*np.pi))**1.5)
    if x < rho_red_fermion[0,0]:
        return dof*math.exp(xi)*pi2*(T**4.)*7./240.
    return dof*(T**4.)*math.exp(xi)*rho_red_fermion_interp(x)

rho_der_red_boson = np.loadtxt(dens_dir + 'rho_der_red_boson.dat')
rho_der_red_boson_interp = utils.LogInterp(rho_der_red_boson[:,0], rho_der_red_boson[:,1])
def rho_der_boson(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > rho_der_red_boson[-1,0] or xi > 700.:
        return dof*math.exp(xi-x)*(T**3.)*((x**3.5) + 3.*(x**2.5) + 3.75*(x**1.5))/((2.*np.pi)**1.5)
    if x < rho_der_red_boson[0,0]:
        return dof*math.exp(xi)*pi2*(T**3.)*2./15.
    return dof*(T**3.)*math.exp(xi)*rho_der_red_boson_interp(x)

rho_der_red_fermion = np.loadtxt(dens_dir + 'rho_der_red_fermion.dat')
rho_der_red_fermion_interp = utils.LogInterp(rho_der_red_fermion[:,0], rho_der_red_fermion[:,1])
def rho_der_fermion(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > rho_der_red_fermion[-1,0] or xi > 700.:
        return dof*math.exp(xi-x)*(T**3.)*((x**3.5) + 3.*(x**2.5) + 3.75*(x**1.5))/((2.*np.pi)**1.5)
    if x < rho_der_red_fermion[0,0]:
        return dof*math.exp(xi)*pi2*(T**3.)*7./60.
    return dof*(T**3.)*math.exp(xi)*rho_der_red_fermion_interp(x)

P_red_boson = np.loadtxt(dens_dir + 'P_red_boson.dat')
P_red_boson_interp = utils.LogInterp(P_red_boson[:,0], P_red_boson[:,1])
def P_boson(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > P_red_boson[-1,0] or xi > 700.:
        return dof*(T - 2.5*(T**2.)/m)*math.exp(xi-x)*((m*T/(2.*np.pi))**1.5)
    if x < P_red_boson[0,0]:
        return dof*math.exp(xi)*pi2*(T**4.)/90.
    return dof*(T**4.)*math.exp(xi)*P_red_boson_interp(x)

P_red_fermion = np.loadtxt(dens_dir + 'P_red_fermion.dat')
P_red_fermion_interp = utils.LogInterp(P_red_fermion[:,0], P_red_fermion[:,1])
def P_fermion(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > P_red_fermion[-1,0] or xi > 700.:
        return dof*(T - 2.5*(T**2.)/m)*math.exp(xi-x)*((m*T/(2.*np.pi))**1.5)
    if x < P_red_fermion[0,0]:
        return dof*math.exp(xi)*pi2*(T**4.)*7./720.
    return dof*(T**4.)*math.exp(xi)*P_red_fermion_interp(x)

rho_3P_diff_red_boson = np.loadtxt(dens_dir + 'rho_3P_diff_red_boson.dat')
rho_3P_diff_red_boson_interp = utils.LogInterp(rho_3P_diff_red_boson[:,0], rho_3P_diff_red_boson[:,1])
def rho_3P_diff_boson(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > rho_3P_diff_red_boson[-1,0] or xi > 700.:
        return dof*math.exp(xi-x)*(math.sqrt((m**5.)*((T/(2.*np.pi))**3.)) + (3./8.)*math.sqrt((T**5.)*((m/(2.*np.pi))**3.)))
    if x < rho_3P_diff_red_boson[0,0]:
        return dof*math.exp(xi)*m*m*T*T/12.
    return dof*((m*T)**2.)*math.exp(xi)*rho_3P_diff_red_boson_interp(x)

rho_3P_diff_red_fermion = np.loadtxt(dens_dir + 'rho_3P_diff_red_fermion.dat')
rho_3P_diff_red_fermion_interp = utils.LogInterp(rho_3P_diff_red_fermion[:,0], rho_3P_diff_red_fermion[:,1])
def rho_3P_diff_fermion(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > rho_3P_diff_red_fermion[-1,0] or xi > 700.:
        return dof*math.exp(xi-x)*(math.sqrt((m**5.)*((T/(2.*np.pi))**3.)) + (3./8.)*math.sqrt((T**5.)*((m/(2.*np.pi))**3.)))
    if x < rho_3P_diff_red_fermion[0,0]:
        return dof*math.exp(xi)*m*m*T*T/24.
    return dof*((m*T)**2.)*math.exp(xi)*rho_3P_diff_red_fermion_interp(x)

n_red_boson = np.loadtxt(dens_dir + 'n_red_boson.dat')
n_red_boson_interp = utils.LogInterp(n_red_boson[:,0], n_red_boson[:,1])
def n_boson(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > n_red_boson[-1,0] or xi > 700.:
        return dof*math.exp(xi-x)*((m*T/(2.*np.pi))**1.5)
    if x <  n_red_boson[0,0]:
        return dof*math.exp(xi)*zeta3*(T**3.)/pi2
    return dof*(T**3.)*math.exp(xi)*n_red_boson_interp(x)

n_red_fermion = np.loadtxt(dens_dir + 'n_red_fermion.dat')
n_red_fermion_interp = utils.LogInterp(n_red_fermion[:,0], n_red_fermion[:,1])
def n_fermion(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > n_red_fermion[-1,0] or xi > 700.:
        return dof*math.exp(xi-x)*((m*T/(2.*np.pi))**1.5)
    if x < n_red_fermion[0,0]:
        return dof*math.exp(xi)*0.75*zeta3*(T**3.)/pi2
    return dof*(T**3.)*math.exp(xi)*n_red_fermion_interp(x)

n_der_red_boson = np.loadtxt(dens_dir + 'n_der_red_boson.dat')
n_der_red_boson_interp = utils.LogInterp(n_der_red_boson[:,0], n_der_red_boson[:,1])
def n_der_boson(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > n_der_red_boson[-1,0] or xi > 700.:
        return dof*math.exp(xi-x)*(T**2.)*((x**2.5)+1.5*(x**1.5))/((2.*np.pi)**1.5)
    if x <  n_der_red_boson[0,0]:
        return dof*math.exp(xi)*3.*zeta3*(T**2.)/pi2
    return dof*(T**2.)*math.exp(xi)*n_der_red_boson_interp(x)

n_der_red_fermion = np.loadtxt(dens_dir + 'n_der_red_fermion.dat')
n_der_red_fermion_interp = utils.LogInterp(n_der_red_fermion[:,0], n_der_red_fermion[:,1])
def n_der_fermion(T, m, dof, xi = 0.):
    x = m/T
    if x - xi > 700.:
        return 0.
    if x > n_der_red_fermion[-1,0] or xi > 700.:
        return dof*math.exp(xi-x)*(T**2.)*((x**2.5)+1.5*(x**1.5))/((2.*np.pi)**1.5)
    if x < n_der_red_fermion[0,0]:
        return dof*math.exp(xi)*3.*0.75*zeta3*(T**2.)/pi2
    return dof*(T**2.)*math.exp(xi)*n_der_red_fermion_interp(x)
