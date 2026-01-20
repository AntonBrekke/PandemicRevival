#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import constants_functions as cf
import pandemolator
import utils

m1keV = np.loadtxt('0612182_dw_fig_7_1keV.dat')
m2keV = np.loadtxt('0612182_dw_fig_7_2keV.dat')
m4keV = np.loadtxt('0612182_dw_fig_7_4keV.dat')
m8keV = np.loadtxt('0612182_dw_fig_7_8keV.dat')
m16keV = np.loadtxt('0612182_dw_fig_7_16keV.dat')
m32keV = np.loadtxt('0612182_dw_fig_7_32keV.dat')

avg_mom_data = np.loadtxt('0612182_dw_fig_8.dat', skiprows=2)
avg_mom_interp_fc = interp1d(np.log(avg_mom_data[:,0]), np.log(avg_mom_data[:,1]))
avg_mom = lambda m: np.exp(avg_mom_interp_fc(np.log(m)))*7.*(np.pi**4.)*1e-3/(180.*cf.zeta3)

f_eq = lambda x: 1./(np.exp(x)+1.)
th = 1e-10
f1keV = m1keV[:,1]*f_eq(m1keV[:,0])*(1e-6**2.)*((1e10*th)**2.)
f2keV = m2keV[:,1]*f_eq(m2keV[:,0])*(2e-6**2.)*((1e10*th)**2.)
f4keV = m4keV[:,1]*f_eq(m4keV[:,0])*(4e-6**2.)*((1e10*th)**2.)
f8keV = m8keV[:,1]*f_eq(m8keV[:,0])*(8e-6**2.)*((1e10*th)**2.)
f16keV = m16keV[:,1]*f_eq(m16keV[:,0])*(16e-6**2.)*((1e10*th)**2.)
f32keV = m32keV[:,1]*f_eq(m32keV[:,0])*(32e-6**2.)*((1e10*th)**2.)

p1keV = m1keV[:,0]*1e-3
p2keV = m2keV[:,0]*1e-3
p4keV = m4keV[:,0]*1e-3
p8keV = m8keV[:,0]*1e-3
p16keV = m16keV[:,0]*1e-3
p32keV = m32keV[:,0]*1e-3

n1keV = np.trapz(f1keV*(p1keV**2.), p1keV)*2./(2.*(np.pi**2.))
n2keV = np.trapz(f2keV*(p2keV**2.), p2keV)*2./(2.*(np.pi**2.))
n4keV = np.trapz(f4keV*(p4keV**2.), p4keV)*2./(2.*(np.pi**2.))
n8keV = np.trapz(f8keV*(p8keV**2.), p8keV)*2./(2.*(np.pi**2.))
n16keV = np.trapz(f16keV*(p16keV**2.), p16keV)*2./(2.*(np.pi**2.))
n32keV = np.trapz(f32keV*(p32keV**2.), p32keV)*2./(2.*(np.pi**2.))

avg_mom_1keV = np.trapz(f1keV*(p1keV**3.), p1keV)*2./(2.*(np.pi**2.)*n1keV)
avg_mom_2keV = np.trapz(f2keV*(p2keV**3.), p2keV)*2./(2.*(np.pi**2.)*n2keV)
avg_mom_4keV = np.trapz(f4keV*(p4keV**3.), p4keV)*2./(2.*(np.pi**2.)*n4keV)
avg_mom_8keV = np.trapz(f8keV*(p8keV**3.), p8keV)*2./(2.*(np.pi**2.)*n8keV)
avg_mom_16keV = np.trapz(f16keV*(p16keV**3.), p16keV)*2./(2.*(np.pi**2.)*n16keV)
avg_mom_32keV = np.trapz(f32keV*(p32keV**3.), p32keV)*2./(2.*(np.pi**2.)*n32keV)

Ttrel = pandemolator.TimeTempRelation()
i_1MeV = np.argmax(Ttrel.T_nu_grid < 1e-3)
i_fs_max = np.argmax(Ttrel.T_SM_grid/cf.T0 - 1. < 15.) # only assume free-streaming until z = 50

ent_grid = np.array([cf.s_SM_no_nu(T)+cf.s_nu(T_nu) for T, T_nu in zip(Ttrel.T_SM_grid, Ttrel.T_nu_grid)])
sf_norm_1MeV = (ent_grid[i_1MeV]/ent_grid)**(1./3.)
sf_norm_today = (cf.s0/ent_grid)**(1./3.)

i_dw_1keV = np.argmax(Ttrel.T_nu_grid < cf.T_d_dw(1e-6))
i_dw_2keV = np.argmax(Ttrel.T_nu_grid < cf.T_d_dw(2e-6))
i_dw_4keV = np.argmax(Ttrel.T_nu_grid < cf.T_d_dw(4e-6))
i_dw_8keV = np.argmax(Ttrel.T_nu_grid < cf.T_d_dw(8e-6))
i_dw_16keV = np.argmax(Ttrel.T_nu_grid < cf.T_d_dw(16e-6))
i_dw_32keV = np.argmax(Ttrel.T_nu_grid < cf.T_d_dw(32e-6))

# norm_32keV = (n32keV/(sf_norm_1MeV[i_dw_32keV]**3.))/(1.5*cf.zeta3*(Ttrel.T_nu_grid[i_dw_32keV]**3.)/cf.pi2)
# f32keV = norm_32keV/(1.+np.exp(p32keV/(Ttrel.T_nu_grid[i_dw_32keV]*sf_norm_1MeV[i_dw_32keV])))
#1.7603541298571859 0.9381245323329195 0.4946138670277145 0.2534925930935097 0.1268067809561153 0.04813452724985895

fs_integrand_1keV = np.zeros(Ttrel.T_SM_grid.size)
fs_integrand_2keV = np.zeros(Ttrel.T_SM_grid.size)
fs_integrand_4keV = np.zeros(Ttrel.T_SM_grid.size)
fs_integrand_8keV = np.zeros(Ttrel.T_SM_grid.size)
fs_integrand_16keV = np.zeros(Ttrel.T_SM_grid.size)
fs_integrand_32keV = np.zeros(Ttrel.T_SM_grid.size)
for i in range(Ttrel.T_SM_grid.size):
    if i >= i_dw_1keV and i <= i_fs_max:
        E1keV = np.sqrt(1e-6**2. + (p1keV/sf_norm_1MeV[i])**2.)
        v1keV = np.trapz(f1keV*(p1keV**3.)/E1keV, p1keV)*2./(2.*(np.pi**2.)*n1keV*sf_norm_1MeV[i])
        fs_integrand_1keV[i] = v1keV/sf_norm_today[i]
    if i >= i_dw_2keV and i <= i_fs_max:
        E2keV = np.sqrt(2e-6**2. + (p2keV/sf_norm_1MeV[i])**2.)
        v2keV = np.trapz(f2keV*(p2keV**3.)/E2keV, p2keV)*2./(2.*(np.pi**2.)*n2keV*sf_norm_1MeV[i])
        fs_integrand_2keV[i] = v2keV/sf_norm_today[i]
    if i >= i_dw_4keV and i <= i_fs_max:
        E4keV = np.sqrt(4e-6**2. + (p4keV/sf_norm_1MeV[i])**2.)
        v4keV = np.trapz(f4keV*(p4keV**3.)/E4keV, p4keV)*2./(2.*(np.pi**2.)*n4keV*sf_norm_1MeV[i])
        fs_integrand_4keV[i] = v4keV/sf_norm_today[i]
    if i >= i_dw_8keV and i <= i_fs_max:
        E8keV = np.sqrt(8e-6**2. + (p8keV/sf_norm_1MeV[i])**2.)
        v8keV = np.trapz(f8keV*(p8keV**3.)/E8keV, p8keV)*2./(2.*(np.pi**2.)*n8keV*sf_norm_1MeV[i])
        fs_integrand_8keV[i] = v8keV/sf_norm_today[i]
    if i >= i_dw_16keV and i <= i_fs_max:
        E16keV = np.sqrt(16e-6**2. + (p16keV/sf_norm_1MeV[i])**2.)
        v16keV = np.trapz(f16keV*(p16keV**3.)/E16keV, p16keV)*2./(2.*(np.pi**2.)*n16keV*sf_norm_1MeV[i])
        fs_integrand_16keV[i] = v16keV/sf_norm_today[i]
    if i >= i_dw_32keV and i <= i_fs_max:
        E32keV = np.sqrt(32e-6**2. + (p32keV/sf_norm_1MeV[i])**2.)
        v32keV = np.trapz(f32keV*(p32keV**3.)/E32keV, p32keV)*2./(2.*(np.pi**2.)*n32keV*sf_norm_1MeV[i])
        fs_integrand_32keV[i] = v32keV/sf_norm_today[i]

fs_length_1keV = utils.simp(Ttrel.t_grid, fs_integrand_1keV) / cf.Mpc
fs_length_2keV = utils.simp(Ttrel.t_grid, fs_integrand_2keV) / cf.Mpc
fs_length_4keV = utils.simp(Ttrel.t_grid, fs_integrand_4keV) / cf.Mpc
fs_length_8keV = utils.simp(Ttrel.t_grid, fs_integrand_8keV) / cf.Mpc
fs_length_16keV = utils.simp(Ttrel.t_grid, fs_integrand_16keV) / cf.Mpc
fs_length_32keV = utils.simp(Ttrel.t_grid, fs_integrand_32keV) / cf.Mpc


print(fs_length_1keV, fs_length_2keV, fs_length_4keV, fs_length_8keV, fs_length_16keV, fs_length_32keV)

plt.loglog([1., 2., 4., 8., 16., 32.], [fs_length_1keV, fs_length_2keV, fs_length_4keV, fs_length_8keV, fs_length_16keV, fs_length_32keV])
# plt.loglog([1., 100.], [fs_length_32keV*32./1., fs_length_32keV*32./100.])
plt.xlabel(r"$m_s \; [\mathrm{keV}]$")
plt.ylabel(r"$\lambda_\mathrm{fs} \; [\mathrm{Mpc}]$")
plt.tight_layout()
# plt.savefig('lambda_fs_dw.pdf')
plt.show()
