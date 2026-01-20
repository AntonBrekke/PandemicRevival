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

n1keV = np.trapz(f1keV*((m1keV[:,0]*1e-3)**2.), m1keV[:,0]*1e-3)*2./(2.*(np.pi**2.))
n2keV = np.trapz(f2keV*((m2keV[:,0]*1e-3)**2.), m2keV[:,0]*1e-3)*2./(2.*(np.pi**2.))
n4keV = np.trapz(f4keV*((m4keV[:,0]*1e-3)**2.), m4keV[:,0]*1e-3)*2./(2.*(np.pi**2.))
n8keV = np.trapz(f8keV*((m8keV[:,0]*1e-3)**2.), m8keV[:,0]*1e-3)*2./(2.*(np.pi**2.))
n16keV = np.trapz(f16keV*((m16keV[:,0]*1e-3)**2.), m16keV[:,0]*1e-3)*2./(2.*(np.pi**2.))
n32keV = np.trapz(f32keV*((m32keV[:,0]*1e-3)**2.), m32keV[:,0]*1e-3)*2./(2.*(np.pi**2.))

avg_mom_1keV = np.trapz(f1keV*((m1keV[:,0]*1e-3)**3.), m1keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n1keV)
avg_mom_2keV = np.trapz(f2keV*((m2keV[:,0]*1e-3)**3.), m2keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n2keV)
avg_mom_4keV = np.trapz(f4keV*((m4keV[:,0]*1e-3)**3.), m4keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n4keV)
avg_mom_8keV = np.trapz(f8keV*((m8keV[:,0]*1e-3)**3.), m8keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n8keV)
avg_mom_16keV = np.trapz(f16keV*((m16keV[:,0]*1e-3)**3.), m16keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n16keV)
avg_mom_32keV = np.trapz(f32keV*((m32keV[:,0]*1e-3)**3.), m32keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n32keV)
print("Momentum")
print(avg_mom_1keV/avg_mom(1e-6))
print(avg_mom_2keV/avg_mom(2e-6))
print(avg_mom_4keV/avg_mom(4e-6))
print(avg_mom_8keV/avg_mom(8e-6))
print(avg_mom_16keV/avg_mom(16e-6))
print(avg_mom_32keV/avg_mom(32e-6))

p1keV = m1keV[:,0]*1e-3
E1keV = np.sqrt(1e-6**2. + p1keV**2.)
p2keV = m2keV[:,0]*1e-3
E2keV = np.sqrt(2e-6**2. + p2keV**2.)
p4keV = m4keV[:,0]*1e-3
E4keV = np.sqrt(4e-6**2. + p4keV**2.)
p8keV = m8keV[:,0]*1e-3
E8keV = np.sqrt(8e-6**2. + p8keV**2.)
p16keV = m16keV[:,0]*1e-3
E16keV = np.sqrt(16e-6**2. + p16keV**2.)
p32keV = m32keV[:,0]*1e-3
E32keV = np.sqrt(32e-6**2. + p32keV**2.)
avg_E_1keV = np.trapz(f1keV*(p1keV**2.)*E1keV, m1keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n1keV)
avg_E_2keV = np.trapz(f2keV*(p2keV**2.)*E2keV, m2keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n2keV)
avg_E_4keV = np.trapz(f4keV*(p4keV**2.)*E4keV, m4keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n4keV)
avg_E_8keV = np.trapz(f8keV*(p8keV**2.)*E8keV, m8keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n8keV)
avg_E_16keV = np.trapz(f16keV*(p16keV**2.)*E16keV, m16keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n16keV)
avg_E_32keV = np.trapz(f32keV*(p32keV**2.)*E32keV, m32keV[:,0]*1e-3)*2./(2.*(np.pi**2.)*n32keV)
print("Energy / momentum")
print(avg_E_1keV/avg_mom_1keV)
print(avg_E_2keV/avg_mom_2keV)
print(avg_E_4keV/avg_mom_4keV)
print(avg_E_8keV/avg_mom_8keV)
print(avg_E_16keV/avg_mom_16keV)
print(avg_E_32keV/avg_mom_32keV)

# sf3 = 1.6878446557055276e-10**3. # T_SM = 1 MeV
sf3 = 1.6828312477726027e-10**3. # T_nu = 1 MeV
Oh2_1keV = n1keV*sf3*1e-6/cf.rho_crit0_h2
Oh2_2keV = n2keV*sf3*2e-6/cf.rho_crit0_h2
Oh2_4keV = n4keV*sf3*4e-6/cf.rho_crit0_h2
Oh2_8keV = n8keV*sf3*8e-6/cf.rho_crit0_h2
Oh2_16keV = n16keV*sf3*16e-6/cf.rho_crit0_h2
Oh2_32keV = n32keV*sf3*32e-6/cf.rho_crit0_h2

print("relic")
print(Oh2_1keV/cf.O_h2_dw(1e-6, th))
print(Oh2_2keV/cf.O_h2_dw(2e-6, th))
print(Oh2_4keV/cf.O_h2_dw(4e-6, th))
print(Oh2_8keV/cf.O_h2_dw(8e-6, th))
print(Oh2_16keV/cf.O_h2_dw(16e-6, th))
print(Oh2_32keV/cf.O_h2_dw(32e-6, th))
exit(1)

ml = np.array([1., 2., 4., 8., 16., 32.])*1e-6
C_e = cf.C_e_dw(ml)
norm = C_e/ml

plt.semilogy(m1keV[:,0], m1keV[:,1]*f_eq(m1keV[:,0])/norm[0], color='darkorange')
plt.semilogy(m2keV[:,0], m2keV[:,1]*f_eq(m2keV[:,0])/norm[1], color='#A300CC')
plt.semilogy(m4keV[:,0], m4keV[:,1]*f_eq(m4keV[:,0])/norm[2], color='#186E8B')
plt.semilogy(m8keV[:,0], m8keV[:,1]*f_eq(m8keV[:,0])/norm[3], color='#83781B')
plt.semilogy(m16keV[:,0], m16keV[:,1]*f_eq(m16keV[:,0])/norm[4], color='#458751')
plt.semilogy(m32keV[:,0], m32keV[:,1]*f_eq(m32keV[:,0])/norm[5], color='#95190C')
plt.show()
