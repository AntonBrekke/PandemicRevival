#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, asin, sqrt, exp
from scipy.special import kn

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import constants_functions as cf
import C_res_scalar
import C_res_scalar_no_spin_stat
import scalar_mediator

m_d = 2e-5
m_a = 0.
m_phi = 5e-5
sin2_2th = 3.5e-15
y = 1.474e-3

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

M2_dd = 2. * y2 * (c_th**4.) * (m_phi2 - 4.*m_d2)
M2_aa = 2. * y2 * (s_th**4.) * (m_phi2 - 4.*m_a2)
M2_da = 2. * y2 * (s_th**2.) * (c_th**2.) * (m_phi2 - ((m_a+m_d)**2.))

vert_fi = y2*y2*(c_th**4.)*(s_th**4.)
vert_tr = y2*y2*(c_th**6.)*(s_th**2.)
vert_el = y2*y2*(c_th**8.)

Gamma_phi = scalar_mediator.Gamma_phi(y, th, m_phi, m_d)
m_Gamma_phi2 = m_phi2*Gamma_phi*Gamma_phi

filename = f'rates_md_{m_d:.4e}_mphi_{m_phi:.4e}_sin22th_{sin2_2th:.4e}_y_{y:.4e}.dat'
if not os.path.isfile('./' + filename):
    filename_evo = f'md_{m_d:.4e}_mphi_{m_phi:.4e}_sin22th_{sin2_2th:.4e}_y_{y:.4e}.dat'
    data = np.loadtxt(filename_evo)
    i_skip = 10
    t_grid = data[::i_skip,0]
    T_SM_grid = data[::i_skip,1]
    T_nu_grid = data[::i_skip,2]
    ent_grid = data[::i_skip, 3]
    H_grid = data[::i_skip,4]
    sf_grid = data[::i_skip,5]
    T_d_grid = data[::i_skip,6]
    xi_d_grid = data[::i_skip,7]
    xi_phi_grid = data[::i_skip,8]
    n_d_grid = data[::i_skip,9]
    n_phi_grid = data[::i_skip,10]
    n_nu_grid = 2.*0.75*(cf.zeta3/cf.pi2)*(T_nu_grid**3.)

    C_p_dd = np.array([-2.*C_res_scalar.C_n_3_12(m_d, m_d, m_phi, k_d, k_d, k_phi, T_d, T_d, T_d, xi_d, xi_d, xi_phi, M2_dd, type=-1) / 2. for T_d, xi_d, xi_phi in zip(T_d_grid, xi_d_grid, xi_phi_grid)])
    C_dd_p = np.array([2.*C_res_scalar.C_n_3_12(m_d, m_d, m_phi, k_d, k_d, k_phi, T_d, T_d, T_d, xi_d, xi_d, xi_phi, M2_dd, type=1) / 2. for T_d, xi_d, xi_phi in zip(T_d_grid, xi_d_grid, xi_phi_grid)])
    C_p_da = np.array([-C_res_scalar.C_n_3_12(m_d, m_a, m_phi, k_d, k_a, k_phi, T_d, T_a, T_d, xi_d, 0., xi_phi, M2_da, type=-1) for T_d, T_a, xi_d, xi_phi in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_phi_grid)])
    C_da_p = np.array([C_res_scalar.C_n_3_12(m_d, m_a, m_phi, k_d, k_a, k_phi, T_d, T_a, T_d, xi_d, 0., xi_phi, M2_da, type=1) for T_d, T_a, xi_d, xi_phi in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_phi_grid)])
    C_p_aa = np.array([-2.*C_res_scalar.C_n_3_12(m_a, m_a, m_phi, k_d, k_a, k_phi, T_a, T_a, T_d, 0., 0., xi_phi, M2_aa, type=-1) / 2. for T_d, T_a, xi_d, xi_phi in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_phi_grid)])
    C_aa_p = np.array([2.*C_res_scalar.C_n_3_12(m_a, m_a, m_phi, k_d, k_a, k_phi, T_a, T_a, T_d, 0., 0., xi_phi, M2_aa, type=1) / 2. for T_d, T_a, xi_d, xi_phi in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_phi_grid)])
    C_pp_dd_both = np.array([2.*C_res_scalar.C_n_pp_dd(m_d, m_phi, k_d, k_phi, T_d, xi_d, xi_phi, vert_el, type=2) / 4. for T_d, xi_d, xi_phi in zip(T_d_grid, xi_d_grid, xi_phi_grid)])
    # C_dd_dd = np.array([C_res_scalar_no_spin_stat.C_12_34(m_d, m_d, m_d, m_d, k_d, k_d, T_d, T_d, xi_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, type=0, res_sub=True) / 4. for T_d, xi_d in zip(T_d_grid, xi_d_grid)])
    # C_da_dd = np.array([C_res_scalar_no_spin_stat.C_12_34(m_d, m_a, m_d, m_d, k_d, k_a, T_d, T_a, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, type=0, res_sub=True) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_dd = np.zeros(T_d_grid.size)#np.array([C_res_scalar.C_34_12(0, 1., 0., m_d, m_d, m_d, m_d, k_d, k_d, k_d, k_d, T_d, T_d, T_d, T_d, xi_d, xi_d, xi_d, xi_d, vert_el, m_phi2, m_Gamma_phi2, res_sub=False) / 4. for T_d, xi_d in zip(T_d_grid, xi_d_grid)])
    C_da_dd = np.zeros(T_d_grid.size)#np.array([C_res_scalar.C_34_12(0, 1., 0., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2, res_sub=False) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_da = np.zeros(T_d_grid.size)#np.array([C_res_scalar.C_34_12(0, 0., 1., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_phi2, m_Gamma_phi2) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_aa_dd = np.zeros(T_d_grid.size)#np.array([C_res_scalar.C_34_12(0, 2., 0., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_phi2, m_Gamma_phi2) / 4. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_aa = np.zeros(T_d_grid.size)#np.array([C_res_scalar.C_34_12(0, 0., 2., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_phi2, m_Gamma_phi2) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_pp_dd = -C_pp_dd_both[:,0]
    C_dd_pp = C_pp_dd_both[:,1]

    np.savetxt(filename, np.column_stack((m_d/T_nu_grid, H_grid, C_p_dd/n_d_grid, C_dd_p/n_d_grid, C_p_da/n_d_grid, C_da_p/n_d_grid, C_p_aa/n_d_grid, C_aa_p/n_d_grid, C_pp_dd/n_d_grid, C_dd_pp/n_d_grid, C_dd_dd/n_d_grid, C_da_dd/n_d_grid, C_dd_da/n_d_grid, C_aa_dd/n_d_grid, C_dd_aa/n_d_grid)))

data = np.loadtxt(filename)
x_grid = data[:,0]
H = data[:,1]
C_p_dd = data[:,2]
C_dd_p = data[:,3]
C_p_da = data[:,4]
C_da_p = data[:,5]
C_p_aa = data[:,6]
C_aa_p = data[:,7]
C_pp_dd = data[:,8]
C_dd_pp = data[:,9]
C_dd_dd = data[:,10]
C_da_dd = data[:,11]
C_dd_da = data[:,12]
C_aa_dd = data[:,13]
C_dd_aa = data[:,14]

# plt.loglog(x_grid, C_da_dd/C_da_p, color='dodgerblue')
# plt.show()
#
# plt.loglog(x_grid, C_dd_dd/C_dd_p, color='dodgerblue')
# plt.loglog(x_grid, C_dd_dd/C_p_dd, color='darkorange', ls='--')
# plt.show()
# exit(1)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

fig = plt.figure(figsize=(0.4*12.0, 0.4*11.0), dpi=150, edgecolor="white")
ax = fig.add_subplot(1,1,1)
ax.tick_params(axis='both', which='both', labelsize=11, direction="in", width=0.5)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)

plt.loglog(x_grid, 1e24*H, color='#83781B', ls='-')
# plt.loglog(x_grid, 1e24*Hnphi, color='darkorange', ls='-')
# plt.loglog(x_grid, 1e24*Hnnu, color='#A300CC', ls='-')
plt.loglog(x_grid, 1e24*C_dd_p, color='#114B5F', ls='-')
plt.loglog(x_grid, 1e24*C_dd_p*5.468e-15/1.01e-22, color='#186E8B', ls='-')
plt.loglog(x_grid, 1e24*C_da_p, color='#458751', ls='-')
# plt.loglog(x_grid, 1e24*C_p_da, color='#53A262', ls='--')
# plt.loglog(x_grid, 1e24*C_aa_p, color='#83781B', ls='-')
# plt.loglog(x_grid, 1e24*C_p_aa, color='#A99C23', ls='--')
plt.loglog(x_grid, 1e24*C_dd_pp, color='#95190C', ls='-')
plt.loglog(x_grid, 1e24*C_pp_dd, color='#D02411', ls='-')
# plt.loglog(x_grid, 1e24*C_dd_dd, color='#114B5F', ls=':')
# plt.loglog(x_grid, 1e24*C_da_dd, color='#458751', ls=':')
# plt.loglog(x_grid, 1e24*C_dd_da, color='#53A262', ls=':')
# plt.loglog(x_grid, 1e24*C_dd_aa, color='#A99C23', ls=':')
# plt.loglog(x_grid, 1e24*C_aa_dd, color='#83781B', ls=':')

ax.text(2e-2, 8e-1, r"$3 H$", color='#83781B', fontsize=12, rotation=-28)
# # # ax.text(0.7e0, 1e24*7e-43, r"$3 H n_\alpha$", color='#A300CC', fontsize=9)
ax.text(2e-2, 3e2, r"$\nu_s \nu_s \to \phi$", color='#114B5F', fontsize=12, rotation=13)
ax.text(1e-1, 1.5e3, r"$\phi \to \nu_s \nu_s$", color='#186E8B', fontsize=12, rotation=13)
# ax.text(5.5e-2, 3e-20, r"$\nu_s \nu_s \to \nu_s \nu_s$", color='#114B5F', fontsize=9, rotation=-12)
ax.text(2e-2, 6e-5, r"$\nu_s \nu_\alpha \to \phi$", color='#458751', fontsize=12, rotation=13)
# ax.text(8e-1, 2e-27, r"$\nu_s \nu_\alpha \to \nu_s \nu_s$", color='#458751', fontsize=9)
# ax.text(1.9e-2, 2.5e-30, r"$\phi \to \nu_s \nu_\alpha$", color='#53A262', fontsize=9, rotation=-12)
# ax.text(1.9e-2, 1e-32, r"$\nu_s \nu_s \to \nu_s \nu_\alpha$", color='#53A262', fontsize=9, rotation=-12)
# # # ax.text(2e-3, 1e24*1e-55, r"$\phi \to \nu_\alpha \nu_\alpha$", color='#A99C23', fontsize=9)
# # # ax.text(2e-3, 1e24*4e-51, r"$\nu_\alpha \nu_\alpha \to \phi$", color='#83781B', fontsize=9)
# ax.text(1.3e-2, 1.3e-12, r"$\nu_s \nu_s \to \phi \phi$", color='#95190C', fontsize=9, rotation=-25)
# ax.text(1.3e-2, 1.5e-12, r"$\phi \phi \to \nu_s \nu_s$", color='#D02411', fontsize=9, rotation=-25)

props = dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=1, edgecolor="0.8")
# ax.text(6e-1, 9e-18, r"$m_s = 15 \, \mathrm{keV}$" + "\n" + r"$m_\phi = 37.5 \, \mathrm{keV}$" + "\n" + r"$\sin^2 (2 \theta) = 1.5 \times 10^{-13}$" + "\n" + r"$y = 2.44 \times 10^{-4}$", horizontalalignment="left", fontsize=10, bbox=props)

ax.xaxis.set_label_text(r"$m_s / T_\nu$")
ax.yaxis.set_label_text(r"$\text{Rate} \;\; [\mathrm{keV}]$")
ax.set_xlim(1e-5, 1e1)
ax.set_ylim(1e-6, 1e8)
plt.tight_layout()
# plt.savefig('rates_2.pdf')
plt.show()
