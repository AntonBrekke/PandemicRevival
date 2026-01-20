#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter

import matplotlib
matplotlib.rcParams['hatch.linewidth'] = 8.0

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import constants_functions as cf

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
# # plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(0.4*12.0, 0.4*11.0), dpi=150, edgecolor="white", gridspec_kw={'height_ratios': [2, 1]})
ax1.tick_params(axis='both', which='both', labelsize=11, direction="in", width=0.5)
ax1.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(0.5)
ax2.tick_params(axis='both', which='both', labelsize=11, direction="in", width=0.5)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax2.spines[axis].set_linewidth(0.5)

xtMajor = np.array([np.log10(10**j) for j in np.linspace(-5, 2, 8)])
xtMinor = np.array([np.log10(i*10**j) for j in xtMajor for i in range(10)[1:10]])
xlMajor = [r"$10^{" + str(int(i)) + "}$" if i in xtMajor else "" for i in xtMajor]
xtMajor = 10**xtMajor
xtMinor = 10**xtMinor
xMajorLocator = FixedLocator(xtMajor)
xMinorLocator = FixedLocator(xtMinor)
xMajorFormatter = FixedFormatter(xlMajor)

# Anton: Formatting is to move y-tick scale from GeV to keV (str(int(i+6)))
ytMajor = np.array([np.log10(10**j) for j in np.linspace(-25, 5, 25-(-5)+1)])
ytMinor = np.array([np.log10(i*10**j) for j in ytMajor for i in range(10)[1:10]])
ylMajor = [r"$10^{" + str(int(i+6)) + "}$" if i in ytMajor[::2] else "" for i in ytMajor]
ytMajor = 10**ytMajor
ytMinor = 10**ytMinor
yMajorLocator = FixedLocator(ytMajor)
yMinorLocator = FixedLocator(ytMinor)
yMajorFormatter = FixedFormatter(ylMajor)

"""
Had to fix: the right entropy 'ent' was not returned, which caused trouble in plots. 
Should be fixes now. 

data = np.loadtxt('./md_1e-5_mX_2.5e-5_sin22th_1e-12_y_5.6e-5_full.dat')
0: t_grid 
1: T_SM_grid
2: T_nu_grid
3: ent_grid
4: hubble_grid
5: sf_grid / sf_grid
6: T_chi_grid_sol
7: xi_chi_grid_sol
8: xi_X_grid_sol
9: n_chi_grid_sol
10: n_X_grid_sol

data: Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_X_grid_sol, pan.n_chi_grid_sol, pan.n_X_grid_sol
"""

load_str_1 = './md_1.35388e-06;mX_4.06163e-06;sin22th_2.06914e-12;y_1.21276e-05;full.dat'
load_str_2 = './md_3.79269e-05;mX_1.13781e-04;sin22th_4.28133e-16;y_2.56926e-03;full_new.dat'
data_1 = np.loadtxt(load_str_1)
data_2 = np.loadtxt(load_str_2)

T_SM_1 = data_1[:,1]
T_nu_1 = data_1[:,2]
ent_1 = data_1[:,3]
Td_1 = data_1[:,6]
xid_1 = data_1[:,7]
xiX_1 = data_1[:,8]
nd_1 = data_1[:,9]
nX_1 = data_1[:,10]

T_SM_2 = data_2[:,1]
T_nu_2 = data_2[:,2]
ent_2 = data_2[:,3]
Td_2 = data_2[:,6]
xid_2 = data_2[:,7]
xiX_2 = data_2[:,8]
nd_2 = data_2[:,9]
nX_2 = data_2[:,10]

# Mass: 1e-6 * X GeV = X keV
var_list_1 = load_str_1.split(';')[:-1]
md_1, mX_1, sin22th_1, y_1 = [eval(s.split('_')[-1]) for s in var_list_1]
print(f'md_1: {md_1:.2e}, mX_1: {mX_1:.2e}, sin22th_1: {sin22th_1:.2e}, y_1: {y_1:.2e}')

var_list_2 = load_str_2.split(';')[:-1]
md_2, mX_2, sin22th_2, y_2 = [eval(s.split('_')[-1]) for s in var_list_2]
print(f'md_2: {md_2:.2e}, mX_2: {mX_2:.2e}, sin22th_2: {sin22th_2:.2e}, y_2: {y_2:.2e}')


mY_relic = cf.omega_d0 * cf.rho_crit0_h2 / cf.s0        # m*Y = m*n/s = Omega * rho_c0 / s0

T_grid_dw = np.logspace(np.log10(1.4e-3), 1, 400)

mYd_dw_1 = cf.O_h2_dw_Tevo(T_grid_dw, md_1, 0.5*np.arcsin(np.sqrt(sin22th_1)))*cf.rho_crit0_h2 / cf.s0     # Anton: mY from Dodelson-Widrow
mYd_dw_2 = cf.O_h2_dw_Tevo(T_grid_dw, md_2, 0.5*np.arcsin(np.sqrt(sin22th_2)))*cf.rho_crit0_h2 / cf.s0     # Anton: mY from Dodelson-Widrow

if True:

    x_dw_1 = md_1/T_grid_dw
    y_dw_1 = mYd_dw_1

    x_dw_2 = md_2/T_grid_dw
    y_dw_2 = mYd_dw_2

    x_tr_1 = md_1/T_nu_1
    y_tr_1 = md_1*nd_1/ent_1          

    x_tr_2 = md_2/T_nu_2
    y_tr_2 = md_2*nd_2/ent_2          

    x_dw0_1, x_tr0_1 = x_dw_1[x_dw_1 < 1e-3], x_tr_1[x_tr_1 > 1e-3]
    y_dw0_1, y_tr0_1 = y_dw_1[x_dw_1 < 1e-3], y_tr_1[x_tr_1 > 1e-3]

    x_dw0_2, x_tr0_2 = x_dw_2[x_dw_2 < 1e-3], x_tr_2[x_tr_2 > 1e-3]
    y_dw0_2, y_tr0_2 = y_dw_2[x_dw_2 < 1e-3], y_tr_2[x_tr_2 > 1e-3]

    x1, y1 = np.array([*x_dw0_1[::-1], *x_tr0_1, 1e3]), np.array([*y_dw0_1[::-1], *y_tr0_1, y_tr0_1[-1]])
    x2, y2 = np.array([*x_dw0_2[::-1], *x_tr0_2, 1e3]), np.array([*y_dw0_2[::-1], *y_tr0_2, y_tr0_2[-1]])

    # ax1.loglog(x_tr_1, y_tr_1, color='#7bc043', ls='-', zorder=-1)
    # ax1.loglog(x_tr_2, y_tr_2, color='#7bc043', ls='--', zorder=-1)
    # ax1.loglog(x_dw_1[x_dw_1 < 1e-3], y_dw_1[x_dw_1 < 1e-3], color='r', ls='-', zorder=-1)
    # ax1.loglog(x_dw_2[x_dw_2 < 1e-3], y_dw_2[x_dw_2 < 1e-3], color='r', ls='--', zorder=-1)
    ax1.loglog(x1, y1, color='#7bc043', ls='-', zorder=-1)
    ax1.loglog(x2, y2, color='#7bc043', ls='--', zorder=-1)

    ax1.fill_betweenx([1e-23, 1e5], 1e-5, 1e-3, color='white', alpha=1, zorder=-3)
    # ax1.fill_betweenx([1e-23, 1e-18], 1e-5, 1e-3, facecolor="white", hatch="\\", edgecolor="0.9", zorder=1)

    #ax1.text(8e-5, 2e-21, 'Thermalization', fontsize=10, color='darkorange')
    #ax1.text(5e-4, 1.3e-19, r'$\rightarrow$', color='darkorange', horizontalalignment='center', verticalalignment='center')
    #ax1.text(5e-4, 1e-22, r'$\rightarrow$', color='darkorange', horizontalalignment='center', verticalalignment='center')
    ax1.plot([1e-3]*2, [1e-25, 2e-8], ls=':', color='0', zorder=-2)
    ax2.plot([1e-3]*2, [1e-2, 2e0], ls=':', color='0', zorder=-2)

    ax1.text(1.5e-4, 8e-21, r'$\mathrm{Dark}$', fontsize=8, color='0', horizontalalignment='center')
    ax1.text(1.5e-4, 8e-22, r'$\mathrm{Thermalization}$', fontsize=8, color='0', horizontalalignment='center')
    ax1.text(1.5e-4, 8e-23, r'$\rightarrow$', fontsize=8, color='0', horizontalalignment='center')
    #ax1.text(4.5e-5, 1e-22, r'$\hspace{-0.55cm}\mathrm{Therma-}\\\mathrm{lization}\\\mathrm{ }\hspace{0.2cm}\rightarrow$', fontsize=10, color='0')


    ax1.loglog(md_1/T_nu_1, mX_1*nX_1/ent_1, color='#f37736', ls='-', zorder=-4)
    ax1.loglog(md_2/T_nu_2, mX_2*nX_2/ent_2, color='#f37736', ls='--', zorder=-4)

    ax1.loglog([1e-8, 1e3], [mY_relic, mY_relic], color='0.65', ls='-.', zorder=-2)
    ax1.text(3e-5, 1e-11, r'$\Omega_s h^2 = 0.12$', fontsize=11, color='0.65')

    ax1.text(2.5, 1e-11, r'$\nu_s$', color='#7bc043', fontsize=11)
    ax1.text(2.0, 1e-16, r'$X$', color='#f37736', fontsize=11)

    ax2.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='-' , color='black', label=r'$\text{BP1}$')
    ax2.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='--', color='black', label=r'$\text{BP2}$')

    ax2.loglog(md_1/T_nu_1, Td_1/T_nu_1, color='0.4', ls='-', zorder=-4)
    ax2.loglog(md_2/T_nu_2, Td_2/T_nu_2, color='0.4', ls='--', zorder=-4)

    ax2.fill_betweenx([1e-1, 1.5e0], 1e-5, 1e-3, color='white', alpha=1, zorder=-3)

    props = dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=1, edgecolor="0.8")

    #plt.text(2e-2, 1e-11, r'$m_X = 2.5m_\chi$', fontsize=9, horizontalalignment='center', bbox=props)
    #plt.text(1e0, 3e-23, r'$m_X = 2.5m_\chi$', fontsize=9, horizontalalignment='center', bbox=props, zorder=5)
    #ax1.text(1e0, 8e-25, r'$m_X = 2.5m_\chi$', fontsize=10, horizontalalignment='center', zorder=5)


    ax2.legend(fontsize=10, framealpha=0.8, edgecolor='1')
    ax2.xaxis.set_label_text(r"$m_s / T_\nu$")
    ax1.yaxis.set_label_text(r"$m\, n / s\;\;\mathrm{[keV]}$")
    ax2.yaxis.set_label_text(r"$T_\text{d}/T_\nu$")


    ax1.xaxis.set_major_locator(xMajorLocator)
    ax1.xaxis.set_minor_locator(xMinorLocator)
    ax1.xaxis.set_major_formatter(xMajorFormatter)
    ax1.yaxis.set_major_locator(yMajorLocator)
    ax1.yaxis.set_minor_locator(yMinorLocator)
    ax1.yaxis.set_major_formatter(yMajorFormatter)

    plt.xlim(2e-5, 20)

    # ylim + 6 will be shown
    ax1.set_ylim(1e-25, 2e-8)
    ax2.set_ylim(1e-2, 2e0)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    # plt.savefig(f'dens_evo_{load_str_1.replace("./", "").replace(".dat","")}_final.pdf')
    plt.show()
elif False:
    plt.loglog(md1/T_nu1, 1e6*Td1, color='dodgerblue', ls='-')
    plt.loglog(md2/T_nu2, 1e6*Td2, color='dodgerblue', ls='--')
    ax.xaxis.set_label_text(r"$m_s / T_\nu$")
    ax.yaxis.set_label_text(r"$T_s \; \; [\mathrm{keV}]$")
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(1e-1, 1e2)
    plt.tight_layout()
    # plt.savefig('Td.pdf')
    plt.show()
else:
    plt.semilogx(md1/T_nu1, xid1, color='dodgerblue', ls='-')
    plt.semilogx(md2/T_nu2, xid2, color='dodgerblue', ls='--')
    ax.xaxis.set_label_text(r"$m_\mathrm{d} / T_\nu$")
    ax.yaxis.set_label_text(r"$\mu_\mathrm{d} / m_\mathrm{d}$")
    ax.set_xlim(1e-3, 1e1)
    # ax.set_ylim(1e-1, 1e2)
    plt.tight_layout()
    # plt.savefig('mud.pdf')
    plt.show()
