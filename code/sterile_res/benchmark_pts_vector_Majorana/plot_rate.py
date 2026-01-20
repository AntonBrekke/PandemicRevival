#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, asin, sqrt, exp
from scipy.special import kn
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
import time

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
import C_res_vector
import C_res_scalar
import C_res_vector_no_spin_stat
import vector_mediator

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
# # plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

load_str = ''

var_list = load_str.split(';')[:-1]
m_d, m_X, sin2_2th, y = [eval(s.split('_')[-1]) for s in var_list]
m_a = 0.

print(m_d, m_X, sin2_2th, y)

md_str = f'{m_d:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_d:.5e}'.split('e')[1].rstrip('0').rstrip('.')
mX_str = f'{m_X:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_X:.5e}'.split('e')[1].rstrip('0').rstrip('.')
sin22th_str = f'{sin2_2th:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{sin2_2th:.5e}'.split('e')[1].rstrip('0').rstrip('.')
y_str = f'{y:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{y:.5e}'.split('e')[1].rstrip('0').rstrip('.')

k_d = 1.
k_a = 1.
k_X = -1.
dof_d = 2.
dof_X = 3.

m_d2 = m_d*m_d
m_a2 = m_a*m_a
m_X2 = m_X*m_X
th = 0.5*asin(sqrt(sin2_2th))
c_th = cos(th)
s_th = sin(th)
y2 = y*y

# M2_X23 = 2*g**2/m_X2 * (m_X2 - (m2 - m3)**2)*(2*m_X2 + (m2 + m3)**2)
# New matrix elements for X --> 23
# M2_dd = 2.*y2*(c_th**4.)/m_X2 * (m_X2)*(2*m_X2 + (m_d + m_d)**2)
# M2_aa = 2.*y2*(s_th**4.)/m_X2 * (m_X2)*(2*m_X2)
# M2_da = 2.*y2*(s_th**2.)*(c_th**2.)/m_X2 * (m_X2 - m_d**2)*(2*m_X2 + m_d**2)

# Anton: Test if new Feynman rules work. M2_da x2 larger, M2_dd change
M2_dd = 4.*y2*(c_th**4.)*(m_X2-4*m_d2)
M2_aa = 4.*y2*(s_th**4.)*m_X2
M2_da = 4.*y2*(s_th**2.)*(c_th**2.)/m_X2 * (m_X2 - m_d2)*(2*m_X2 + m_d2)

print(f'M2_dd: {M2_dd:3e}, M2_da: {M2_da:3e}, M2_aa: {M2_aa:3e}')

vert_fi = y2*y2*(c_th**4.)*(s_th**4.)
vert_tr = y2*y2*(c_th**6.)*(s_th**2.)
vert_el = y2*y2*(c_th**8.)

Gamma_X = vector_mediator.Gamma_X_new(y, th, m_X, m_d)
m_Gamma_X2 = m_X2*Gamma_X*Gamma_X

data_evo = np.loadtxt(load_str)
"""
0: t_grid 
1: T_SM_grid
2: T_nu_grid
3: ent_grid
4: hubble_grid
5: sf_grid / sf_grid
6: T_chi_grid_sol
7: xi_chi_grid_sol
8: xi_X_grid_sol (xi_phi_grid_sol)
9: n_chi_grid_sol
10: n_X_grid_sol (n_phi_grid_sol)
"""

i_skip = 10
t_grid = data_evo[::i_skip,0]
T_SM_grid = data_evo[::i_skip,1]
T_nu_grid = data_evo[::i_skip,2]
ent_grid = data_evo[::i_skip, 3]
H_grid = data_evo[::i_skip,4]
sf_grid = data_evo[::i_skip,5]
T_d_grid = data_evo[::i_skip,6]
xi_d_grid = data_evo[::i_skip,7]
xi_X_grid = data_evo[::i_skip,8]
n_d_grid = data_evo[::i_skip,9]
n_X_grid = data_evo[::i_skip,10]
n_nu_grid = 2.*0.75*(cf.zeta3/cf.pi2)*(T_nu_grid**3.)

filename = f'rates_md_{md_str};mX_{mX_str};sin22th_{sin22th_str};y_{y_str};full.dat'
# Anton: Check if file exists -- if not, create it 
# if not os.path.isfile('./' + filename):
if True:
    time_start = time.time()
    print('Get C_X_dd')
    C_X_dd = np.array([-2.*C_res_vector.C_n_3_12(m_d, m_d, m_X, k_d, k_d, k_X, T_d, T_d, T_d, xi_d, xi_d, xi_X, M2_dd, type=-1) / 2. for T_d, xi_d, xi_X in zip(T_d_grid, xi_d_grid, xi_X_grid)])
    print('Get C_dd_X')
    C_dd_X = np.array([2.*C_res_vector.C_n_3_12(m_d, m_d, m_X, k_d, k_d, k_X, T_d, T_d, T_d, xi_d, xi_d, xi_X, M2_dd, type=1) / 2. for T_d, xi_d, xi_X in zip(T_d_grid, xi_d_grid, xi_X_grid)])
    print('Get C_X_da')
    C_X_da = np.array([-C_res_vector.C_n_3_12(m_d, m_a, m_X, k_d, k_a, k_X, T_d, T_a, T_d, xi_d, 0., xi_X, M2_da, type=-1) for T_d, T_a, xi_d, xi_X in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid)])
    print('Get C_da_X')
    C_da_X = np.array([C_res_vector.C_n_3_12(m_d, m_a, m_X, k_d, k_a, k_X, T_d, T_a, T_d, xi_d, 0., xi_X, M2_da, type=1) for T_d, T_a, xi_d, xi_X in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid)])
    print('Get C_X_aa')
    C_X_aa = np.array([-2.*C_res_vector.C_n_3_12(m_a, m_a, m_X, k_d, k_a, k_X, T_a, T_a, T_d, 0., 0., xi_X, M2_aa, type=-1) / 2. for T_d, T_a, xi_d, xi_X in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid)])
    print('Get C_aa_X')
    C_aa_X = np.array([2.*C_res_vector.C_n_3_12(m_a, m_a, m_X, k_d, k_a, k_X, T_a, T_a, T_d, 0., 0., xi_X, M2_aa, type=1) / 2. for T_d, T_a, xi_d, xi_X in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid)])
    print('Get C_XX_dd_both')
    C_XX_dd_both = np.array([2.*C_res_vector.C_n_XX_dd(m_d, m_X, k_d, k_X, T_d, xi_d, xi_X, vert_el, type=2) / 4. for T_d, xi_d, xi_X in zip(T_d_grid, xi_d_grid, xi_X_grid)])
    # C_dd_dd = np.array([C_res_vector_no_spin_stat.C_12_34(m_d, m_d, m_d, m_d, k_d, k_d, T_d, T_d, xi_d, xi_d, vert_el, m_X2, m_Gamma_X2, type=0, res_sub=True) / 4. for T_d, xi_d in zip(T_d_grid, xi_d_grid)])
    # C_da_dd = np.array([C_res_vector_no_spin_stat.C_12_34(m_d, m_a, m_d, m_d, k_d, k_a, T_d, T_a, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, type=0, res_sub=True) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_dd = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 1., 0., m_d, m_d, m_d, m_d, k_d, k_d, k_d, k_d, T_d, T_d, T_d, T_d, xi_d, xi_d, xi_d, xi_d, vert_el, m_X2, m_Gamma_X2, res_sub=False) / 4. for T_d, xi_d in zip(T_d_grid, xi_d_grid)])
    C_da_dd = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 1., 0., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, res_sub=False) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_da = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 0., 1., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_X2, m_Gamma_X2) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_aa_dd = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 2., 0., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_X2, m_Gamma_X2) / 4. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_aa = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 0., 2., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_X2, m_Gamma_X2) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_XX_dd = -C_XX_dd_both[:,0]
    C_dd_XX = C_XX_dd_both[:,1]

    np.savetxt(filename, np.column_stack((m_d/T_nu_grid, H_grid, C_X_dd/n_d_grid, C_dd_X/n_d_grid, C_X_da/n_d_grid, C_da_X/n_d_grid, C_X_aa/n_d_grid, C_aa_X/n_d_grid, C_XX_dd/n_d_grid, C_dd_XX/n_d_grid, C_dd_dd/n_d_grid, C_da_dd/n_d_grid, C_dd_da/n_d_grid, C_aa_dd/n_d_grid, C_dd_aa/n_d_grid)))
    print(f'Saved file in {time.time()-time_start}s')

data = np.loadtxt(filename)

x_grid = data[:,0]
H = data[:,1]
C_X_dd = data[:,2]
C_dd_X = data[:,3]
C_X_da = data[:,4]
C_da_X = data[:,5]
C_X_aa = data[:,6]
C_aa_X = data[:,7]
C_XX_dd = data[:,8]
C_dd_XX = data[:,9]
C_dd_dd = data[:,10]
C_da_dd = data[:,11]
C_dd_da = data[:,12]
C_aa_dd = data[:,13]
C_dd_aa = data[:,14]

# plt.loglog(x_grid, C_da_dd/C_da_X, color='dodgerblue')
# plt.show()
#
# plt.loglog(x_grid, C_dd_dd/C_dd_X, color='dodgerblue')
# plt.loglog(x_grid, C_dd_dd/C_X_dd, color='darkorange', ls='--')
# plt.show()
# exit(1)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

fig = plt.figure(figsize=(0.4*12.0, 0.4*11.0), dpi=150, edgecolor="white")
ax = fig.add_subplot(1,1,1)
ax.tick_params(axis='both', which='both', labelsize=11, direction="in", width=0.5)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)

ytMajor = np.array([np.log10(10**j) for j in np.linspace(-28, 3, 32)])
ytMinor = np.array([np.log10(i*10**j) for j in ytMajor for i in range(10)[1:10]])
ylMajor = [r"$10^{" + str(int(i)) + "}$" if i in ytMajor[::2] else "" for i in ytMajor]
ytMajor = 10**ytMajor
ytMinor = 10**ytMinor
yMajorLocator = FixedLocator(ytMajor)
yMinorLocator = FixedLocator(ytMinor)
yMajorFormatter = FixedFormatter(ylMajor)

ch = 'crimson'
c1 = '#797ef6' #'#5170d7'
c2 = '#1aa7ec' #'mediumorchid'
c3 = '#4adede' #'crimson'
c5 = '#1e2f97' #'#f0944d'

c4 = '#ffa62b'

print(np.max(abs(C_X_aa)))
print(np.max(abs(C_aa_X)))

# Anton: 1e6 to make GeV to keV 
plt.loglog(x_grid, 1e6*3*H, color=ch, ls='-', zorder=0) #83781B
plt.loglog(x_grid, 1e6*abs(C_dd_X), color=c1, ls='-', zorder=-4) #114B5F
plt.loglog(x_grid, 1e6*abs(C_da_X), color=c2, ls='-', zorder=-4) #458751
plt.loglog(x_grid, 1e6*abs(C_X_aa), color='brown', ls='-')
plt.loglog(x_grid, 1e6*abs(C_aa_X), color='purple', ls='-')
plt.loglog(x_grid, 1e6*abs(C_dd_XX), color=c3, ls='-', zorder=-4) #95190C
plt.loglog(x_grid, 1e6*abs(C_XX_dd), color=c5, ls='-', zorder=-4) #D02411

plt.text(1.5e-4, 8e-21, r'$\mathrm{Dark}$', fontsize=8, color='0', horizontalalignment='center')
plt.text(1.5e-4, 8e-22, r'$\mathrm{Thermalization}$', fontsize=8, color='0', horizontalalignment='center')
plt.text(1.5e-4, 8e-23, r'$\rightarrow$', fontsize=8, color='0', horizontalalignment='center')
#plt.text(4.5e-5, 1e-22, r'$\hspace{-0.55cm}\mathrm{Therma-}\\\mathrm{lization}\\\mathrm{ }\hspace{0.2cm}\rightarrow$', fontsize=10, color='0')


ax.text(2e-4, 3e-14, r"$H$", color=ch, fontsize=10, rotation=0)
ax.text(9e-2, 2e-13, r"$\nu_s \nu_s \leftrightarrow X$", color=c1, fontsize=10, rotation=0)
ax.text(1.5e-3, 3e-25, r"$\nu_s \nu_\alpha \to X$", color=c2, fontsize=10, rotation=0)
ax.text(1.5e-3, 0.3e-19, r"$\nu_s \nu_s \to X X$", color=c3, fontsize=10, rotation=0)
ax.text(2.2e-3, 8e-28, r"$X X \to \nu_s \nu_s$", color=c5, fontsize=10, rotation=0)

plt.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='-', color='black', label=r'$\text{BP1}$')
plt.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='--', color='black', label=r'$\text{BP2}$')


# plt.fill_betweenx([1e-28, 1e0], 1e-5, 1e-3, color='white', alpha=1, zorder=-3)
plt.loglog([1e-3]*2, [1e-28, 1e0], ls=':', color='0', zorder=-2)
#plt.fill_betweenx([1e-28, 1e-8], 1e-5, 1e-3, facecolor="white", hatch="\\", edgecolor="0.9", zorder=1)

# N = 1000
# for i in range(N)[:-1]:
#     x1 = -7 + (-3.5 - (-7))*i/(N-1)
#     x2 = -7 + (-3.5 - (-7))*(i+1)/(N-1)
#
#     plt.fill_betweenx([1e-28, 1e-6], 10**x1, 10**x2, color='white', alpha=1-i/N, zorder=1)
# plt.fill_betweenx([1e-28, 1e-6], 1e-5, 3e-5, color='white', alpha=1, zorder=1)


props = dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=1, edgecolor="0.8")
# ax.text(6e-1, 9e-18, r"$m_s = 15 \, \mathrm{keV}$" + "\n" + r"$m_\phi = 37.5 \, \mathrm{keV}$" + "\n" + r"$\sin^2 (2 \theta) = 1.5 \times 10^{-13}$" + "\n" + r"$y = 2.44 \times 10^{-4}$", horizontalalignment="left", fontsize=10, bbox=props)

plt.legend(fontsize=9, framealpha=0.8, edgecolor='none')

ax.xaxis.set_label_text(r"$m_s / T_\nu$")
ax.yaxis.set_label_text(r"$\text{Rate} \;\; [\mathrm{keV}]$")

ax.yaxis.set_major_locator(yMajorLocator)
ax.yaxis.set_minor_locator(yMinorLocator)
ax.yaxis.set_major_formatter(yMajorFormatter)

plt.xlim(2e-5, 20)
ax.set_ylim(1e-28, 1e-8)
plt.tight_layout()
fig_str = f'rates_evo_md_{md_str};mX_{mX_str};sin22th_{sin22th_str};y_{y_str}.pdf'
print(f'saved {fig_str}')
plt.savefig(fig_str)
plt.show()

