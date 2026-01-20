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
import scalar_mediator

def get_figsize(columnwidth, wf=1.0, hf=(5.**0.5-1.0)/2.0):
    """Parameters:
    - wf [float]:  width fraction in columnwidth units
    - hf [float]:  height fraction in columnwidth units.
                    Set by default to golden ratio.
    - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                            using \\showthe\\columnwidth
    Returns:  [fig_width, fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]

columnwidth = 418.25368     # pt, given by \showthe\textwidth in LaTeX

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
params = {'axes.labelsize': 13,
            'axes.titlesize': 12,
            'font.size': 11} # extend as needed
# print(plt.rcParams.keys())
plt.rcParams.update(params)

x_therm = 1e-3
save_fig = True
data_skip_pan = 2
data_skip_rate = 10
plot_data_skip = 1
force_write = False
### Benchmark Points ###:
BP = 0
if BP == 1:
    # BP1
    load_str = './md_1.12884e-05;mX_5.64419e-05;mh_3.38651e-05;sin22th_1.83298e-13;y_1.93457e-04;full_new.dat' 
    x_therm = 2e-3
if BP  == 2:
    # BP2
    load_str = './md_2.1e-05;mX_1.05e-04;mh_6.3e-05;sin22th_1.5e-15;y_1.313e-03;full_new.dat'
    x_therm = 2e-3
if BP  == 3:
    # BP3
    load_str = './md_4e-06;mX_2e-05;mh_1.2e-05;sin22th_3e-15;y_8.36e-04;full_new.dat'  
    x_therm = 2e-3
if BP == 4:
    # BP4
    load_str = './md_5.13483e-05;mX_2.56742e-04;mh_1.54045e-04;sin22th_3.66524e-16;y_3.36087e-03;full_new.dat' 
    x_therm = 7e-4
if BP == 5:
    # BP4
    load_str = './md_1e-05;mX_2.5e-05;mh_6.25e-05;sin22th_5e-14;y_4e-07;full_new.dat' 
    x_therm = 2e-2
else: 
    BP = None

# load_str = './md_5e-05;mX_1.5e-04;mh_6e-05;sin22th_1e-15;y_2e-03;full_new.dat' 
load_str = './md_1e-05;mX_2.5e-05;sin22th_1e-16;y_2.5e-03;full_new.dat' 
# load_str = './md_1e-05;mX_2.5e-05;sin22th_1e-14;y_4e-04;full_new.dat' 
data = np.loadtxt(load_str)
T_SM = data[::data_skip_pan, 1]
T_nu = data[::data_skip_pan, 2]
ent = data[::data_skip_pan, 3]
Td = data[::data_skip_pan, 6]
xid = data[::data_skip_pan, 7]
xiX = data[::data_skip_pan, 8]
nd = data[::data_skip_pan, 9]
nX = data[::data_skip_pan, 10]

c1 = '#7bc043'      # green
c2 = '#f37736'      # orange

# Mass: 1e-6 * X GeV = X keV
var_list = load_str.split(';')[:-1]
md, mX, sin22th, y = [eval(s.split('_')[-1]) for s in var_list]
print(f'md: {md:.2e}, mX: {mX:.2e}, sin22th: {sin22th:.2e}, y: {y:.2e}')
mY_relic = cf.omega_d0 * cf.rho_crit0_h2 / cf.s0        # m*Y = m*n/s = Omega * rho_c0 / s0

T_grid_dw = np.logspace(np.log10(1.4e-3), 1, 400)
mYd_dw = cf.O_h2_dw_Tevo(T_grid_dw, md, 0.5*np.arcsin(np.sqrt(sin22th)))*cf.rho_crit0_h2 / cf.s0     # Anton: mY from Dodelson-Widrow

fig = plt.figure(figsize=get_figsize(columnwidth, wf=1.5, hf=0.5), dpi=150, edgecolor="white")

gridspec = fig.add_gridspec(2,2, height_ratios=[2, 1], width_ratios=[1, 1], hspace=0.0, wspace=0.0)
ax1 = fig.add_subplot(gridspec[0, 0])
ax2 = fig.add_subplot(gridspec[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gridspec[:, 1])

ax1.tick_params(axis='both', which='both', direction="in", width=0.5)
# ax1.xaxis.set_ticks_position('top')
ax1.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax1.spines[axis].set_linewidth(0.5)
ax2.tick_params(axis='both', which='both', direction="in", width=0.5)
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

x1_dw = md/T_grid_dw
y1_dw = mYd_dw

x1_tr = md/T_nu
y1_tr = md*nd/ent

x1_dw0, x1_tr0 = x1_dw[x1_dw < x_therm*3e-1], x1_tr[x1_tr > x_therm*3e-1]
y1_dw0, y1_tr0 = y1_dw[x1_dw < x_therm*3e-1], y1_tr[x1_tr > x_therm*3e-1]

x1, y1 = np.array([*x1_dw0[::-1], *x1_tr0, 1e3]), np.array([*y1_dw0[::-1], *y1_tr0, y1_tr0[-1]])

ax1.loglog(x1, y1, color=c1, zorder=-1)
# import pandemolator
# Ttrel = pandemolator.TimeTempRelation()
# T_nu_dec = 1.4e-3
ax1.fill_betweenx([1e-28, 1e5], x1[0], x_therm, color='white', alpha=1, zorder=-3)

ax1.axvline([x_therm], ls=':', color='0', zorder=-2)
ax2.axvline([x_therm], ls=':', color='0', zorder=-2)

dark_therm_x = 10**((np.log10(x_therm) + np.log10(2e-5))/2)
dark_therm_x_index = np.where(np.min(np.abs(x1-2e-4)) == np.abs(x1-2e-4))
x_therm_index = np.where(np.min(np.abs(x1-x_therm)) == np.abs(x1-x_therm))
ax1.text(dark_therm_x, 8e-21, r'$\mathrm{Dark}$', color='0', ha='center')
ax1.text(dark_therm_x, 8e-22, r'$\mathrm{Thermalization}$', color='0', ha='center')
ax1.text(dark_therm_x, 8e-23, r'$\rightarrow$', color='0', ha='center')
#ax1.text(4.5e-5, 1e-22, r'$\hspace{-0.55cm}\mathrm{Therma-}\\\mathrm{lization}\\\mathrm{ }\hspace{0.2cm}\rightarrow$', fontsize=10, color='0')

ax1.loglog(md/T_nu, mX*nX/ent, color=c2, ls='-', zorder=-4)

ax1.loglog([1e-8, 1e3], [mY_relic, mY_relic], color='0.55', ls='-.', zorder=-2)
ax1.text(3e-5, 1e-11, r'$\Omega_s h^2 = 0.12$', color='0.55')

YX_max = np.max(mX*nX/ent)
Ys_max = np.max(y1)
if BP != 5:
    ax1.text(md/T_nu[np.where(mX*nX/ent==YX_max)], Ys_max*1e-1, r'$\nu_s$', color=c1, ha='center', va='top')
    ax1.text(md/T_nu[np.where(mX*nX/ent==YX_max)], YX_max*1e-1, r'$X_\mu$', color=c2, ha='center', va='top')
else: 
    ax1.text(md/T_nu[np.where(mX*nX/ent==YX_max)]*20, Ys_max*1e-1, r'$\nu_s$', color=c1, ha='center', va='top')
    ax1.text(md/T_nu[np.where(mX*nX/ent==YX_max)]*6, YX_max*1e-1, r'$X_\mu$', color=c2, ha='center', va='top')

if BP != 5:
    BP_str = r'$\textit{BP' + f'{BP}' + r'}$'
    legend_plot = ax2.plot(0, 0, color=None, ls=None)
    legend_BP = ax2.legend(legend_plot, [BP_str], loc='lower left', handlelength=0, handletextpad=0, edgecolor='gray')
    for item in legend_BP.legend_handles:
        item.set_visible(False)
    # plt.gca().add_artist(legend_BP)

ax2.loglog(md/T_nu, Td/T_nu, color='0.4', ls='-', zorder=-4)

ax2.fill_betweenx([1e-1, 1.5e0], x1[0], x_therm, color='white', alpha=1, zorder=-3)

props = dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=1, edgecolor="0.8")

# ax2.legend(framealpha=0.8, edgecolor='1')
ax2.xaxis.set_label_text(r"$m_s / T_\nu$")
ax1.yaxis.set_label_text(r"$m\, n / s\;\;\mathrm{[keV]}$")
ax2.yaxis.set_label_text(r"$T_\text{d}/T_\nu$")

ax1.xaxis.set_major_locator(xMajorLocator)
ax1.xaxis.set_minor_locator(xMinorLocator)
ax1.xaxis.set_major_formatter(xMajorFormatter)
ax1.yaxis.set_major_locator(yMajorLocator)
ax1.yaxis.set_minor_locator(yMinorLocator)
ax1.yaxis.set_major_formatter(yMajorFormatter)

ax1.set_xlim(2e-5, 20)
ax2.set_xlim(2e-5, 20)

# ylim + 6 will be shown
ax1.set_ylim(1e-25, 2e-8)
ax2.set_ylim(np.min(Td/T_nu)*0.5, np.max(Td/T_nu)*2)


m_d = md
m_X = mX
m_a = 0.
sin2_2th = sin22th

k_d = 1.
k_a = 1.
k_X = -1.
k_h = -1.
dof_d = 2.
dof_X = 3.
dof_h = 1.

m_d2 = m_d*m_d
m_a2 = m_a*m_a
m_X2 = m_X*m_X
th = 0.5*asin(sqrt(sin2_2th))
c_th = cos(th)
s_th = sin(th)
y2 = y*y

m_N1 = m_d
m_N2 = m_d
m_nu = m_a

m0 = 1e3
m12 = m_d
m2 = 1e-2
ma = m12*1e-2

# sin2_2th = m2/m0 * ma/m12
C_1nu = np.sqrt(sin2_2th)
C_10 = C_1nu * m12/ma

m_N12 = m_N1*m_N1
m_N22 = m_N2*m_N2

# M2_X_dd = 4*y2*(c_th**4.)*(m_X2-4*m_d2)
# M2_X_da = 4*y2*(s_th**2.)*(c_th**2.)*(m_X2-m_d2)*(1 + m_d2/(2*m_X2))
# M2_X_aa = 4.*y2*(s_th**4.)*m_X2

M2_X_12 = 2.*y2 * (m_X+m_N1-m_N2)*(m_X-m_N1+m_N2)*(2*m_X2 + (m_N1+m_N2)**2)/m_X2
M2_X_10 = 2.*y2*C_10**2 * (m_X+m_N1-m0)*(m_X-m_N1+m0)*(2*m_X2 + (m_N1+m0)**2)/m_X2
M2_X_1nu = 2.*y2*C_1nu**2 * (m_X2-m_N12)*(2*m_X2 + m_N12)/m_X2

print(f'M2_X_dd: {M2_X_12:3e}, M2_X_da: {M2_X_1nu:3e}, M2_X_d0: {M2_X_10:3e}')

vert_fi = y2*y2*(c_th**4.)*(s_th**4.)
vert_tr = y2*y2*(c_th**6.)*(s_th**2.)
vert_el = y2*y2

Gamma_X = vector_mediator.Gamma_X_new(y=y, m_X=m_X, m_N1=m_N1, m_N2=m_N2, m0=m0, m12=m12, m2=m2, ma=ma)
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

md_str = f'{m_d:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_d:.5e}'.split('e')[1].rstrip('0').rstrip('.')
mX_str = f'{m_X:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_X:.5e}'.split('e')[1].rstrip('0').rstrip('.')
sin22th_str = f'{sin2_2th:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{sin2_2th:.5e}'.split('e')[1].rstrip('0').rstrip('.')
y_str = f'{y:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{y:.5e}'.split('e')[1].rstrip('0').rstrip('.')

filename = f'rates_md_{md_str};mX_{mX_str};sin22th_{sin22th_str};y_{y_str};full.dat'
# Anton: Check if file exists -- if not, create it 
if not os.path.isfile('./' + filename) or force_write:
# if True:
    time_start = time.time()
    print('Get C_X_dd')
    C_X_dd = np.array([-C_res_vector.C_n_3_12(m_d, m_d, m_X, k_d, k_d, k_X, T_d, T_d, T_d, xi_d, xi_d, xi_X, M2_X_12, type=-1) for T_d, xi_d, xi_X in zip(T_d_grid, xi_d_grid, xi_X_grid)])
    
    print('Get C_dd_X')
    C_dd_X = np.array([C_res_vector.C_n_3_12(m_d, m_d, m_X, k_d, k_d, k_X, T_d, T_d, T_d, xi_d, xi_d, xi_X, M2_X_12, type=1) for T_d, xi_d, xi_X in zip(T_d_grid, xi_d_grid, xi_X_grid)])
    
    print('Get C_X_da')
    C_X_da = np.array([-C_res_vector.C_n_3_12(m_d, m_a, m_X, k_d, k_a, k_X, T_d, T_a, T_d, xi_d, 0., xi_X, M2_X_1nu, type=-1) for T_d, T_a, xi_d, xi_X in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid)])
    
    print('Get C_da_X')
    C_da_X = np.array([C_res_vector.C_n_3_12(m_d, m_a, m_X, k_d, k_a, k_X, T_d, T_a, T_d, xi_d, 0., xi_X, M2_X_1nu, type=1) for T_d, T_a, xi_d, xi_X in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid)])
    
    print('Get C_XX_dd_both')
    C_XX_dd_both = np.array([2.*C_res_vector.C_n_XX_dd(m_d, m_X, 0, k_d, k_X, T_d, xi_d, xi_X, vert_el, th, 0, type=2) / 4. for T_d, xi_d, xi_X in zip(T_d_grid, xi_d_grid, xi_X_grid)])

    print('Get C_11_22')

    C_11_22_ut = np.array([C_res_vector.C_34_12(type=0, nFW=2, nBW=2, m1=m_N1, m2=m_N1, m3=m_N2, m4=m_N2, k1=k_d, k2=k_d, k3=k_d, k4=k_d, T1=T_d, T2=T_d, T3=T_d, T4=T_d, xi1=xi_d, xi2=xi_d, xi3=xi_d, xi4=xi_d, vert=vert_el, m_d2=m_d2, m_X2=m_X2, m_h2=0, m_Gamma_X2=m_Gamma_X2, m_Gamma_h2=0, res_sub=False, thermal_width=True) / 4. for T_d, xi_d in zip(T_d_grid, xi_d_grid)])

    # C_dd_dd = np.array([C_res_vector_no_spin_stat.C_12_34(m_d, m_d, m_d, m_d, k_d, k_d, T_d, T_d, xi_d, xi_d, vert_el, m_X2, m_Gamma_X2, type=0, res_sub=True) / 4. for T_d, xi_d in zip(T_d_grid, xi_d_grid)])
    # C_da_dd = np.array([C_res_vector_no_spin_stat.C_12_34(m_d, m_a, m_d, m_d, k_d, k_a, T_d, T_a, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, type=0, res_sub=True) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_dd = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 1., 0., m_d, m_d, m_d, m_d, k_d, k_d, k_d, k_d, T_d, T_d, T_d, T_d, xi_d, xi_d, xi_d, xi_d, vert_el, m_X2, m_Gamma_X2, res_sub=False) / 4. for T_d, xi_d in zip(T_d_grid, xi_d_grid)])
    C_da_dd = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 1., 0., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, res_sub=False) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_da = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 0., 1., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_X2, m_Gamma_X2) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_aa_dd = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 2., 0., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_X2, m_Gamma_X2) / 4. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_aa = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 0., 2., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_X2, m_Gamma_X2) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_XX_dd = -C_XX_dd_both[:,0]
    C_dd_XX = C_XX_dd_both[:,1]

    np.savetxt(filename, np.column_stack((
        m_d/T_nu_grid, 
        H_grid, 
        C_X_dd/n_d_grid, 
        C_dd_X/n_d_grid,  
        C_X_da/n_d_grid, 
        C_da_X/n_d_grid, 
        C_XX_dd/n_d_grid, 
        C_dd_XX/n_d_grid, 
        C_dd_dd/n_d_grid, 
        C_da_dd/n_d_grid, 
        C_dd_da/n_d_grid, 
        C_aa_dd/n_d_grid, 
        C_dd_aa/n_d_grid,
        C_11_22_ut/n_d_grid)))
    print(f'Saved file in {time.time()-time_start}s')

data = np.loadtxt(filename)

x_grid  = data[::plot_data_skip, 0]
H       = data[::plot_data_skip, 1]
C_X_dd  = data[::plot_data_skip, 2]
C_dd_X  = data[::plot_data_skip, 3]
C_X_da  = data[::plot_data_skip, 4]
C_da_X  = data[::plot_data_skip, 5]
C_XX_dd = data[::plot_data_skip, 6]
C_dd_XX = data[::plot_data_skip, 7]
C_dd_dd = data[::plot_data_skip, 8]
C_da_dd = data[::plot_data_skip, 9]
C_dd_da = data[::plot_data_skip, 10]
C_aa_dd = data[::plot_data_skip, 11]
C_dd_aa = data[::plot_data_skip, 12]
C_11_22_ut = data[::plot_data_skip, 13]


ax3.tick_params(axis='both', which='both', direction="in", width=0.5)
ax3.xaxis.set_ticks_position('both')
ax3.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax3.spines[axis].set_linewidth(0.5)

ytMajor = np.array([np.log10(10**j) for j in np.linspace(-28, 3, 32)])
ytMinor = np.array([np.log10(i*10**j) for j in ytMajor for i in range(10)[1:10]])
ylMajor = [r"$10^{" + str(int(i)) + "}$" if i in ytMajor[::2] else "" for i in ytMajor]
ytMajor = 10**ytMajor
ytMinor = 10**ytMinor
yMajorLocator = FixedLocator(ytMajor)
yMinorLocator = FixedLocator(ytMinor)
yMajorFormatter = FixedFormatter(ylMajor)

ch = 'crimson' # crimson
c1 = '#797ef6' # orchid
c2 = '#1aa7ec' # sky blue
c3 = '#4adede' # turquoise
c4 = '#ffa62b' # gold
c5 = '#1e2f97' # dark blue

# Anton: 1e6 to make GeV to keV 
# plt.loglog(x_grid, 1e6*H, color=ch, ls='-', zorder=0) #83781B
ax3.loglog(x_grid, 1e6*H, color=ch, ls='-', zorder=0) #83781B

ax3.loglog(x_grid, 1e6*abs(C_dd_X), color=c1, ls='-', zorder=-4) #114B5F
ax3.loglog(x_grid, 1e6*abs(C_da_X), color=c2, ls='-', zorder=-4) #458751
ax3.loglog(x_grid, 1e6*abs(C_dd_XX), color=c3, ls='-', zorder=-4) #95190C
ax3.loglog(x_grid, 1e6*abs(C_XX_dd), color=c5, ls='-', zorder=-4) #D02411
ax3.loglog(x_grid, 1e6*abs(C_11_22_ut), color=c4, ls='-', zorder=-4) #D02411

x_therm_index = np.where(np.min(np.abs(x_grid-x_therm)) == np.abs(x_grid-x_therm))
dark_therm_x = 10**((np.log10(x_therm) + np.log10(x_grid)[0]) / 2)
dark_therm_x_index = np.where(np.min(np.abs(x_grid-2e-4)) == np.abs(x_grid-2e-4))
ax3.text(dark_therm_x, 8e-21, r'$\mathrm{Dark}$', color='0', horizontalalignment='center')
ax3.text(dark_therm_x, 8e-22, r'$\mathrm{Thermalization}$', color='0', horizontalalignment='center')
ax3.text(dark_therm_x, 8e-23, r'$\rightarrow$', color='0', horizontalalignment='center')

Hubble_x = 10**((np.log10(x_therm) + np.log10(x_grid)[0]) / 2)
Hubble_x_index = np.where(np.min(np.abs(x_grid-2e-4)) == np.abs(x_grid-2e-4))
if BP != 5:
    ax3.text(Hubble_x, 1e5*H[Hubble_x_index], r"$H$", color=ch, rotation=0, va='top')
else:
    ax3.text(Hubble_x, 1e-15, r"$H$", color=ch, rotation=0, va='top')

max_nus_nus = 1e6*abs(np.max(C_X_dd))
max_nus_nus_index = np.where(np.min(np.abs(1e6*abs(C_X_dd)-max_nus_nus)) == np.abs(1e6*abs(C_X_dd)-max_nus_nus))
ypos_nus = 10**((np.log10(max_nus_nus) + np.log10(1e6*abs(C_X_dd[x_therm_index])))/2 - 2)

if BP == 1: 
    xpos_nus = 6e-1*x_grid[max_nus_nus_index]
elif BP == 2: 
    xpos_nus = 3e-1
elif BP == 3: 
    xpos_nus = x_grid[max_nus_nus_index]
elif BP == 4: 
    xpos_nus = 3e-1
elif BP == 5: 
    xpos_nus = 2e0
else: 
    xpos_nus = 3e-1

ax3.text(xpos_nus, ypos_nus, r"$N_1 N_2 \leftrightarrow X$", color=c1, rotation=0, ha='center', va='top')

x_mult = 1.3
y_PP_ss = 2e-28
if BP == 1: 
    x_ss_PP = 30.3
    x_PP_ss = 30.3
    x_sa_P = 1.3

    y_ss_PP = 1e-7
    y_sa_P = 0.9
elif BP == 2: 
    x_ss_PP = 25
    x_PP_ss = 30
    x_sa_P = 20

    y_ss_PP = 2e-1
    y_sa_P = 2e2
elif BP == 3: 
    x_ss_PP = 1.3
    x_PP_ss = 10
    x_sa_P = 1.3

    y_ss_PP = 5e-3
    y_sa_P = 0.9
elif BP == 4: 
    x_ss_PP = 30.3
    x_PP_ss = 3.3
    x_sa_P = 3.3

    y_ss_PP = 1e-2
    y_sa_P = 1e1
elif BP == 5: 
    x_ss_PP = 3.3
    x_PP_ss = 3.3
    x_sa_P = 1/x_therm * 0.8

    y_ss_PP = 0
    y_sa_P = 5e2
    y_PP_ss = 0
else: 
    x_ss_PP = 30.3
    x_PP_ss = 30.3
    x_sa_P = 1.3

    y_ss_PP = 1e-2
    y_sa_P = 0.9

ax3.text(x_therm*x_ss_PP, y_ss_PP*np.abs(1e6*C_dd_XX[x_therm_index]), r"$NN \to XX$", color=c3, rotation=0, ha='left', va='bottom')
ax3.text(x_therm*x_PP_ss, y_PP_ss, r"$XX \to NN$", color=c5, rotation=0, va='bottom')

# ax3.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='-', color='black', label=r'$\Phi=X_\mu$')
BP_str = r'$\textit{BP' + f'{BP}' + r'}$'

legend_plot = ax3.plot(0, 0, color=None, ls=None)
legend_BP = ax3.legend(legend_plot, [BP_str], loc='lower left', handlelength=0, handletextpad=0, edgecolor='gray')
for item in legend_BP.legend_handles:
    item.set_visible(False)
# plt.gca().add_artist(legend_BP)

ax3.fill_betweenx([1e-28, 1e0], 1e-5, x_therm, color='white', alpha=1, zorder=-3)
ax3.axvline(x_therm, ls=':', color='0', zorder=-2)

props = dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=1, edgecolor="0.8")
# ax3.text(6e-1, 9e-18, r"$m_s = 15 \, \mathrm{keV}$" + "\n" + r"$m_\phi = 37.5 \, \mathrm{keV}$" + "\n" + r"$\sin^2 (2 \theta) = 1.5 \times 10^{-13}$" + "\n" + r"$y = 2.44 \times 10^{-4}$", horizontalalignment="left", fontsize=10, bbox=props)

ax3.legend(framealpha=0.8, edgecolor='none', loc='upper left')

ax3.xaxis.set_label_text(r"$m_s / T_\nu$")
ax3.yaxis.set_label_text(r"$\text{Rate} \;\; [\mathrm{keV}]$")

ax3.yaxis.set_major_locator(yMajorLocator)
ax3.yaxis.set_minor_locator(yMinorLocator)
ax3.yaxis.set_major_formatter(yMajorFormatter)
ax3.yaxis.set_label_position('right')
ax3.yaxis.set_ticks_position('right')

ax3.set_xlim(2e-5, 20)
ymax = max(np.max(1e6*abs(C_X_dd)), np.max(1e6*abs(C_dd_X)))
ax3.set_ylim(1e-28, max(np.max(1e6*H)*2e3, ymax*1e1))
fig.tight_layout()

if BP is None:
    fig_str = f'./saved_benchmarks/combined_pandemic_rate{load_str[2:].split(';full_new')[0]}.pdf'
else:
    fig_str = f'./saved_benchmarks/combined_pandemic_rate_BP{BP}.pdf'
if save_fig:
    plt.savefig(fig_str, bbox_inches='tight', dpi=300)
plt.show()
