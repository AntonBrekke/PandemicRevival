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
                            using \showthe\columnwidth
    Returns:  [fig_width, fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]

ggplot_red = "#E24A33"
ch = 'crimson' # crimson
c1 = '#797ef6' # orchid
c2 = '#1aa7ec' # sky blue
c3 = '#4adede' # turquoise
c4 = '#ffa62b' # gold
c5 = '#1e2f97' # dark blue

columnwidth = 418.25368     # pt, given by \showthe\textwidth in LaTeX

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
params = {'axes.labelsize': 10,
            'axes.titlesize': 10,
            'font.size': 8} # extend as needed
# print(plt.rcParams.keys())
plt.rcParams.update(params)



load_str = './md_1.08264e-05;mX_3.24791e-05;sin22th_1.1721e-16;y_3.8676e-03;full_new.dat'

force_write = False
save_fig = False
sample_data_skip = 10
plot_data_skip = 1
x_therm = 1e-3

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
else: None  

# load_str = './md_2.33572e-04;mX_1.16786e-03;mh_7.00716e-04;sin22th_1.26638e-14;y_8.76604e-04;full.dat'
# load_str = './md_2.33572e-04;mX_1.16786e-03;mh_7.00716e-04;sin22th_1.26638e-14;y_8.76604e-04;full.dat'
var_list = load_str.split(';')[:-1]
m_d, m_X, m_h, sin2_2th, y = [eval(s.split('_')[-1]) for s in var_list]
m_a = 0.

print(m_d, m_X, m_h, sin2_2th, y, np.sqrt(2)*m_d/m_X*y)

md_str = f'{m_d:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_d:.5e}'.split('e')[1].rstrip('0').rstrip('.')
mX_str = f'{m_X:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_X:.5e}'.split('e')[1].rstrip('0').rstrip('.')
mh_str = f'{m_h:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_h:.5e}'.split('e')[1].rstrip('0').rstrip('.')
sin22th_str = f'{sin2_2th:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{sin2_2th:.5e}'.split('e')[1].rstrip('0').rstrip('.')
y_str = f'{y:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{y:.5e}'.split('e')[1].rstrip('0').rstrip('.')

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
m_h2 = m_h*m_h
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
# M2_dd = 4.*y2*(c_th**4.)*(m_X2-4*m_d2)
# M2_aa = 4.*y2*(s_th**4.)*m_X2
# M2_da = 4.*y2*(s_th**2.)*(c_th**2.)/m_X2 * (m_X2 - m_d2)*(2*m_X2 + m_d2)

M2_X_dd = 4*y2*(c_th**4.)*(m_X2-4*m_d2)
M2_X_da = 4*y2*(s_th**2.)*(c_th**2.)*(m_X2-m_d2)*(1 + m_d2/(2*m_X2))
M2_X_aa = 4.*y2*(s_th**4.)*m_X2

M2_h_dd = 2*(4*y2*m_d2/m_X2)*(c_th**4)*(m_h2-4*m_d2)
M2_h_da = 2*(4*y2*m_d2/m_X2)*(c_th**2)*(s_th**2)*(m_h2-m_d2)
M2_h_aa = 2*(4*y2*m_d2/m_X2)*(s_th**4)*m_h2

M2_h_XX = 4*y2*(m_h2**2/m_X2-4*m_h2+12*m_X2)

print(f'M2_X_dd: {M2_X_dd:3e}, M2_X_da: {M2_X_da:3e}, M2_X_aa: {M2_X_aa:3e}')
print(f'M2_h_dd: {M2_h_dd:3e}, M2_h_da: {M2_h_da:3e}, M2_h_aa: {M2_h_aa:3e}')

vert_fi = y2*y2*(c_th**4.)*(s_th**4.)
vert_tr = y2*y2*(c_th**6.)*(s_th**2.)
vert_el = y2*y2*(c_th**8.)

Gamma_X = vector_mediator.Gamma_X_new(y=y, th=th, m_X=m_X, m_d=m_d)
Gamma_h = scalar_mediator.Gamma_phi(y=y, th=th, m_phi=m_h, m_d=m_d, m_X=m_X)
m_Gamma_X2 = m_X2*Gamma_X*Gamma_X
m_Gamma_h2 = m_h2*Gamma_h*Gamma_h

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

t_grid = data_evo[::sample_data_skip,0]
T_SM_grid = data_evo[::sample_data_skip,1]
T_nu_grid = data_evo[::sample_data_skip,2]
ent_grid = data_evo[::sample_data_skip, 3]
H_grid = data_evo[::sample_data_skip,4]
sf_grid = data_evo[::sample_data_skip,5]
T_d_grid = data_evo[::sample_data_skip,6]
xi_d_grid = data_evo[::sample_data_skip,7]
xi_X_grid = data_evo[::sample_data_skip,8]
xi_h_grid = data_evo[::sample_data_skip,9]
n_d_grid = data_evo[::sample_data_skip,10]
n_X_grid = data_evo[::sample_data_skip,11]
n_h_grid = data_evo[::sample_data_skip,12]
n_nu_grid = 2.*0.75*(cf.zeta3/cf.pi2)*(T_nu_grid**3.)

filename = f'rates_md_{md_str};mX_{mX_str};mh_{mh_str};sin22th_{sin22th_str};y_{y_str};full.dat'
# Anton: Check if file exists -- if not, create it 
if not os.path.isfile('./' + filename) or force_write:
# if True:
    time_start = time.time()
    print('Get C_X_dd')
    C_X_dd = np.array([-2.*C_res_vector.C_n_3_12(m_d, m_d, m_X, k_d, k_d, k_X, T_d, T_d, T_d, xi_d, xi_d, xi_X, M2_X_dd, type=-1) / 2. for T_d, xi_d, xi_X in zip(T_d_grid, xi_d_grid, xi_X_grid)])
    print('Get C_h_dd')
    C_h_dd = np.array([-2.*C_res_vector.C_n_3_12(m_d, m_d, m_h, k_d, k_d, k_h, T_d, T_d, T_d, xi_d, xi_d, xi_h, M2_h_dd, type=-1) / 2. for T_d, xi_d, xi_h in zip(T_d_grid, xi_d_grid, xi_h_grid)])
    
    print('Get C_dd_X')
    C_dd_X = np.array([2.*C_res_vector.C_n_3_12(m_d, m_d, m_X, k_d, k_d, k_X, T_d, T_d, T_d, xi_d, xi_d, xi_X, M2_X_dd, type=1) / 2. for T_d, xi_d, xi_X in zip(T_d_grid, xi_d_grid, xi_X_grid)])
    print('Get C_dd_h')
    C_dd_h = np.array([2.*C_res_vector.C_n_3_12(m_d, m_d, m_h, k_d, k_d, k_h, T_d, T_d, T_d, xi_d, xi_d, xi_h, M2_h_dd, type=1) / 2. for T_d, xi_d, xi_h in zip(T_d_grid, xi_d_grid, xi_h_grid)])
    
    print('Get C_X_da')
    C_X_da = np.array([-C_res_vector.C_n_3_12(m_d, m_a, m_X, k_d, k_a, k_X, T_d, T_a, T_d, xi_d, 0., xi_X, M2_X_da, type=-1) for T_d, T_a, xi_d, xi_X in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid)])
    print('Get C_h_da')
    C_h_da = np.array([-C_res_vector.C_n_3_12(m_d, m_a, m_h, k_d, k_a, k_X, T_d, T_a, T_d, xi_d, 0., xi_h, M2_h_da, type=-1) for T_d, T_a, xi_d, xi_h in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_h_grid)])
    
    print('Get C_da_X')
    C_da_X = np.array([C_res_vector.C_n_3_12(m_d, m_a, m_X, k_d, k_a, k_X, T_d, T_a, T_d, xi_d, 0., xi_X, M2_X_da, type=1) for T_d, T_a, xi_d, xi_X in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid)])
    print('Get C_da_h')
    C_da_h = np.array([C_res_vector.C_n_3_12(m_d, m_a, m_h, k_d, k_a, k_h, T_d, T_a, T_d, xi_d, 0., xi_h, M2_h_da, type=1) for T_d, T_a, xi_d, xi_h in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_h_grid)])
    
    print('Get C_X_aa')
    C_X_aa = np.array([-2.*C_res_vector.C_n_3_12(m_a, m_a, m_X, k_d, k_a, k_X, T_a, T_a, T_d, 0., 0., xi_X, M2_X_aa, type=-1) / 2. for T_d, T_a, xi_d, xi_X in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid)])
    print('Get C_h_aa')
    C_h_aa = np.array([-2.*C_res_vector.C_n_3_12(m_a, m_a, m_h, k_d, k_a, k_h, T_a, T_a, T_d, 0., 0., xi_h, M2_h_aa, type=-1) / 2. for T_d, T_a, xi_d, xi_h in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_h_grid)])
    
    print('Get C_aa_X')
    C_aa_X = np.array([2.*C_res_vector.C_n_3_12(m_a, m_a, m_X, k_d, k_a, k_X, T_a, T_a, T_d, 0., 0., xi_X, M2_X_aa, type=1) / 2. for T_d, T_a, xi_d, xi_X in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid)])
    print('Get C_aa_h')
    C_aa_h = np.array([2.*C_res_vector.C_n_3_12(m_a, m_a, m_h, k_d, k_a, k_h, T_a, T_a, T_d, 0., 0., xi_h, M2_h_aa, type=1) / 2. for T_d, T_a, xi_d, xi_h in zip(T_d_grid, T_nu_grid, xi_d_grid, xi_h_grid)])
    
    print('Get C_XX_dd_both')
    C_XX_dd_both = np.array([2.*C_res_vector.C_n_XX_dd(m_d, m_X, m_h, k_d, k_X, T_d, xi_d, xi_X, vert_el, th, m_Gamma_h2, type=2) / 4. for T_d, xi_d, xi_X in zip(T_d_grid, xi_d_grid, xi_X_grid)])
    print('Get C_hh_dd_both')
    C_hh_dd_both = np.array([2.*C_res_scalar.C_n_pp_dd(m_d, m_h, k_d, k_h, T_d, xi_d, xi_h, vert_el*(4*m_d2/m_X2)**2, type=2) / 4. for T_d, xi_d, xi_h in zip(T_d_grid, xi_d_grid, xi_h_grid)])

    print('Get C_h_XX')
    C_h_XX = np.array([-C_res_vector.C_n_3_12(m_X, m_X, m_h, k_X, k_X, k_h, T_d, T_d, T_d, xi_X, xi_X, xi_h, M2_h_XX, type=-1) / 2. for T_d, xi_h, xi_X in zip(T_d_grid, xi_h_grid, xi_X_grid)])
    print('Get C_XX_h')
    C_XX_h = np.array([C_res_vector.C_n_3_12(m_X, m_X, m_h, k_X, k_X, k_h, T_d, T_d, T_d, xi_X, xi_X, xi_h, M2_h_XX, type=1) / 2. for T_d, xi_h, xi_X in zip(T_d_grid, xi_h_grid, xi_X_grid)])

    # C_dd_dd = np.array([C_res_vector_no_spin_stat.C_12_34(m_d, m_d, m_d, m_d, k_d, k_d, T_d, T_d, xi_d, xi_d, vert_el, m_X2, m_Gamma_X2, type=0, res_sub=True) / 4. for T_d, xi_d in zip(T_d_grid, xi_d_grid)])
    # C_da_dd = np.array([C_res_vector_no_spin_stat.C_12_34(m_d, m_a, m_d, m_d, k_d, k_a, T_d, T_a, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, type=0, res_sub=True) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_dd = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 1., 0., m_d, m_d, m_d, m_d, k_d, k_d, k_d, k_d, T_d, T_d, T_d, T_d, xi_d, xi_d, xi_d, xi_d, vert_el, m_X2, m_Gamma_X2, res_sub=False) / 4. for T_d, xi_d in zip(T_d_grid, xi_d_grid)])
    C_da_dd = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 1., 0., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, res_sub=False) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_da = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 0., 1., m_d, m_d, m_d, m_a, k_d, k_d, k_d, k_a, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_X2, m_Gamma_X2) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_aa_dd = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 2., 0., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_X2, m_Gamma_X2) / 4. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_dd_aa = np.zeros(T_d_grid.size)#np.array([C_res_vector.C_34_12(0, 0., 2., m_d, m_d, m_a, m_a, k_d, k_d, k_a, k_a, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_X2, m_Gamma_X2) / 2. for T_d, T_a, xi_d in zip(T_d_grid, T_nu_grid, xi_d_grid)])
    C_XX_dd = -C_XX_dd_both[:,0]
    C_dd_XX = C_XX_dd_both[:,1]
    C_hh_dd = -C_hh_dd_both[:,0]
    C_dd_hh = C_hh_dd_both[:,1]

    np.savetxt(filename, np.column_stack((
        m_d/T_nu_grid, 
        H_grid, 
        C_X_dd/n_d_grid, 
        C_dd_X/n_d_grid, 
        C_h_dd/n_d_grid, 
        C_dd_h/n_d_grid, 
        C_X_da/n_d_grid, 
        C_da_X/n_d_grid, 
        C_h_da/n_d_grid, 
        C_da_h/n_d_grid, 
        C_X_aa/n_d_grid, 
        C_aa_X/n_d_grid, 
        C_h_aa/n_d_grid, 
        C_aa_h/n_d_grid, 
        C_XX_dd/n_d_grid, 
        C_dd_XX/n_d_grid, 
        C_hh_dd/n_d_grid, 
        C_dd_hh/n_d_grid, 
        C_dd_dd/n_d_grid, 
        C_da_dd/n_d_grid, 
        C_dd_da/n_d_grid, 
        C_aa_dd/n_d_grid, 
        C_dd_aa/n_d_grid,
        C_h_XX/n_d_grid,
        C_XX_h/n_d_grid)))
    print(f'Saved file in {time.time()-time_start}s')

data = np.loadtxt(filename)

x_grid  = data[::plot_data_skip, 0]
H       = data[::plot_data_skip, 1]
C_X_dd  = data[::plot_data_skip, 2]
C_dd_X  = data[::plot_data_skip, 3]
C_h_dd  = data[::plot_data_skip, 4]
C_dd_h  = data[::plot_data_skip, 5]
C_X_da  = data[::plot_data_skip, 6]
C_da_X  = data[::plot_data_skip, 7]
C_h_da  = data[::plot_data_skip, 8]
C_da_h  = data[::plot_data_skip, 9]
C_X_aa  = data[::plot_data_skip, 10]
C_aa_X  = data[::plot_data_skip, 11]
C_h_aa  = data[::plot_data_skip, 12]
C_aa_h  = data[::plot_data_skip, 13]
C_XX_dd = data[::plot_data_skip, 14]
C_dd_XX = data[::plot_data_skip, 15]
C_hh_dd = data[::plot_data_skip, 16]
C_dd_hh = data[::plot_data_skip, 17]
C_dd_dd = data[::plot_data_skip, 18]
C_da_dd = data[::plot_data_skip, 19]
C_dd_da = data[::plot_data_skip, 20]
C_aa_dd = data[::plot_data_skip, 21]
C_dd_aa = data[::plot_data_skip, 22]
C_h_XX  = data[::plot_data_skip, 23]
C_XX_h  = data[::plot_data_skip, 24]

# plt.loglog(x_grid, C_da_dd/C_da_X, color='dodgerblue')
# plt.show()
#
# plt.loglog(x_grid, C_dd_dd/C_dd_X, color='dodgerblue')
# plt.loglog(x_grid, C_dd_dd/C_X_dd, color='darkorange', ls='--')
# plt.show()
# exit(1)

fig = plt.figure(figsize=get_figsize(columnwidth, wf=0.6, hf=0.9), dpi=150, edgecolor="white")
ax = fig.add_subplot()

ax.tick_params(axis='both', which='both', direction="in", width=0.5)
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

ch = 'crimson' # crimson
c1 = '#797ef6' # orchid
c2 = '#1aa7ec' # sky blue
c3 = '#4adede' # turquoise
c4 = '#ffa62b' # gold
c5 = '#1e2f97' # dark blue

# Anton: 1e6 to make GeV to keV 
# plt.loglog(x_grid, 1e6*H, color=ch, ls='-', zorder=0) #83781B
plt.loglog(x_grid, 1e6*H, color=ch, ls='-', zorder=0) #83781B

plt.loglog(x_grid, 1e6*abs(C_dd_X), color=c1, ls='-', zorder=-4) #114B5F
plt.loglog(x_grid, 1e6*abs(C_da_X), color=c2, ls='-', zorder=-4) #458751
plt.loglog(x_grid, 1e6*abs(C_X_aa), color='brown', ls='-')
plt.loglog(x_grid, 1e6*abs(C_aa_X), color='purple', ls='-')
plt.loglog(x_grid, 1e6*abs(C_dd_XX), color=c3, ls='-', zorder=-4) #95190C
plt.loglog(x_grid, 1e6*abs(C_XX_dd), color=c5, ls='-', zorder=-4) #D02411

plt.loglog(x_grid, 1e6*abs(C_dd_h), color=c1, ls='--', zorder=-4) #114B5F
plt.loglog(x_grid, 1e6*abs(C_da_h), color=c2, ls='--', zorder=-4) #458751
plt.loglog(x_grid, 1e6*abs(C_h_aa), color='brown', ls='--')
plt.loglog(x_grid, 1e6*abs(C_aa_h), color='purple', ls='--')
plt.loglog(x_grid, 1e6*abs(C_dd_hh), color=c3, ls='--', zorder=-4) #95190C
plt.loglog(x_grid, 1e6*abs(C_hh_dd), color=c5, ls='--', zorder=-4) #D02411

# Funny how close this got to T_nu_dec! 
# T_nu_dec = (24*np.pi*(m_X/m_d)**(3/2)*m_X**2*Gamma_X)**(1/5)
# plt.axvline(m_d/T_nu_dec)
# T_nu_dec = 1.38e-3       # GeV
# plt.axvline(m_d/T_nu_dec, ls='--', color='k')

# # Happened at 2*T_nu_dec  
# T_nu_dec = 2*T_nu_dec       # GeV
# T_nu_dec = 55*m_d       # GeV
# plt.axvline(m_d/T_nu_dec, color='gray')
# plt.plot(m_d/T_nu_grid, sf_grid**(-3))
# T_nu_dec = 8*m_X
# plt.axvline(m_d/T_nu_dec)
# T_nu_dec = 16*m_X
# plt.axvline(m_d/T_nu_dec)


# plt.loglog(x_grid, 1e6*abs(C_h_XX), color='r', ls='--', zorder=-4) 
# plt.loglog(x_grid, 1e6*abs(C_XX_h), color='blue', ls='--', zorder=-4) 

x_therm_index = np.where(np.min(np.abs(x_grid-x_therm)) == np.abs(x_grid-x_therm))
dark_therm_x = 10**((np.log10(x_therm) + np.log10(x_grid)[0]) / 2)
dark_therm_x_index = np.where(np.min(np.abs(x_grid-2e-4)) == np.abs(x_grid-2e-4))
plt.text(dark_therm_x, 8e-21, r'$\mathrm{Dark}$', color='0', horizontalalignment='center')
plt.text(dark_therm_x, 8e-22, r'$\mathrm{Thermalization}$', color='0', horizontalalignment='center')
plt.text(dark_therm_x, 8e-23, r'$\rightarrow$', color='0', horizontalalignment='center')
#plt.text(4.5e-5, 1e-22, r'$\hspace{-0.55cm}\mathrm{Therma-}\\\mathrm{lization}\\\mathrm{ }\hspace{0.2cm}\rightarrow$', fontsize=10, color='0')


Hubble_x = 10**((np.log10(x_therm) + np.log10(x_grid)[0]) / 2)
Hubble_x_index = np.where(np.min(np.abs(x_grid-2e-4)) == np.abs(x_grid-2e-4))
ax.text(Hubble_x, 1e5*H[Hubble_x_index], r"$H$", color=ch, rotation=0, va='top')

max_nus_nus = 1e6*max(np.max(abs(C_X_dd)), np.max(abs(C_h_dd)))
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
else: 
    xpos_nus = 3e-1

ax.text(xpos_nus, ypos_nus, r"$\nu_s \nu_s \leftrightarrow \Phi$", color=c1, rotation=0, ha='center', va='top')

x_mult = 1.3
if BP == 1: 
    x_ss_PP = 30.3
    x_PP_ss = 30.3
    x_sa_P = 1.3

    y_ss_PP = 1e-7
    y_sa_P = 0.9
elif BP == 2: 
    x_ss_PP = 1.3
    x_PP_ss = 1.3
    x_sa_P = 1.3

    y_ss_PP = 1e-2
    y_sa_P = 0.9
elif BP == 3: 
    x_ss_PP = 1.3
    x_PP_ss = 30.3
    x_sa_P = 1.3

    y_ss_PP = 1e-2
    y_sa_P = 0.9
elif BP == 4: 
    x_ss_PP = 30.3
    x_PP_ss = 3.3
    x_sa_P = 3.3

    y_ss_PP = 1e-2
    y_sa_P = 1e1
else: 
    y_ss_PP = 1e-2

ax.text(x_therm*x_sa_P, y_sa_P*np.abs(1e6*C_da_h[x_therm_index]), r"$\nu_s \nu_\alpha \to \Phi$", color=c2, rotation=0, ha='left', va='top')
ax.text(x_therm*x_ss_PP, y_ss_PP*np.abs(1e6*C_dd_XX[x_therm_index]), r"$\nu_s \nu_s \to \Phi \Phi$", color=c3, rotation=0, ha='left', va='bottom')
ax.text(x_therm*x_PP_ss, 2e-28, r"$\Phi \Phi \to \nu_s \nu_s$", color=c5, rotation=0, va='bottom')

plt.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='-', color='black', label=r'$\Phi=X_\mu$')
plt.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='--', color='black', label=r'$\Phi=h_\phi$')
BP_str = r'$\textit{BP' + f'{BP}' + r'}$'

legend_plot = plt.plot(0, 0, color=None, ls=None)
legend_BP = ax.legend(legend_plot, [BP_str], loc='lower left', handlelength=0, handletextpad=0, edgecolor='gray')
for item in legend_BP.legend_handles:
    item.set_visible(False)
plt.gca().add_artist(legend_BP)

plt.fill_betweenx([1e-28, 1e0], 1e-5, x_therm, color='white', alpha=1, zorder=-3)
plt.axvline(x_therm, ls=':', color='0', zorder=-2)
# plt.loglog([1e-3]*2, [1e-28, 1e0], ls=':', color='0', zorder=-2)
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

plt.legend(framealpha=0.8, edgecolor='none', loc='upper left')

ax.xaxis.set_label_text(r"$m_s / T_\nu$")
ax.yaxis.set_label_text(r"$\text{Rate} \;\; [\mathrm{keV}]$")

ax.yaxis.set_major_locator(yMajorLocator)
ax.yaxis.set_minor_locator(yMinorLocator)
ax.yaxis.set_major_formatter(yMajorFormatter)
ax.yaxis.set_label_position('left')
ax.yaxis.set_ticks_position('left')

plt.xlim(2e-5, 20)
ymax = max(np.max(1e6*abs(C_X_dd)), np.max(1e6*abs(C_dd_X)), np.max(1e6*abs(C_h_dd)), np.max(1e6*abs(C_dd_h)))
ax.set_ylim(1e-28, max(np.max(1e6*H)*2e3, ymax*1e1))
plt.tight_layout()
fig_str = f'./saved_benchmarks/rates_evo_md_{md_str};mX_{mX_str};mh_{mh_str};sin22th_{sin22th_str};y_{y_str}_BP{BP}.pdf'
print(f'saved {fig_str}')
if save_fig:
    plt.savefig(fig_str, bbox_inches='tight', dpi=300)
plt.show()