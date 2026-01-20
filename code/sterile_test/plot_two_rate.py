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
import scalar_mediator 

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def make_rate_file(m_d, m_X, m_h, M2_X_dd, M2_X_da, M2_X_aa, M2_h_dd, M2_h_da, M2_h_aa, M2_h_XX, vert_el, th, n_d_grid, T_d_grid, T_nu_grid, xi_d_grid, xi_X_grid, xi_h_grid, H_grid, m_Gamma_h2, filename):
    m_d2 = m_d*m_d
    m_X2 = m_X*m_X
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
    print(f'Made file {filename} in {time.time()-time_start}s')

# load_str_1 = './md_2.48163e-06;mX_7.44489e-06;sin22th_1.4251e-12;y_2.76291e-05;full.dat'
# load_str_2 = './md_5.13483e-05;mX_1.54045e-04;sin22th_1.19378e-15;y_2.23145e-03;full.dat'

# load_str_1 = './md_1.35388e-06;mX_4.06163e-06;sin22th_4.64159e-12;y_1.12987e-05;full_new.dat'
# load_str_2 = './md_5.13483e-05;mX_1.54045e-04;sin22th_1.19378e-15;y_2.23145e-03;full_new.dat'

# load_str_1 = './md_3.35982e-06;mX_1.00795e-05;sin22th_7.01704e-15;y_5.43211e-04;full_new.dat'
# load_str_2 = './md_3.79269e-05;mX_1.13781e-04;sin22th_6.61474e-16;y_2.65322e-03;full_new.dat'

load_str_1 = './md_1.12884e-05;mX_3.38651e-05;sin22th_2.42446e-13;y_1.34284e-04;full_new.dat'
load_str_2 = './md_5.13483e-05;mX_1.54045e-04;sin22th_1.19378e-15;y_2.23145e-03;full_new.dat'

load_str_1 = './md_1.12884e-05;mX_3.38651e-05;sin22th_2.42446e-13;y_1.34284e-04;full_new.dat'
load_str_2 = './md_2.06914e-05;mX_6.20741e-05;sin22th_3.66524e-16;y_2.89428e-03;full_new.dat'

load_str_1 = './md_2.06914e-05;mX_1.03457e-04;mh_6.20741e-05;sin22th_6.61474e-16;y_1.83218e-03;full_new.dat'     # Perfect
load_str_2 = './md_1.35388e-06;mX_6.76938e-06;mh_4.06163e-06;sin22th_7.01704e-15;y_4.45923e-04;full_new.dat'     # Perfect

data_1 = np.loadtxt(load_str_1)
data_2 = np.loadtxt(load_str_2)

var_list_1 = load_str_1.split(';')[:-1]
m_d_1, m_X_1, m_h_1, sin2_2th_1, y_1 = [eval(s.split('_')[-1]) for s in var_list_1]
print(f'md_1: {m_d_1:.2e}, mX_1: {m_X_1:.2e}, mh_1: {m_h_1:.2e}, sin22th_1: {sin2_2th_1:.2e}, y_1: {y_1:.2e}')

var_list_2 = load_str_2.split(';')[:-1]
m_d_2, m_X_2, m_h_2, sin2_2th_2, y_2 = [eval(s.split('_')[-1]) for s in var_list_2]
print(f'md_2: {m_d_2:.2e}, mX_2: {m_X_2:.2e}, mh_2: {m_h_2:.2e}, sin22th_2: {sin2_2th_2:.2e}, y_2: {y_2:.2e}')

md_str_1 = f'{m_d_1:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_d_1:.5e}'.split('e')[1].rstrip('0').rstrip('.')
mX_str_1 = f'{m_X_1:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_X_1:.5e}'.split('e')[1].rstrip('0').rstrip('.')
mh_str_1 = f'{m_h_1:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_h_1:.5e}'.split('e')[1].rstrip('0').rstrip('.')
sin22th_str_1 = f'{sin2_2th_1:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{sin2_2th_1:.5e}'.split('e')[1].rstrip('0').rstrip('.')
y_str_1 = f'{y_1:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{y_1:.5e}'.split('e')[1].rstrip('0').rstrip('.')

md_str_2 = f'{m_d_2:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_d_2:.5e}'.split('e')[1].rstrip('0').rstrip('.')
mX_str_2 = f'{m_X_2:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_X_2:.5e}'.split('e')[1].rstrip('0').rstrip('.')
mh_str_2 = f'{m_h_2:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_h_2:.5e}'.split('e')[1].rstrip('0').rstrip('.')
sin22th_str_2 = f'{sin2_2th_2:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{sin2_2th_2:.5e}'.split('e')[1].rstrip('0').rstrip('.')
y_str_2 = f'{y_2:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{y_2:.5e}'.split('e')[1].rstrip('0').rstrip('.')

k_d = 1.
k_a = 1.
k_X = -1.
k_h = -1
dof_d = 2.
dof_X = 3.
dof_h = 1.
m_a = 0.

m_d2_1 = m_d_1*m_d_1
m_X2_1 = m_X_1*m_X_1
m_h2_1 = m_h_1*m_h_1
th_1 = 0.5*asin(sqrt(sin2_2th_1))
c_th_1 = cos(th_1)
s_th_1 = sin(th_1)
y2_1 = y_1*y_1

m_d2_2 = m_d_2*m_d_2
m_X2_2 = m_X_2*m_X_2
m_h2_2 = m_h_2*m_h_2
th_2 = 0.5*asin(sqrt(sin2_2th_2))
c_th_2 = cos(th_2)
s_th_2 = sin(th_2)
y2_2 = y_2*y_2

m_a2 = m_a*m_a

# M2_X23 = 2*g**2/m_X2 * (m_X2 - (m2 - m3)**2)*(2*m_X2 + (m2 + m3)**2)
# New matrix elements for X --> 23
# Only vector coupling 
# M2_dd_1 = 2.*y2_1*(c_th_1**4.)/m_X2_1 * (m_X2_1)*(2*m_X2_1 + 4*m_d2_1)
# M2_aa_1 = 2.*y2_1*(s_th_1**4.)/m_X2_1 * (m_X2_1)*(2*m_X2_1)
# M2_da_1 = 2.*y2_1*(s_th_1**2.)*(c_th_1**2.)/m_X2_1 * (m_X2_1 - m_d2_1)*(2*m_X2_1 + m_d2_1)
# M2_dd_2 = 2.*y2_2*(c_th_2**4.)/m_X2_2 * (m_X2_2)*(2*m_X2_2 + 4*m_d2_2)
# M2_aa_2 = 2.*y2_2*(s_th_2**4.)/m_X2_2 * (m_X2_2)*(2*m_X2_2)
# M2_da_2 = 2.*y2_2*(s_th_2**2.)*(c_th_2**2.)/m_X2_2 * (m_X2_2 - m_d2_2)*(2*m_X2_2 + m_d2_2)

# Anton: Test if new Feynman rules work.
M2_X_dd_1 = 4*y2_1*(c_th_1**4.)*(m_X2_1-4*m_d2_1)
M2_X_da_1 = 4*y2_1*(s_th_1**2.)*(c_th_1**2.)*(m_X2_1-m_d2_1)*(1 + m_d2_1/(2*m_X2_1))
M2_X_aa_1 = 4.*y2_1*(s_th_1**4.)*m_X2_1
M2_h_dd_1 = 2*(4*y2_1*m_d2_1/m_X2_1)*(c_th_1**4)*(m_h2_1-4*m_d2_1)
M2_h_da_1 = 2*(4*y2_1*m_d2_1/m_X2_1)*(c_th_1**2)*(s_th_1**2)*(m_h2_1-m_d2_1)
M2_h_aa_1 = 2*(4*y2_1*m_d2_1/m_X2_1)*(s_th_1**4)*m_h2_1
M2_h_XX_1 = 4*y2_1*(m_h2_1**2/m_X2_1-4*m_h2_1+12*m_X2_1)

M2_X_dd_2 = 4*y2_2*(c_th_2**4.)*(m_X2_2-4*m_d2_2)
M2_X_da_2 = 4*y2_2*(s_th_2**2.)*(c_th_2**2.)*(m_X2_2-m_d2_2)*(1 + m_d2_2/(2*m_X2_2))
M2_X_aa_2 = 4.*y2_2*(s_th_2**4.)*m_X2_2
M2_h_dd_2 = 2*(4*y2_2*m_d2_2/m_X2_2)*(c_th_2**4)*(m_h2_2-4*m_d2_2)
M2_h_da_2 = 2*(4*y2_2*m_d2_2/m_X2_2)*(c_th_2**2)*(s_th_2**2)*(m_h2_2-m_d2_2)
M2_h_aa_2 = 2*(4*y2_2*m_d2_2/m_X2_2)*(s_th_2**4)*m_h2_2
M2_h_XX_2 = 4*y2_2*(m_h2_2**2/m_X2_2-4*m_h2_2+12*m_X2_2)

# print(f'M2_dd_1: {M2_X_dd_1:3e}, M2_da_1: {M2_da_1:3e}, M2_aa_1: {M2_aa_1:3e}')
# print(f'M2_dd_2: {M2_dd_2:3e}, M2_da_2: {M2_da_2:3e}, M2_aa_2: {M2_aa_2:3e}')

vert_fi_1 = y2_1*y2_1*(c_th_1**4.)*(s_th_1**4.)
vert_tr_1 = y2_1*y2_1*(c_th_1**6.)*(s_th_1**2.)
vert_el_1 = y2_1*y2_1*(c_th_1**8.)

vert_fi_2 = y2_2*y2_2*(c_th_2**4.)*(s_th_2**4.)
vert_tr_2 = y2_2*y2_2*(c_th_2**6.)*(s_th_2**2.)
vert_el_2 = y2_2*y2_2*(c_th_2**8.)

Gamma_X_1 = vector_mediator.Gamma_X_new(y=y_1, th=th_1, m_X=m_X_1, m_d=m_d_1)
Gamma_h_1 = scalar_mediator.Gamma_phi(y=y_1, th=th_1, m_phi=m_h_1, m_d=m_d_1, m_X=m_X_1)
m_Gamma_X2_1 = m_X2_1*Gamma_X_1*Gamma_X_1
m_Gamma_h2_1 = m_h2_1*Gamma_h_1*Gamma_h_1

Gamma_X_2 = vector_mediator.Gamma_X_new(y=y_2, th=th_2, m_X=m_X_2, m_d=m_d_2)
Gamma_h_2 = scalar_mediator.Gamma_phi(y=y_2, th=th_2, m_phi=m_h_2, m_d=m_d_2, m_X=m_X_2)
m_Gamma_X2_2 = m_X2_2*Gamma_X_2*Gamma_X_2
m_Gamma_h2_2 = m_h2_2*Gamma_h_2*Gamma_h_2

data_evo_1 = np.loadtxt(load_str_1)
data_evo_2 = np.loadtxt(load_str_2)
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

i_skip = 15

t_grid_1 = data_evo_1[::i_skip,0]
T_SM_grid_1 = data_evo_1[::i_skip,1]
T_nu_grid_1 = data_evo_1[::i_skip,2]
ent_grid_1 = data_evo_1[::i_skip, 3]
H_grid_1 = data_evo_1[::i_skip,4]
sf_grid_1 = data_evo_1[::i_skip,5]
T_d_grid_1 = data_evo_1[::i_skip,6]
xi_d_grid_1 = data_evo_1[::i_skip,7]
xi_X_grid_1 = data_evo_1[::i_skip,8]
xi_h_grid_1 = data_evo_1[::i_skip,9]
n_d_grid_1 = data_evo_1[::i_skip,10]
n_X_grid_1 = data_evo_1[::i_skip,11]
n_h_grid_1 = data_evo_1[::i_skip,12]
n_nu_grid_1 = 2.*0.75*(cf.zeta3/cf.pi2)*(T_nu_grid_1**3.)


t_grid_2 = data_evo_2[::i_skip,0]
T_SM_grid_2 = data_evo_2[::i_skip,1]
T_nu_grid_2 = data_evo_2[::i_skip,2]
ent_grid_2 = data_evo_2[::i_skip, 3]
H_grid_2 = data_evo_2[::i_skip,4]
sf_grid_2 = data_evo_2[::i_skip,5]
T_d_grid_2 = data_evo_2[::i_skip,6]
xi_d_grid_2 = data_evo_2[::i_skip,7]
xi_X_grid_2 = data_evo_2[::i_skip,8]
xi_h_grid_2 = data_evo_2[::i_skip,9]
n_d_grid_2 = data_evo_2[::i_skip,10]
n_X_grid_2 = data_evo_2[::i_skip,11]
n_h_grid_2 = data_evo_2[::i_skip,12]
n_nu_grid_2 = 2.*0.75*(cf.zeta3/cf.pi2)*(T_nu_grid_2**3.)

filename_1 = f'rates_md_{md_str_1};mX_{mX_str_1};mh_{mh_str_1};sin22th_{sin22th_str_1};y_{y_str_1};full.dat'
filename_2 = f'rates_md_{md_str_2};mX_{mX_str_2};mh_{mh_str_2};sin22th_{sin22th_str_2};y_{y_str_2};full.dat'

make_rates_file_1_by_force = False
make_rates_file_2_by_force = False

# Anton: Check if file exists -- if not, create it 
if not os.path.isfile('./' + filename_1) or make_rates_file_1_by_force:
    make_rate_file(m_d=m_d_1, m_X=m_X_1, m_h=m_h_1, 
                   M2_X_dd=M2_X_dd_1, M2_X_da=M2_X_da_1, M2_X_aa=M2_X_aa_1, 
                   M2_h_dd=M2_h_dd_1, M2_h_da=M2_h_da_1, M2_h_aa=M2_h_aa_1, M2_h_XX=M2_h_XX_1,
                   vert_el=vert_el_1, th=th_1, 
                   n_d_grid=n_d_grid_1, 
                   T_d_grid=T_d_grid_1, T_nu_grid=T_nu_grid_1, 
                   xi_d_grid=xi_d_grid_1, xi_X_grid=xi_X_grid_1, xi_h_grid=xi_h_grid_1, 
                   H_grid=H_grid_1, 
                   m_Gamma_h2=m_Gamma_h2_1, 
                   filename=filename_1)
    
if not os.path.isfile('./' + filename_2) or make_rates_file_2_by_force:
    make_rate_file(m_d=m_d_2, m_X=m_X_2, m_h=m_h_2, 
                   M2_X_dd=M2_X_dd_2, M2_X_da=M2_X_da_2, M2_X_aa=M2_X_aa_2, 
                   M2_h_dd=M2_h_dd_2, M2_h_da=M2_h_da_2, M2_h_aa=M2_h_aa_2, M2_h_XX=M2_h_XX_2,
                   vert_el=vert_el_2, th=th_2, 
                   n_d_grid=n_d_grid_2, 
                   T_d_grid=T_d_grid_2, T_nu_grid=T_nu_grid_2, 
                   xi_d_grid=xi_d_grid_2, xi_X_grid=xi_X_grid_2, xi_h_grid=xi_h_grid_2, 
                   H_grid=H_grid_2, 
                   m_Gamma_h2=m_Gamma_h2_2, 
                   filename=filename_2)

data_1 = np.loadtxt(filename_1)
data_2 = np.loadtxt(filename_2)

data_skip = 1
x_grid_1 = data_1[::data_skip,0]
H_1 = data_1[::data_skip,1]
C_X_dd_1 = data_1[::data_skip,2]
C_dd_X_1 = data_1[::data_skip,3]
C_h_dd_1 = data_1[::data_skip,4]
C_dd_h_1 = data_1[::data_skip,5]
C_X_da_1 = data_1[::data_skip,6]
C_da_X_1 = data_1[::data_skip,7]
C_h_da_1 = data_1[::data_skip,8]
C_da_h_1 = data_1[::data_skip,9]
C_X_aa_1 = data_1[::data_skip,10]
C_aa_X_1 = data_1[::data_skip,11]
C_h_aa_1 = data_1[::data_skip,12]
C_aa_h_1 = data_1[::data_skip,13]
C_XX_dd_1 = data_1[::data_skip,14]
C_dd_XX_1 = data_1[::data_skip,15]
C_hh_dd_1 = data_1[::data_skip,16]
C_dd_hh_1 = data_1[::data_skip,17]
C_dd_dd_1 = data_1[::data_skip,18]
C_da_dd_1 = data_1[::data_skip,19]
C_dd_da_1 = data_1[::data_skip,20]
C_aa_dd_1 = data_1[::data_skip,21]
C_dd_aa_1 = data_1[::data_skip,22]
C_h_XX_1 = data_1[::data_skip,23]
C_XX_h_1 = data_1[::data_skip,24]

x_grid_2 = data_2[::data_skip,0]
H_2 = data_2[::data_skip,1]
C_X_dd_2 = data_2[::data_skip,2]
C_dd_X_2 = data_2[::data_skip,3]
C_h_dd_2 = data_2[::data_skip,4]
C_dd_h_2 = data_2[::data_skip,5]
C_X_da_2 = data_2[::data_skip,6]
C_da_X_2 = data_2[::data_skip,7]
C_h_da_2 = data_2[::data_skip,8]
C_da_h_2 = data_2[::data_skip,9]
C_X_aa_2 = data_2[::data_skip,10]
C_aa_X_2 = data_2[::data_skip,11]
C_h_aa_2 = data_2[::data_skip,12]
C_aa_h_2 = data_2[::data_skip,13]
C_XX_dd_2 = data_2[::data_skip,14]
C_dd_XX_2 = data_2[::data_skip,15]
C_hh_dd_2 = data_2[::data_skip,16]
C_dd_hh_2 = data_2[::data_skip,17]
C_dd_dd_2 = data_2[::data_skip,18]
C_da_dd_2 = data_2[::data_skip,19]
C_dd_da_2 = data_2[::data_skip,20]
C_aa_dd_2 = data_2[::data_skip,21]
C_dd_aa_2 = data_2[::data_skip,22]
C_h_XX_2 = data_2[::data_skip,23]
C_XX_h_2 = data_2[::data_skip,24]

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

ch = 'crimson' # crimson
c1 = '#797ef6' # orchid
c2 = '#1aa7ec' # sky blue
c3 = '#4adede' # turquoise
c4 = '#1e2f97' # dark blue

c5 = '#ffa62b' # gold
c6 = '#5db04f' # green
c7 = '#61929e' # teal
c8 = '#c53a80' # pink

# Anton: 1e6 to make GeV to keV 
plt.loglog(x_grid_1, 1e6*H_1, color=ch, ls='-', zorder=0) #83781B

plt.loglog(x_grid_1, 1e6*abs(C_dd_X_1), color=c1, ls='-', zorder=-4) #114B5F
plt.loglog(x_grid_1, 1e6*abs(C_da_X_1), color=c2, ls='-', zorder=-4) #458751
# plt.loglog(x_grid_1, 1e6*abs(C_X_aa_1), color='brown', ls='-')
# plt.loglog(x_grid_1, 1e6*abs(C_aa_X_1), color='purple', ls='-')
plt.loglog(x_grid_1, 1e6*abs(C_dd_XX_1), color=c3, ls='-', zorder=-4) #95190C
plt.loglog(x_grid_1, 1e6*abs(C_XX_dd_1), color=c4, ls='-', zorder=-4) #D02411

plt.loglog(x_grid_1, 1e6*abs(C_dd_h_1), color=c5, ls='-', zorder=-4) #114B5F
plt.loglog(x_grid_1, 1e6*abs(C_da_h_1), color=c6, ls='-', zorder=-4) #458751
# plt.loglog(x_grid_1, 1e6*abs(C_h_aa_1), color='brown', ls='--')
# plt.loglog(x_grid_1, 1e6*abs(C_aa_h_1), color='purple', ls='--')
plt.loglog(x_grid_1, 1e6*abs(C_dd_hh_1), color=c7, ls='-', zorder=-4) #95190C
plt.loglog(x_grid_1, 1e6*abs(C_hh_dd_1), color=c8, ls='-', zorder=-4) #D02411

plt.loglog(x_grid_2, 1e6*H_2, color=ch, ls='--', zorder=0) #83781B

plt.loglog(x_grid_2, 1e6*abs(C_dd_X_2), color=c1, ls='--', zorder=-4) #114B5F
plt.loglog(x_grid_2, 1e6*abs(C_da_X_2), color=c2, ls='--', zorder=-4) #458751
# plt.loglog(x_grid_2, 1e6*abs(C_X_aa_2), color='brown', ls='-')
# plt.loglog(x_grid_2, 1e6*abs(C_aa_X_2), color='purple', ls='-')
plt.loglog(x_grid_2, 1e6*abs(C_dd_XX_2), color=c3, ls='--', zorder=-4) #95190C
plt.loglog(x_grid_2, 1e6*abs(C_XX_dd_2), color=c4, ls='--', zorder=-4) #D02411

plt.loglog(x_grid_2, 1e6*abs(C_dd_h_2), color=c5, ls='--', zorder=-4) #114B5F
plt.loglog(x_grid_2, 1e6*abs(C_da_h_2), color=c6, ls='--', zorder=-4) #458751
# plt.loglog(x_grid_2, 1e6*abs(C_h_aa_2), color='brown', ls='--')
# plt.loglog(x_grid_2, 1e6*abs(C_aa_h_2), color='purple', ls='--')
plt.loglog(x_grid_2, 1e6*abs(C_dd_hh_2), color=c7, ls='--', zorder=-4) #95190C
plt.loglog(x_grid_2, 1e6*abs(C_hh_dd_2), color=c8, ls='--', zorder=-4) #D02411

plt.text(1.5e-4, 8e-21, r'$\mathrm{Dark}$', fontsize=8, color='0', horizontalalignment='center')
plt.text(1.5e-4, 8e-22, r'$\mathrm{Thermalization}$', fontsize=8, color='0', horizontalalignment='center')
plt.text(1.5e-4, 8e-23, r'$\rightarrow$', fontsize=8, color='0', horizontalalignment='center')
#plt.text(4.5e-5, 1e-22, r'$\hspace{-0.55cm}\mathrm{Therma-}\\\mathrm{lization}\\\mathrm{ }\hspace{0.2cm}\rightarrow$', fontsize=10, color='0')


ax.text(2e-4, 8e-13, r"$3H$", color=ch, fontsize=10, rotation=0)
ax.text(9e-2, 2e-13, r"$\nu_s \nu_s \leftrightarrow X$", color=c1, fontsize=10, rotation=0)
ax.text(1.5e-3, 3e-25, r"$\nu_s \nu_\alpha \to X$", color=c2, fontsize=10, rotation=0)
ax.text(1.5e-3, 0.3e-19, r"$\nu_s \nu_s \to X X$", color=c3, fontsize=10, rotation=0)
ax.text(1.5e-3, 8e-28, r"$X X \to \nu_s \nu_s$", color=c5, fontsize=10, rotation=0)

plt.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='-', color='black', label=r'$\text{BP1}$')
plt.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='--', color='black', label=r'$\text{BP2}$')


plt.fill_betweenx([1e-28, 1e0], 1e-5, 1e-3, color='white', alpha=1, zorder=-3)
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
fig_str = f'rates_evo_BP1_BP2.pdf'
print(f'saved {fig_str}')
# plt.savefig(fig_str)
plt.show()

