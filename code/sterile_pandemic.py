#! /usr/bin/env python3

import numpy as np
import numba as nb
from math import cos, sin, asin, sqrt, exp

import constants_functions as cf

import vector_mediator
# import resonant_pandemolator as pandemolator
import pandemolator as pandemolator
import C_res_vector
import C_res_scalar
import time

"""
Anton: 
theta << 1
cos(theta) ~ 1
sin^2(2*theta) = 4*cos^2(theta)*sin^2(theta) ~ 4*sin^2(theta)
Decays dominate due to scale of y^2:
X <--> da ~ y^2 * cos^2(theta)*sin^2(theta) ~ y^2 * sin^2(2*theta)
X <--> dd ~ y^2 * cos^4(theta) ~ y^2
For larger y, 2-to-2 also becomes important 
XX <--> dd ~ y^4 * cos^4(theta) ~ y^4

Larger y must be compansated with smaller sin^2(2*theta) to 
not make pandemic growth to strong. 
Generically roughly expect: lower/increase y by one order <--> increase(lower) sin22th by two orders
"""

m_ratio = 3           # If m_d = 10 keV, m_X = 10*1e4 keV = 100 MeV
# C = (1.5e-3)**4 * 3.5e-15       # Anton: Tested, this value gives close to Omega*h^2 = 0.12

# BP1 
# m_d = 12e-6          # 1e-6*M GeV = M keV, 2e-5 GeV = 20 keV
# m_a = 0.
# m_X = m_ratio*m_d
# sin2_2th = 2.5e-13
# y = 1.905e-4

# BP2 
m_d = 20e-6          # 1e-6*M GeV = M keV, 2e-5 GeV = 20 keV
m_a = 0.
m_X = m_ratio*m_d
sin2_2th = 3e-15
y = 1.602e-3

# load_str = './md_3.79269e-05;mX_1.13781e-04;sin22th_3.66524e-16;y_3.42622e-03;full.dat'
# load_str = './md_2.06914e-05;mX_6.20741e-05;sin22th_1.12534e-16;y_4.80046e-03;full.dat'
# var_list = load_str.split(';')[:-1]
# m_d, m_X, sin2_2th, y = [eval(s.split('_')[-1]) for s in var_list]
m_a = 0.

# Manual  
# m_d = 20e-6          # 1e-6*M GeV = M keV, 2e-5 GeV = 20 keV
# m_a = 0.
# m_X = m_ratio*m_d
# sin2_2th = 3e-15
# y = 1.602e-3

print(f'md: {m_d:.3e}, mX: {m_X:.3e}, y: {y:.3e}, sin22th: {sin2_2th:.3e}')

# Anton: fermion = 1, boson = -1
k_d = 1.
k_a = 1.
k_X = -1.

dof_d = 2.      # Anton: Fermions have 2 spin dofs. 
dof_X = 3.      # Anton: Massive vector boson has 3 polarization dof., removed longitudinal component

m_d2 = m_d*m_d
m_a2 = m_a*m_a
m_X2 = m_X*m_X
th = 0.5*asin(sqrt(sin2_2th))
c_th = cos(th)
s_th = sin(th)
y2 = y*y

# Anton: Matrix elements added here for some reason
# M2_dd = 2. * y2 * (c_th**4.) * (m_X2 - 4.*m_d2)
# M2_da = 2. * y2 * (s_th**2.) * (c_th**2.) * (m_X2 - ((m_a+m_d)**2.))
# M2_aa = 2. * y2 * (s_th**4.) * (m_X2 - 4.*m_a2)

# M2_X23 = 2*g^2/m_X^2 * (m_X2 - (m2 - m3)**2)*(2*m_X2 + (m2 + m3)**2)
# Anton: Vector coupling only 
# M2_dd = 2.*y2*(c_th**4.)/m_X2 * (m_X2)*(2*m_X2 + (2*m_d)**2)
# M2_da = 2.*y2*(s_th**2.)*(c_th**2.)/m_X2 * (m_X2 - m_d2)*(2*m_X2 + m_d2)
# M2_aa = 2.*y2*(s_th**4.)/m_X2 * (m_X2)*(2*m_X2)

# Anton: Test if new Feynman rules work. M2_da x2 larger, M2_dd change
# Anton: Vector and/or axial coupling gamma^mu * (gV - gamma^5)
# M2_dd = 4.*y2*(c_th**4.)*(m_X2-4*m_d2)
# M2_da = 4.*y2*(s_th**2.)*(c_th**2.)/m_X2 * (m_X2 - m_d2)*(2*m_X2 + m_d2)
# M2_aa = 4.*y2*(s_th**4.)*m_X2

# Anton: Removed longitudinal component of spin sum
# M2_da = 8*y2*(s_th**2.)*(c_th**2.)*(m_X2-m_d2)
M2_dd = 4*y2*(c_th**4.)*(m_X2-6*m_d2)
M2_da = 4*y2*(s_th**2.)*(c_th**2.)*(m_X2-m_d2)
M2_aa = 4.*y2*(s_th**4.)*m_X2

vert_fi_da = y2*y2*(c_th**2.)*(s_th**6.)       # aa <--> da
vert_fi_dd = y2*y2*(c_th**4.)*(s_th**4.)       # aa <--> dd
vert_tr = y2*y2*(c_th**6.)*(s_th**2.)       # ad <--> dd
vert_el = y2*y2*(c_th**8.)                  # dd <--> dd

print(f'vert_el:{vert_el:.3e}, vert_tr:{vert_tr:.3e}, vert_fi_dd:{vert_fi_dd:.3e}, vert_fi_da:{vert_fi_da:.3e}')

Gamma_X = vector_mediator.Gamma_X_new(y, th, m_X, m_d)
m_Gamma_X2 = m_X2*Gamma_X*Gamma_X

"""
Anton:
res_sub = True: |D_off-shell|^2 is used -- on-shell contribution is subtracted, and decays should be included manually
res_sub = False: |D_BW|^2 is used -- already counts decay/inverse decay contributions via s-channel resonance

nFW = # occurence of particle in final - # occurence of particle in inital for forward process (34 --> 12)
nBW = # occurence of particle in final - # occurence of particle in inital for backward process (12 --> 34)

nFW = -nBW if they are for same process
Same procedure should be done for all C_n, i.e. sum collision operator for each occurence of particle.  

Collision-operator C_XX_dd, is set up for evolution of X, C_XX_dd = C[X]_XX_dd = -C[d]_XX_dd.
Collision-operator C_3_12, is set up for evolution of 3 (X), C_12 = C[X]_X_12 = -C[1,2]_X_12.
Collision-operator C_34_12, is set up for evolution of 1,2, C_34_12 = C[1,2]_34_12 = -C[3,4]_34_12.

Multiple occurences of particle A in inital/final state is summed over, i.e. X --> AA' + X --> A'A.
n = n_d + 2*n_X, both d and X is counted. 
C[d]_XX_dd' + C[d]_XX_d'd + 2*(C[X]_XX'_dd + C[X]_X'X_dd) = -2*C[X]_XX_dd + 4*C[X]_XX_dd = 2*C[X]_XX_dd = 2*C_XX_dd
or 
C[d]_X->da + 2*C[X]_X->da = -C[X]_X->da + 2*C[X]_X->da = C[X]_X_da = C_da,
while
C[d]_X_dd' + C[d]_X_d'd + 2*C[X]_X_dd = 0 

Only for n_d: 
For 34 --> 12, nFW and nBW takes care of the counting. E.g.
Writing C[3]_12<->34 = C[3]_12->34 - C[3]_34->12 (- since 3 is in initial state), 
C[d]_ad''->dd' + C[d]_ad''->d'd - C[d]_dd'->ad'' - C[d]_d'd->ad'' - C[d]_ad->d'd'' + C[d]_d'd''->ad = C[d]_ad<->dd
--> nFW=1, nBW=-1
C[d]_aa->dd' + C[d]_aa->d'd - C[d]_dd'->aa - C[d]_d'd->aa = 2*C[d]_aa<->dd 
--> nFW=2, nBW=-2

If we neglige t,u-channel contributions for 34 <--> 12, we can set C_34_12 = 0 and instead calculate decay-rates, 
as these equal to the on-shell contribution of the s-channel. 
This speeds up the runtime of the code significantly, ~1.5-2h --> ~10-20m by experience. 
"""
# n = n_d + 2.*n_X
def C_n(T_a, T_d, xi_d, xi_X):
    # Decay/inverse decay X <--> 23
    C_aa = C_res_vector.C_n_3_12(m1=m_a, m2=m_a, m3=m_X, k1=k_a, k2=k_a, k3=k_X, T1=T_a, T2=T_a, T3=T_d, xi1=0., xi2=0., xi3=xi_X, M2=M2_aa, type=0) / 2.
    C_da = C_res_vector.C_n_3_12(m1=m_d, m2=m_a, m3=m_X, k1=k_d, k2=k_a, k3=k_X, T1=T_d, T2=T_a, T3=T_d, xi1=xi_d, xi2=0., xi3=xi_X, M2=M2_da, type=0)      # type=0 include reaction both ways 
    C_dd = 0        # Cancels 

    # 2-to-2
    # dd <--> XX
    C_XX_dd = C_res_vector.C_n_XX_dd(m_d=m_d, m_X=m_X, k_d=k_d, k_X=k_X, T_d=T_d, xi_d=xi_d, xi_X=xi_X, vert=vert_el, type=0) / 4. # symmetry factor 1/4, type=0 include reaction both ways
    # Pandemic growth
    # C_da_dd = C_res_vector.C_34_12(type=0, nFW=1., nBW=-1., m1=m_d, m2=m_d, m3=m_d, m4=m_a, k1=k_d, k2=k_d, k3=k_d, k4=k_a, T1=T_d, T2=T_d, T3=T_d, T4=T_a, xi1=xi_d, xi2=xi_d, xi3=xi_d, xi4=0., vert=vert_tr, m_X2=m_X2,m_Gamma_X2=m_Gamma_X2, gV1=1, gV2=0, res_sub=False) / 2.
    # Freeze-in
    # C_aa_dd = C_res_vector.C_34_12(type=0, nFW=2., nBW=-2., m1=m_d, m2=m_d, m3=m_a, m4=m_a, k1=k_d, k2=k_d, k3=k_a, k4=k_a, T1=T_d, T2=T_d, T3=T_a, T4=T_a, xi1=xi_d, xi2=xi_d, xi3=0., xi4=0., vert=vert_fi_dd, m_X2=m_X2, m_Gamma_X2=m_Gamma_X2, gV1=0, gV2=0, res_sub=False) / 4.
    # C_aa_da = C_res_vector.C_34_12(type=0, nFW=1., nBW=-1., m1=m_d, m2=m_a, m3=m_a, m4=m_a, k1=k_d, k2=k_a, k3=k_a, k4=k_a, T1=T_d, T2=T_a, T3=T_a, T4=T_a, xi1=xi_d, xi2=0., xi3=0., xi4=0., vert=vert_fi_da, m_X2=m_X2, m_Gamma_X2=m_Gamma_X2, res_sub=True) / 2.

    C_da_dd = 0
    C_aa_dd = 0
    C_aa_da = 0

    print("C_ns:  ", f'{C_da:.5e}', f'{2.*C_aa:.5e}', f'{C_da_dd}', f'{C_aa_da}', f'{2.*C_XX_dd:.5e}')
    return C_da + 2*C_aa + C_dd + C_da_dd + 2*C_aa_dd + C_aa_da + 2*C_XX_dd

# rho = rho_d + rho_X
def C_rho(T_a, T_d, xi_d, xi_X):
    # Decay/inverse decay
    # Anton: type=2: C[X] + C[d] = int (E_x - E_d)*delta(E_x-E_d-E_a)*(f_s*f_a*(1-kX*f_X) - f_X*(1-kd*f_d)*(1-ka*f_a)) = int E_a*delta(E_x-E_d-E_a)*(f_s*f_a*(1-kX*f_X) - f_X*(1-kd*f_d)*(1-ka*f_a)) = C[a]
    # Trick to avoid computation for both X and d, and only do for a once 
    C_da = C_res_vector.C_rho_3_12(type=2, m1=m_d, m2=m_a, m3=m_X, k1=k_d, k2=k_a, k3=k_X, T1=T_d, T2=T_a, T3=T_d, xi1=xi_d, xi2=0., xi3=xi_X, M2=M2_da)
    C_aa = C_res_vector.C_rho_3_12(type=3, m1=m_a, m2=m_a, m3=m_X, k1=k_d, k2=k_a, k3=k_X, T1=T_a, T2=T_a, T3=T_d, xi1=0., xi2=0., xi3=xi_X, M2=M2_aa) / 2. # symmetry factor 1/2

    # 2-to-2
    # Pandemic growth only included 
    # C_da_dd = C_res_vector.C_34_12(type=1, nFW=1., nBW=-1., m1=m_d, m2=m_d, m3=m_d, m4=m_a, k1=k_d, k2=k_d, k3=k_d, k4=k_a, T1=T_d, T2=T_d, T3=T_d, T4=T_a, xi1=xi_d, xi2=xi_d, xi3=xi_d, xi4=0., vert=vert_tr, m_X2=m_X2, m_Gamma_X2=m_Gamma_X2, gV1=1, gV2=0, res_sub=False) / 2.
    # C_aa_dd = C_res_vector.C_34_12(type=1, nFW=2., nBW=-2., m1=m_d, m2=m_d, m3=m_a, m4=m_a, k1=k_d, k2=k_d, k3=k_a, k4=k_a, T1=T_d, T2=T_d, T3=T_a, T4=T_a, xi1=xi_d, xi2=xi_d, xi3=0., xi4=0., vert=vert_fi_dd, m_X2=m_X2, m_Gamma_X2=m_Gamma_X2, gV1=0, gV2=0, res_sub=False) / 2.

    C_da_dd = 0
    C_aa_dd = 0
    C_aa_da = 0 

    print("C_rhos:", f'{C_da:.5e}', f'{C_aa:.5e}', f'{C_da_dd}', f'{C_aa_da}', f'{C_aa_dd}')
    return C_da + C_aa + C_da_dd + C_aa_da + C_aa_dd

def C_xi0(T_a, T_d, xi_d, xi_X):
    C_XX_dd = C_res_vector.C_n_XX_dd(m_d=m_d, m_X=m_X, k_d=k_d, k_X=k_X, T_d=T_d, xi_d=xi_d, xi_X=xi_X, vert=vert_el, type=1) / 4.      # type=1, only dd --> XX
    return 2.*C_XX_dd

Ttrel = pandemolator.TimeTempRelation()
ent_grid = np.array([cf.s_SM_no_nu(T)+cf.s_nu(T_nu) for T, T_nu in zip(Ttrel.T_SM_grid, Ttrel.T_nu_grid)])
# np.savetxt('sterile_test/ent_grid.dat', ent_grid)
T_d_DW = 0.133*((1e6*m_d)**1./3.) # temperature of maximal d production by Dodelson-Widrow mechanism
i_ic = np.argmax(Ttrel.T_nu_grid < T_d_DW)
i_end = np.argmax(Ttrel.T_nu_grid < 0.01*m_d)
sf_ic_norm_d_DW = (cf.s_SM_before_nu_dec(T_d_DW)/(cf.s_SM_no_nu(Ttrel.T_SM_grid[i_ic]) + cf.s_nu(Ttrel.T_nu_grid[i_ic])))**(1./3.)
O_d_h2 = 0.3*1e10*sin2_2th*((1e4*m_d)**2.)
norm_f_d_0 = 4.*cf.pi2*cf.s_SM_before_nu_dec(T_d_DW)*O_d_h2*cf.rho_crit0_h2/(3.*cf.zeta3*(T_d_DW**3.)*m_d*cf.s0)
T_d_ic = ((norm_f_d_0/(1.+8.*dof_X/(7.*dof_d)))**(1./3.))*T_d_DW/sf_ic_norm_d_DW
xi_d_ic = 0.
xi_X_ic = 0.

sf_ic_norm_0 = (cf.s0/(cf.s_SM_no_nu(Ttrel.T_SM_grid[i_ic]) + cf.s_nu(Ttrel.T_nu_grid[i_ic])))**(1./3.)
n_ic = cf.n_0_dw(m_d, th) / (sf_ic_norm_0**3.)
rho_ic = n_ic * cf.avg_mom_0_dw(m_d) / sf_ic_norm_0

# pan = pandemolator.Pandemolator(m_d, k_d, dof_d, m_X, k_X, dof_X, m_a, k_a, C_n, C_rho, C_xi0, Ttrel.t_grid, Ttrel.T_nu_grid, Ttrel.dTnu_dt_grid, ent_grid, Ttrel.hubble_grid, Ttrel.sf_grid, i_ic, T_d_ic, xi_d_ic, xi_X_ic, i_end)
# self, m_chi, k_chi, dof_chi, m_X, k_X, dof_X, m_psi, k_psi, C_n, C_rho, C_xi0, t_grid, T_grid, dT_dt_grid, ent_grid, hubble_grid, sf_grid, i_ic, n_ic, rho_ic, i_end
print('sterile_pandemic.py start pandemolator')
time_now = time.localtime()
print(f'Estimated finish {(time_now.tm_hour + 1 + (time_now.tm_min + 30)//60)%24}:{(time_now.tm_min + 30)%60} -- {time_now.tm_hour + 2}:{time_now.tm_min}')
pan = pandemolator.Pandemolator(m_chi=m_d, k_chi=k_d, dof_chi=dof_d, m_X=m_X, k_X=k_X, dof_X=dof_X, m_psi=m_a, k_psi=k_a, C_n=C_n, C_rho=C_rho, C_xi0=C_xi0, t_grid=Ttrel.t_grid, T_grid=Ttrel.T_nu_grid, dT_dt_grid=Ttrel.dTnu_dt_grid, ent_grid=ent_grid, hubble_grid=Ttrel.hubble_grid, sf_grid=Ttrel.sf_grid, i_ic=i_ic, n_ic=n_ic, rho_ic=rho_ic, i_end=i_end)
start = time.time()
pan.pandemolate()
end = time.time()

T = end - start
print(f'pandemolator ran in {T//60//60}h, {T//60%60}m, {T%60:.0f}s')

md_str = f'{m_d:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_d:.5e}'.split('e')[1].rstrip('0').rstrip('.')
mX_str = f'{m_X:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_X:.5e}'.split('e')[1].rstrip('0').rstrip('.')
sin22th_str = f'{sin2_2th:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{sin2_2th:.5e}'.split('e')[1].rstrip('0').rstrip('.')
y_str = f'{y:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{y:.5e}'.split('e')[1].rstrip('0').rstrip('.')

file_str = f'sterile_test/md_{md_str};mX_{mX_str};sin22th_{sin22th_str};y_{y_str};full_new.dat'
np.savetxt(file_str, np.column_stack((Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], ent_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_X_grid_sol, pan.n_chi_grid_sol, pan.n_X_grid_sol)))

print(f'Saved data to {file_str}')
