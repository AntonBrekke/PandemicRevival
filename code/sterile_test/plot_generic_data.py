#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
from math import cos, sin, asin, sqrt, exp
from scipy.interpolate import make_interp_spline
from scipy.differentiate import derivative

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

import vector_mediator
import scalar_mediator
import C_res_vector
import C_res_scalar
import densities as dens

import constants_functions as cf


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
            'font.size': 10 } # extend as needed
# print(plt.rcParams.keys())
plt.rcParams.update(params)

fig = plt.figure(figsize=get_figsize(columnwidth, wf=1.0), dpi=150, edgecolor="white")
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


BP = 3
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
else: None  
if BP == 4:
    # BP4
    load_str = './md_5.13483e-05;mX_2.56742e-04;mh_1.54045e-04;sin22th_3.66524e-16;y_3.36087e-03;full_new.dat' 
    x_therm = 7e-3
else: None  

data = np.loadtxt(load_str)

var_list = load_str.split(';')[:-1]
m_d, m_X, m_h, sin2_2th, y = [eval(s.split('_')[-1]) for s in var_list]
m_a = 0 

m_d2 = m_d*m_d
m_X2 = m_X*m_X
m_h2 = m_h*m_h
th = 0.5*asin(sqrt(sin2_2th))
c_th = cos(th)
s_th = sin(th)
y2 = y*y

k_d = 1
k_a = 1
k_h = -1
k_X = -1

dof_d = 2.      # Anton: Fermion 2 spin dof.
dof_a = 2.      # Anton: Fermion 2 spin dof.
dof_X = 3.      # Anton: Massive vector boson 3 polarization dof.
dof_h = 1.      # Anton: Scalar has 1 dof.

M2_X_dd = 4*y2*(c_th**4.)*(m_X2-4*m_d2)
M2_X_da = 4*y2*(s_th**2.)*(c_th**2.)*(m_X2-m_d2)*(1 + m_d2/(2*m_X2))
M2_X_aa = 4.*y2*(s_th**4.)*m_X2

M2_h_dd = 2*(4*y2*m_d2/m_X2)*(c_th**4)*(m_h2-4*m_d2)
M2_h_da = 2*(4*y2*m_d2/m_X2)*(c_th**2)*(s_th**2)*(m_h2-m_d2)
M2_h_aa = 2*(4*y2*m_d2/m_X2)*(s_th**4)*m_h2

M2_h_XX = 4*y2*(m_h2**2/m_X2-4*m_h2+12*m_X2)

vert_fi = y2*y2*(c_th**4.)*(s_th**4.)
vert_tr = y2*y2*(c_th**6.)*(s_th**2.)
vert_el = y2*y2*(c_th**8.)

# X --> dd, ad, aa
# h --> dd, ad, aa, XX
Gamma_X = vector_mediator.Gamma_X_new(y=y, th=th, m_X=m_X, m_d=m_d)
Gamma_h = scalar_mediator.Gamma_phi(y=y, th=th, m_phi=m_h, m_d=m_d, m_X=m_X)
m_Gamma_X2 = m_X2*Gamma_X*Gamma_X
m_Gamma_h2 = m_h2*Gamma_h*Gamma_h
def C_n(T_a, T_d, xi_d, xi_X, xi_h):
    if T_a < min(m_X / 50., m_h/50.):
        return 0.
    # Done RIS subtraction for Higgs diagram in C_n_XX_dd
    CX_XX_dd = C_res_vector.C_n_XX_dd(m_d=m_d, m_X=m_X, m_h=m_h, k_d=k_d, k_X=k_X, T_d=T_d, xi_d=xi_d, xi_X=xi_X, vert=vert_el, th=th, m_Gamma_h2=m_Gamma_h2, type=0) / 4. # symmetry factor 1/4
    # Higgs-fermion vertex: 2i*md/mx * (cth**2, sth**2, -sth*cth) * g
    Ch_hh_dd = C_res_scalar.C_n_pp_dd(m_d=m_d, m_phi=m_h, k_d=k_d, k_phi=k_h, T_d=T_d, xi_d=xi_d, xi_phi=xi_h, vert=vert_el*(4*m_d2/m_X2)**2, type=0) / 4.      # symmetry factor 1/4

    # Anton: C_dd cancels/vanish for n = n_d + 2*n_X + 2*n_h
    CX_X_da = C_res_vector.C_n_3_12(m1=m_d, m2=m_a, m3=m_X, k1=k_d, k2=k_a, k3=k_X, T1=T_d, T2=T_a, T3=T_d, xi1=xi_d, xi2=0., xi3=xi_X, M2=M2_X_da, type=0)
    CX_X_aa = C_res_vector.C_n_3_12(m1=m_a, m2=m_a, m3=m_X, k1=k_a, k2=k_a, k3=k_X, T1=T_a, T2=T_a, T3=T_d, xi1=0., xi2=0., xi3=xi_X, M2=M2_X_aa, type=0) / 2.

    Ch_h_da = C_res_vector.C_n_3_12(m1=m_d, m2=m_a, m3=m_h, k1=k_d, k2=k_a, k3=k_h, T1=T_d, T2=T_a, T3=T_d, xi1=xi_d, xi2=0., xi3=xi_h, M2=M2_h_da, type=0)
    Ch_h_aa = C_res_vector.C_n_3_12(m1=m_a, m2=m_a, m3=m_h, k1=k_a, k2=k_a, k3=k_h, T1=T_a, T2=T_a, T3=T_d, xi1=0., xi2=0., xi3=xi_h, M2=M2_h_aa, type=0) / 2.

    Ch_h_XX = C_res_vector.C_n_3_12(m1=m_X, m2=m_X, m3=m_h, k1=k_X, k2=k_X, k3=k_h, T1=T_d, T2=T_d, T3=T_d, xi1=xi_X, xi2=xi_X, xi3=xi_h, M2=M2_h_XX, type=0) / 2.
    
    C_da = CX_X_da + Ch_h_da
    C_aa = CX_X_aa + Ch_h_aa

    C_da_dd = 0.
    C_aa_dd = 0.

    return C_da + 2.*C_aa + C_da_dd + C_aa_dd + 2.*CX_XX_dd + 2.*Ch_hh_dd - 2.*Ch_h_XX

# rho = rho_d + rho_X + rho_h
def C_rho(T_a, T_d, xi_d, xi_X, xi_h):
    if T_a < min(m_X/50., m_h/50.):
        return 0.
    # Decay/inverse decay
    # Anton: type=2: C[X] + C[d] = int (E_x - E_d)*delta(E_x-E_d-E_a)*(f_s*f_a*(1-kX*f_X) - f_X*(1-kd*f_d)*(1-ka*f_a)) = int E_a*delta(E_x-E_d-E_a)*(f_s*f_a*(1-kX*f_X) - f_X*(1-kd*f_d)*(1-ka*f_a)) = C[a]

    # Trick to avoid computation for both X and d, and only do for a once 
    # C[X]_X<->ad + C[d]_X<->ad = -C[a]_X<->ad = C_rho_3_12(type=2)
    C_X_da = C_res_vector.C_rho_3_12(type=2, m1=m_d, m2=m_a, m3=m_X, k1=k_d, k2=k_a, k3=k_X, T1=T_d, T2=T_a, T3=T_d, xi1=xi_d, xi2=0., xi3=xi_X, M2=M2_X_da)
    C_X_aa = C_res_vector.C_rho_3_12(type=3, m1=m_a, m2=m_a, m3=m_X, k1=k_d, k2=k_a, k3=k_X, T1=T_a, T2=T_a, T3=T_d, xi1=0., xi2=0., xi3=xi_X, M2=M2_X_aa) / 2. # symmetry factor 1/2

    C_h_da = C_res_vector.C_rho_3_12(type=2, m1=m_d, m2=m_a, m3=m_h, k1=k_d, k2=k_a, k3=k_h, T1=T_d, T2=T_a, T3=T_d, xi1=xi_d, xi2=0., xi3=xi_h, M2=M2_h_da)
    C_h_aa = C_res_vector.C_rho_3_12(type=3, m1=m_a, m2=m_a, m3=m_h, k1=k_d, k2=k_a, k3=k_h, T1=T_a, T2=T_a, T3=T_d, xi1=0., xi2=0., xi3=xi_h, M2=M2_h_aa) / 2. # symmetry factor 1/2

    C_da = C_X_da + C_h_da
    C_aa = C_X_aa + C_h_aa

    C_da_dd = 0.
    C_aa_dd = 0.
    return C_da + C_aa + C_da_dd + C_aa_dd



# t, T_SM, T_nu, ent, H, sf, T_d, xi_d, xi_X, xi_h, n_d, n_X, n_h
data_skip = 10
t = data[::data_skip, 0]
T_SM = data[::data_skip, 1]
T_nu = data[::data_skip, 2]
ent = data[::data_skip, 3]
H = data[::data_skip, 4]
sf = data[::data_skip, 5]
Td = data[::data_skip, 6]
xid = data[::data_skip, 7]
xiX = data[::data_skip, 8]
xih = data[::data_skip, 9]
nd = data[::data_skip, 10]
nX = data[::data_skip, 11]
nh = data[::data_skip, 12]

# plt.loglog(m_d/T_nu, Td)
# plt.show()

mu_s = xid*Td
mu_h = xih*Td
mu_X = xiX*Td

mu_s = make_interp_spline(t, mu_s, k=10)
mu_h = make_interp_spline(t, mu_h, k=10)
mu_X = make_interp_spline(t, mu_X, k=10)

dmu_sdt = mu_s.derivative(nu=1)(t)
dmu_hdt = mu_h.derivative(nu=1)(t)
dmu_Xdt = mu_X.derivative(nu=1)(t)


# dmu_sdt = np.diff(mu_s, axis=0)/np.diff(t, axis=0)
# dmu_hdt = np.diff(mu_h, axis=0)/np.diff(t, axis=0)
# dmu_Xdt = np.diff(mu_X, axis=0)/np.diff(t, axis=0)

mudot_n_tot = nd*dmu_sdt + nX*dmu_Xdt + nh*dmu_hdt

data_skip = 50
t = data[::data_skip, 0]
T_SM = data[::data_skip, 1]
T_nu = data[::data_skip, 2]
ent = data[::data_skip, 3]
H = data[::data_skip, 4]
sf = data[::data_skip, 5]
Td = data[::data_skip, 6]
xid = data[::data_skip, 7]
xiX = data[::data_skip, 8]
xih = data[::data_skip, 9]
nd = data[::data_skip, 10]
nX = data[::data_skip, 11]
nh = data[::data_skip, 12]

mu_s = xid*Td
mu_h = xih*Td
mu_X = xiX*Td

mudot_n_tot = mudot_n_tot[::data_skip]

C_ns = np.array([-2*C_res_vector.C_n_3_12(m1=m_d, m2=m_d, m3=m_X, k1=k_d, k2=k_d, k3=k_X, T1=Td, T2=Td, T3=Td, xi1=xid, xi2=xid, xi3=xiX, M2=M2_X_dd, type=0) / 2
        - C_res_vector.C_n_3_12(m1=m_d, m2=m_a, m3=m_X, k1=k_d, k2=k_a, k3=k_X, T1=Td, T2=Ta, T3=Td, xi1=xid, xi2=0., xi3=xiX, M2=M2_X_da, type=0) 
        -2*C_res_vector.C_n_3_12(m1=m_d, m2=m_d, m3=m_h, k1=k_d, k2=k_d, k3=k_h, T1=Td, T2=Td, T3=Td, xi1=xid, xi2=xid, xi3=xih, M2=M2_h_dd, type=0) / 2
        - C_res_vector.C_n_3_12(m1=m_d, m2=m_a, m3=m_h, k1=k_d, k2=k_a, k3=k_h, T1=Td, T2=Ta, T3=Td, xi1=xid, xi2=0., xi3=xih, M2=M2_h_da, type=0) 
        -2*C_res_vector.C_n_XX_dd(m_d=m_d, m_X=m_X, m_h=m_h, k_d=k_d, k_X=k_X, T_d=Td, xi_d=xid, xi_X=xiX, vert=vert_el, th=th, m_Gamma_h2=m_Gamma_h2, type=0) / 4.
        -2*C_res_scalar.C_n_pp_dd(m_d=m_d, m_phi=m_h, k_d=k_d, k_phi=k_h, T_d=Td, xi_d=xid, xi_phi=xih, vert=vert_el*(4*m_d2/m_X2)**2, type=0) / 4.
        for Td, Ta, xid, xiX, xih in zip(Td, T_nu, xid, xiX, xih)])

C_nX = np.array([C_res_vector.C_n_3_12(m1=m_d, m2=m_d, m3=m_X, k1=k_d, k2=k_d, k3=k_X, T1=Td, T2=Td, T3=Td, xi1=xid, xi2=xid, xi3=xiX, M2=M2_X_dd, type=0) / 2
        + C_res_vector.C_n_3_12(m1=m_d, m2=m_a, m3=m_X, k1=k_d, k2=k_a, k3=k_X, T1=Td, T2=Ta, T3=Td, xi1=xid, xi2=0., xi3=xiX, M2=M2_X_da, type=0)
        + C_res_vector.C_n_3_12(m1=m_a, m2=m_a, m3=m_X, k1=k_a, k2=k_a, k3=k_X, T1=Ta, T2=Ta, T3=Td, xi1=0., xi2=0., xi3=xiX, M2=M2_X_aa, type=0) 
        + 2*C_res_vector.C_n_XX_dd(m_d=m_d, m_X=m_X, m_h=m_h, k_d=k_d, k_X=k_X, T_d=Td, xi_d=xid, xi_X=xiX, vert=vert_el, th=th, m_Gamma_h2=m_Gamma_h2, type=0) / 4.
        for Td, Ta, xid, xiX in zip(Td, T_nu, xid, xiX)])

C_nh = np.array([C_res_vector.C_n_3_12(m1=m_d, m2=m_d, m3=m_h, k1=k_d, k2=k_d, k3=k_h, T1=Td, T2=Td, T3=Td, xi1=xid, xi2=xid, xi3=xih, M2=M2_h_dd, type=0) / 2
        + C_res_vector.C_n_3_12(m1=m_d, m2=m_a, m3=m_h, k1=k_d, k2=k_a, k3=k_h, T1=Td, T2=Ta, T3=Td, xi1=xid, xi2=0., xi3=xih, M2=M2_h_da, type=0) 
        + C_res_vector.C_n_3_12(m1=m_a, m2=m_a, m3=m_h, k1=k_a, k2=k_a, k3=k_h, T1=Ta, T2=Ta, T3=Td, xi1=0., xi2=0., xi3=xih, M2=M2_h_aa, type=0) 
        + 2*C_res_scalar.C_n_pp_dd(m_d=m_d, m_phi=m_h, k_d=k_d, k_phi=k_h, T_d=Td, xi_d=xid, xi_phi=xih, vert=vert_el*(4*m_d2/m_X2)**2, type=0) / 4.
        for Td, Ta, xid, xih in zip(Td, T_nu, xid, xih)])

# C_na = [-C_res_vector.C_n_3_12(m1=m_d, m2=m_a, m3=m_X, k1=k_d, k2=k_a, k3=k_X, T1=Td, T2=Ta, T3=Td, xi1=xid, xi2=0., xi3=xiX, M2=M2_X_da, type=0)
#         - 2*C_res_vector.C_n_3_12(m1=m_a, m2=m_a, m3=m_X, k1=k_a, k2=k_a, k3=k_X, T1=Ta, T2=Ta, T3=Td, xi1=0., xi2=0., xi3=xiX, M2=M2_X_aa, type=0)
#         - C_res_vector.C_n_3_12(m1=m_d, m2=m_a, m3=m_h, k1=k_d, k2=k_a, k3=k_h, T1=Td, T2=Ta, T3=Td, xi1=xid, xi2=0., xi3=xih, M2=M2_h_da, type=0) 
#         - 2*C_res_vector.C_n_3_12(m1=m_a, m2=m_a, m3=m_h, k1=k_a, k2=k_a, k3=k_h, T1=Ta, T2=Ta, T3=Td, xi1=0., xi2=0., xi3=xih, M2=M2_h_aa, type=0) 
#         for Td, Ta, xid, xiX, xih in zip(Td, T_nu, xid, xiX, xih)]

mu_Cn = mu_s*C_ns + mu_X*C_nX + mu_h*C_nh

# C_rho = C_rho,h + C_rho,X + C_rho,s
C_rho = np.array([C_rho(T_a=Td, T_d=Td, xi_d=xid, xi_X=xiX, xi_h=xih) for Td, xid, xiX, xih in zip(Td, xid, xiX, xih)])
C_n = C_ns + C_nX + C_nh

# d/dt(a^3 s) = a^3/Td * (C_\rho - C_n*mu - dmu/dt * n)
# dimless: 1/H * d/dt(a^3 * s)
# H = 1
# C_rho_kin = C_rho - m_d*C_ns - m_X*C_nX - m_h*C_nh

rho_s = np.array([dens.rho(k_d, Td, m_d, dof_d, xid) for Td, xid in zip(Td, xid)])
rho_X = np.array([dens.rho(k_X, Td, m_X, dof_X, xiX) for Td, xiX in zip(Td, xiX)])
rho_h = np.array([dens.rho(k_h, Td, m_h, dof_h, xih) for Td, xih in zip(Td, xih)])
rho_d = rho_s + rho_X + rho_h

drho_sdT = np.array([dens.rho_der_T(k_d, Td, m_d, dof_d, xid) for Td, xid in zip(Td, xid)])
drho_XdT = np.array([dens.rho_der_T(k_X, Td, m_X, dof_X, xiX) for Td, xiX in zip(Td, xiX)])
drho_hdT = np.array([dens.rho_der_T(k_h, Td, m_h, dof_h, xih) for Td, xih in zip(Td, xih)])
drho_dT = drho_sdT + drho_XdT + drho_hdT

dn_sdT = np.array([dens.n_der_T(k_d, Td, m_d, dof_d, xid) for Td, xid in zip(Td, xid)])
dn_XdT = np.array([dens.n_der_T(k_X, Td, m_X, dof_X, xiX) for Td, xiX in zip(Td, xiX)])
dn_hdT = np.array([dens.n_der_T(k_h, Td, m_h, dof_h, xih) for Td, xih in zip(Td, xih)])
dn_dT = dn_sdT + dn_XdT + dn_hdT

P_s = np.array([dens.P(k_d, Td, m_d, dof_d, xid) for Td, xid in zip(Td, xid)])
P_X = np.array([dens.P(k_X, Td, m_X, dof_X, xiX) for Td, xiX in zip(Td, xiX)])
P_h = np.array([dens.P(k_h, Td, m_h, dof_h, xih) for Td, xih in zip(Td, xih)])
P_d = P_s + P_X + P_h

rho_mass = m_d*nd + m_X*nX + m_h*nh
rho_kin = rho_d - rho_mass

drho_mass_dT = m_d*dn_sdT + m_X*dn_XdT + m_h*dn_hdT
drho_kin_dT = drho_dT - drho_mass_dT
C_rho_mass = m_d*C_ns + m_X*C_nX + m_h*C_nh
C_rho_kin = C_rho - m_d*C_ns - m_X*C_nX - m_h*C_nh

rho_plus_P = rho_d + P_d
mu_s_nd = mu_s*nd + mu_X*nX + mu_h*nh
drhodt = C_rho - 3*H*rho_plus_P

ent_DS = sf**3/Td*(rho_plus_P - mu_s_nd) # entropy dark sector
dSdt_pr_H = sf**3/Td * (C_rho - mu_Cn) / (ent_DS)    # time derivative entropy pr. H 

dTdt = (C_rho - 3*H*rho_plus_P)/drho_dT
heat_transfer_pr_H = (C_rho - mu_Cn) / H
heat_capacity = heat_transfer_pr_H*H / dTdt         # dq/dT

ent_SM = sf**3*ent

drho_kin_dT_1 = (C_rho_kin - 3*H*(rho_kin + P_d)) / dTdt
drho_kin_dT_2 = drho_kin_dT

dTdt_1 = (C_rho - 3*H*rho_plus_P)/drho_dT
dTdt_2 = (C_rho_kin - 3*H*(rho_kin + P_d)) / drho_kin_dT

drho_kin_dt = (C_rho_kin - 3*H*(rho_kin + P_d))
dn_s_dt = C_ns - 3*H*nd
dn_X_dt = C_nX - 3*H*nX
dn_h_dt = C_nh - 3*H*nh


rho_s_kin = rho_s - m_d*nd
rho_X_kin = rho_X - m_X*nX
rho_h_kin = rho_h - m_h*nh

C_rho_s = dTdt*drho_sdT + 3*H*(rho_s + P_s)
C_rho_X = dTdt*drho_XdT + 3*H*(rho_X + P_X)
C_rho_h = dTdt*drho_hdT + 3*H*(rho_h + P_h)

drho_s_kin_dt = C_rho_s - m_d*C_ns
drho_X_kin_dt = C_rho_X - m_X*C_nX
drho_h_kin_dt = C_rho_h - m_h*C_nh

Tdot_pr_Td = (drho_s_kin_dt / rho_s_kin - dn_s_dt / nd)

drho_s_kin_dt = (C_rho_s - m_d*C_ns - 3*H*(rho_s_kin + P_s))
drho_X_kin_dt = (C_rho_X - m_X*C_nX - 3*H*(rho_X_kin + P_X))
drho_h_kin_dt = (C_rho_h - m_h*C_nh - 3*H*(rho_h_kin + P_h))

plt.loglog(m_d/T_nu, Td)
plt.loglog(m_d/T_nu, T_nu)
plt.show()

plt.loglog(m_d/T_nu, abs((rho_plus_P - mu_s_nd)))
# plt.loglog(m_d/T_nu, abs(dSdt_pr_H))
# plt.loglog(m_d/T_nu, abs(drho_s_kin_dt / rho_s_kin - dn_s_dt / nd))
# plt.loglog(m_d/T_nu, abs(dTdt_2 / (H*Td)))
# plt.loglog(m_d/T_nu, abs(dTdt_2/(Td*H)))
# plt.loglog(m_d/T_nu, abs(dTdt/Td + dn_s_dt/nd))
# plt.loglog(m_d/T_nu, abs(dTdt/Td + dn_X_dt/nX))
# plt.loglog(m_d/T_nu, 1/H*abs(dsrho_s_kin_dt/rho_s_kin - dn_s_dt/nd))
# plt.loglog(m_d/T_nu, abs(drho_s_kin_dt/rho_s*nd/dn_s_dt))
# plt.loglog(m_d/T_nu, abs(drho_X_kin_dt/rho_X*nX/dn_X_dt))
# plt.loglog(m_d/T_nu, abs(drho_h_kin_dt/rho_h*nh/dn_h_dt))
# plt.loglog(m_d/T_nu, abs(dn_s_dt/nd))
# plt.loglog(m_d/T_nu, rho_s_kin/nd)
# plt.loglog(m_d/T_nu, Td)

# plt.loglog(m_d/T_nu, (rho_s_kin), color='tab:blue', ls='--')
# plt.loglog(m_d/T_nu, (m_d*nd), color='tab:orange', ls='--')

# plt.loglog(m_d/T_nu, nd)
# plt.loglog((m_d/T_nu), abs(rho_d))
# plt.loglog((m_d/T_nu), abs(P_d))
# plt.loglog((m_d/T_nu), abs(mu_s*(nd + 2*nX + 2*nh)))
plt.loglog((m_d/T_nu), abs(sf**(-3)*((rho_plus_P - mu_s_nd))[-1]/(sf**(-3))[-1]))
plt.loglog((m_d/T_nu), abs(sf**(-4)*((rho_plus_P - mu_s_nd))[0]/(sf**(-4))[0]))
# plt.loglog((m_d/T_nu), abs(rho_kin))
# plt.loglog((m_d/T_nu), abs(rho_mass))
# plt.loglog((m_d/T_nu), abs(C_rho), ls='--', color='tab:blue')
# plt.loglog((m_d/T_nu), abs(C_rho_kin), ls='--', color='tab:orange')
# plt.loglog((m_d/T_nu), abs(C_rho_mass), ls='--', color='tab:green')
# plt.loglog((m_d/T_nu), abs(Tdot_pr_Td / H))
# plt.loglog((m_d/T_nu), abs(drho_kin_dt / (H*rho_kin)))
# plt.loglog((m_d/T_nu), abs(dn_s_dt / (H*nd)), ls='--', color='tab:blue')
# plt.loglog((m_d/T_nu), (dn_dT), ls='--', color='tab:blue')
# plt.loglog((m_d/T_nu), abs(Tdot_pr_Td))
# plt.loglog((m_d/T_nu), abs(dTdt_2 / Td), ls='--', color='tab:orange')
# plt.loglog((m_d/T_nu), (rho_d), ls='--', color='tab:orange')
# plt.loglog((m_d/T_nu), (rho_mass), ls='--', color='tab:green')
# plt.loglog((m_d/T_nu), (rho_kin), ls='--', color='tab:blue')
# plt.loglog((m_d/T_nu), (rho_d), ls='--', color='tab:orange')
# plt.loglog((m_d/T_nu), (rho_mass), ls='--', color='tab:green')



# plt.loglog((m_d/T_nu), abs(heat_capacity))
# plt.loglog((m_d/T_nu), Td)
# plt.loglog((m_d/T_nu), abs(drho_kin_dT/H))
# plt.loglog((m_d/T_nu), abs(dSdt_pr_H), label=r'$\left|H^{-1}S_d^{-1}\frac{d}{dt}(S_d)\right|$')
# plt.loglog((m_d/T_nu), abs(S_d), label=r'$\left|S_d\right|$')
# plt.loglog((m_d/T_nu), abs(heat_transfer_pr_H), label=r'$\dot q / H$')
# plt.loglog((m_d/T_nu), abs(drhodt_pr_H), label=r'$C_{\rm rho,kin}$')
# plt.semilogx((m_d/T_nu), abs(ent))
# plt.semilogx((m_d/T_nu), rho_d)
# plt.semilogx((m_d/T_nu), P_d)
# plt.semilogx((m_d/T_nu), sf**3/H*abs(C_rho - 3*H*(rho_d + P_d)))
# plt.semilogx((m_d/T_nu), abs(s_d))
plt.xlabel(r'$m_s/T_\nu$')
plt.axvline(1e-3, ls='--', color='k')
plt.legend()
plt.show()