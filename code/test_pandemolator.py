#! /usr/bin/env python3

import numpy as np
from math import asin, sqrt, exp, log
from scipy.integrate import solve_ivp
import time

import constants_functions as cf
import utils

import vector_mediator
import pandemolator as pandemolator

import matplotlib.pyplot as plt

GF = 1.166378e-5
mZ = 91.1876
mW = 80.379

def call(
        m_N1, m_N2, m_X, m_nu, m0, m12, m2, ma,
        k_d, k_X, k_nu,
        dof_d, dof_X,
        y,
        sin2_2th=None,
        spin_facs=True,
        off_shell=False
    ):
    m_N12 = m_N1*m_N1
    m_X2 = m_X*m_X
    m_h = 0.
    m_h2 = m_h*m_h
    if sin2_2th is None:
        sin2_2th = (m2*ma/(2*m0*m12))**2
        print(f"sin^2(2th) = {sin2_2th:.3e} (calculated from m0, m12, m2, ma)")
    th = 1/2*np.arcsin(sqrt(sin2_2th))
    sin2_th = np.sin(th)**2
    y2 = y*y

    print(f'y: {y:.3e}, sin^2(th): {sin2_2th:.3e}')
    print(f'y^4: {y**4:.3e}, y^2 sin^2(2th): {y**2*sin2_2th:.3e}')

    M2_X_12 = 2.*y2 * (m_X+m_N1-m_N2)*(m_X-m_N1+m_N2)*(2*m_X2 + (m_N1+m_N2)**2)/m_X2
    M2_X_1nu = 2.*y2*sin2_th * (m_X-m_N1-m_nu)*(m_X+m_N1+m_nu)*(2*m_X2+(m_N1-m_nu)**2)/m_X2

    vert_el = y2*y2

    # Anton: X --> 12, 1nu
    Gamma_X = vector_mediator.Gamma_X_new(
        y=y,
        m_X=m_X, m_N1=m_N1, m_N2=m_N2, m_nu=m_nu,
        sin2_2th=sin2_2th
    )
    m_Gamma_X2 = m_X2*Gamma_X*Gamma_X
    m_Gamma_h2 = 0

    if spin_facs:       # Anton: If spin statistics is important
        import C_res_vector
        if m_X > m_N1 + m_N2:
            import C_res_vector_no_spin_stat as C_res_vector_no_spin_stat
            call.count = 0
            call.x_list = []
            call.C_list = []
            # n = n_N1 + n_N2 + 2*n_X
            def C_n(T_a, T_d, xi_d, xi_X):
                """
                Anton: A lot of processes do not contriubute due to equilibrium or no change in particle number. 

                Collision operator describing particle alpha: 
                Cn[alpha]_{I_r -> F_r} = eps^alpha_r int dPI |M|^2 prod_{i in I_r} f_i * prod_{j in F_r} (1+k_j*f_j) / kappa_r 

                kappa_r : symmetry factor 
                eps^alpha_r : = 1 if alpha in F_r, = -1 if alpha in I_r

                From this, have Cn[alpha in I_r] = -Cn[beta in F_r], so 
                Cn[alpha in I_r] + Cn[beta in F_r] = 0

                Code: C_n_3_12(type=0) = C[3]_{3<->12} = int dPI |M|^2 * [f1*f2*(1+f3) - f3*(1+k1*f1)*(1+k2*f2)]

                n = n1 + n2 + 2*nX
                Then for example the processes 
                X -> 12:
                Cn[1]_{X->12} + Cn[2]_{X->12} + 2*Cn[X]_{X->12} = 0
                11 -> 22:
                2*Cn[1]_{11->22} + 2*Cn[2]_{11->22} = 0 
                But for 11 <-> XX, 
                2*Cn[1]_{11<->XX} + 4*Cn[X]_{11<->XX} = 2*Cn[X]_{11<->XX}
                and X <-> 1nu
                C[1]_{X<->1nu} + 2*C[X]_{X<->1nu} = C[X]_{X<->1nu}

                Thus in total, our Boltzmann equation is 
                n + 3Hn = C[X]_{X<->1nu} + 2*C[X]_{11<->XX} + 2*C[X]_{22<->XX}
                """
                if T_a < m_X / 50:
                    return 0.

                # th, m_Gamma_h2 do not matter anymore
                # as long as mN1 = mN2, xiN1 = xiN2, TN1 = TN2, do not need CX_XX_22 separately -- just add factor 2 
                # TODO: Why divide by 4 and multiply by 4 in return statement? Symmetry factor (2*2)
                CX_XX_11 = C_res_vector.C_n_XX_dd(
                    m_d=m_N1, m_X=m_X,
                    k_d=k_d, k_X=k_X,
                    T_d=T_d,
                    xi_d=xi_d, xi_X=xi_X,
                    vert=vert_el,
                    type=0
                ) / 4.

                if not off_shell:
                    CX_X_1nu = C_res_vector.C_n_3_12(m1=m_N1, m2=m_nu, m3=m_X, k1=k_d, k2=k_nu, k3=k_X, T1=T_d, T2=T_a, T3=T_d, xi1=xi_d, xi2=0., xi3=xi_X, M2=M2_X_1nu, type=0)
                else:
                    # Anton: NOT UPDATED 
                    print("off_shell=True is not implemented in minimal version.")
                    exit(1)

                return CX_X_1nu + 4*CX_XX_11

            # rho = rho_N1 + rho_N2 + rho_X
            def C_rho(T_a, T_d, xi_d, xi_X):
                """
                Anton: Internal processes of the DS does not contribute due to energy conservation. 
                C1_X->12 + C2_X->12 + CX_X->12 
                = int dPi * (2pi)^4 delta(E1+E2+E3) (E1 + E2 - E3)*fX*(1+k1*f1)*(1+k2*f2) = 0 

                Hence the only contributions come from energy transer between the SM and the DS

                Code: C_rho_3_12(type) = int dPI E_type |M|^2 * [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]

                Trick to save calculation: 
                C[1]_{3<->12} + C[3]_{3<->12}
                = int dPI (E3 - E1) |M|^2 [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]
                = int dPI E_2 |M|^2 [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]
                = C_rho_3_12(type=2, ...)
                """
                if T_a < m_X/50.:
                    return 0.
                if not off_shell:

                    C_X_1nu = C_res_vector.C_rho_3_12(type=2, m1=m_N1, m2=m_nu, m3=m_X, k1=k_d, k2=k_nu, k3=k_X, T1=T_d, T2=T_a, T3=T_d, xi1=xi_d, xi2=0., xi3=xi_X, M2=M2_X_1nu)

                else:
                    # Anton: NOT UPDATED
                    print("off_shell=True is not implemented in minimal version.")
                    exit(1)
                return C_X_1nu

            def C_xi0(T_a, T_d, xi_d, xi_X):
                # Anton: C_n with xi=0
                if T_a < m_X / 50.:
                    return 0.
                C_XX_dd = np.abs(C_res_vector.C_n_XX_dd(
                    m_d=m_N1, m_X=m_X,
                    k_d=k_d, k_X=k_X,
                    T_d=T_d,
                    xi_d=xi_d, xi_X=xi_X,
                    vert=vert_el,
                    th=th,
                    m_Gamma_h2=m_Gamma_h2,
                    type=1
                ) / 4.)

                C_dd_XX = np.abs(C_res_vector.C_n_XX_dd(
                    m_d=m_N1, m_X=m_X,
                    k_d=k_d, k_X=k_X,
                    T_d=T_d,
                    xi_d=xi_d, xi_X=xi_X,
                    vert=vert_el,
                    th=th,
                    m_Gamma_h2=m_Gamma_h2,
                    type=-1
                ) / 4.)

                # Anton: Decay-rates will always be larger than 2-to-2
                return min(4.*C_XX_dd, 4.*C_dd_XX)

            def C_therm(T_d, xi_d, xi_X):

                C_dd_X = C_res_vector.C_n_3_12(m1=m_N1, m2=m_N2, m3=m_X, k1=k_d, k2=k_d, k3=k_X, T1=T_d, T2=T_d, T3=T_d, xi1=xi_d, xi2=xi_d, xi3=xi_X, M2=M2_X_12, type=1) / 2.

                return 4.*(C_dd_X)

            def C_therm_kd(T_d, xi_d, xi_X):
                if T_d > m_X:
                    C_X_12 = C_res_vector.C_n_3_12(m1=m_N1, m2=m_N2, m3=m_X, k1=k_d, k2=k_d, k3=k_X, T1=T_d, T2=T_d, T3=T_d, xi1=xi_d, xi2=xi_d, xi3=xi_X, M2=M2_X_12, type=1)

                    return 2*C_X_12

                elif m_d / T_d - xi_d < 4.:
                    C_12_12 = C_res_vector.C_34_12(type=0, nFW=1., nBW=0., m1=m_N1, m2=m_N2, m3=m_N1, m4=m_N2, k1=k_d, k2=k_d, k3=k_d, k4=k_d, T1=T_d, T2=T_d, T3=T_d, T4=T_d, xi1=xi_d, xi2=xi_d, xi3=xi_d, xi4=xi_d, vert=vert_el, m_d2=m_N12, m_X2=m_X2, m_h2=m_h2, m_Gamma_X2=m_Gamma_X2, m_Gamma_h2=m_Gamma_h2, res_sub=False, thermal_width=True)

                    return 2*C_12_12

                C_dd_X_dd_gon_gel = C_res_vector_no_spin_stat.C_dd_dd_gon_gel(m_d=m_d, k_d=k_d, T_d=T_d, xi_d=xi_d, vert_el=vert_el, m_X2=m_X2, m_h2=m_h2, m_Gamma_X2=m_Gamma_X2, m_Gamma_h2=m_Gamma_h2, res_sub=False) / 4.

                # Anton: Lacks the cross-term 
                # return 2.*(C_dd_X_dd_gon_gel + C_dd_h_dd_gon_gel)
                return 2.*(C_dd_X_dd_gon_gel)
        else:
            # Anton: THIS IS NOT UPDATEED/IMPLEMENTED
            print("This mass relation is not implemented in minimal version.")
            exit(1)
    else:
        # Anton: THIS IS NEVER CALLED AS LONG AS spin_fac=True
        print("spin_fac=false is not implemented in minimal version.")
        exit(1)


    # Anton: Calculate SM neutrino temperature
    Ttrel = pandemolator.TimeTempRelation()
    ent_grid = np.array([cf.s_SM_no_nu(T)+cf.s_nu(T_nu) for T, T_nu in zip(Ttrel.T_SM_grid, Ttrel.T_nu_grid)])
    sf_norm_today = (cf.s0/ent_grid)**(1./3.)
    T_d_dw = cf.T_d_dw(m_d) # temperature of maximal d production by Dodelson-Widrow mechanism
    i_ic = np.argmax(Ttrel.T_nu_grid < T_d_dw)      # Anton: Start when T_nu < T_dw
    i_end = np.argmax(Ttrel.T_nu_grid < m_d/2e1)    # Anton: End when T_nu < m_d/20 <--> 20 < m_d/T_nu
    sf_ic_norm_0 = (cf.s0/(cf.s_SM_no_nu(Ttrel.T_SM_grid[i_ic]) + cf.s_nu(Ttrel.T_nu_grid[i_ic])))**(1./3.)
    n_ic = cf.n_0_dw(m_d, th) / (sf_ic_norm_0**3.)
    rho_ic = n_ic * cf.avg_mom_0_dw(m_d) / sf_ic_norm_0

    print(f"T_d_dw = {T_d_dw:.3e}")
    print(f"i_ic = {i_ic}")
    print(f"i_end = {i_end}")
    print(f"n_ic = {n_ic:.3e}")
    print(f"rho_ic = {rho_ic:.3e}")

    # Anton: Run main computation
    pan = pandemolator.Pandemolator(
        m_N1, m_N2, m_X, m_h, m_nu,
        k_d, k_X, k_nu,
        dof_d, dof_X,
        C_n, C_rho, C_xi0,
        Ttrel.t_grid,
        Ttrel.T_nu_grid,
        Ttrel.dTnu_dt_grid,
        ent_grid,
        Ttrel.hubble_grid,
        Ttrel.sf_grid,
        i_ic, n_ic, rho_ic, i_end
    )
    return pan


def compare_results(pan):
    x_pan = pan.m_N1 / pan.T_grid
    print(f"x_0 = {x_pan[0]:.3e}, x_end = {x_pan[-1]:.3e}")
    print(f"T_0 = {pan.T_grid[0]:.3e}, T_end = {pan.T_grid[-1]:.3e}")

    x = np.geomspace(x_pan[0], x_pan[-1], 1000)
    T_nu = pan.m_N1 / x
    dT_dt = pan.dT_dt_interp_T(T_nu)
    ent = pan.ent_interp_T(T_nu)
    hubble = pan.H_interp_T(T_nu)

    pan_jul = np.genfromtxt("../julia/pandemic/tmp/test_pandemolator.csv", delimiter=',', skip_header=1)

    x_jul = pan_jul[:, 0]
    T_nu_jul = pan_jul[:, 1]
    dT_dt_jul = pan_jul[:, 2]
    ent_jul = pan_jul[:, 3]
    hubble_jul = pan_jul[:, 4]

    plot_results(x, T_nu, x_jul, T_nu_jul, "T_nu")
    plot_results(x, dT_dt, x_jul, dT_dt_jul, "dT_dt", sign=True)
    plot_results(x, ent, x_jul, ent_jul, "ent")
    plot_results(x, hubble, x_jul, hubble_jul, "hubble")
    return 0

def plot_results(x, u, x_jul, u_jul, str, sign=False):
    if sign:
        u = -u
        u_jul = -u_jul
    fig, ax = plt.subplots()
    ax.plot(x, u)
    ax.plot(x_jul, u_jul, ':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f"figures/test_pand_{str}.pdf")


if __name__ == '__main__':
    # Anton: Ignore these masses 
    m_h = 0
    m_nu = 0

    # Compare with julia
    m_d = 1e-5
    m_N1 = m_d
    m_N2 = m_d
    m_X = 2.5*m_d
    y = 1e-4
    sin2_2th = 5e-16

    m12 = m_d
    # Anton: Need m0 >> mi, m12 >> mi^2 / m0, i = a,1,2,(12), m1 = 0
    # y = 3e-3
    m0 = 1e3
    m2 = m0*10**(-15./2)
    ma = m12

    # sin2_2th = (m2*ma/(2*m0*m12))**2

    print('mi/m0 << 1 :', f'{ma/m0:.3e}, {m2/m0:.3e}, {m12/m0:.3e}')
    print('mi^2/m0 << m12 :', f'{ma**2/m12:.3e}, {m2**2/m12:.3e}')

    # Anton: fermion = 1, boson = -1 (I did not choose this convention..)
    k_d = 1.
    k_X = -1.
    k_nu = 1.

    # Anton: Fermion has 2 dof, massive vector has 3 dof 
    dof_d = 2.
    dof_X = 3.

    # Anton: If spin-statistics (1+k*f) matters, if the mediator is off-shell or not 
    # HM: spin_facs=True and off_shell=False are the only implemented options.
    spin_facs = True
    off_shell = False

    print('Start sterile_caller')
    start = time.time()
    pan = call(
        m_N1, m_N2, m_X, m_nu, m0, m12, m2, ma,
        k_d, k_X, k_nu,
        dof_d, dof_X,
        y,
        sin2_2th,
        spin_facs=True,
        off_shell=False
    )
    end = time.time()
    print(f'pandemolator test ran in {end-start:.5f}s')

    compare_results(pan)
