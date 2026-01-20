#! /usr/bin/env python3

import numpy as np
from math import cos, sin, asin, sqrt, exp, log
from scipy.integrate import solve_ivp
import time

import constants_functions as cf
import utils
import densities as dens

import scalar_mediator
import vector_mediator
import pandemolator as pandemolator

import matplotlib.pyplot as plt 

GF = 1.166378e-5
mZ = 91.1876
mW = 80.379

def call(m_N1, m_N2, m_X, m0, m12, m2, ma, k_d, k_X, k_nu, dof_d, dof_X, y, spin_facs=True, off_shell=False):
    m_N12 = m_N1*m_N1
    m_N22 = m_N2*m_N2
    m_nu = 0 
    m_X2 = m_X*m_X
    m_h = 0
    m_h2 = m_h*m_h
    C_10 = m2/m0
    # Anton: C_1nu = sin(2*theta)
    C_1nu = C_10 * ma/m12
    y2 = y*y

    print(y**4, (y*C_1nu)**2)
    
    M2_X_12 = 2.*y2 * (m_X+m_N1-m_N2)*(m_X-m_N1+m_N2)*(2*m_X2 + (m_N1+m_N2)**2)/m_X2
    # Anton: M2_X_10 IS NEVER USED 
    M2_X_10 = 2.*y2*C_10**2 * (m_X+m_N1-m0)*(m_X-m_N1+m0)*(2*m_X2 + (m_N1+m0)**2)/m_X2
    M2_X_1nu = 2.*y2*C_1nu**2 * (m_X2-m_N12)*(2*m_X2 + m_N12)/m_X2

    vert_fi = y2*y2*(C_1nu**4.)
    vert_tr = y2*y2*(C_1nu**2.)
    vert_el = y2*y2

    # Anton: X --> 12, 1nu
    Gamma_X = vector_mediator.Gamma_X_new(y=y, m_X=m_X, m_N1=m_N1, m_N2=m_N2, m0=m0, m12=m12, m2=m2, ma=ma)
    m_Gamma_X2 = m_X2*Gamma_X*Gamma_X
    # Anton: THERE IS NO h -> (..)
    # Gamma_h = scalar_mediator.Gamma_phi(y=y, th=th, m_phi=m_h, m_d=m_d, m_X=m_X)
    # m_Gamma_h2 = m_h2*Gamma_h*Gamma_h
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
                Anton: A lot of processes do not contriubute due to equilibrium or no change
                in particle number. 

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
                CX_XX_11 = C_res_vector.C_n_XX_dd(m_d=m_N1, m_X=m_X, m_h=m_h, k_d=k_d, k_X=k_X, T_d=T_d, xi_d=xi_d, xi_X=xi_X, vert=vert_el, th=0, m_Gamma_h2=0, type=0) / 4.

                if not off_shell:
                    CX_X_1nu = C_res_vector.C_n_3_12(m1=m_N1, m2=m_nu, m3=m_X, k1=k_d, k2=k_nu, k3=k_X, T1=T_d, T2=T_a, T3=T_d, xi1=xi_d, xi2=0., xi3=xi_X, M2=M2_X_1nu, type=0)
                else:
                    # Anton: NOT UPDATED 
                    C_da = 0.
                    C_aa = 0.
                    C_da_dd = C_res_vector.C_34_12(type=0, nFW=1., nBW=-1., m1=m_d, m2=m_d, m3=m_d, m4=m_nu, k1=k_d, k2=k_d, k3=k_d, k4=k_nu, T1=T_d, T2=T_d, T3=T_d, T4=T_a, xi1=xi_d, xi2=xi_d, xi3=xi_d, xi4=0., vert=vert_tr, m_d2=m_d**2, m_X2=m_X2, m_h2=m_h2, m_Gamma_X2=m_Gamma_X2, m_Gamma_h2=m_Gamma_h2, res_sub=False, thermal_width=True) / 2.
                    
                    C_aa_dd = C_res_vector.C_34_12(type=0, nFW=2., nBW=-2., m1=m_d, m2=m_d, m3=m_nu, m4=m_nu, k1=k_d, k2=k_d, k3=k_nu, k4=k_nu, T1=T_d, T2=T_d, T3=T_a, T4=T_a, xi1=xi_d, xi2=xi_d, xi3=0., xi4=0., vert=vert_fi, m_d2=m_d**2, m_X2=m_X2, m_h2=m_h2, m_Gamma_X2=m_Gamma_X2, m_Gamma_h2=m_Gamma_h2, res_sub=False, thermal_width=True) / 4.

                x = m_d / T_a
                if call.count % 10 == 0:
                    call.x_list.append(x)
                    # call.C_list.append([CX_XX_dd, Ch_hh_dd, Ch_h_XX])
                    call.C_list.append([CX_X_1nu, CX_XX_11])
                    i = call.count // 10
                    if i > 0: 
                        plt.loglog([call.x_list[i-1], call.x_list[i]], [abs(call.C_list[i-1][0]), abs(call.C_list[i][0])], color='r', marker='o', markersize=3)
                        plt.loglog([call.x_list[i-1], call.x_list[i]], [abs(call.C_list[i-1][1]), abs(call.C_list[i][1])], color='tab:blue', marker='o', markersize=3)
                        # plt.loglog([call.x_list[i-1], call.x_list[i]], [abs(call.C_list[i-1][2]), abs(call.C_list[i][2])], color='tab:green', marker='o', markersize=3)
                        plt.pause(0.05)
                call.count += 1

                print("C_ns:", f'{CX_XX_11:.5e}', f'{CX_X_1nu:.5e}')
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
                    C_da = 0.
                    C_aa = 0.
                    # C_da_X_dd = C_res_vector.C_34_12(type=4, nFW=1., nBW=-1., m1=m_d, m2=m_d, m3=m_d, m4=m_nu, k1=k_d, k2=k_d, k3=k_d, k4=k_nu, T1=T_d, T2=T_d, T3=T_d, T4=T_a, xi1=xi_d, xi2=xi_d, xi3=xi_d, xi4=0., vert=vert_tr, m_d2=m_d2, m_X2=m_X2, m_h2=m_h2, m_Gamma_X2=m_Gamma_X2, m_Gamma_h2=m_Gamma_h2, res_sub=False, thermal_width=True) / 2.
                    # C_da_h_dd = C_res_scalar.C_34_12(type=4, nFW=1, nBW=-1, m1=m_d, m2=m_d, m3=m_d, m4=m_nu, k1=k_d, k2=k_d, k3=k_d, k4=k_nu, T1=T_d, T2=T_d, T3=T_d, T4=T_a, xi1=xi_d, xi2=xi_d, xi3=xi_d, xi4=xi_d, vert=vert_tr*(4*m_d2/m_X2)**2, m_phi2=m_h2, m_Gamma_phi2=m_Gamma_h2, res_sub=False, thermal_width=True) / 2.
                    # C_da_dd = C_da_X_dd + C_da_h_dd
                    C_aa_dd = 0. #C_res_vector.C_34_12(12, 1., -1., m_d, m_d, m_nu, m_nu, k_d, k_d, k_nu, k_nu, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi, m_X2, m_Gamma_X2, res_sub=False, thermal_width=True) / 4.
                # print("C_rhos:", f'{C_X_da:.5e}', f'{C_h_da:.5e}', f'{2.*C_X_aa:.5e}', f'{C_h_aa:.5e}')
                return C_X_1nu
            
            def C_xi0(T_a, T_d, xi_d, xi_X):
                # Anton: C_n with xi=0
                if T_a < m_X / 50.:
                    return 0.
                C_XX_dd = np.abs(C_res_vector.C_n_XX_dd(m_d=m_N1, m_X=m_X, m_h=m_h, k_d=k_d, k_X=k_X, T_d=T_d, xi_d=xi_d, xi_X=xi_X, vert=vert_el, th=th, m_Gamma_h2=m_Gamma_h2, type=1) / 4.)

                C_dd_XX = np.abs(C_res_vector.C_n_XX_dd(m_d=m_N1, m_X=m_X, m_h=m_h, k_d=k_d, k_X=k_X, T_d=T_d, xi_d=xi_d, xi_X=xi_X, vert=vert_el, th=th, m_Gamma_h2=m_Gamma_h2, type=-1) / 4.)                
                
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
                # C_dd_h_dd_gon_gel = C_res_scalar_no_spin_stat.C_dd_dd_gon_gel(m_d=m_d, k_d=k_d, T_d=T_d, xi_d=xi_d, vert_el=vert_el*(4*m_d2/m_X2)**2, m_phi2=m_h2, m_Gamma_phi2=m_Gamma_h2, res_sub=False) / 4.

                # Anton: Lacks the cross-term 
                # return 2.*(C_dd_X_dd_gon_gel + C_dd_h_dd_gon_gel)
                return 4.*(C_dd_X_dd_gon_gel)
        else:
            # Anton: THIS IS NOT UPDATEED/IMPLEMENTED
            print("Implementation needs to be updated...")
            exit(1)
            import C_res_vector_no_spin_stat as C_res_vector_no_spin_stat
            # n = n_d + n_X
            def C_n(T_a, T_d, xi_d, xi_X):
                C_da = 0.# vanishes since net number change is zero
                C_aa = 0.#C_res_vector.C_n_3_12(m_nu, m_nu, m_X, k_nu, k_nu, k_X, T_a, T_a, T_d,   0.,   0., xi_X, M2_aa) / 2.
                C_da_dd = C_res_vector.C_34_12(0, 1., -1., m_d, m_d, m_d, m_nu, k_d, k_d, k_d, k_nu, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, res_sub=False) / 2.
                C_aa_da = 0.#C_res_vector.C_34_12(0, 1., -1., m_d, m_nu, m_nu, m_nu, k_d, k_nu, k_nu, k_nu, T_d, T_a, T_a, T_a, xi_d, 0., 0., 0., vert_fi_1, m_X2, m_Gamma_X2, res_sub=False) / 2.
                C_aa_dd = 0.#C_res_vector.C_34_12(0, 2., -2., m_d, m_d, m_nu, m_nu, k_d, k_d, k_nu, k_nu, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi_2, m_X2, m_Gamma_X2, res_sub=False) / 4.
                return C_da + C_aa + C_da_dd + C_aa_da + C_aa_dd
            # rho = rho_d + rho_X
            def C_rho(T_a, T_d, xi_d, xi_X):
                C_da = C_res_vector.C_rho_3_12(2, m_d, m_nu, m_X, k_d, k_nu, k_X, T_d, T_a, T_d, xi_d, 0., xi_X, M2_X_da)
                C_aa = 0.#C_res_vector.C_rho_3_12(3, m_nu, m_nu, m_X, k_d, k_nu, k_X, T_a, T_a, T_d,   0., 0., xi_X, M2_aa) / 2. # symmetry factor 1/2
                C_da_dd = C_res_vector.C_34_12(4, 1., -1., m_d, m_d, m_d, m_nu, k_d, k_d, k_d, k_nu, T_d, T_d, T_d, T_a, xi_d, xi_d, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, res_sub=False) / 2.
                C_aa_da = 0.#C_res_vector.C_34_12(1, 1., -1., m_d, m_nu, m_nu, m_nu, k_d, k_nu, k_nu, k_nu, T_d, T_a, T_a, T_a, xi_d, 0., 0., 0., vert_fi_1, m_X2, m_Gamma_X2, res_sub=False) / 2.
                C_aa_dd = 0.#C_res_vector.C_34_12(12, 1., -1., m_d, m_d, m_nu, m_nu, k_d, k_d, k_nu, k_nu, T_d, T_d, T_a, T_a, xi_d, xi_d, 0., 0., vert_fi_2, m_X2, m_Gamma_X2, res_sub=False) / 4.
                return C_da + C_aa + C_da_dd + C_aa_da + C_aa_dd
            def C_xi0(T_a, T_d, xi_d, xi_X):
                return 0.
            def C_therm(T_d, xi_d, xi_X): # use collision operators without spin-stat. factors, only proxy here
                return 0.
                C_dd_XX = C_res_vector_no_spin_stat.C_12_34(m_d, m_d, m_d, m_d, k_d, k_d, T_d, T_d, xi_d, xi_d, vert_el, m_X2, m_Gamma_X2, type=0, res_sub=False) / 4.
                C_pp_dd = C_res_vector_no_spin_stat.C_pp_dd(m_d, m_X, k_X, T_d, xi_X, vert_el, type=0) / 4.
                return min(2.*C_dd_dd, 2.*C_pp_dd)
            def C_therm_kd(T_d, xi_d, xi_X):
                # C_dd_dd = C_res_vector_no_spin_stat.C_12_34(m_d, m_d, m_d, m_d, k_d, k_d, T_d, T_d, xi_d, xi_d, vert_el, m_X2, m_Gamma_X2, type=0, res_sub=False) / 4.
                C_dd_dd = C_res_vector_no_spin_stat.C_dd_dd_gon_gel(m_d, k_d, T_d, xi_d, vert_el, m_X2, m_Gamma_X2, res_sub=False) / 4.
                return 2.*C_dd_dd
        # def G_d(T_d, xi_d, xi_X):
        #     return C_res_vector.Gamma_scat(T_d, m_d, m_d, m_X, k_d, k_X, T_d, T_d, xi_d, xi_X, M2_dd)
    else:
        # Anton: THIS IS NEVER CALLED AS LONG AS spin_fac=True
        import C_res_vector_no_spin_stat as C_res_vector
        if m_X > 2.*m_d:
            # n = n_d + 2.*n_X
            def C_n(T_a, T_d, xi_d, xi_X):
                if T_a < m_X / 50.:
                    return 0.
                C_XX_dd = (-C_res_vector.C_XX_dd(m_d, m_X, k_X, T_d, xi_X, vert_el, type=0) + C_res_vector.C_dd_XX(m_d, m_X, k_d, T_d, xi_d, vert_el, type=0)) / 4. # symmetry factor 1/4
                if not off_shell:
                    C_da = C_res_vector.C_12_3(m_d, m_nu, m_X, k_d, k_nu, T_d, T_a, xi_d, 0., M2_X_da, type=0)
                    C_da_dd = 0.
                else:
                    C_da = 0.
                    C_da_dd = C_res_vector.C_12_34(m_d, m_nu, m_d, m_d, k_d, k_nu, T_d, T_a, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, type=0, res_sub=False) / 2.
                return C_da + C_da_dd + 2.*C_XX_dd
            # rho = rho_d + rho_X
            def C_rho(T_a, T_d, xi_d, xi_X):
                if T_a < m_X / 50.:
                    return 0.
                if not off_shell:
                    C_da = C_res_vector.C_12_3(m_d, m_nu, m_X, k_d, k_nu, T_d, T_a, xi_d, 0., M2_X_da, type=1)
                    C_da_dd = 0.
                else:
                    C_da = 0.
                    C_da_dd = C_res_vector.C_12_34(m_d, m_nu, m_d, m_d, k_d, k_nu, T_d, T_a, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, type=1, res_sub=False) / 2.
                return C_da + C_da_dd
            def C_xi0(T_a, T_d, xi_d, xi_X):
                if T_a < m_X / 50.:
                    return 0.
                C_XX_dd = C_res_vector.C_XX_dd(m_d, m_X, k_X, T_d, xi_X, vert_el, type=0) / 4.
                return 2.*C_XX_dd
            def C_therm(T_d, xi_d, xi_X):
                C_dd_X = C_res_vector.C_12_3(m_d, m_d, m_X, k_d, k_d, T_d, T_d, xi_d, xi_d, M2_X_dd, type = 0) / 2.
                return 2. * C_dd_X
            def C_therm_kd(T_d, xi_d, xi_X):
                C_dd_dd = C_res_vector.C_dd_dd_gon_gel(m_d, k_d, T_d, xi_d, vert_el, m_X2, m_Gamma_X2, res_sub=False) / 4.
                return 2.*C_dd_dd
        else:
            print("Implementation needs to be updated...")
            exit(1)
            # n = n_d + n_X
            def C_n(T_a, T_d, xi_d, xi_X):
                C_da = 0.# vanishes since net number change is zero
                C_da_dd = C_res_vector.C_12_34(m_d, m_nu, m_d, m_d, k_d, k_nu, T_d, T_a, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, type=0, res_sub=False) / 2.
                return C_da_dd
            # rho = rho_d + rho_X
            def C_rho(T_a, T_d, xi_d, xi_X):
                C_da = C_res_vector.C_12_3(m_d, m_nu, m_X, k_d, k_nu, T_d, T_a, xi_d, 0., M2_X_da, type=1)
                C_da_dd = C_res_vector.C_12_34(m_d, m_nu, m_d, m_d, k_d, k_nu, T_d, T_a, xi_d, 0., vert_tr, m_X2, m_Gamma_X2, type=1, res_sub=False) / 2.
                return C_da + C_da_dd
            def C_xi0(T_a, T_d, xi_d, xi_X):
                return 0.
            def C_therm(T_d, xi_d, xi_X):
                return 0.
                # return C_res_vector.C_12_3(m_d, m_nu, m_X, k_d, k_nu, T_d, T_a, xi_d, 0., M2_da, type=0)
                C_dd_dd = C_res_vector.C_12_34(m_d, m_d, m_d, m_d, k_d, k_d, T_d, T_d, xi_d, xi_d, vert_el, m_X2, m_Gamma_X2, type=0, res_sub=False) / 4.
                C_pp_dd = C_res_vector.C_pp_dd(m_d, m_X, k_X, T_d, xi_X, vert_el, type=0) / 4.
                return [2.*C_dd_dd, 2.*C_pp_dd]
            def C_therm_kd(T_d, xi_d, xi_X):
                C_dd_dd = C_res_vector.C_dd_dd_gon_gel(m_d, k_d, T_d, xi_d, vert_el, m_X2, m_Gamma_X2, res_sub=False) / 4.
                return 2.*C_dd_dd
        # def G_d(T_d, xi_d, xi_X):
        #     return C_res_vector.Gamma_scat(T_d, m_d, m_d, m_X, k_d, T_d, xi_X, M2_dd)

    # Anton: Calculate SM neutrino temperature
    Ttrel = pandemolator.TimeTempRelation()
    ent_grid = np.array([cf.s_SM_no_nu(T)+cf.s_nu(T_nu) for T, T_nu in zip(Ttrel.T_SM_grid, Ttrel.T_nu_grid)])
    sf_norm_today = (cf.s0/ent_grid)**(1./3.)
    T_d_dw = cf.T_d_dw(m_d) # temperature of maximal d production by Dodelson-Widrow mechanism
    i_ic = np.argmax(Ttrel.T_nu_grid < T_d_dw)      # Anton: Start when T_nu < T_dw
    i_end = np.argmax(Ttrel.T_nu_grid < m_d/2e1)    # Anton: End when T_nu < 20/m_d <--> 20 < m_d/T_nu
    sf_ic_norm_0 = (cf.s0/(cf.s_SM_no_nu(Ttrel.T_SM_grid[i_ic]) + cf.s_nu(Ttrel.T_nu_grid[i_ic])))**(1./3.)
    n_ic = 2 * cf.n_0_dw(m_d, th) / (sf_ic_norm_0**3.)      # 2* due to 2 species of neutrinos
    rho_ic = n_ic * cf.avg_mom_0_dw(m_d) / sf_ic_norm_0

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
        i_ic, n_ic, rho_ic, i_end)

    # time1 = time.time()
    # print("Running Pandemolator.pandemolate ")
    pan.pandemolate()
    # print(f"Pandemolator.pandemolate ran in {time.time() - time1}s ")

    try:
        C_therm_grid = np.array([C_therm(T_d, xi_d, xi_X) for T_d, xi_d, xi_X in zip(pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_X_grid_sol)])
    except:
        C_therm_grid = np.zeros(pan.T_chi_grid_sol.size)

    O_d_h2 = pan.n_chi_grid_sol[-1]*m_d*cf.s0/(ent_grid[pan.i_end]*cf.rho_crit0_h2)

    if pan.T_chi_grid_sol.size < i_end - i_ic + 1: # integration of ode was stopped (abundance too large); issues calculating fs_length
        return Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], ent_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_X_grid_sol, pan.n_chi_grid_sol, pan.n_X_grid_sol, C_therm_grid, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False
    elif O_d_h2 < 1e-2*cf.omega_d0 or O_d_h2 > 1e1*cf.omega_d0: # computation of lambda_fs does not work for very small O_d_h2
        return Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], ent_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_X_grid_sol, pan.n_chi_grid_sol, pan.n_X_grid_sol, C_therm_grid, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, True

    try:
        T_d_grid = np.empty(Ttrel.t_grid.size - i_ic)
        xi_d_grid = np.empty(Ttrel.t_grid.size - i_ic)
        xi_X_grid = np.empty(Ttrel.t_grid.size - i_ic)
        n_d_grid = np.empty(Ttrel.t_grid.size - i_ic)
        n_X_grid = np.empty(Ttrel.t_grid.size - i_ic)
        P_grid = np.empty(Ttrel.t_grid.size - i_ic)
        rho_grid = np.empty(Ttrel.t_grid.size - i_ic)

        n_sol = pan.T_chi_grid_sol.size
        T_d_grid[:n_sol] = pan.T_chi_grid_sol
        xi_d_grid[:n_sol] = pan.xi_chi_grid_sol
        xi_X_grid[:n_sol] = pan.xi_X_grid_sol
        n_d_grid[:n_sol] = pan.n_chi_grid_sol
        n_X_grid[:n_sol] = pan.n_X_grid_sol
        P_grid[:n_sol] = np.array([pan.P(T_d, xi_d, xi_X) for T_d, xi_d, xi_X in zip(T_d_grid[:n_sol], xi_d_grid[:n_sol], xi_X_grid[:n_sol])])
        rho_grid[:n_sol] = np.array([pan.rho(T_d, xi_d, xi_X) for T_d, xi_d, xi_X in zip(T_d_grid[:n_sol], xi_d_grid[:n_sol], xi_X_grid[:n_sol])])

        if n_sol < T_d_grid.size:
            print('n_sol > T_d_grid.size')
            H_interp = utils.LogInterp(Ttrel.t_grid, Ttrel.hubble_grid)
            sf_grid_tmp = (ent_grid[i_ic + n_sol - 1]/ent_grid)**(1./3.)
            sf_interp = utils.LogInterp(Ttrel.t_grid, sf_grid_tmp)
            def der(log_t, y):
                t = exp(log_t)
                H = H_interp(t) if t < Ttrel.t_grid[-1] else Ttrel.hubble_grid[-1]
                sf = sf_interp(t) if t < Ttrel.t_grid[-1] else sf_grid_tmp[-1]
                T = y[0]/(sf*sf)
                xi_diff = y[1]
                x = T/m_d
                x2 = x*x
                x3 = x2*x
                x4 = x3*x
                x5 = x4*x
                x6 = x5*x

                T_der = t*sf*sf*H*T*(x*5.-x2*20.+x3*595./8.-x4*2125/8.+x5*117475./128.-x6*49025./16.)
                xi_diff_der = t*H*(-x*15./4.+x2*135./8.-x3*4005./64.+x4*28035./128.-x5*378675./512.+x6*2463075./1024.)

                return [T_der, xi_diff_der]
            sol = solve_ivp(der, [log(Ttrel.t_grid[n_sol+i_ic-1]), log(Ttrel.t_grid[-1])], [T_d_grid[n_sol-1], xi_d_grid[n_sol-1]-m_d/T_d_grid[n_sol-1]], t_eval=np.log(Ttrel.t_grid[n_sol+i_ic:]), rtol=1e-6, atol=0., method='RK45', max_step=1.)
            T_d_grid[n_sol:] = sol.y[0, :] / (sf_grid_tmp[n_sol+i_ic:]**2.)
            xi_d_grid[n_sol:] = sol.y[1, :] + m_d / T_d_grid[n_sol:]
            xi_X_grid[n_sol:] = 2.*xi_d_grid[n_sol:]
            n_d_grid[n_sol:] = n_d_grid[n_sol - 1] * ent_grid[i_ic + n_sol:] / ent_grid[i_ic + n_sol - 1]
            n_X_grid[n_sol:] = 0.
            P_grid[n_sol:] = T_d_grid[n_sol:] * n_d_grid[n_sol:]
            rho_grid[n_sol:] = (m_d + 1.5*T_d_grid[n_sol:]) * n_d_grid[n_sol:]

        dPdt_grid = utils.fpsder(Ttrel.t_grid[i_ic:], P_grid)
        drhodt_grid = utils.fpsder(Ttrel.t_grid[i_ic:], rho_grid)
        mask = (dPdt_grid < 0.) * (drhodt_grid < 0.) * (dPdt_grid / drhodt_grid < 0.3)
        c_sound_grid = np.zeros(T_d_grid.size)
        c_sound_grid[mask] = np.sqrt(dPdt_grid[mask] / drhodt_grid[mask])
        # c_sound_all = np.zeros(T_d_grid.size)
        # c_sound_all[dPdt_grid/drhodt_grid > 0.] = np.sqrt(dPdt_grid[dPdt_grid/drhodt_grid > 0.] / drhodt_grid[dPdt_grid/drhodt_grid > 0.])
        # c_sound_no_pan = np.zeros(T_d_grid.size)
        # c_sound_no_pan[(dPdt_grid < 0.) * (drhodt_grid < 0.) * (dPdt_grid / drhodt_grid < 0.34)] = np.sqrt(dPdt_grid[(dPdt_grid < 0.) * (drhodt_grid < 0.) * (dPdt_grid / drhodt_grid < 0.34)] / drhodt_grid[(dPdt_grid < 0.) * (drhodt_grid < 0.) * (dPdt_grid / drhodt_grid < 0.34)])
        # # dPdrho = np.diff(P_grid)/np.diff(rho_grid)
        # import matplotlib.pyplot as plt
        # plt.loglog(m_d / Ttrel.T_nu_grid[i_ic:], c_sound_all)
        # plt.loglog(m_d / Ttrel.T_nu_grid[i_ic:], c_sound_no_pan)
        # plt.loglog(m_d / Ttrel.T_nu_grid[i_ic:], c_sound_grid)
        # plt.xlabel('1e-16')
        # plt.show()

        i_kd = np.argmax(Ttrel.T_nu_grid[i_ic:] < 0.1*m_X)
        found_kd_coarse = False
        i_kd_start = i_kd
        C_therm_kd_last = C_therm_kd(T_d_grid[i_kd], xi_d_grid[i_kd], xi_X_grid[i_kd])
        T_d_last = T_d_grid[i_kd]
        while i_kd < Ttrel.t_grid.size - i_ic - 1:
            try:
                C_therm_kd_cur = C_therm_kd(T_d_grid[i_kd], xi_d_grid[i_kd], xi_X_grid[i_kd])
            except:
                C_therm_kd_cur = 0.
            if C_therm_kd_cur <= 0.:
                C_therm_kd_cur = C_therm_kd_last * ((T_d_grid[i_kd]/T_d_last)**3.5)
            C_therm_kd_last = C_therm_kd_cur
            T_d_last = T_d_grid[i_kd]
            ratio_kd = C_therm_kd_cur / (3.*Ttrel.hubble_grid[i_ic+i_kd]*n_d_grid[i_kd])
            # print('kd', m_d/T_d_grid[i_kd], m_d/T_d_grid[i_kd] - xi_d_grid[i_kd], ratio_kd)
            if ratio_kd < 1. and found_kd_coarse:
                break
            elif ratio_kd < 1.:
                found_kd_coarse = True
                i_kd -= 99
            elif found_kd_coarse:
                i_kd += 1
            else:
                i_kd += 100
        n_d_kd = n_d_grid[i_kd]
        T_d_kd = T_d_grid[i_kd]
        xi_d_kd = xi_d_grid[i_kd]
        T_kd_3 = Ttrel.T_SM_grid[i_ic+i_kd]
        T_d_kd_3 = T_d_kd
        r_sound_3 = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_grid[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # r_sound_3_all = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_all[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # r_sound_3_no_pan = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_no_pan[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # print(T_kd_3, T_d_kd_3, r_sound_3/cf.Mpc, r_sound_3_all/cf.Mpc, r_sound_3_no_pan/cf.Mpc)

        if m_d/T_d_kd > 1e2:
            # only valid if already in non-/semi-relativistic regime
            lo_v_kd = (dof_d*exp(xi_d_kd-m_d/T_d_kd)*(T_d_kd**2.)/(m_d*(np.pi**2.)*n_d_kd)) * \
             ((m_d**2.)+3.*m_d*T_d_kd+3.*(T_d_kd**2.))
            nlo_v_kd = -(dof_d*exp(xi_d_kd-m_d/T_d_kd)*(T_d_kd**3.)*2./((m_d**3.)*(np.pi**2.)*n_d_kd)) * \
             ((m_d**3.)+6.*(m_d**2.)*T_d_kd+15.*m_d*(T_d_kd**2.)+15.*(T_d_kd**3.))
            nnlo_v_kd = (dof_d*exp(xi_d_kd-m_d/T_d_kd)*(T_d_kd**4.)*9./((m_d**5.)*(np.pi**2.)*n_d_kd)) * \
             ((m_d**4.)+10.*(m_d**3.)*T_d_kd+45.*(m_d**2.)*(T_d_kd**2.)+105.*m_d*(T_d_kd**3.)+105.*(T_d_kd**4.))

            sf_kd_norm_today = (cf.s0/ent_grid[pan.i_ic+i_kd])**(1./3.)
            i_fs_max = np.argmax(Ttrel.T_SM_grid/cf.T0 - 1. < 50.) # only assume free-streaming until z = 50
            sf_norm_kd = (ent_grid[pan.i_ic+i_kd]/ent_grid[pan.i_ic+i_kd:i_fs_max])**(1./3.)#Ttrel.sf_grid[i_ic+i_kd:]/Ttrel.sf_grid[i_ic+i_kd]
            integrand_fs_length = (lo_v_kd/(sf_norm_kd**2.) + nlo_v_kd/(sf_norm_kd**4.) + nnlo_v_kd/(sf_norm_kd**6.))/sf_kd_norm_today
            fs_length_3 = utils.simp(Ttrel.t_grid[pan.i_ic+i_kd:i_fs_max], integrand_fs_length)
        else:
            # always valid
            i_fs_max = np.argmax(Ttrel.T_SM_grid/cf.T0 - 1. < 50.) # only assume free-streaming until z = 50
            p_star_grid = np.logspace(np.log10(1e-6*max(T_d_kd, sqrt(2.*m_d*T_d_kd))), np.log10(1e3*max(T_d_kd, sqrt(2.*m_d*T_d_kd))), 5000)
            f_d_grid = 1./(np.exp(np.sqrt(m_d*m_d+p_star_grid*p_star_grid)/T_d_kd - xi_d_kd) + k_d)
            integrand_fs_length = np.zeros(Ttrel.t_grid.size)
            sf_norm_kd = (ent_grid[pan.i_ic+i_kd]/ent_grid)**(1./3.)#Ttrel.sf_grid[i_ic+i_kd:]/Ttrel.sf_grid[i_ic+i_kd]
            sf_norm_today = (cf.s0/ent_grid)**(1./3.)
            for i in range(i_kd+i_ic, i_fs_max):
                p_cur = p_star_grid/sf_norm_kd[i]
                E_cur = np.sqrt(m_d*m_d + p_cur*p_cur)
                v = utils.simp(p_star_grid, f_d_grid*(p_star_grid**3.)/E_cur)*2./(2.*cf.pi2*n_d_kd*sf_norm_kd[i])
                integrand_fs_length[i] = v/sf_norm_today[i]
            fs_length_3 = utils.simp(Ttrel.t_grid, integrand_fs_length)

        i_kd = i_kd - 1
        i_kd_start = i_kd
        try:
            C_therm_kd_last = C_therm_kd(T_d_grid[i_kd], xi_d_grid[i_kd], xi_X_grid[i_kd])
        except:
            C_therm_kd_last = 0.
        if C_therm_kd_last <= 0.:
            C_therm_kd_last = C_therm_kd_cur * ((T_d_grid[i_kd]/T_d_last)**3.5)
        T_d_last = T_d_grid[i_kd]
        while i_kd < Ttrel.t_grid.size - i_ic - 1:
            try:
                C_therm_kd_cur = C_therm_kd(T_d_grid[i_kd], xi_d_grid[i_kd], xi_X_grid[i_kd])
            except:
                C_therm_kd_cur = 0.
            if C_therm_kd_cur <= 0.:
                C_therm_kd_cur = C_therm_kd_last * ((T_d_grid[i_kd]/T_d_last)**3.5)
            C_therm_kd_last = C_therm_kd_cur
            T_d_last = T_d_grid[i_kd]
            ratio_kd = C_therm_kd_cur / (Ttrel.hubble_grid[i_ic+i_kd]*n_d_grid[i_kd])
            # print('kd', m_d/T_d_grid[i_kd], m_d/T_d_grid[i_kd] - xi_d_grid[i_kd], ratio_kd)
            if ratio_kd < 1.:
                break
            i_kd += 1
        n_d_kd = n_d_grid[i_kd]
        T_d_kd = T_d_grid[i_kd]
        xi_d_kd = xi_d_grid[i_kd]
        T_kd = Ttrel.T_SM_grid[i_ic+i_kd]
        T_d_kd = T_d_kd
        r_sound = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_grid[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # r_sound_all = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_all[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # r_sound_no_pan = utils.simp(Ttrel.t_grid[i_ic:i_ic+i_kd], c_sound_no_pan[:i_kd]/sf_norm_today[i_ic:i_ic+i_kd])
        # print(T_kd, T_d_kd, r_sound/cf.Mpc, r_sound_all/cf.Mpc, r_sound_no_pan/cf.Mpc)

        if m_d/T_d_kd > 1e2:
            # only valid if already in non-/semi-relativistic regime
            lo_v_kd = (dof_d*exp(xi_d_kd-m_d/T_d_kd)*(T_d_kd**2.)/(m_d*(np.pi**2.)*n_d_kd)) * \
             ((m_d**2.)+3.*m_d*T_d_kd+3.*(T_d_kd**2.))
            nlo_v_kd = -(dof_d*exp(xi_d_kd-m_d/T_d_kd)*(T_d_kd**3.)*2./((m_d**3.)*(np.pi**2.)*n_d_kd)) * \
             ((m_d**3.)+6.*(m_d**2.)*T_d_kd+15.*m_d*(T_d_kd**2.)+15.*(T_d_kd**3.))
            nnlo_v_kd = (dof_d*exp(xi_d_kd-m_d/T_d_kd)*(T_d_kd**4.)*9./((m_d**5.)*(np.pi**2.)*n_d_kd)) * \
             ((m_d**4.)+10.*(m_d**3.)*T_d_kd+45.*(m_d**2.)*(T_d_kd**2.)+105.*m_d*(T_d_kd**3.)+105.*(T_d_kd**4.))

            sf_kd_norm_today = (cf.s0/ent_grid[pan.i_ic+i_kd])**(1./3.)
            i_fs_max = np.argmax(Ttrel.T_SM_grid/cf.T0 - 1. < 50.) # only assume free-streaming until z = 50
            sf_norm_kd = (ent_grid[pan.i_ic+i_kd]/ent_grid[pan.i_ic+i_kd:i_fs_max])**(1./3.)#Ttrel.sf_grid[i_ic+i_kd:]/Ttrel.sf_grid[i_ic+i_kd]
            integrand_fs_length = (lo_v_kd/(sf_norm_kd**2.) + nlo_v_kd/(sf_norm_kd**4.) + nnlo_v_kd/(sf_norm_kd**6.))/sf_kd_norm_today
            fs_length = utils.simp(Ttrel.t_grid[pan.i_ic+i_kd:i_fs_max], integrand_fs_length)
        else:
            # always valid
            i_fs_max = np.argmax(Ttrel.T_SM_grid/cf.T0 - 1. < 50.) # only assume free-streaming until z = 50
            p_star_grid = np.logspace(np.log10(1e-6*max(T_d_kd, sqrt(2.*m_d*T_d_kd))), np.log10(1e3*max(T_d_kd, sqrt(2.*m_d*T_d_kd))), 5000)
            f_d_grid = 1./(np.exp(np.sqrt(m_d*m_d+p_star_grid*p_star_grid)/T_d_kd - xi_d_kd) + k_d)
            integrand_fs_length = np.zeros(Ttrel.t_grid.size)
            sf_norm_kd = (ent_grid[pan.i_ic+i_kd]/ent_grid)**(1./3.)#Ttrel.sf_grid[i_ic+i_kd:]/Ttrel.sf_grid[i_ic+i_kd]
            sf_norm_today = (cf.s0/ent_grid)**(1./3.)
            for i in range(i_kd+i_ic, i_fs_max):
                p_cur = p_star_grid/sf_norm_kd[i]
                E_cur = np.sqrt(m_d*m_d + p_cur*p_cur)
                v = utils.simp(p_star_grid, f_d_grid*(p_star_grid**3.)/E_cur)*2./(2.*cf.pi2*n_d_kd*sf_norm_kd[i])
                integrand_fs_length[i] = v/sf_norm_today[i]
            fs_length = utils.simp(Ttrel.t_grid, integrand_fs_length)

        return Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], ent_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_X_grid_sol, pan.n_chi_grid_sol, pan.n_X_grid_sol, C_therm_grid, fs_length/cf.Mpc, fs_length_3/cf.Mpc, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound/cf.Mpc, r_sound_3/cf.Mpc, True#, V_SM_grid, V_d_grid, G_a_grid, G_d_grid
    except:
        return Ttrel.t_grid[pan.i_ic:pan.i_end+1], Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1], ent_grid[pan.i_ic:pan.i_end+1], Ttrel.hubble_grid[pan.i_ic:pan.i_end+1], Ttrel.sf_grid[pan.i_ic:pan.i_end+1]/Ttrel.sf_grid[pan.i_ic], pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_X_grid_sol, pan.n_chi_grid_sol, pan.n_X_grid_sol, C_therm_grid, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, True

    # print(m_d/T_d_kd, lo_v_kd + nlo_v_kd +nnlo_v_kd)
    # fs_length_cum_int = utils.cumsimp(Ttrel.t_grid[pan.i_ic+i_kd:i_fs_max], integrand_fs_length[:i_fs_max])
    # print(fs_length_cum_int[-1])
    # utils.simp(Ttrel.t_grid[pan.i_ic+i_kd:], sf_kd_norm_today*nlo_v_kd/(sf_norm_kd**4.)), utils.simp(Ttrel.t_grid[pan.i_ic+i_kd:], sf_kd_norm_today*nnlo_v_kd/(sf_norm_kd**6.)))

    # np.savetxt('tmp.dat', np.column_stack((Ttrel.t_grid[i_ic:], Ttrel.T_SM_grid[i_ic:], Ttrel.T_nu_grid[i_ic:], Ttrel.sf_grid[i_ic:], ent_grid[i_ic:], T_d_grid, xi_d_grid, n_d_grid)))
    # np.savetxt('tmp_2.dat', np.array([i_kd, lo_v_kd, nlo_v_kd, nnlo_v_kd]))

    # plt.loglog(Ttrel.T_SM_grid[i_ic+i_kd:]/cf.T0, fs_length_cum_int/cf.Mpc)
    # plt.show()
    # plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:], T_d_grid*(Ttrel.sf_grid[i_ic:]**2.))
    # plt.show()
    # plt.semilogx(m_d/Ttrel.T_nu_grid[i_ic:], xi_d_grid - m_d/T_d_grid)
    # plt.show()
    # plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:], 3.*Ttrel.hubble_grid[i_ic:]*n_d_grid, color='dodgerblue')
    # plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:], C_therm_grid, color='#458751')
    # plt.show()
    # def V_SM(T_SM, T_nu):
    #     rho_nu = 2.*(7./8.)*(cf.pi2/30.)*(T_nu**4.)
    #     rho_a = cf.rho_fermion(T_SM, 0.511e-3, 2.)
    #     return -(8.*sqrt(2)*GF*T_nu/3.)*(rho_nu/(mZ**2.) + rho_a/(mW**2.))
    #
    # def V_d(T_d):
    #     return y2*T_d*T_d/(16.*np.sqrt(T_d*T_d+m_d*m_d)) if T_d > m_X else -y2*np.sqrt(T_d*T_d+m_d*m_d)*0.75*2.*m_d*m_d*T_d*T_d*np.exp(-m_d/T_d)/(cf.pi2*m_X2*m_X2)#-7.*cf.pi2*y2*np.sqrt(T_d*T_d+m_d*m_d)*(T_d**4.)/(90.*m_X2*m_X2)
    #
    # def G_a(T_nu):
    #     return 1.27*(GF**2.)*(T_nu**5.)
    #
    # V_SM_grid = np.array([V_SM(T_SM, T_nu) for T_SM, T_nu in zip(Ttrel.T_SM_grid[pan.i_ic:pan.i_end+1], Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1])])
    # V_d_grid = np.array([V_d(T_d) for T_d in pan.T_chi_grid_sol])
    # G_a_grid = np.array([G_a(T_nu) for T_nu in Ttrel.T_nu_grid[pan.i_ic:pan.i_end+1]])
    # G_d_grid = np.array([G_d(T_d, xi_d, xi_X) for T_d, xi_d, xi_X in zip(pan.T_chi_grid_sol, pan.xi_chi_grid_sol, pan.xi_X_grid_sol)])
    
if __name__ == '__main__':



    # load_str = './md_2.1e-05;mX_1.05e-04;mh_6.3e-05;sin22th_1.5e-15;y_1.4e-03;full_new.dat'    
    # var_list = load_str.split(';')[:-1]
    # m_d, m_X, m_h, sin2_2th, y = [eval(s.split('_')[-1]) for s in var_list]
    # Anton: Ignore these masses 
    m_h = 0
    m_nu = 0
    
    # Anton: Actual masses - assume m_N1 = m_N2
    m_d = 1e-5
    m_N1 = m_d
    m_N2 = m_d
    m_X = 2.5*m_d

    # Anton: fermion = 1, boson = -1 (I did not choose this convention..)
    k_d = 1.
    k_X = -1.
    k_nu = 1.

    # Anton: Fermion has 2 dof, massive vector has 3 dof 
    dof_d = 2.
    dof_X = 3.

    # Anton: Need m0 >> mi, m12 >> mi^2 / m0, i = a,1,2,(12)
    y = 4e-4
    m0 = 1e3
    m12 = m_d
    m2 = 1e-2
    ma = m12*1e-2

    sin2_2th = (m2*ma/(m0*m12))**2
    th = 1/2 * asin(sqrt(sin2_2th))
    print(sin2_2th)

    # Anton: If spin-statistics (1+k*f) matters, if the mediator is off-shell or not 
    spin_facs = True
    off_shell = False

    print('Start sterile_caller')
    start = time.time()
    t, T_SM, T_nu, ent, H, sf, T_d, xi_d, xi_X, n_d, n_X, C_therm, fs_length, fs_length_3, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound, r_sound_3, reached_integration_end = call(m_N1, m_N2, m_X, m0, m12, m2, ma, k_d, k_X, k_nu, dof_d, dof_X, y, spin_facs=True, off_shell=False)
    # print(fs_length, fs_length_3, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound, r_sound_3)
    end = time.time()
    print(f'sterile_caller ran in {end-start:.5f}s')

    print(r_sound)
    print(fs_length)


    # Anton: Save data to file 
    md_str = f'{m_d:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_d:.5e}'.split('e')[1].rstrip('0').rstrip('.')
    mX_str = f'{m_X:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_X:.5e}'.split('e')[1].rstrip('0').rstrip('.')
    mh_str = f'{m_h:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_h:.5e}'.split('e')[1].rstrip('0').rstrip('.')
    sin22th_str = f'{sin2_2th:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{sin2_2th:.5e}'.split('e')[1].rstrip('0').rstrip('.')
    y_str = f'{y:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{y:.5e}'.split('e')[1].rstrip('0').rstrip('.')

    # file_str = f'sterile_test/md_{md_str};mX_{mX_str};mh_{mh_str};sin22th_{sin22th_str};y_{y_str};full_new.dat'
    file_str = f'sterile_test/md_{md_str};mX_{mX_str};sin22th_{sin22th_str};y_{y_str};full_new.dat'
    np.savetxt(file_str, np.column_stack((t, T_SM, T_nu, ent, H, sf, T_d, xi_d, xi_X, n_d, n_X)))

    print(f'Saved data to {file_str}')
