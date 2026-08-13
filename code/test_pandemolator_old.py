import time
import numpy as np
import matplotlib.pyplot as plt

import pandemolator as pan
import constants_functions as cf
import C_res_vector



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

    x = m_d / T_a

    # print("C_ns:", f'{CX_XX_11:.5e}', f'{CX_X_1nu:.5e}')
    # Factor 2 as N_1 and N_2 both contribute. Second factor of two from number change in n_d = n_N + 2*n_X
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


def test_dw():
    m_N = 1e-5
    k_N = 1
    dof_N = 2

    m_A = 2.5 * m_N
    k_A = -1
    dof_A = 3

    m_nu = 0.
    k_nu = 1
    dof_nu = 2

    sin2_2th = 5e-16
    th = np.arcsin(np.sqrt(sin2_2th)) / 2.

    tT_rel = pan.TimeTempRelation()

    # Dodelson-Widrow
    T_d_dw = cf.T_d_dw(m_N) # temperature of maximal d production by Dodelson-Widrow mechanism
    i_ic = np.argmax(tT_rel.T_nu_grid < T_d_dw)      # Anton: Start when T_nu < T_dw
    i_end = np.argmax(tT_rel.T_nu_grid < m_N/2e1)    # Anton: End when T_nu < m_d/20 <--> 20 < m_d/T_nu
    T_ic = tT_rel.T_nu_grid[i_ic]
    T_end = tT_rel.T_nu_grid[i_end]
    sf_ic_norm_0 = (cf.s0/(cf.s_SM_no_nu(tT_rel.T_SM_grid[i_ic]) + cf.s_nu(tT_rel.T_nu_grid[i_ic])))**(1./3.)
    n_ic = cf.n_0_dw(m_N, th) / (sf_ic_norm_0**3.)
    rho_ic = n_ic * cf.avg_mom_0_dw(m_N) / sf_ic_norm_0

    ent_grid = np.array([cf.s_SM_no_nu(T)+cf.s_nu(T_nu) for T, T_nu in zip(tT_rel.T_SM_grid, tT_rel.T_nu_grid)])


    m_N1 = m_N
    m_N2 = m_N

    # TODO: [12.08.26] Not used?
    m_h = 0.

    pan = pan.Pandemolator(
        m_N1, m_N2, m_A, m_h, m_nu,
        k_N, k_A, k_nu,
        dof_N, dof_A,
        C_n, C_rho, C_xi0,
        tT_rel.t_grid,
        tT_rel.T_nu_grid,
        tT_rel.dTnu_dt_grid,
        ent_grid,
        tT_rel.hubble_grid,
        tT_rel.sf_grid,
        i_ic, n_ic, rho_ic, i_end
    )



test_dw()