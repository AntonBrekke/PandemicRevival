import time

import numpy as np
import matplotlib.pyplot as plt

import C_res_vector



def test_collision_integrals(m_nu, m_N, m_A, y, theta):

    x = np.geomspace(1e-6, 1e2, 1000)
    temp_d = m_N / x
    xi_N = -20.
    xi_A = 2. * xi_N

    """
    c_AA_NN = np.zeros_like(x)
    for i in range(len(x)):
        c_AA_NN[i] = C_res_vector.C_n_XX_dd(
            m_d=m_N, m_X=m_A,
            k_d=1., k_X=-1.,
            T_d=temp_d[i],
            xi_d=xi_N, xi_X=xi_A,
            vert=y**4,
            type=0,
        )
    print(c_AA_NN)
    """


    amplitude_squared = 2. * y**2 * np.sin(theta)**2 * (m_A-m_N-m_nu)*(m_A+m_N+m_nu)*(2*m_A**2+(m_N-m_nu)**2)/m_A**2

    start = time.time()
    c_A_Nnu = np.zeros_like(x)
    for i in range(len(x)):
        c_A_Nnu[i] = C_res_vector.C_n_3_12(
            m1=m_N, m2=m_nu, m3=m_A,
            k1=1., k2=1., k3=-1.,
            T1=temp_d[i], T2=temp_d[i], T3=temp_d[i],
            xi1=xi_N, xi2=0., xi3=xi_A,
            M2=amplitude_squared,
            type=0
        )
    # print(c_A_Nnu)
    end = time.time()
    print(f'Integration ran in {end-start:.5f}s')

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-100, 1e-20)
    # ax.plot(x, c_AA_NN, label=r"$C_{n}^{AA\to NN}$")
    ax.scatter(x, c_A_Nnu, label=r"$C_{n}^{A\to N\nu}$")
    ax.set_xlabel(r"$x = m_N / T$")
    ax.set_ylabel(r"Collision integral")
    ax.legend()
    fig.savefig("figures/collision_integrals.pdf")



def main():
    print("Testing collision integrals")

    y = 1e-5
    sin2_2th = 2e-11
    theta = 0.5 * np.arcsin(np.sqrt(sin2_2th))

    m_nu = 0.
    m_N = 1e-5
    m_A = 2.5 * m_N

    test_collision_integrals(m_nu, m_N, m_A, y, theta)

if __name__ == "__main__":
    main()
