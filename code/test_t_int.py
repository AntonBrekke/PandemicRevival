import numpy as np
import matplotlib.pyplot as plt

import C_res_vector

def test_t_int():
    y = 1e-4
    vert = y**4

    m_N = 1e-5
    m_A = 2.5 * m_N

    m1 = m_N
    m3 = m_A

    e1 = 4. * m3
    e2 = e1

    rest_energy = e1 + e2 - m3 - m3
    e3 = m3 + rest_energy / 2.
    e4 = e1 + e2 - e3

    print("e1 = ", e1, ", e2 = ", e2, ", e3 = ", e3, ", e4 = ", e4)

    p1 = np.sqrt(e1**2 - m1**2)
    p2 = np.sqrt(e2**2 - m1**2)
    p3 = np.sqrt(e3**2 - m3**2)
    p4 = np.sqrt(e4**2 - m3**2)

    s12_min = np.fmax(2.*m1*m1+2.*(e1*e2-p1*p2), 4.*m1*m1)
    s12_max = 2.*m1*m1+2.*e1*e2+2.*p1*p2
    s34_min = np.fmax(2.*m3*m3+2.*e3*(e4-p3*p4/e3), 4.*m3*m3)
    s34_max = 2.*m3*m3+2.*e3*e4+2.*p3*p4

    s_min = np.fmax(s12_min, s34_min)
    s_max = np.fmin(s12_max, s34_max)

    print("s_min = ", s_min)
    print("s_max = ", s_max)

    reg = (s_max - s_min) / 1e5
    n = 100
    s = np.linspace(s_min + reg , s_max - reg , n)
    print("s = ", s)

    a = np.fmin(-4.*p3**2*((e1+e2)*(e1+e2) - s), -1e-200)
    print("a = ", a)
    b = 2.*(p3/p1)*(s-2.*e1*(e1+e2))*(s-2.*e3*(e1+e2))
    sqrt_arg = 4.*(p3**2/p1**2)*(s-s12_min)*(s-s12_max)*(s-s34_min)*(s-s34_max)
    sqrt_fac = np.sqrt(np.fmax(sqrt_arg, 0.))

    ct_p = (-b + sqrt_fac)/(2.*a)
    ct_m = (-b - sqrt_fac)/(2.*a)

    ct_min = np.fmin(np.fmax(-1., ct_p), 1.)
    ct_max = np.fmax(np.fmin(1., ct_m), ct_min)

    sol = C_res_vector.ker_C_n_XX_dd_s_t_integral_revival(
        ct_min, ct_max,
        ct_p, ct_m,
        a,
        s,
        e1, e3,
        p1, p3,
        m_N, m_A,
        vert
    )

    # sol_new = C_res_vector.ker_C_n_XX_dd_s_t_integral_revival_new(
    #     ct_min, ct_max,
    #     ct_p, ct_m,
    #     a,
    #     s,
    #     e1, e3,
    #     p1, p3,
    #     m_N, m_A,
    #     vert
    # )

    # print("Integral result:", sol)
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-9, 1e-7)
    ax.set_ylim(1e-8, 1e-2)
    ax.scatter(s, sol)
    # ax.scatter(s, sol_new)
    fig.savefig("figures/test_t_int.pdf")


test_t_int()