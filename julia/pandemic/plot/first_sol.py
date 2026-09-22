import time
import numpy as np
import matplotlib.pyplot as plt

def plot_x_frac(jul, pyt):
    x_min = min(jul.x_nu[0], pyt.x_nu[0])
    x_max = max(jul.x_nu[-1], pyt.x_nu[-1])

    fig_x_N, ax_x_N = plt.subplots()
    fig_x_N.set_layout_engine('constrained')
    ax_x_N.set_xscale("log")
    ax_x_N.set_yscale("log")
    ax_x_N.set_xlim((x_min/2, x_max*2))
    # ax_x_N.set_ylim(0., 5.)
    ax_x_N.plot(jul.x_nu, jul.x_N / jul.x_nu, label="Jul")
    ax_x_N.plot(pyt.x_nu, pyt.x_N / pyt.x_nu, ls='--', c='r', label="Pyt")
    ax_x_N.set_xlabel(r"$x$")
    ax_x_N.set_ylabel(r"$x_N/x$")
    ax_x_N.grid()
    ax_x_N.grid(which='minor', ls=':', alpha=0.5)
    ax_x_N.legend()
    fig_x_N.suptitle(fr"$m_N=1\cdot 10^{{-5}}\, \textrm{{GeV}},\ m_A=2.5\, m_N,\ y=2\sqrt{{3}}\cdot 10^{{-5}},\ \sin^2(2\theta)=2.65\cdot 10^{{-13}}$")
    fig_x_N.savefig("figures/x_N.pdf")

    m_N = 1e-5
    T_nu_jul = m_N / jul.x_nu
    T_nu_pyt = m_N / pyt.x_nu
    T_N_jul = m_N / jul.x_N
    T_N_pyt = m_N / pyt.x_N

    # fig_T, ax_T = plt.subplots()
    # fig_T.set_layout_engine('constrained')
    # ax_T.set_xscale("log")
    # ax_T.set_yscale("log")
    # ax_T.plot(jul.x_nu, T_nu_jul, label=r"$T_\nu$")

def plot_y_n(jul, pyt=None, m_N1=1e-5, m_A_over_m_N=2.5, path="figures/y_n.pdf", title=None, ax=None):
    """Saves its own figure to `path`, or only draws onto `ax` if one is given."""
    x_min = jul.x_nu[0] if pyt is None else min(jul.x_nu[0], pyt.x_nu[0])
    x_max = jul.x_nu[-1] if pyt is None else max(jul.x_nu[-1], pyt.x_nu[-1])
    y_min = 1e-30
    y_max = 1e-8

    # See constant_functions.jl
    # mY_relic = omega_d0 * rho_crit0_h2 * s0
    mY_relic = 4.354e-10

    m_N2 = m_N1
    m_A = m_A_over_m_N * m_N1

    if ax is None:
        fig_n, ax_n = plt.subplots()
        fig_n.set_layout_engine('constrained')
    else:
        fig_n, ax_n = None, ax
    ax_n.set_xscale("log")
    ax_n.set_yscale("log")
    ax_n.set_xlim((x_min/2, x_max*2))
    ax_n.set_ylim(y_min, y_max)
    ax_n.hlines(mY_relic, x_min, x_max, ls='--', color='k', label="Correct abund.")
    # ax_n.plot(jul.x_nu, jul.y_n, label="y_n")
    ax_n.plot(jul.x_nu, m_N1 * jul.y_N1, label=r"$N_1$")
    ax_n.plot(jul.x_nu, m_N2 * jul.y_N2, ls=':', label=r"$N_2$")
    ax_n.plot(jul.x_nu, m_A * jul.y_A, label=r"$A$")

    my_tot = m_N1 * jul.y_N1 + m_N2 * jul.y_N2 + m_A * jul.y_A
    ax_n.plot(jul.x_nu , my_tot, label="Total abund.")
    # ax_n.plot(pyt.x_nu, pyt.y_N, ls='--')
    ax_n.set_xlabel(r"$x = m_N/T_\nu$")
    ax_n.set_ylabel(r"$mY$")

    ax_n.grid()
    ax_n.legend()

    # fig_n.suptitle(fr"$m_N={md_str},\ m_A={mX_str}\, m_N,\ y={y_str},\ \sin^2(2\theta)={sin22th_str}$")
    # fig_n.suptitle(fr"$m_N=1\cdot 10^{{-5}}\, \textrm{{GeV}},\ m_A=2.5\, m_N,\ y=2\sqrt{{3}}\cdot 10^{{-5}},\ \sin^2(2\theta)=5.3\cdot 10^{{-13}}$")
    if fig_n is not None:
        if title is not None:
            fig_n.suptitle(title)
        fig_n.savefig(path)
        plt.close(fig_n)

def plot_rho(jul, pyt):
    x_min = min(jul.x_nu[0], pyt.x_nu[0])
    x_max = max(jul.x_nu[-1], pyt.x_nu[-1])

    fig_rho, ax_rho = plt.subplots()
    fig_rho.set_layout_engine('constrained')
    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_rho.set_xlim((x_min/2, x_max*2))
    # ax_rho.set_xlim(1e-5, 1e-3)
    # ax_rho.set_ylim(0., 2.)
    ax_rho.plot(jul.x_nu, jul.y_rho)
    ax_rho.plot(pyt.x_nu, pyt.y_rho, ls='--')
    ax_rho.set_xlabel("x")
    ax_rho.set_ylabel("y_rho")
    ax_rho.grid()
    ax_rho.grid(which='minor', ls=':', alpha=0.5)
    fig_rho.savefig("figures/y_rho.pdf")

def plot_xi_N(jul, pyt):
    x_min = min(jul.x_nu[0], pyt.x_nu[0])
    x_max = max(jul.x_nu[-1], pyt.x_nu[-1])

    fig_xi_N, ax_xi_N = plt.subplots()
    fig_xi_N.set_layout_engine('constrained')
    ax_xi_N.set_xscale("log")
    # ax_xi_N.set_yscale("log")
    ax_xi_N.set_xlim((x_min/2, x_max*2))
    # ax_xi_N.set_xlim(1e-5, 1e-3)
    # ax_xi_N.set_ylim(0., 2.)
    ax_xi_N.plot(jul.x_nu, jul.xi_N)
    ax_xi_N.plot(pyt.x_nu, pyt.xi_N, ls='--')
    ax_xi_N.set_xlabel("x")
    ax_xi_N.set_ylabel("xi_N")
    fig_xi_N.savefig("figures/xi_N.pdf")

def plot_hubble_coll(jul, path="figures/hubble.pdf", title=None, ylim=(1e-32, 1e-20), ax=None):
    """Saves its own figure to `path`, or only draws onto `ax` if one is given."""
    x_min = jul.x_nu[0]
    x_max = jul.x_nu[-1]
    y_min, y_max = ylim

    n = jul.ent * jul.y_n

    coll_over_n = jul.coll_n / n
    coll_over_A_N2nu = jul.coll_A_N2nu / n
    coll_over_AA_NN = jul.coll_AA_NN / n

    if ax is None:
        fig_hubble, ax_hubble = plt.subplots()
        fig_hubble.set_layout_engine('constrained')
    else:
        fig_hubble, ax_hubble = None, ax
    ax_hubble.set_xscale("log")
    ax_hubble.set_yscale("log")
    ax_hubble.set_xlim((x_min/2, x_max*2))
    ax_hubble.set_ylim((y_min, y_max))
    ax_hubble.plot(jul.x_nu, jul.hubble, label=r"$H$")
    ax_hubble.plot(jul.x_nu, abs(coll_over_n), label=r"$|C_n|/n$")
    ax_hubble.plot(jul.x_nu, abs(coll_over_A_N2nu), label=r"$|C_{N2\nu}|/n$")
    ax_hubble.plot(jul.x_nu, abs(coll_over_AA_NN), label=r"$|C_{AA}|/n$")
    ax_hubble.set_xlabel(r"$x = m_N/T_\nu$")
    ax_hubble.set_ylabel(r"Rate [GeV]")
    ax_hubble.grid()
    ax_hubble.legend()
    # fig_hubble.suptitle(fr"$m_N=1\cdot 10^{{-5}}\, \textrm{{GeV}},\ m_A=2.5\, m_N,\ y=2\sqrt{{3}}\cdot 10^{{-5}},\ \sin^2(2\theta)=5.3\cdot 10^{{-13}}$")
    if fig_hubble is not None:
        if title is not None:
            fig_hubble.suptitle(title)
        fig_hubble.savefig(path)
        plt.close(fig_hubble)

def plot_coll_n(jul, pyt):
    x_min = min(jul.x_nu[0], pyt.x_coll[0])
    x_max = max(jul.x_nu[-1], pyt.x_coll[-1])

    fig_coll_n, ax_coll_n = plt.subplots()
    fig_coll_n.set_layout_engine('constrained')
    ax_coll_n.set_xscale("log")
    ax_coll_n.set_yscale("log")
    ax_coll_n.set_xlim((x_min/2, x_max*2))
    # ax_coll_n.set_xlim(1e-5, 1e-3)
    # ax_coll_n.set_ylim(0., 2.)
    ax_coll_n.plot(jul.x_nu, jul.coll_n)
    ax_coll_n.scatter(pyt.x_coll, pyt.coll_n, ls='--')
    ax_coll_n.set_xlabel(r"$x$")
    ax_coll_n.set_ylabel(r"$C_N$")
    fig_coll_n.savefig("figures/coll_n.pdf")

    log_coll_n_jul_interp = np.interp(np.log(pyt.x_coll), np.log(jul.x_nu), np.log(jul.coll_n))

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlim((x_min/2, x_max*2))
    ax.set_ylim((0, 20))
    # ax.set_ylim(())
    ax.scatter(pyt.x_coll, np.exp(log_coll_n_jul_interp) / pyt.coll_n)
    fig.savefig("figures/coll_frac.pdf")


class PytRes:
    def __init__(
            self,
            x_nu, x_N,
            y_N, y_A, y_rho,
            xi_N,
            x_coll, coll_n,
        ):
        self.x_nu = x_nu
        self.x_N = x_N
        self.y_N = y_N
        self.y_A = y_A
        self.y_rho = y_rho
        self.xi_N = xi_N
        self.x_coll = x_coll
        self.coll_n = coll_n


def python_results():
    load_str = "./../../code/sterile_test/md_1e-05;mX_2.5e-05;sin22th_2.65e-11;y_1e-05;full_new.dat"
    c_str = "./../../code/sterile_test/md_1e-05;mX_2.5e-05;sin22th_2.65e-11;y_1e-05;c_list.dat"

    var_list = load_str.split(';')[:-1]
    m_N, m_A, sin22th, y = [eval(s.split('_')[-1]) for s in var_list]

    data = np.loadtxt(load_str)
    T_SM = data[:, 1]
    T_nu = data[:, 2]
    ent = data[:, 3]
    Td = data[:, 6]
    xi_N = data[:, 7]
    xi_A = data[:, 8]
    n_N = data[:, 9]
    n_A = data[:, 10]
    rho = data[:, 11]

    x_nu = m_N / T_nu
    x_N = m_N / Td

    y_N = n_N / ent
    y_A = n_A / ent

    y_rho = rho / ent**(4/3)

    c_data = np.loadtxt(c_str)
    x_coll = c_data[:, 0]
    coll_n = c_data[:, 1]

    return PytRes(
        x_nu, x_N,
        y_N, y_A, y_rho,
        xi_N,
        x_coll, coll_n,
    )


class JulRes:
    def __init__(
            self,
            x_nu, x_N,
            hubble, ent,
            y_n, y_rho,
            y_N1, y_N2, y_A,
            xi_N,
            coll_n,
            coll_A_N2nu,
            coll_AA_NN,
        ):
        self.x_nu = x_nu
        self.x_N = x_N
        self.hubble = hubble
        self.ent = ent
        self.y_n = y_n
        self.y_rho = y_rho
        self.y_N1 = y_N1
        self.y_N2 = y_N2
        self.y_A = y_A
        self.xi_N = xi_N
        self.coll_n = coll_n
        self.coll_A_N2nu = coll_A_N2nu
        self.coll_AA_NN = coll_AA_NN

def julia_results(path="tmp/sol.csv"):
    sol = np.genfromtxt(path, delimiter=',', skip_header=1)

    x_nu = sol[:, 0]
    x_N = sol[:, 1]
    hubble = sol[:, 2]
    ent = sol[:, 3]

    y_n = sol[:, 4]
    y_rho = sol[:, 5]
    xi_N = sol[:, 6]

    y_N1 = sol[:, 8]
    y_N2 = sol[:, 9]
    y_A = sol[:, 10]

    coll_n = sol[:, 11]
    coll_A_N2nu = sol[:, 12]
    coll_AA_NN = sol[:, 13]

    return JulRes(
        x_nu, x_N,
        hubble, ent,
        y_n, y_rho,
        y_N1, y_N2, y_A,
        xi_N,
        coll_n,
        coll_A_N2nu,
        coll_AA_NN,
    )


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "lines.markersize": 2.,
        #"lines.linewidth": .3
    })

    jul_res = julia_results()
    pyt_res = python_results()

    plot_x_frac(jul_res, pyt_res)
    plot_y_n(jul_res, pyt_res)
    plot_rho(jul_res, pyt_res)
    plot_xi_N(jul_res, pyt_res)

    plot_coll_n(jul_res, pyt_res)
    plot_hubble_coll(jul_res)

if __name__ == "__main__":
    main()
