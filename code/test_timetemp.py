import time
import numpy as np
import matplotlib.pyplot as plt

import pandemolator as pan

def main():
    start = time.time()
    for i in range(100):
        rel = pan.TimeTempRelation()
    end = time.time()
    print(f"TimeTempRelation ran in {(end-start):.5f}s")


def compare():
    rel = pan.TimeTempRelation()

    rel_jul = np.genfromtxt("../julia/pandemic/tmp/test_time_temp.csv", delimiter=',', skip_header=1)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e20, 1e35)
    ax.set_ylim(1e-10, 1e-2)
    ax.plot(rel.t_grid, rel.T_SM_grid)
    ax.plot(rel.t_grid, rel.T_nu_grid)
    ax.plot(rel_jul[:, 0], rel_jul[:, 1], ':')
    ax.plot(rel_jul[:, 0], rel_jul[:, 2], ':')
    fig.savefig("figures/test_time_temp.pdf")

    fig2, ax2 = plt.subplots()
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.plot(rel.t_grid, rel.hubble_grid)
    ax2.plot(rel_jul[:, 0], rel_jul[:, 3], ':')
    fig2.savefig("figures/test_time_temp_hubble.pdf")

    fig3, ax3 = plt.subplots()
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.plot(rel.t_grid, rel.nu_dec_grid)
    ax3.plot(rel_jul[:, 0], rel_jul[:, 4], ':')
    fig3.savefig("figures/test_time_temp_nu_dec.pdf")

    fig4, ax4 = plt.subplots()
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlim(1e20, 1e35)
    ax4.set_ylim(1e-40, 1e-23)
    ax4.plot(rel.t_grid, -rel.dTSM_dt_grid)
    ax4.plot(rel_jul[:, 0], -rel_jul[:, 5], ':')
    ax4.plot(rel.t_grid, -rel.dTnu_dt_grid)
    ax4.plot(rel_jul[:, 0], -rel_jul[:, 6], ':')
    fig4.savefig("figures/test_time_temp_dT_SM_dt.pdf")

    ent_norm_julia = rel_jul[:, 7] / rel_jul[0, 7]
    ent_norm_python = (rel.sf_grid / rel.sf_grid[0])**(-3)

    fig5, ax5 = plt.subplots()
    ax5.set_xscale("log")
    ax5.set_yscale("log")
    ax5.plot(rel.t_grid, ent_norm_python)
    ax5.plot(rel_jul[:, 0], ent_norm_julia, ':')
    fig5.savefig("figures/test_time_temp_ent.pdf")




# main()
compare()