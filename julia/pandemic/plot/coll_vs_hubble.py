import numpy as np
import matplotlib.pyplot as plt

def main():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "lines.markersize": .8,
        #"lines.linewidth": .3
    })

    sol = np.genfromtxt("tmp/sol.csv", delimiter=',', skip_header=1)

    x_nu = sol[:, 0]
    y_n = sol[:, 1]
    coll_n = sol[:, 5]
    hubble = sol[:, 6]
    ent = sol[:, 7]
    n = ent * y_n

    rate_coll_n = coll_n / n
    print(x_nu)
    print(coll_n)
    print(rate_coll_n)

    fig, ax = plt.subplots()
    fig.set_layout_engine('constrained')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(x_nu, hubble)
    ax.scatter(x_nu, rate_coll_n)
    fig.savefig("figures/hubble.pdf")

main()