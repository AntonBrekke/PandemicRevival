#! /usr/bin/env python3
"""
Abundance (mY) and collision-rate vs. Hubble plots for every history written by
src/run_abundance_history.jl, using plot_y_n and plot_hubble_coll from
first_sol.py. For <name>.csv in the input directory, writes <name>.pdf to the
output directory, with the abundances on top and the rates below.

Run from julia/pandemic:
    conda run -n pandemic python plot/abundance_history.py [history_dir] [figure_dir]
"""

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from first_sol import julia_results, plot_y_n, plot_hubble_coll  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
IN_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tmp", "abundance_history")
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures", "abundance_history")

M_A_OVER_M_N = 2.5
NAME = re.compile(r"m_N_(?P<m>[^_]+)keV_y_(?P<y>[^_]+)_sin22th_(?P<s>[^_]+)\.csv$")


def latex_num(v):
    """4.151e-15 -> 4.151\\cdot 10^{-15}"""
    mant, exp = f"{v:.3e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    exp = int(exp)
    if exp == 0:
        return mant
    return (f"{mant}\\cdot " if mant != "1" else "") + f"10^{{{exp}}}"


def main():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "lines.markersize": 2.,
    })
    os.makedirs(OUT_DIR, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(IN_DIR, "m_N_*keV_y_*_sin22th_*.csv")))
    if not paths:
        raise SystemExit(f"no abundance histories in {IN_DIR}")

    for path in paths:
        match = NAME.search(os.path.basename(path))
        if match is None:
            print(f"skipping {path}: unexpected file name")
            continue
        m_keV, y, s = (float(match[k]) for k in ("m", "y", "s"))
        m_N = m_keV * 1e-6
        jul = julia_results(path)

        title = (fr"$m_N={m_keV:g}\ \textrm{{keV}},\ m_A={M_A_OVER_M_N}\, m_N,\ "
                 fr"y={latex_num(y)},\ \sin^2(2\theta)={latex_num(s)}$")
        stem = os.path.join(OUT_DIR, os.path.basename(path)[:-len(".csv")])
        fig, (ax_n, ax_rate) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 8.))
        fig.set_layout_engine('constrained')
        plot_y_n(jul, m_N1=m_N, m_A_over_m_N=M_A_OVER_M_N, ax=ax_n)
        # H ~ T^2, so at fixed x = m_N/T the rates scale as m_N^2: keep the
        # window of the 10 keV default.
        scale = (m_N / 1e-5)**2
        plot_hubble_coll(jul, ylim=(1e-32 * scale, 1e-20 * scale), ax=ax_rate)
        ax_n.set_xlabel("")
        fig.suptitle(title)
        fig.savefig(stem + ".pdf")
        plt.close(fig)
        print(f"{stem}.pdf")


if __name__ == "__main__":
    main()
