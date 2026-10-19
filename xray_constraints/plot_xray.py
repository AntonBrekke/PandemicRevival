#! /usr/bin/env python3
"""
The combined X-ray / gamma-ray limit on sin^2(2 theta) of sterile-neutrino decay N -> nu gamma: the
minimum over the current limits below, written to overall_constraint.dat (m_s [GeV], sin^2 2theta)
on a log grid. The limits assume that all the dark matter decays; the relic plot rescales by the
decaying fraction (plot/relic_scan_theta.py in julia/pandemic).

    python plot_xray.py [--plot]      # --plot: draw the limits and their minimum (xray_limits.pdf)

Superseded and therefore not used: INTEGRAL/SPI 2008 (0710.4922, by SPI 2023), and NuSTAR 2020
(1908.09037, by NuSTAR 2023, same group and data) except above 39.6 keV, where the 2023/2024
NuSTAR analyses end and SPI starts only at 40.5 keV (its digitised values there sit at the 1e-14
floor of the compilation figure they were read from, i.e. are conservative). Not used: eROSITA eFEDS (2401.16747; weaker than
older limits and than eROSITA LMC in its range), SRG/ART-XC (2303.12673; weaker than NuSTAR 2023
everywhere), XRISM (2510.24560; one mass, 7.1 keV, inside the NuSTAR limits).

INTEGRAL/SPI template (2209.06299) is recomputed from the authors' data and notebook (Zenodo
7984451) in spi_2209.06299/ (README there); the curve of the paper's Fig. 5 is 2x too strong in
the Majorana convention used here.
"""
import sys
import numpy as np

LIMITS = {   # file: label
    "CBS_2008.02283.dat": "Chandra blank sky (2008.02283)",
    "Dwarfs_1408.3531.dat": "dwarf spheroidals (1408.3531)",
    "M31_1311.0282.dat": "M31 (1311.0282)",
    "XMMBlankSky_2102.02207.dat": "XMM-Newton blank sky (2102.02207)",
    "eROSITA_LMC_2603.19109.dat": "eROSITA LMC (2603.19109)",
    "NuSTAR_2207.04572.dat": "NuSTAR halo (2207.04572)",
    "NuSTAR_straylight_2405.17861.dat": "NuSTAR stray light (2405.17861)",
    "INTEGRAL_SPI_2211.06200.dat": "INTEGRAL/SPI (2211.06200)",
    "INTEGRAL_SPI_2209.06299.dat": "INTEGRAL/SPI template (2209.06299)",
    "NuSTAR_1908.09037.dat": "NuSTAR 2020 above 39.6 keV (1908.09037)",
}
M_MIN = {"NuSTAR_1908.09037.dat": 3.96e-5}   # GeV: used only above the newer NuSTAR limits


def load(f):
    return np.loadtxt(f, comments="#")[:, :2]     # SPI 2211.06200: column 1 is the reference profile


lims = {f: load(f) for f in LIMITS}
m_lo = min(d[0, 0] for d in lims.values())
m_hi = max(d[-1, 0] for d in lims.values())
m = np.logspace(np.log10(m_lo), np.log10(m_hi), 10000)
logs = np.array([np.where(m >= M_MIN.get(f, 0.0),
                          np.interp(np.log(m), np.log(d[:, 0]), np.log(d[:, 1]), left=np.inf, right=np.inf), np.inf)
                 for f, d in lims.items()])
overall = np.exp(logs.min(axis=0))
np.savetxt("overall_constraint.dat", np.column_stack((m, overall)), fmt="%.6e",
           header="minimum over: " + ", ".join(LIMITS) + "\nm_s [GeV]    sin^2 (2 theta)")
which = np.where(np.isfinite(logs.min(axis=0)), np.array(list(LIMITS.values()))[logs.argmin(axis=0)], "none")
print("overall_constraint.dat: %.3g-%.3g keV; set by:" % (m_lo * 1e6, m_hi * 1e6))
start = 0
for i in range(1, len(m) + 1):
    if i == len(m) or which[i] != which[start]:
        if m[i - 1] / m[start] > 1.02:
            print(f"  {m[start] * 1e6:8.3g} - {m[i - 1] * 1e6:8.3g} keV  {which[start]}")
        start = i

if "--plot" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for f, lab in LIMITS.items():
        ax.loglog(lims[f][:, 0] * 1e6, lims[f][:, 1], lw=1, label=lab)
    ax.loglog(m * 1e6, overall, "k", lw=2.5, alpha=0.5, label="minimum")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(1e-24, 1e-6)
    ax.set_xlabel(r"$m_s$ [keV]")
    ax.set_ylabel(r"$\sin^2 2\theta$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("xray_limits.pdf")
    print("written xray_limits.pdf")
