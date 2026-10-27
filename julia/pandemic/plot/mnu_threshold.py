"""
How small the light-neutrino mass must be before the N1-N2 splitting it implies,
delta = sin^2 theta_1 m_N/m_nu (mnu_splitting in src/thermal_rates.jl), matters, at every relic
root of the m_A' = 2.5 m_N scan (tmp/combined_scan). Two thresholds:

- depletion of N2 (late N1 N1 <-> N2 N2, with_conversions in src/lyman_alpha.jl): N2 follows
  N1 e^(-delta m_N/T_N) until the conversions freeze out at x_N,fo = m_N/T_N, so
  ln(f_N1/f_N2) grows by delta x_N,fo. x_N,fo per root from the scans at delta = 1e-4
  (tmp/delta_runs/delta1e-4/combined) and delta = 0: [ln(f1/f2)_1e-4 - ln(f1/f2)_0]/1e-4. N2 is
  depleted by 1/e for m_nu < m_nu,dep = sin^2 theta_1 m_N x_N,fo.
- production (phi N1 N1 Yukawa (g/sqrt2)(m_N/m_A') delta): the delta_*(g) at which the root moves
  by 1%, from test/check_mnu_threshold.jl (tmp/mnu_threshold/*.csv, interpolated in log g,
  constant beyond); m_nu,prod = sin^2 theta_1 m_N (1 + delta_*)/delta_*.

    python plot/mnu_threshold.py [fig_out_without_extension]

Writes tmp/mnu_threshold/thresholds.csv (m_N, y, s2, delta at 0.1 eV, x_N,fo, m_nu_dep_eV,
m_nu_prod_eV) and the figure (default figures/mnu_threshold.pdf).
"""
import csv
import glob
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_NU = 0.1e-9   # GeV, M_NU of src/thermal_rates.jl
SHIFT = 0.01    # root shift that counts as relevant
G_DEP_MIN = 1e-5
key = lambda r: (round(float(r["m_N"]) * 1e6, 3), round(math.log10(float(r["y"])), 3))


def load(path):
    return {key(r): r for r in csv.DictReader(open(path))}


def delta_star():
    """(log10 g, log10 delta_*) where the root moves by SHIFT, from the threshold study."""
    pts = {}
    for f in glob.glob("tmp/mnu_threshold/mnu_threshold_*keV.csv"):
        rows = list(csv.DictReader(open(f)))
        for g in sorted({float(r["g"]) for r in rows}):
            rs = sorted((float(r["delta"]), abs(float(r["root_shift"]))) for r in rows
                        if float(r["g"]) == g and math.isfinite(float(r["root_shift"])))
            d_star = math.nan
            for (d0, s0), (d1, s1) in zip(rs[:-1], rs[1:]):
                if s0 < SHIFT <= s1:
                    w = math.log(SHIFT / max(s0, 1e-12)) / math.log(s1 / max(s0, 1e-12))
                    d_star = d0 * (d1 / d0) ** min(max(w, 0.0), 1.0)
                    break
            if math.isnan(d_star) and rs and rs[-1][1] < SHIFT:
                d_star = rs[-1][0]   # a lower bound: no 1% shift up to the largest delta tried
            if math.isfinite(d_star):
                pts.setdefault(round(math.log10(g), 3), []).append(math.log10(d_star))
    lg = np.array(sorted(pts))
    return lg, np.array([np.mean(pts[k]) for k in lg])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "figures/mnu_threshold"
    base = load("tmp/combined_scan/lyman_alpha.csv")
    d4 = load("tmp/delta_runs/delta1e-4/combined/lyman_alpha.csv")
    lg, ld = delta_star()
    rows = []
    for k, r in base.items():
        m, s2 = float(r["m_N"]), float(r["sin2_2theta"])
        th1sq = math.sin(math.asin(math.sqrt(s2)) / 2) ** 2
        f2_0 = float(r["f_N2"])
        x_fo = math.nan
        # below g ~ 1e-5 the conversions (rate ∝ g^4) never run, and the f_N2 difference of the
        # two scans is that of their production, not a depletion
        if k in d4 and float(r["y"]) >= G_DEP_MIN:
            f2_4 = float(d4[k]["f_N2"])
            if 0 < f2_4 < 1 and 0 < f2_0 < 1:
                x_fo = (math.log((1 - f2_4) / f2_4) - math.log((1 - f2_0) / f2_0)) / 1e-4
        m_dep = th1sq * m * x_fo if x_fo > 0 else math.nan
        dst = 10 ** np.interp(math.log10(float(r["y"])), lg, ld) if len(lg) else math.nan
        m_prod = th1sq * m * (1 + dst) / dst
        rows.append((m, float(r["y"]), s2, th1sq * m / M_NU, x_fo, m_dep * 1e9, m_prod * 1e9))
    os.makedirs("tmp/mnu_threshold", exist_ok=True)
    with open("tmp/mnu_threshold/thresholds.csv", "w") as io:
        io.write("m_N,y,sin2_2theta,delta_at_0.1eV,x_N_fo,m_nu_dep_eV,m_nu_prod_eV\n")
        for row in sorted(rows):
            io.write(",".join(f"{v:.5g}" for v in row) + "\n")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    decades = sorted({y for _, y, *_ in rows if abs(math.log10(y) - round(math.log10(y))) < 1e-6})
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(decades)))
    for y, c in zip(decades, cols):
        pts = sorted((m * 1e6, dep, prod) for m, yy, _, _, _, dep, prod in rows if yy == y)
        m_ = np.array([p[0] for p in pts])
        if math.log10(y) < lg[0] - 1e-3:
            continue   # below the smallest g of the threshold study: not extrapolated in the figure
        ax.loglog(m_, [p[1] for p in pts], color=c, ls="--")
        ax.loglog(m_, [p[2] for p in pts], color=c, ls="-", label=rf"$g = 10^{{{round(math.log10(y))}}}$")
    ax.axhline(0.1, color="k", lw=0.8)
    ax.text(1.05, 0.12, r"$m_\nu = 0.1$ eV (current)", fontsize=8)
    for v, lab in ((0.05, r"$\sqrt{\Delta m^2_{31}}$"), (0.0087, r"$\sqrt{\Delta m^2_{21}}$")):
        ax.axhline(v, color="0.5", lw=0.6, ls=":")
        ax.text(200, v * 1.15, lab, fontsize=7, color="0.4", ha="right")
    ax.set_xlabel(r"$m_{N_1}$ [keV]")
    ax.set_ylabel(r"$m_\nu$ below which the splitting matters [eV]")
    ax.set_title(r"solid: relic root moves by 1\%;  dashed: $N_2$ depleted by $1/e$", fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.set_xlim(0.6, 300)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out + ".pdf")
    print("written", out + ".pdf and tmp/mnu_threshold/thresholds.csv")
    for y in decades:
        sel = [r for r in rows if r[1] == y]
        dep = [r[5] for r in sel if math.isfinite(r[5])]
        prod = [r[6] for r in sel if math.isfinite(r[6])]
        print(f"g = {y:.0e}: delta(0.1 eV) {min(r[3] for r in sel):.2g} - {max(r[3] for r in sel):.2g}; "
              f"m_nu,dep {min(dep, default=math.nan):.2g} - {max(dep, default=math.nan):.2g} eV; "
              f"m_nu,prod {min(prod, default=math.nan):.2g} - {max(prod, default=math.nan):.2g} eV")


if __name__ == "__main__":
    main()
