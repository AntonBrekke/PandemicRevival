"""
Late decays against the relic roots: summary numbers and figures (from src/run_late_decays.jl).

    python plot/late_decays.py [tmp/combined_scan] [tmp/late_decays/late_decays.csv]
                               [tmp/mphi_scans/mphi1.5/late_decays.csv] [figures/late_decays]

Uses the bounds of the money plot (relic_scan_theta.py) for "allowed". Prints
1. N1 -> 3 nu against the invisible-decay bound f_N1/tau < 1/(246 Gyr), at all roots;
2. the X-ray reach: q = f_N1 sin^2 2theta / L(m) at the allowed roots (q = 1 is the current
   limit; 1/q is the improvement needed to probe a root), per g;
3. lifetimes of A' and phi (decays into the dark sector);
4. the long-lived phi of m_phi = 1.5 m_N (phi -> N1 nu): lifetime, coupling at decay, warm N1;
5. m_phi < m_N (Sec. VI of the draft), with the m_phi = 100 roots as a proxy: N1 -> phi nu,
   the free streaming of that phi, and phi -> nu nu against the decaying-DM bounds.
Writes <out>_xray_reach.pdf (q at the allowed roots) and <out>_light_phi.pdf (fate of each root
for m_phi < m_N).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relic_scan_theta import (load_roots, load_lyman_alpha, scaled_xray_limit, load_xray_limit, read_csv, key,
                              LAMBDA_FS_MAX_MPC, R_S_MAX_MPC, SIGMA_M_MAX, M_A_OVER_M_N, M_LIM, S_LIM)
from self_interaction import sigma_over_m

SEC_PER_GYR = 3.15576e16
TAU_INV_BOUND = 246 * SEC_PER_GYR      # Planck + lensing + BAO, arXiv:2205.05636
R_LIGHT = (0.1, 0.3, 0.5, 0.7, 0.9)

args = sys.argv[1:]
scan_dir = args[0] if len(args) > 0 else "tmp/combined_scan"
ld_path = args[1] if len(args) > 1 else "tmp/late_decays/late_decays.csv"
ld15_path = args[2] if len(args) > 2 else "tmp/mphi_scans/mphi1.5/late_decays.csv"
out = args[3] if len(args) > 3 else "figures/late_decays"

roots = load_roots(scan_dir)
lam, r_s, mix = load_lyman_alpha(scan_dir)
sig_m = {k: sigma_over_m(k[0] * 1e-6, k[1], M_A_OVER_M_N, *mix[k]) for k in roots if k in mix}
m_x, s_x = scaled_xray_limit(roots, mix)
xray_b = lambda m: np.exp(np.interp(np.log(m), np.log(m_x), np.log(s_x), left=np.inf, right=np.inf))
x_excl = {k: roots[k] > xray_b(k[0]) for k in roots}
other_excl = {k: lam.get(k, 0.0) > LAMBDA_FS_MAX_MPC or r_s.get(k, 0.0) > R_S_MAX_MPC
              or sig_m.get(k, 0.0) > SIGMA_M_MAX for k in roots}
allowed = {k for k in roots if not x_excl[k] and not other_excl[k]}
ld = {key(r): r for r in read_csv(ld_path)}
m_L, L = load_xray_limit()
L_of = lambda m: np.exp(np.interp(np.log(m), np.log(m_L), np.log(L)))
print(f"{len(roots)} roots, {len(allowed)} allowed by X-rays, Lyman-alpha and sigma/m")

# 1. invisible decays N1 -> 3 nu
R3 = {k: ld[k]["f_N1"] * TAU_INV_BOUND / ld[k]["tau_3nu"] for k in roots if k in ld}
kmax = max(R3, key=R3.get)
print(f"\n1. N1 -> 3nu: f_N1 (246 Gyr)/tau_3nu at most {R3[kmax]:.2g} (all roots; m = {kmax[0]:.3g} keV, g = {kmax[1]:.3g}),"
      f" {max(R3[k] for k in allowed):.2g} at the allowed ones; min tau_3nu = {min(ld[k]['tau_3nu'] for k in roots if k in ld):.2g} s")

# 2. X-ray reach
q = {k: ld[k]["f_N1"] * roots[k] / L_of(k[0]) for k in allowed}
qs = np.array(list(q.values()))
print(f"\n2. X-ray reach at the {len(q)} allowed roots: q = f_N1 s2 / L from {qs.min():.2g} to {qs.max():.2g}")
for thr in (0.5, 0.1, 0.01):
    print(f"   q > {thr:5g} (an improvement by {1 / thr:g}x probes it): {np.mean(qs > thr):.0%} of the allowed roots")
print("   per g (whole decades): allowed masses, q range")
for e in range(-8, 0):
    ks = sorted(k for k in q if abs(np.log10(k[1]) - e) < 1e-6)
    if ks:
        print(f"     g = 1e{e:+d}: m = {ks[0][0]:6.3g}-{ks[-1][0]:6.3g} keV, q = {min(q[k] for k in ks):.2g}-{max(q[k] for k in ks):.2g}")
tg = min(ld[k]["tau_gamma"] / ld[k]["f_N1"] for k in allowed)
print(f"   shortest effective nu gamma lifetime (tau_gamma/f_N1) allowed: {tg:.2g} s")

# 3. A' and phi
print(f"\n3. tau_A' at most {max(ld[k]['tau_A'] for k in roots):.2g} s, tau_phi at most {max(ld[k]['tau_phi'] for k in roots):.2g} s"
      " (m_phi = 100 m_N; decays to N1 N2, N2 nu and A'A')")

# 4. long-lived phi at m_phi = 1.5 m_N
if os.path.exists(ld15_path):
    d = read_csv(ld15_path)
    ok = np.isfinite(d["tau_phi"])
    coupled = (d["x_c"] < d["x_phi_dec"]) & (d["x_phi_dec"] < d["x_kd"])
    print(f"\n4. m_phi = 1.5 m_N ({ok.sum()} roots): tau_phi = {d['tau_phi'][ok].min():.2g}-{d['tau_phi'][ok].max():.2g} s,"
          f" m/T_nu at decay = {np.nanmin(d['x_phi_dec']):.3g}-{np.nanmax(d['x_phi_dec']):.3g};"
          f" dark sector still kinetically coupled at the decay at {coupled.sum()} roots")
    F = d["F_phi"]
    print(f"   largest phi share F_phi = {np.nanmin(F):.2g}-{np.nanmax(F):.2g};"
          f" free streaming of the N1 from phi decays = {np.nanmin(d['lambda_fs_phi_dec_Mpc']):.2g}-{np.nanmax(d['lambda_fs_phi_dec_Mpc']):.2g} Mpc;"
          f" Delta N_eff <= {np.nanmax(d['dNeff_phi_dec']):.2g}")
    big = (d["lambda_fs_phi_dec_Mpc"] > LAMBDA_FS_MAX_MPC) & ~coupled
    print(f"   lambda_fs > {LAMBDA_FS_MAX_MPC} Mpc and decoupled at the decay: {big.sum()} roots"
          + ("" if not big.any() else f" (F_phi there {np.nanmin(F[big]):.2g}-{np.nanmax(F[big]):.2g})"))

# 5. m_phi < m_N
print("\n5. m_phi < m_N (proxy: the m_phi = 100 roots). Roots not excluded by Lyman-alpha/sigma/m"
      " (X-rays no longer apply: N1 decays early):")
cand = [k for k in roots if not other_excl[k] and k in ld]
fate = {}
for r in R_LIGHT:
    s = f"_r{r:.1f}".replace(".", "")   # genfromtxt drops the "." of the column names
    f = {}
    for k in cand:
        row = ld[k]
        if row["tau_N1phinu" + s] > 0.01 * TAU_INV_BOUND:
            f[k] = "N1 long-lived"
        elif row["lambda_fs_phi" + s + "_Mpc"] > LAMBDA_FS_MAX_MPC:
            f[k] = "hot phi"
        elif row["ddm" + s] != "allowed":
            f[k] = "phi->nunu " + row["ddm" + s]
        else:
            f[k] = "allowed"
    fate[r] = f
    tN = np.array([ld[k]["tau_N1phinu" + s] for k in cand])
    tp = np.array([ld[k]["tau_phinunu" + s] for k in cand])
    cnt = {c: sum(1 for v in f.values() if v == c) for c in sorted(set(f.values()))}
    print(f"   m_phi/m_N = {r}: tau(N1->phi nu) = {tN.min():.2g}-{tN.max():.2g} s, tau(phi->nu nu) = {tp.min():.2g}-{tp.max():.2g} s; {cnt}")
    ga = sorted({k[1] for k, v in f.items() if v == "allowed"})
    if ga:
        print(f"      allowed at g = {ga[0]:.3g}-{ga[-1]:.3g}, m = {min(k[0] for k, v in f.items() if v == 'allowed'):.3g}-"
              f"{max(k[0] for k, v in f.items() if v == 'allowed'):.3g} keV")

# figures
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4.5))
ks = sorted(q)
sc = ax.scatter([k[0] for k in ks], [roots[k] for k in ks], c=np.log10([q[k] for k in ks]), cmap="viridis",
                s=10, vmin=-4, vmax=0)
ax.loglog(m_x, s_x, "k-", lw=1.2, label="X-ray limit / $f_{N_1}$")
for fac, ls in ((10, "--"), (100, ":")):
    ax.loglog(m_x, s_x / fac, "k", ls=ls, lw=0.8, label=f"limit / {fac}")
ax.set_xlim(*M_LIM); ax.set_ylim(*S_LIM); ax.minorticks_off()
ax.set_xlabel(r"$m_{N_1}$ [keV]"); ax.set_ylabel(r"$\sin^2 2\theta_1$")
fig.colorbar(sc, ax=ax, label=r"$\log_{10}\, f_{N_1}\sin^2 2\theta_1 / L(m)$")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("allowed roots: distance to the X-ray limit", fontsize=9)
fig.tight_layout(); fig.savefig(out + "_xray_reach.pdf"); plt.close(fig)

colors = {"allowed": "#1b9e77", "hot phi": "#d95f02", "N1 long-lived": "#7570b3", "phi->nunu neff": "#e7298a",
          "phi->nunu cmb": "#a6761d", "phi->nunu late": "#666666"}
shown = (0.1, 0.5, 0.9)
fig, axs = plt.subplots(1, len(shown), figsize=(12, 4), sharey=True)
for ax, r in zip(axs, shown):
    for c, col in colors.items():
        ks = [k for k, v in fate[r].items() if v == c]
        if ks:
            ax.scatter([k[0] for k in ks], [roots[k] for k in ks], s=8, color=col, label=c)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*M_LIM); ax.set_ylim(*S_LIM); ax.minorticks_off()
    ax.set_title(rf"$m_\phi = {r}\,m_{{N_1}}$", fontsize=10)
    ax.set_xlabel(r"$m_{N_1}$ [keV]")
axs[0].set_ylabel(r"$\sin^2 2\theta_1$ (roots at $m_\phi = 100\,m_N$)")
axs[0].legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(out + "_light_phi.pdf"); plt.close(fig)
print(f"\nwritten {out}_xray_reach.pdf, {out}_light_phi.pdf")
