"""Sterile-neutrino limit of Calore et al. (2209.06299) from the SPI line fits of fit_channels.py.

    python make_limit.py [samples_dir]     # -> channel_limits.dat, ../INTEGRAL_SPI_2209.06299.dat

The DM model of DM_2gamma.ipynb is a box of height Norm [1/(cm^2 s keV)] over the channel holding
E = m/2, so the 95% upper limit K on Norm depends on the channel only. With the paper's D-factor
D = 1.289e29 keV/cm^2 (|l|, |b| <= 47.5 deg) and the SPI resolution sigma = 0.5 keV of the notebook,
    Gamma_2gamma < K dE / (2 D p / (4 pi m)),   p = fraction of N(m/2, sigma) inside the channel,
i.e. the paper's flux formula integrated over the channel. The notebook omits the factor dE (its Gamma is in
1/(s keV)); with it, the limit reproduces Fig. 1 (red solid) of the paper. N -> nu gamma gives one
photon per decay, Gamma_nugamma = 2 Gamma_2gamma (as the paper's text says), and
Gamma_nugamma = 9 alpha G_F^2 m^5 sin^2 2theta / (1024 pi^4), the Majorana rate used by the other
limits and by julia/pandemic/src/late_decays.jl. The paper's Fig. 5 is a factor 2 below this
(it matches Gamma_2gamma instead of 2 Gamma_2gamma; see README).
Within a channel sin^2 2theta ~ m^-4 / p(m); the file samples it densely near the channel edges.
"""
import glob
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.special import erf

here = os.path.dirname(os.path.abspath(__file__))
sdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "samples")
eb = fits.open(os.path.join(here, "spectral_response.rmf.fits.gz"))[3].data
Emin, Emax = np.array(eb["E_MIN"], float), np.array(eb["E_MAX"], float)
D, SIG = 1.289e29, 0.5                      # keV/cm^2, keV
ALPHA, G_F, HBAR = 1 / 137.035999, 1.1663788e-5, 6.582119569e-25     # -, GeV^-2, GeV s
M_LO, M_HI = 61.0, 16000.0                  # keV: the paper's mass range


def gamma_nugamma_unit(m_kev):             # 1/s at sin^2 2theta = 1
    m = m_kev * 1e-6
    return 9 * ALPHA * G_F**2 * m**5 / (1024 * np.pi**4) / HBAR


rows = []
for c in range(len(Emin)):
    files = sorted(glob.glob(os.path.join(sdir, f"c{c:02d}_s*.npy")))
    if not files:
        continue
    per_seed = np.array([np.percentile(np.load(f), 95) for f in files])
    pooled = np.percentile(np.concatenate([np.load(f) for f in files]), 95)
    rows.append((c, Emin[c], Emax[c], pooled, per_seed.std(ddof=1) / np.sqrt(len(files)) / pooled, len(files)))
rows = np.array(rows)
np.savetxt(os.path.join(here, "channel_limits.dat"), rows, fmt=["%3d", "%8.1f", "%8.1f", "%.5e", "%.3f", "%2d"],
           header="channel  E_min [keV]  E_max [keV]  K_95 [1/(cm^2 s keV)] (pooled seeds)  "
                  "rel. MC error of K_95 (seed scatter / sqrt(n))  n_seeds")

m_all, s_all = [], []
for c, e0, e1, K, _, _ in rows:
    E = np.unique(np.concatenate([e0 + np.array([1e-6, 0.25, 0.5, 1, 1.5, 2, 3]) * SIG,
                                  np.geomspace(e0 + 3 * SIG, e1 - 3 * SIG, 12),
                                  e1 - np.array([3, 2, 1.5, 1, 0.5, 0.25, 0]) * SIG]))
    E = E[(E > e0) & (E <= e1)]
    m = 2 * E
    p = 0.5 * (erf((e1 - E) / (np.sqrt(2) * SIG)) - erf((e0 - E) / (np.sqrt(2) * SIG)))
    gamma_2g = K * (e1 - e0) / (2 * D * p / (4 * np.pi * m))
    s2 = 2 * gamma_2g / gamma_nugamma_unit(m)
    keep = (m >= M_LO) & (m <= M_HI)
    m_all.append(m[keep]); s_all.append(s2[keep])
m_all, s_all = np.concatenate(m_all), np.concatenate(s_all)
np.savetxt(os.path.join(here, "..", "INTEGRAL_SPI_2209.06299.dat"), np.column_stack((m_all * 1e-6, s_all)),
           fmt="%.6e", header=(
    "Calore, Dekker, Serpico, Siegert, MNRAS 520 (2023) 4167, arXiv:2209.06299: INTEGRAL/SPI 16 yr with a DM\n"
    "spatial template. Recomputed from the published data and notebook (Zenodo 10.5281/zenodo.7984451,\n"
    "DM_2gamma.ipynb: total-spectrum fit, 1D, 95% Bayesian UL) per SPI channel, pooled over seeds; see\n"
    "spi_2209.06299/README.md. Gamma_nugamma = 2 Gamma_2gamma = 9 alpha G_F^2 m^5 s2/(1024 pi^4) (Majorana).\n"
    "Unsmoothed: steps at the channel edges (E = m/2), weaker near the edges where the line straddles two.\n"
    "m_s [GeV]    sin^2 (2 theta)"))
print(f"{len(rows)} channels -> INTEGRAL_SPI_2209.06299.dat ({len(m_all)} points, "
      f"{m_all.min():.0f}-{m_all.max():.0f} keV); max MC error of K_95: {rows[:, 4].max():.1%}")
