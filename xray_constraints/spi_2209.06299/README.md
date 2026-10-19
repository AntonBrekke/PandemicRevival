# INTEGRAL/SPI template limit (Calore et al. 2023, arXiv:2209.06299) from the published data

`../INTEGRAL_SPI_2209.06299.dat` is recomputed here from the authors' data release instead of
being read off Fig. 5 of the paper.

Source: T. Siegert, Zenodo record 7984451 (doi:10.5281/zenodo.7984451, CC-BY-4.0):
`total_spectrum.fits.gz` (SPI 16 yr total spectrum, 30 keV - 8 MeV, 24 channels),
`spectral_response.rmf.fits.gz` and `Positronium_Spectrum.py` are copied unchanged;
`spi_channel.py` reproduces `DM_2gamma.ipynb` (same model, priors, emcee settings, fixed
parameters), with `DM_2gamma` rewritten to take explicit paths.

    python fit_channels.py [NPROC] [NSEEDS]   # env with threeML, astromodels, emcee; ~10 min (16, 8)
    python make_limit.py                       # -> channel_limits.dat, ../INTEGRAL_SPI_2209.06299.dat
    cd .. && python plot_xray.py               # -> overall_constraint.dat

(On the ThinkStation: conda env `spi`, `conda create -n spi python=3.11 && pip install threeml
astromodels emcee`; threeML 2.6.1, astromodels 2.6.0, emcee 3.1.6.) The posterior samples
(`samples/`, 60 MB) are not tracked; `channel_limits.dat` holds the per-channel result.

## Method, and where it departs from the notebook

- The DM line is a box over the channel holding E = m/2, so the 95% limit on its height K
  depends on the channel only. Each of the 24 channels is fitted once per seed (8 seeds) and the
  samples are pooled; the MC error of K_95 is listed in `channel_limits.dat`. (The notebook runs
  50 masses with one chain each; the 95th percentile then scatters by a factor 0.4-1.4 between
  seeds, and the published curve is smoothed over 3 grid points.)
- **Channel width.** The notebook's `Gamma = K_ub / A_int` lacks the channel width dE (it has
  units 1/(s keV)). With dE, the reproduced Gamma_2gamma agrees with the red solid line of the
  paper's Fig. 1 (median ratio 1.01 over 61-2000 keV, scatter 0.14 dex, as large as the MC noise
  of a single chain), so the paper includes it.
- **Simpson integral.** The notebook integrates the sigma = 0.5 keV Gaussian with 2000 Simpson
  points across the channel; above ~1 MeV the step exceeds sigma and the integral is wrong (even
  negative above 6 MeV). Here the Gaussian fraction inside the channel is taken from erf.
- **Recast to sin^2 2theta.** One photon per N -> nu gamma decay: Gamma_nugamma = 2 Gamma_2gamma
  (the paper's text), with Gamma_nugamma = 9 alpha G_F^2 m^5 sin^2 2theta / (1024 pi^4)
  = 1.36e-29/s (sin^2 2theta/1e-7)(m/keV)^5, the Majorana rate used by the other limits and by
  `julia/pandemic/src/late_decays.jl`. (The paper calls this formula Dirac and says Majorana
  widths are twice as large.) **The paper's Fig. 5 is a factor 0.49 (constant over the mass range)
  below this recast of its own Fig. 1**, so the curve previously extracted from Fig. 5 was 2x too
  strong in our convention.
- No smoothing: the limit has steps at the channel edges (sin^2 2theta ~ m^-4 inside a channel)
  and is weaker within ~1 keV of an edge in m, where the line straddles two channels.
