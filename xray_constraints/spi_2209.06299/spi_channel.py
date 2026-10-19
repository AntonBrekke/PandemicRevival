"""DM_2gamma.ipynb (Zenodo 7984451; Calore et al. 2023, arXiv:2209.06299) for one SPI channel.

Same model, priors and sampler as the notebook; saves the posterior samples of the line Norm.
usage: python spi_channel.py DATA_DIR CHANNEL OUT_DIR [seed]
"""
import sys, os, json
import numpy as np
data, mass, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
np.random.seed(seed)
from astropy.io import fits
import astropy.units as u
from astromodels.functions.function import Function1D, FunctionMeta
from threeML import (OGIPLike, Cutoff_powerlaw, Powerlaw, Gaussian, PointSource, Model, DataList,
                     BayesianAnalysis, Log_uniform_prior, Truncated_gaussian, Uniform_prior, silence_warnings)
from scipy import integrate
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Positronium_Spectrum import Positronium_Spectrum
silence_warnings()

resp_f = os.path.join(data, "spectral_response.rmf.fits.gz")
eb = fits.open(resp_f)[3].data
Emin, Emax = np.array(eb["E_MIN"], float), np.array(eb["E_MAX"], float)


class DM_2gamma(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        Differential flux for decaying dark matter into 2 photons (box in the channel of E = m/2).
    parameters :
        mass :
            desc : Dark matter mass
            initial value : 100
            min : 61
            max : 16000
            fix : True
        Norm :
            desc : Flux normalization within 1 energy bin
            initial value : 1e-6
            min : 1e-8
            max : 1e-1
    """

    def _set_units(self, x_unit, y_unit):
        self.mass.unit = u.keV
        self.Norm.unit = 1 / (u.cm**2 * u.s * u.keV)

    def evaluate(self, x, mass, Norm):
        dflux = np.full(len(x), 1e-50)
        Emin_ = Emin[np.where(mass / 2 > Emin)[0][-1]]
        Emax_ = Emax[np.where(mass / 2 <= Emax)[0][0]]
        dflux[np.where(np.logical_and(x >= Emin_, x <= Emax_))] = Norm
        return dflux


ds = OGIPLike("ig_data", observation=os.path.join(data, "total_spectrum.fits.gz"), response=resp_f)
ds.set_active_measurements("c0-c24")
chan = int(mass)          # here: the channel index; the DM box depends on the mass only through it
Mass = 2 * np.sqrt(Emin[chan] * Emax[chan])
spec = (Cutoff_powerlaw(K=1e-5, piv=50, index=0, xc=7.) + Powerlaw(K=1e-5, piv=1000, index=-1.3)
        + Gaussian(F=1e-3, mu=1809, sigma=1.7) + Positronium_Spectrum(Flux=1.7e-3, f_Ps=1.0)
        + DM_2gamma(mass=Mass))
model = Model(PointSource("ig", l=0, b=0, spectral_shape=spec))
c = model.ig.spectrum.main.composite
c.index_1.fix = True; c.mu_3.fix = True; c.sigma_3.fix = True; c.mass_5.fix = True
c.K_1.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1e0)
c.xc_1.prior = Truncated_gaussian(F=1, mu=7, sigma=3, lower_bound=1, upper_bound=100)
c.K_2.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e0)
c.index_2.prior = Uniform_prior(lower_bound=-3, upper_bound=0)
c.F_3.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e0)
c.Flux_4.prior = Log_uniform_prior(lower_bound=1e-6, upper_bound=1e-1)
c.f_Ps_4.prior = Truncated_gaussian(F=1, mu=1, sigma=0.2, lower_bound=0, upper_bound=1)
c.Norm_5.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=1e-1)
bs = BayesianAnalysis(model, DataList(ds))
bs.set_sampler("emcee")
bs.sampler.setup(n_iterations=2000, n_burn_in=5000, n_walkers=20)
bs.sample(quiet=True)
smp = bs.results.samples
names = list(bs.results.optimized_model.free_parameters.keys())
iN = [i for i, n in enumerate(names) if n.endswith("Norm_5")][0]
assert iN == 7, names     # the notebook takes samples[7]
i = np.where(Mass / 2 <= Emax)[0][0]
assert i == chan
np.save(os.path.join(out, f"c{chan:02d}_s{seed}.npy"), np.asarray(smp[iN], np.float64))
print(chan, Mass, np.percentile(smp[iN], 95))
