"""Run spi_channel.py for every SPI channel and seed in parallel (samples -> samples/, ~10 min at 16).

    python fit_channels.py [NPROC] [NSEEDS]       # defaults 16, 8; needs threeML, astromodels, emcee

Finished (channel, seed) pairs are skipped, so an interrupted run can be resumed.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

here = os.path.dirname(os.path.abspath(__file__))
nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
out = os.path.join(here, "samples")
os.makedirs(os.path.join(out, "logs"), exist_ok=True)


def run(job):
    c, s = job
    if os.path.exists(os.path.join(out, f"c{c:02d}_s{s}.npy")):
        return
    with open(os.path.join(out, "logs", f"c{c:02d}_s{s}.log"), "w") as log:
        subprocess.run([sys.executable, "-I", os.path.join(here, "spi_channel.py"), here, str(c), out, str(s)],
                       stdout=log, stderr=log, cwd=here)


with ThreadPoolExecutor(nproc) as ex:
    list(ex.map(run, [(c, s) for s in range(1, nseeds + 1) for c in range(24)]))
print("samples:", len([f for f in os.listdir(out) if f.endswith(".npy")]))
