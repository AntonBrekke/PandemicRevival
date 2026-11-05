# Reduced network checks (2026-11-05)

Checks of the reduced network (src/network_reduced.jl: A' and phi eliminated by
mu_A' = mu_N1 + mu_N2, mu_phi = 2 mu_A') against the stored roots of every scan, made before the
base scans were regenerated with it. Results and their interpretation:
notes/implementation_summary.tex, Sec. "The reduced network".

    python tmp/reduced_network/make_seeds.py    # seeds/<set>_<solver>.csv from the stored combined roots
    bash tmp/reduced_network/run.sh             # roots (+ Lyman-alpha for base) per set, ~1-3 h
    python tmp/reduced_network/compare.py       # shifts per decade of g, worst points, Lyman-alpha
    bash tmp/reduced_network/regen_base.sh      # tmp/network_scan{,_fine}, their lyman_alpha, combined_scan

`<set>_<solver>/network_roots.csv`: the s2_reduced column is the stored root (the seed), s2_network
the reduced network's. Sets: base (m_phi = 100 m_N), mphi50/10/5.5 (coarse), mA3, delta1e-4;
`base_fluid` runs the reduced network from the fluid roots (g >= 1e-3), to see how far up in g it
applies. run.sh passes `--reduced`, the default since; regen_base.sh uses the default.

All network roots converge; the root shift is 0.3% (95%), at most 4% (300 keV, g = 7.5e-4), where
the full network was wrong (fixed since: `reservoir_correction!` in src/network.jl).
