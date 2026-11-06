# m_A' = 50 m_N, m_phi = 150 m_N, m_nu = 0.01 eV (2026-11-06, coarse scan)

Tables: `tmp/rate_tables/rate_tables_mA50.0_mphi150.0.jls`. m_phi = 100 m_N would sit exactly at the
singular phi -> A'A' threshold (2 m_A'); 150 keeps it open with the quartic lambda = g^2 (m_phi/m_A')^2/2
= 4.5 g^2 perturbative up to g ~ 1.7.

## Checks before the scan
- Perturbativity (`test/check_perturbativity.jl 50 150`): the phi width binds first, Gamma_phi/m_phi = 10%
  at g ~ 1.46 (lambda = 4 pi at 1.67). Top coupling g = 1.333 (Gamma_phi/m_phi 8%, lambda/4pi 0.63).
- Tables (`TABLE_MA=50 TABLE_MPHI=150 TABLE_ALL=1 test/test_rate_tables_eval.jl`): all 16 reactions pass.
  Narrow width vs Breit-Wigner (`test/check_narrow_width.jl`): <= 0.25% up to g = 1.33. Fluid quadrature
  (`test/check_legendre_ratio.jl`): 4e-9.
- Pilot roots (`pilot/`): reduced network 15/15, fluid 24/24 up to 1 MeV and g = 1.333; network vs fluid
  at 10.6 keV, g = 1e-3: 0.1%. The kappa~ fluid of branch reduced-network failed here once the N turn
  degenerate; the fluid is back to rho.
- Benchmarks at 11 pilot roots (`benchmarks/`, `check_benchmarks.py`): thermal from the start and
  mu_N1 = mu_N2 (N1 N1 <-> N2 N2 >= 1e3 H) during growth at every fluid root; <= 6% of Omega made before
  the end of DW production; N nearly degenerate at large g (gap 0.003 at 10.6 keV, g = 1).
- At large g the dark sector thermalises at T_dw with mu ~ 0 and T_N/T_nu ~ 1e-4 - 1e-3 (A' is light
  then), i.e. a cold relativistic gas: MB (network) stops, the fluid applies.
- Portal 2 -> 2 rates (MB tables, no Pauli blocking of the final N): `portal_scale/`. Root shift for
  portal rates x0.5 / x0.25: 10.6 keV g = 0.1: +1% / +1.5%; 10.6 keV g = 1: x1.9 / x3.6; 300 keV g = 1:
  x1.4 / x1.7. With f_N ~ 0.5 at those roots the g ~ 1 roots are likely too low by up to that much.

## Coarse scan
13 masses 0.6 - 754 keV, half-decades of g from 1e-8, plus 1.333 (`make_coarse_seeds.py`); couplings with
sigma/m > 1 cm^2/g at any mixing (f_N2 = 1/2) more than a half-decade above the bound left out.
Upper mass: the X-ray limit / f_N1 closes the region at ~670 keV for g = 1.333 (pilot roots); above
1.022 MeV N1 -> nu e+ e- opens, which the code does not include.

- `network_scan/`: 130/130 + 8 points where the fluid only bracketed the root (`network_scan_bracketed/`,
  appended): 138 roots.
- `fluid_scan/`: 73 converged, 10 bracketed (Omega jumps at the take-off; excluded as in the other scans).
- `combined/`: 138 network + 73 fluid, 2 failed; Lyman-alpha for all 211.

    python plot/relic_scan_theta.py tmp/mA50_scan/combined figures/mA50/relic_contours --mA 50 --mlim 1 800 \
        --slim 1e-24 1e-8 --g-pert 1.333 [--all-points]

`--g-pert 1.333` hatches the couplings above the largest scanned one (below its line of roots) as not
computed. Allowed region: ~7 - 600 keV, g ~ 1e-3 - 1, closed by self-interactions below, X-rays above and
at high mass, Lyman-alpha at low mass. Stopped at the coarse scan (user, 2026-11-06): the g ~ 1 roots
carry the Pauli-blocking systematic above.

Commands: `run_pilot.sh`, `run_pilot_benchmarks.sh`, `run_portal_scale.sh`, then the network scan
(`src/run_network_scan.jl <tables> tmp/mA50_scan/network_scan tmp/mA50_scan/seeds/network_seeds.csv`,
`run_network_lyman_alpha.jl`) and `run_coarse_fluid.sh`.
