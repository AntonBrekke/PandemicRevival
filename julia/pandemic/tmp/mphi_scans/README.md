# Coarse scans at smaller m_phi (2026-10-16)

Test of how the dark-Higgs mass changes the relic contours. Main scan: m_phi = 100 m_N
(tmp/network_scan, tmp/fluid_scan). Here m_phi/m_N = 50, 10, 5.5 and 1.5, with
m_A' = 2.5 m_N throughout.

- 5.5 instead of 5: m_phi = 5 m_N = 2 m_A' is exactly the phi -> A'A' threshold, where the s-channel
  phi pole of N1 nu -> A'A' sits at the kinematic endpoint and the zero-width table integral
  diverges (NaN). 5.5 keeps phi -> A'A' open (lifetime <= 1e-6 s), like 10 and 50.
- 1.5 is inside the draft's light window m_N1 < m_phi < 2 m_N1 (and below m_A'): phi -> A'A'
  and phi -> N1 N1 are closed, phi decays only to N1 nu (lifetime 1e3-1e5 s, i.e. around
  x ~ 1-100); each phi ends as one N1 (`phi_weight`). The fluid solver does not apply
  (mu_phi = 2 mu_A' needs phi <-> A'A'), and the network becomes degenerate from
  g ~ 3e-3 - 1e-2: only the network part (g <~ 1e-3) exists.

Grid: 10 masses (every third of the main scan's 26, plus 10.57 keV) and half-decades of g,
seeded with the m_phi = 100 roots (`plot/mphi_seeds.py` -> seeds/). The network runs up to
g = 3.16e-3 (the overlap with the fluid; up to 0.1 for 1.5), the fluid from 1e-3.
Before scanning, every table set passed test/test_network.jl and test/check_mphi_validity.jl
(single solves at the m_phi = 100 roots; network and fluid agree to 1-8% in Omega h^2 at
g = 1e-3 and 3.16e-3 for 50, 10, 5.5).

    ./tmp/mphi_scans/run_coarse.sh                          # all scans, ~3 h (log: run_coarse.log)
    python plot/combined_scan_dir.py tmp/mphi_scans/mphi10/combined 1e-3 tmp/mphi_scans/mphi10
    python plot/relic_scan_theta.py tmp/mphi_scans/mphi10/combined figures/relic_contours_mphi10
    python plot/mphi_compare.py                             # roots vs m_phi = 100

Layout: mphi<r>/{network_scan,fluid_scan}/ (roots, lyman_alpha.csv, logs), mphi<r>/combined/.
