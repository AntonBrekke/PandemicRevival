# m_A' = 3 m_N, splitting at m_nu = 0.01 eV (2026-11-05, ThinkStation)

Lyman-alpha post-processing of the m_A' = 3 roots of `tmp/mA3_scan` at m_nu = 0.01 eV instead of 0.1 eV
(`run_mnu0.01.sh`, 11 fluid workers, 1 h 48 min). The roots are copied unchanged, not
re-searched; `combined/` is the result of `plot/combined_scan_dir.py`.

## Result against 0.1 eV (`tmp/mA3_scan/combined`)

- 1274 points in both, identical roots, no failed lengths, no point changes its Lyman-alpha status.
- delta_m x10 (max 2.2e-3, at 0.6 keV, g = 1e-8). lambda_fs,kd and r_s: median ratio 1, extremes
  0.989 - 1.012, which is at the level of the fluid's ODE noise. f_N2: 0.998 - 1.003.
- Roots not re-searched: by `tmp/mnu_threshold/thresholds.csv` (m_A' = 2.5 roots, 1% shift of the root)
  14 of 1274 roots would move by more than 1% at 0.01 eV, all at m_N <= 1.3 keV and g < 1e-7, with
  sin^2 2theta >= 3e-8.

## Commands (from julia/pandemic)

    ./tmp/mA3_scan/mnu0.01/run_mnu0.01.sh 11
    conda run -n pandemic python plot/relic_scan_theta.py tmp/mA3_scan/mnu0.01/combined figures/mA3/relic_contours_mnu0.01_bp \
        --mA 3 --bp tmp/mA3_scan/bp_round/bps.csv [--xray-proj]
