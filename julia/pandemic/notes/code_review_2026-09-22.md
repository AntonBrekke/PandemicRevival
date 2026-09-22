# Julia codebase review — 2026-09-22

Requested by the user during a long autonomous session (scan extension +
physics review). Scope: `julia/pandemic/src/`, focused on correctness of the
physics and robustness of the scan machinery, with lighter comments on code
quality. I did not touch anything not listed below as "fixed".

## Fixed this session (see git log on `worktree-lyman-alpha`)

### 1. `check_plateau_z` checked the wrong quantity (real correctness bug)

`relic_scan_z.jl`'s `check_plateau_z` judged convergence from the raw solver
state `Y_n = exp(sol[1,:])` (with `n ≡ n_N1 + n_N2 + 2 n_A`), not from
`Omega h^2` itself. Whenever A' is a non-negligible fraction of `n` right up
to `x = m_N/T_nu = 100` (the end of every solve), the N/A' split can still be
visibly relaxing while the *physically relevant*, mass-weighted `Omega h^2`
has already frozen (A' contributes little to it once
`fac_n_A m_A Y_A ≪ m_N1 Y_N1`). This is exactly what happens right after
pandemic thermalization at large `y`: I traced a concrete case (1 keV,
y = 1e-2) where the sharp "spike"/thermalization at `x ≈ 3e-3` leaves the
sector oscillating in `Y_n` at the few-percent level all the way to `x = 100`
while `Omega h^2` has been stable to 5 significant figures since long before.

The consequence was worse than a slightly-too-strict convergence check:
`theta_point` (`relic_scan_theta_z.jl`) returns `NaN` for the residual
whenever `!converged`, and both `march_to_bracket` and `coarse_bracket`
require *two adjacent finite* residuals to see a sign change. A single
plateau false alarm can therefore delete an entire bracket, and I found this
was silently producing spurious `:NoRoot` results at large `y`
(reproduced at `m_N` = 100, 160, 250 keV, `y = 0.1`, all three "no root"
before the fix). This is not a cosmetic issue: any (m_N, y) column affected
this way is simply *missing* from the scan output, with no error raised.

Fix: `check_plateau_z` now checks `Omega h^2(z)` (via a new shared helper
`omega_h2_of(pan, u, z)`, also used by `final_omega_h2_z`) over the same tail
window, which is both more physically correct and *at least as strict* for
genuine non-convergence (Omega h^2 cannot plateau while its dominant
ingredient is still evolving). As a second line of defense for the rare case
where Omega h^2 genuinely has not settled by `x = 100` (dark sector
thermalizes/relaxes very late), `theta_point` now retries with a longer
integration (`x_end` = 300, then 1000 — costs 1.3-5x per retry, only paid
when needed) before giving up. `DodelsonWidrow`'s `x_end` (previously
hardcoded to 100) is now an optional keyword for this purpose.

**Action for you:** the old Y_n-based check was silently dropping columns at
high y in any past scan (with `first_root_only=true`, a dropped bracket just
means the freeze-in branch appears to vanish, and the driver would print
"no root" and move on) — if you have relied on "no root at this (m,y)" to
mean "the freeze-in branch genuinely doesn't reach the target" anywhere
(rather than re-deriving it), it is worth re-running those specific points.

### 2. `C_n_AA`'s `/4` and `-2, -2` (TODO since 01.06.26 / 15.07.26) — verified correct

`coll_12_34_legendre`'s docstring states plainly: "No symmetry factors for
identical particles are included" — it returns the *raw*, unsymmetrized
integral for a generic (possibly-identical) 4-particle process, labeled as if
distinguishable. I independently re-derived the correct combination from
first principles (standard reaction-rate bookkeeping for the A'A'↔N_iN_i
channel, cross-checked two ways: against the Maxwell-Boltzmann limit quoted
in the docstring itself with the standard "extra 1/2 for an identical
*final*-state pair, no extra factor for an identical *initial*-state pair
because it cancels against the dn/dt ∝ n² multiplicity" convention used for
e.g. self-conjugate WIMP annihilation dn/dt = -⟨σv⟩n²; and against the
explicit particle-number-flow identity already documented in `C_n`'s
docstring for the `n ≡ n_N1+n_N2+2n_A` convention). Both give exactly
`dn/dt|_{channel} = -2 × (raw/4)`, i.e. the code as written. **You can remove
both TODO comments** (lines 261-262) — I did not do this myself since it's a
comment-only change adjacent to code I'm not otherwise touching, but the
derivation is in the session log if you want to check it or add it as a
permanent comment.

### 3. `C_AA_N2N2 = C_AA_N1N1` (16.09.27 TODO) — confirmed harmless *today*, but a landmine

Every current run uses `m_N1 = m_N2` exactly (every `PandemolatorZ`
construction in the codebase builds `N1`/`N2` with the identical mass), so
reusing the N1N1 computation for N2N2 is mathematically *exact*, not an
approximation, right now. It only becomes wrong the moment someone sets
`m_N1 ≠ m_N2` (needed for any δm ≠ 0 physics: the mass-splitting velocity
threshold for self-interactions, a real N1↔N2 kinetic-decoupling rate, etc.)
without also implementing the real `coll_12_34_legendre(..., N2, N2, ...)`
call (already sketched, commented out, right below). **Before any
mass-splitting work starts, this needs to be uncommented and given its own
`sq_amp_AA_NN` valid for m_N1 ≠ m_N2** — the paper's eq. (A9) is stated only
in the m_N1 = m_N2 limit "for displaying purposes" but the full expression is
used in the paper's own rate calculations, so it exists and just needs
porting.

### 4. `fac_n_A` (01.07.26 / 16.07.26 TODOs, `pandemolator.jl` and its z-counterpart)

The `A.m > N1.m + N2.m ? 2 : 1` switch is choosing between two different
"exactly conserved under the dominant channel" quantities: weight 2 makes
`n ≡ n_N1+n_N2+2n_A` exactly conserved under A'A'↔N_iN_i and A'↔N1N2 (both
convert one A' into *two* DM particles or vice versa — I verified this is
the same `2` I derived independently in item 2 above), while weight 1 would
be the natural choice if A'↔N2ν were the only relevant channel. Given the
paper always uses `m_A' = 2.5 m_N` (so `A.m > N1.m+N2.m` always holds, the
weight-2 branch is always taken), the `else` branch is confirmed dead code
in every scenario studied so far, matching the "[16.07.26] Not used" comment
already there. Not a bug, but if `m_A'/m_N` is ever brought close to or below
2, both channels can matter simultaneously and neither fixed weight exactly
conserves `n`; this would need revisiting.

### 5. `pandemolator_common.jl`'s general `C_n_3_12`/`C_rho_3_12` docstring has the blocking-factor sign backwards (documentation only)

This is very likely *why* the `coll_3_12.jl:128,505` "double check sign in
front of k" TODOs below were never closed: two descriptions of the same
convention disagree, and I don't think anyone went back to work out which
one was right. `Particle.k` is `+1` for fermions, `-1` for bosons
(`utils.jl`). Standard kinetic theory needs Pauli blocking `(1-f)` for a
fermion in the final state and Bose enhancement `(1+f)` for a boson, i.e.
uniformly `(1 - k·f)` for this sign convention. `pandemolator_common.jl`'s
docstring (lines 313, 321, 348, 352, 356-357 — one repeated explanatory
block) instead writes the general formula and the `C_n_3_12`/`C_rho_3_12`
"Code:" line with `(1 + k_j·f_j)`, which is backwards for *both* statistics
given this `k`. The actually-executed code
(`coll_3_12_ker`'s `dist_fac_3_12`/`dist_fac_12_3`, `(1. - p.p1.k*f1)` etc.)
already uses the correct `(1-k·f)`, matching the old Python reference
(`C_res_vector.ker_C_rho_3_12_E2`: `f1*f2*(1-k3*f3) - f3*(1-k1*f1)*(1-k2*f2)`)
exactly. So this is a **documentation bug, not a physics bug** — but it's
worth fixing the docstring text (`1+k` → `1-k`, 6 occurrences) precisely
*because* it's what's making the two genuinely-open `coll_3_12.jl` sign
TODOs hard to close: right now anyone comparing the code to that docstring
would (wrongly) conclude the code has the sign backwards.

## Not independently re-verified (flagged by TODOs already in the code, still open)

I did not have time to re-derive these from scratch; I list them so they are
not lost among the ones above.

- `coll_3_12.jl:128,505` — "Double check sign in front of k and of total
  expression" in `coll_3_12_ker`'s `dist_fac`. Partially addressed by item 5
  above (the sign-in-front-of-k half): the code's `(1-k·f)` matches the old
  Python reference exactly, and is the textbook one, so I'm fairly confident
  this part of the TODO is fine. I did not separately re-derive the "sign of
  total expression" (i.e. whether `dist_fac_3_12 - dist_fac_12_3`, decay
  minus inverse-decay, is used with the right overall sign at the call site,
  as opposed to just the relative sign of the two terms) — this is the A'↔N2ν
  collision kernel used throughout the live solver (`A_N2nu_moments`), so
  it's worth prioritizing over the items below if you want it fully closed.
- `coll_3_12.jl:260,383` — "Check prefactors!" on the overall `pre = 1/(2^5
  pi^3)` normalization of `coll_3_12`/`coll_3_12_moments`. The same kind of
  cross-check I used for `C_n_AA` (compare against the textbook
  thermally-averaged-rate limit, or a direct numerical phase-space Monte
  Carlo for one simple test process) would settle this relatively quickly if
  you want it resolved.
- `pandemolator.jl:82,345` — "Why this factor?" / "Double-check the signs"
  on the T-parameterized solver's RHS. **This file appears to be superseded**
  by `pandemolator_z.jl` for all current work (same physics via
  `pandemolator_common.jl`, validated against it once in
  `test/test_pandemolator_z.jl`) — the equivalent line in the z-file
  (`du[1] = x * coll_n / (n * dx_dt)`) does *not* carry this warning, which
  suggests it may already have been resolved there and the T-file's comment
  is stale. Since both files are still present and someone could edit the
  wrong one, I'd suggest either deleting `pandemolator.jl` (keeping the test
  that validated the two agree, as a git-history pointer) or adding a
  one-line banner pointing to `pandemolator_z.jl` as the maintained version.
- `pandemolator.jl:237` — "This initial condition makes little sense. Ask
  Torsten." on a commented-out `T_N_0` line that isn't even used (the actual
  `ln_x_N_0` comes from the nonlinear solve just above it) — looks like
  dead code from an earlier attempt; safe to delete if the T-file is kept.

## Other observations (not from grepping TODOs)

- `run_relic_scan_theta_all.jl` and `run_lyman_alpha.jl` both build one
  `PandemolatorZ`/`TimeTempRelation` per worker and cache it across points on
  that worker (`_TT_REL[]`, `_PANS`) — good, this is why the per-point cost
  is dominated by the ODE solve rather than setup. Nothing to change.
- `ScanConfigThetaZ.log10s_max = -6.0` (i.e. the coarse-grid search never
  looks above `sin^2 2theta = 1e-6`) was set for the original 1.5-300 keV,
  y ≤ 1e-2 grid. It was not a problem for the wide low-mass/low-y scan I ran
  today (every root found was well below 1e-6), but if the grid is ever
  extended to very small masses *and* very small y at the same time (where
  the DW-dominated branch can need a larger mixing angle), it's worth
  sanity-checking that the true root isn't sitting above this ceiling before
  trusting a "no root" result there.
- The `DodelsonWidrow` constructor's new `x_end` keyword (this session's
  addition, item 1) has a **hardcoded x_end=1e2 default** matching the old
  behavior everywhere except the two call sites in `theta_point`'s retry —
  every other call site (`lyman_alpha.jl`, `run_abundance_history.jl`,
  `relic_scan.jl`, `relic_scan_mN_z.jl`, tests) is unaffected. I checked
  this explicitly (`grep -rn "DodelsonWidrow{"`) before making the change.
