"""
lyman_alpha.jl

Free-streaming length of the dark matter N (= N1, N2) for the Lyman-alpha
bound, evaluated on a solved relic point.

PHYSICS (as in the Python code)
===============================
Same definitions as code/sterile_caller.py and data/dw/0612182_fs_length.py,
remapping the warm-DM Lyman-alpha limit as in Bringmann et al.
(arXiv:2206.10630):

    lambda_fs = ∫_{t_i}^{t(z_red)} <v>(t) / a(t) dt,    a = (s0 / s)^(1/3), a_today = 1,
    <v>       = ∫ d^3p f p/E / ∫ d^3p f               (mean speed of N),

integrated up to redshift z_red = `LYA_Z_RED_END` = 50 (defined, as in the
Python code, by T_SM / T0 - 1), and excluded if lambda_fs > `LYA_LAMBDA_FS_MAX_MPC`
= 0.24 Mpc.

DIFFERENCES TO THE PYTHON CODE
==============================
- No kinetic decoupling (option (i)): the solver assumes kinetic equilibrium
  of the dark sector throughout, so <v> is taken from the Fermi-Dirac
  distribution with the solved (T_N, xi_N) over the whole solution, from
  T_nu = T_dw (start of the solution) to x = m_N/T_nu = 100. The Python code
  instead started the integral at T_kd, with a frozen spectrum from there on,
  and added the sound horizon r_s before T_kd. There is therefore no r_s here.
- After the end of the solution the spectrum is frozen (free streaming,
  p ∝ 1/a) and redshifted to z_red, as the Python code did after T_kd.
- The integral is done by adaptive quadrature in z = log(m_N/T_nu) on the
  dense ODE output (breakpoints at the solver steps), and the momentum
  moments with `thermal_moment`, rather than Simpson on the time grid with a
  fixed 5000-point momentum grid at every time step.
- It is computed once per relic point, after the root finding
  (src/run_lyman_alpha.jl), not inside every solve of the root search.
"""

include(joinpath(@__DIR__, "relic_scan_z.jl"))

# Free streaming is followed until this redshift (sterile_caller.py: z = 50).
const LYA_Z_RED_END = 50.0
# Lyman-alpha limit on lambda_fs in Mpc (Bringmann et al. 2206.10630, as in
# code/sterile_res/plotter.py).
const LYA_LAMBDA_FS_MAX_MPC = 0.24

"""Speed q/E of a particle of mass m and momentum q (exactly 1 for m = 0)."""
_speed(q, m) = q / sqrt(q * q + m * m)

"""
    mean_speed(p::Particle, temp, gap; r=1.0)

Mean speed <|v|> of particle `p` with occupation f = 1/(e^((E-m)/temp + gap) + k)
per dof, after all momenta have been scaled by `r` (free streaming with f
fixed in comoving momentum: r = a_f/a). With r = 1, the value in kinetic
equilibrium at (temp, gap).

In the Maxwell-Boltzmann limit the ratio does not depend on the gap, which is
therefore capped at `m_T_r_MB` (a relative change below e^-m_T_r_MB): for
large gaps both moments would otherwise underflow to 0.
"""
function mean_speed(p::Particle, temp, gap; r=1.0)
    g = min(gap, m_T_r_MB)
    # d^3p = 4 pi p^2 dp = 4 pi E p dE
    num = thermal_moment(p, temp, g, (E, mom) -> E * mom * _speed(r * mom, p.m))
    den = thermal_moment(p, temp, g, (E, mom) -> E * mom)
    return num / den
end

"""
    N_temp_gap(pan, u)

(T_N, (m_N - mu_N)/T_N) of N1 (= N2) in the solver state u = [ln Y_n, ln Y_rho,
ln x_N, eta]. The gap is built from e^eta directly,
x_N (1 - m_A/(fac_n_A m_N)) + e^eta/fac_n_A, rather than as x_N - xi_N.
"""
function N_temp_gap(pan, u)
    ln_x_N, eta = u[3], u[4]
    x_N = exp(ln_x_N)
    gap = x_N * (1 - pan.A.m / (pan.fac_n_A * pan.N1.m)) + gap_A_from_eta(eta) / pan.fac_n_A
    return T_N_from_ln_x_N(pan, ln_x_N), gap
end

"""
    z_at_redshift(m_N, tT_rel, z_red)

z = log(m_N / T_nu) at redshift z_red, with 1 + z_red = T_SM / T0 as in the
Python code (log-linear interpolation on the time grid).
"""
function z_at_redshift(m_N, tT_rel::TimeTempRelation, z_red)
    target = log((1 + z_red) * T0)
    i = findfirst(<(target), log.(tT_rel.T_SM_grid))
    (i === nothing || i == 1) && error("TimeTempRelation grid does not reach redshift $z_red")
    l_hi, l_lo = log(tT_rel.T_SM_grid[i-1]), log(tT_rel.T_SM_grid[i])
    w = (l_hi - target) / (l_hi - l_lo)
    ln_Tnu = (1 - w) * log(tT_rel.T_nu_grid[i-1]) + w * log(tT_rel.T_nu_grid[i])
    return log(m_N) - ln_Tnu
end

"""Scale factor today = 1, from entropy conservation, at z = log(m_N/T_nu)."""
scale_factor_z(pan, z) = cbrt(s0 / pan.ent_interp_z(z))

"""dt/dz = x / (dx/dt), with z = log x."""
dt_dz(pan, z) = exp(z) / pan.dx_dt_interp_z(z)

"""
    comoving_distance(pan, v_of_z, z_lo, z_hi; breakpoints=(), rtol=1e-6)

∫ v(t)/a(t) dt between z_lo and z_hi, in GeV^-1 (divide by `Mpc` for Mpc).
`v_of_z(z)` is the mean speed at z = log(m_N/T_nu).
"""
function comoving_distance(pan, v_of_z, z_lo, z_hi; breakpoints=(), rtol=1e-6)
    z_hi > z_lo || return 0.0
    pts = sort!(unique!([z_lo; [b for b in breakpoints if z_lo < b < z_hi]; z_hi]))
    integrand(z) = v_of_z(z) / scale_factor_z(pan, z) * dt_dz(pan, z)
    # Segment by segment: splatting O(100) breakpoints into one quadgk call
    # makes Julia compile it for a tuple of that length (minutes).
    return sum(QuadGK.quadgk(integrand, pts[i], pts[i+1]; rtol=rtol, atol=0)[1] for i in 1:length(pts)-1)
end

"""
    free_streaming_length(pan, tT_rel, sol; z_red_end=LYA_Z_RED_END, rtol=1e-6)

Free-streaming length of N for the solved point `sol` (from `pandemolate_z`).
Returns a NamedTuple with, all in Mpc:
- `lambda_fs`: the total;
- `lambda_fs_eq`: the part during the solution, from <v> in kinetic
  equilibrium at the solved (T_N, xi_N);
- `lambda_fs_free`: the part after it, with the final spectrum frozen and
  redshifted to z_red_end;
and `v_end`, the mean speed at the end of the solution.
"""
function free_streaming_length(pan::PandemolatorZ, tT_rel::TimeTempRelation, sol;
        z_red_end=LYA_Z_RED_END, rtol=1e-6)
    z_i, z_f = sol.t[1], sol.t[end]
    z_end = z_at_redshift(pan.N1.m, tT_rel, z_red_end)
    z_end > z_f || error("the solution ends after redshift $z_red_end")

    # Hide the ODESolution type behind Ref{Any}: specialising quadgk on a
    # closure that carries it takes minutes to compile (one dynamic call per
    # evaluation is negligible next to `mean_speed`).
    sol_ref = Ref{Any}(sol)
    v_eq(z) = mean_speed(pan.N1, N_temp_gap(pan, sol_ref[](z)::Vector{Float64})...)
    L_eq = comoving_distance(pan, v_eq, z_i, z_f; breakpoints=sol.t, rtol=rtol)

    T_f, gap_f = N_temp_gap(pan, sol.u[end])
    a_f = scale_factor_z(pan, z_f)
    v_free(z) = mean_speed(pan.N1, T_f, gap_f; r=a_f / scale_factor_z(pan, z))
    L_free = comoving_distance(pan, v_free, z_f, z_end; rtol=rtol)

    return (
        lambda_fs=(L_eq + L_free) / Mpc,
        lambda_fs_eq=L_eq / Mpc,
        lambda_fs_free=L_free / Mpc,
        v_end=mean_speed(pan.N1, T_f, gap_f),
    )
end

"""
    solve_relic_point(pan, tT_rel, y, theta; reltol=1e-4, wall_limit=600.0)

Solves one point as `solve_point_z` does (same checks), but also returns the
solution: (sol, omega_h2, converged, retcode), with sol = nothing on failure.
"""
function solve_relic_point(pan::PandemolatorZ{T}, tT_rel::TimeTempRelation{T}, y, theta;
        reltol=1e-4, wall_limit=600.0, plateau_frac=0.05, plateau_tol=1e-3) where T <: Real
    dw = DodelsonWidrow{T}(pan.N1.m, theta, tT_rel)
    pan.mp = ModelParams{T}(y, theta)
    pan.deadline = time() + wall_limit
    try
        sol = pandemolate_z(tT_rel, dw, pan; reltol=reltol)
        DE.successful_retcode(sol) || return nothing, NaN, false, Symbol(sol.retcode)
        omega_h2 = final_omega_h2_z(pan, sol)
        converged = check_plateau_z(pan, sol; frac=plateau_frac, tol=plateau_tol) && isfinite(omega_h2) && omega_h2 > 0
        return sol, omega_h2, converged, Symbol(sol.retcode)
    catch e
        e isa SolveDeadlineExceeded && return nothing, NaN, false, :WallLimit
        @warn "pandemolate_z failed at y=$y, theta=$theta" exception = (e, catch_backtrace())
        return nothing, NaN, false, :Exception
    finally
        pan.deadline = Inf
    end
end
