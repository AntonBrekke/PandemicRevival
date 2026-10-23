"""
lyman_alpha.jl

Lyman-alpha lengths of the dark matter at a relic root, for both solvers (the network,
network.jl, and the one-fluid solver, fluid.jl), as in Bringmann et al.
(arXiv:2206.10630), who recast the thermal-WDM bound m_WDM > 1.9 keV (Garzilli et al.)
into lambda_fs < 0.24 Mpc and r_s < 0.34 Mpc:

    lambda_fs = ∫ <v>/a dt   while the dark matter free-streams, up to redshift 50,
    r_s       = ∫ c_s/a dt   while the dark sector is kinetically coupled,

with a = (s0/s)^(1/3), <v> the mean speed of N1 and c_s^2 = dP/drho of the dark sector.

KINETIC COUPLING
================
Gamma_kd = C_kd/n_N1 (kinetic_decoupling.jl) against H, from the start of the solution:
the onset x_c is the first time Gamma_kd > H, the decoupling x_kd the last time
Gamma_kd drops below H (short dips in between count as coupled). The fluid starts
thermal and is coupled from the start (x_c = start); at small g the network sector is
produced free-streaming by the Dodelson-Widrow mechanism and couples only later, or (if
Gamma_kd < H throughout) never. Then

    lambda_fs_kd = free streaming before x_c + after x_kd,   r_s = sound horizon x_c -> x_kd,

and, never coupled, lambda_fs_kd = lambda_fs, r_s = 0. `lambda_fs` alone (no kinetic
decoupling: free streaming over the whole history) overstates the bound wherever the
sector is coupled.

Free streaming uses <v> at the dark temperature during the solution (x <= 100), and a
frozen spectrum, redshifted with p ∝ 1/a, after it and after x_kd. After the solution the
dark state continues as a non-relativistic gas with conserved number, T ∝ a^-2 at fixed
chemical-potential gaps. The sound speed is the derivative of the background P(rho) on a
grid, kept only where dP/dt < 0, drho/dt < 0 and c_s^2 < 0.3 (as the old code; this drops
the relativistic phase and energy injection, a few-% effect on r_s); after the solution
c_s^2 = 5T/(3m + 7.5T).
"""

include(joinpath(@__DIR__, "fluid.jl"))
include(joinpath(@__DIR__, "network.jl"))
include(joinpath(@__DIR__, "kinetic_decoupling.jl"))
using Printf

# Free streaming is followed until this redshift (as the old code).
const LYA_Z_RED_END = 50.0

"""Speed q/E of a particle of mass m and momentum q (exactly 1 for m = 0)."""
_speed(q, m) = q / sqrt(q * q + m * m)

"""
    mean_speed(p::Particle, temp, gap; r=1.0)

Mean speed <|v|> of particle `p` with occupation f = 1/(e^((E-m)/temp + gap) + k) per
dof, after all momenta have been scaled by `r` (free streaming with f fixed in comoving
momentum: r = a_f/a). With r = 1, the value in kinetic equilibrium at (temp, gap). For
Maxwell-Boltzmann (and gap >= m_T_r_MB, where the ratio no longer depends on the gap) at
r = 1 the closed form 2(x + 1)/(x^2 e^x K_2(x)), x = m/T.
"""
function mean_speed(p::Particle, temp, gap; r=1.0)
    if r == 1 && (p.k == 0 || gap >= m_T_r_MB) && p.m > 0
        x = p.m / temp
        return 2 * (x + 1) / (x^2 * SF.besselkx(2, x))
    end
    g = min(gap, m_T_r_MB)
    # d^3p = 4 pi p^2 dp = 4 pi E p dE
    num = thermal_moment(p, temp, g, (E, mom) -> E * mom * _speed(r * mom, p.m))
    den = thermal_moment(p, temp, g, (E, mom) -> E * mom)
    return num / den
end

"""z = log(m_N/T_nu) at redshift z_red, with 1 + z_red = T_SM/T0 (log-linear on the time grid)."""
function z_at_redshift(m_N, tT_rel::TimeTempRelation, z_red)
    target = log((1 + z_red) * T0)
    i = findfirst(<(target), log.(tT_rel.T_SM_grid))
    (i === nothing || i == 1) && error("TimeTempRelation grid does not reach redshift $z_red")
    l_hi, l_lo = log(tT_rel.T_SM_grid[i-1]), log(tT_rel.T_SM_grid[i])
    w = (l_hi - target) / (l_hi - l_lo)
    ln_Tnu = (1 - w) * log(tT_rel.T_nu_grid[i-1]) + w * log(tT_rel.T_nu_grid[i])
    return log(m_N) - ln_Tnu
end

"""Scale factor (today = 1, from entropy conservation) at z; `bg` has `ent_interp_z`."""
scale_factor_z(bg, z) = cbrt(s0 / bg.ent_interp_z(z))

"""dt/dz = x/(dx/dt), with z = log x."""
dt_dz(bg, z) = exp(z) / bg.dx_dt_interp_z(z)

"""
    comoving_distance(bg, v_of_z, z_lo, z_hi; breakpoints=(), rtol=1e-6)

∫ v/a dt between z_lo and z_hi, in GeV^-1 (divide by `Mpc` for Mpc).
"""
function comoving_distance(bg, v_of_z, z_lo, z_hi; breakpoints=(), rtol=1e-6)
    z_hi > z_lo || return 0.0
    pts = sort!(unique!([z_lo; [b for b in breakpoints if z_lo < b < z_hi]; z_hi]))
    integrand(z) = v_of_z(z) / scale_factor_z(bg, z) * dt_dz(bg, z)
    # segment by segment: splatting O(100) breakpoints into one quadgk call makes Julia
    # compile it for a tuple of that length (minutes)
    return sum(QuadGK.quadgk(integrand, pts[i], pts[i+1]; rtol=rtol, atol=0)[1] for i in 1:length(pts)-1)
end

# --------------------------------------------------------------------------- dark history
"""
The dark sector along a solution and after it. `state(u, z)` gives
(T_N, gap of N1, gap of N2, n_N1, P, rho) of the dark sector in the solver state u;
`N1` carries the statistics of the dark matter (Fermi-Dirac in the fluid, MB (k = 0) in
the network). `bg` has the background interpolants `ent_interp_z`, `dx_dt_interp_z`,
`H_interp_z`.
"""
struct DarkHistory{B,F}
    bg::B
    N1::Particle{Float64}
    state::F
    sol::Any                  # untyped: specialising quadgk on it compiles for minutes
    tab::CkdTable
    z_i::Float64
    z_f::Float64
    a_f::Float64
    fin::NTuple{4,Float64}    # (T, gap1, gap2, n1) at z_f
end

function DarkHistory(bg, N1, m_A, g, sol, state; delta=0.0)
    z_f = sol.t[end]
    fin = state(sol.u[end], z_f)[1:4]
    return DarkHistory(bg, N1, state, sol, CkdTable(N1.m, m_A, g; delta=delta), sol.t[1], z_f,
                       scale_factor_z(bg, z_f), fin)
end

"""History of a fluid solution (fluid.jl): one T_N and one chemical potential for N1 and N2
(with a splitting the N2 gap is larger by (m_N2 - m_N1)/T_N)."""
function DarkHistory(pan::FluidSolver, sol)
    function state(u, z)
        T, gap = N_temp_gap(pan, u)
        xi = xi_from_eta(pan, u[4], u[3])
        gA = gap_A_from_eta(u[4])
        return (T, gap, N2_gap(pan, u), number_density(pan.N1, T, 0.0; gap=gap),
                pressure_dens(pan, T, xi; gap_A=gA), energy_dens(pan, T, xi; gap_A=gA))
    end
    return DarkHistory(pan, pan.N1, pan.A.m, pan.mp.y, sol, state; delta=pan.N2.m / pan.N1.m - 1)
end

"""History of a network solution (network.jl): MB species, separate N1 and N2 densities."""
function DarkHistory(net::Network, sol)
    function state(u, z)
        n, kappa = state_densities(net, u, z)
        T = dark_temperature(net, n, kappa)
        gap(k) = log(n_scaled_safe(net.masses[k], net.dofs[k], T) / n[k])
        return (T, gap(1), gap(2), n[1], sum(n) * T, sum(net.masses[k] * n[k] for k in 1:NSPECIES) + kappa)
    end
    return DarkHistory(net, Particle{Float64}(net.p.m1, 0, dof=2), net.masses[3], net.p.g, sol, state;
                       delta=splitting(net.p))
end

"""(T, gap1, gap2, n1) at z: from the solution, then non-relativistic with conserved number."""
function dark_state(h::DarkHistory, z)
    z <= h.z_f && return h.state(h.sol(z)::Vector{Float64}, z)[1:4]
    r = h.a_f / scale_factor_z(h.bg, z)
    T, g1, g2, n1 = h.fin
    return (T * r^2, g1, g2, n1 * r^3)
end

"""
    f_N2_final(h, itp, p, m2; frozen_at_end, z_hi)

N2/(N1 + N2) after N1 N1 <-> N2 N2 freezes out, for a splitting m_N2 = m2 > m_N1 (h.N1.m).
With r = n_N2/n_N1 and the total N = n_N1 + n_N2 and T_N of the history,

    dr/dz = (x/xdot) (1 + r) 2 (G_f - G_r)/n_N1,   G_f = R_f (n_N1/nh_N1)^2,  G_r = R_r (n_N2/nh_N2)^2,

the conversions alone (MB, the network's rates from the tables at parameters `p`; nh the MB
density at zero chemical potential). After the solution the dark sector continues as
`dark_state` has it (T ∝ a^-2, N ∝ a^-3), without the heat of the conversions.
- `frozen_at_end = true` (the network, which follows N2 itself): the history's value at its
  end, continued with this equation if the conversions are still faster than H there.
- `frozen_at_end = false` (the fluid, which holds N1 <-> N2 in equilibrium): the equation is
  started from the history at the last z (after x = 1) where the relaxation rate
  lambda = 2 G_f/n_N1 + 2 G_r/n_N2 exceeds 1e3 H, where the equilibrium still holds.
Integrated to z_hi. Returns (f_N2, x where lambda last exceeds H; NaN if it never does).
"""
function f_N2_final(h::DarkHistory, itp::RateInterpolants, p, m2; frozen_at_end, z_hi, dz=0.02)
    m1 = h.N1.m
    r = reaction(:N1N1_N2N2)
    t = itp.dark[r.name]
    lnR = ln_norm(r, p, itp.p_ref)
    # (T, total N, history's r) at z
    function hist(z)
        T, g1, g2, n1 = dark_state(h, z)
        rh = n_scaled_safe(m2, 2, T) / n_scaled_safe(m1, 2, T) * exp(g1 - g2)
        return T, n1 * (1 + rh), rh
    end
    # (G_f/n1, G_r/n1) at z for ratio r: reactions per N1 and time
    function rates(z, rr)
        T, N, _ = hist(z)
        n1 = N / (1 + rr)
        nh1, nh2 = n_scaled_safe(m1, 2, T), n_scaled_safe(m2, 2, T)
        lx = log(m1 / T)
        return exp(t.lnRf(lx) + lnR) * n1 / nh1^2, exp(t.lnRr(lx) + lnR) * rr^2 * n1 / nh2^2
    end
    lam_H(z, rr) = (gf = rates(z, rr); 2 * (gf[1] + gf[2] / rr) / h.bg.H_interp_z(z))
    zs = collect(range(h.z_i, z_hi, length=max(2, ceil(Int, (z_hi - h.z_i) / dz) + 1)))
    lam_hist = [lam_H(z, hist(z)[3]) for z in zs]
    i_fo = findlast(>=(1), lam_hist)
    x_fo = i_fo === nothing ? NaN : exp(zs[i_fo])
    if frozen_at_end
        z0 = h.z_f
        lam_H(z0, hist(z0)[3]) < 1 && (rh = hist(z0)[3]; return rh / (1 + rh), x_fo)
    else
        i0 = findlast(i -> zs[i] >= 0 && lam_hist[i] >= 1e3, eachindex(zs))
        z0 = i0 === nothing ? max(h.z_i, 0.0) : zs[i0]
    end
    # in l = ln r; the rate is stiff (lambda up to 1e3 per unit z at the start)
    function rhs!(du, u, _, z)
        rr = exp(u[1])
        gf, gr = rates(z, rr)
        du[1] = dt_dz(h.bg, z) * (1 + rr) * 2 * (gf - gr) / rr
    end
    prob = DE.ODEProblem(rhs!, [log(hist(z0)[3])], (z0, z_hi))
    sol = DE.solve(prob, ODER.Rodas4P(autodiff=AutoFiniteDiff()); reltol=1e-6, abstol=1e-8)
    rr = exp(sol.u[end][1])
    return rr / (1 + rr), x_fo
end

"""Gamma_kd/H of N1 at z."""
function kd_ratio(h::DarkHistory, z)
    T, g1, g2, n1 = dark_state(h, z)
    return C_kd(h.tab, T, g1; gap2=g2) / (h.bg.H_interp_z(z) * n1)
end

"""
(z_c, z_kd, coupled): the first z with Gamma_kd > H and the last at which it drops below
H, on a grid of step `dz` from z_lo to z_hi, refined by bisection to 1e-4; coupled =
false if Gamma_kd < H throughout, z_kd = z_hi if it is still coupled there.
"""
function find_kd(h::DarkHistory, z_lo, z_hi; dz=0.05)
    r(z) = kd_ratio(h, z)
    function crossing(a, b)
        up = r(a) < 1
        while b - a > 1e-4
            c = (a + b) / 2
            (r(c) < 1) == up ? (a = c) : (b = c)
        end
        return b
    end
    zs = collect(range(z_lo, z_hi, length=max(2, ceil(Int, (z_hi - z_lo) / dz) + 1)))
    on = [r(z) >= 1 for z in zs]
    i_c = findfirst(on)
    i_c === nothing && return z_lo, z_lo, false
    i_kd = findlast(on)
    z_c = i_c == 1 ? z_lo : crossing(zs[i_c-1], zs[i_c])
    z_kd = i_kd == length(zs) ? z_hi : crossing(zs[i_kd], zs[i_kd+1])
    return z_c, z_kd, true
end

"""
∫ <v>/a dt from z_lo to z_hi in GeV^-1, with <v> at the dark temperature during the
solution and the spectrum at z_f frozen after it.
"""
function free_stream(h::DarkHistory, z_lo, z_hi)
    z_hi > z_lo || return 0.0
    L = 0.0
    if z_lo < h.z_f
        v_eq(z) = mean_speed(h.N1, dark_state(h, z)[1:2]...)
        L += comoving_distance(h.bg, v_eq, z_lo, min(z_hi, h.z_f); breakpoints=h.sol.t)
    end
    z_hi > h.z_f && (L += free_stream_frozen(h, max(z_lo, h.z_f), z_hi; z_freeze=h.z_f))
    return L
end

"""∫ <v>/a dt from z_lo to z_hi in GeV^-1 for the spectrum at z_freeze (<= z_lo) redshifted freely."""
function free_stream_frozen(h::DarkHistory, z_lo, z_hi; z_freeze=z_lo)
    T, g1 = dark_state(h, z_freeze)[1:2]
    a0 = scale_factor_z(h.bg, z_freeze)
    v(z) = mean_speed(h.N1, T, g1; r=a0 / scale_factor_z(h.bg, z))
    return comoving_distance(h.bg, v, z_lo, z_hi)
end

"""∫ c_s/a dt from z_lo to z_hi in GeV^-1 (see the header for c_s)."""
function sound_horizon(h::DarkHistory, z_lo, z_hi; n_grid=2000)
    z_hi > z_lo || return 0.0
    r = 0.0
    if z_lo < h.z_f
        zs = range(z_lo, min(z_hi, h.z_f), length=n_grid)
        PR = [h.state(h.sol(z)::Vector{Float64}, z)[5:6] for z in zs]
        lP, lR = log.(first.(PR)), log.(last.(PR))
        f_prev = 0.0
        for i in eachindex(zs)
            j0, j1 = max(i - 1, 1), min(i + 1, n_grid)
            dlP, dlR = lP[j1] - lP[j0], lR[j1] - lR[j0]
            cs2 = (PR[i][1] * dlP) / (PR[i][2] * dlR)
            f = (dlP < 0 && dlR < 0 && cs2 < 0.3) ? sqrt(cs2) / scale_factor_z(h.bg, zs[i]) * dt_dz(h.bg, zs[i]) : 0.0
            i > 1 && (r += (f + f_prev) / 2 * step(zs))
            f_prev = f
        end
    end
    if z_hi > h.z_f
        cs(z) = (T = dark_state(h, z)[1]; sqrt(5T / (3 * h.N1.m + 7.5T)))
        r += comoving_distance(h.bg, cs, max(z_lo, h.z_f), z_hi)
    end
    return r
end

"""
    lyman_alpha_lengths(h, tT_rel)

NamedTuple, lengths in Mpc: `lambda_fs` (free streaming over the whole history, no
kinetic decoupling), `x_c`, `x_kd` (m_N/T_nu at the coupling onset and at decoupling; NaN
if never coupled, Inf if still coupled at the end of the time grid), `lambda_fs_pre`
(before x_c), `lambda_fs_kd` (before x_c plus after x_kd), `r_s`, `coupled`.
"""
function lyman_alpha_lengths(h::DarkHistory, tT_rel::TimeTempRelation)
    m = h.N1.m
    z_hi = log(m / tT_rel.T_nu_grid[end])
    z_end = z_at_redshift(m, tT_rel, LYA_Z_RED_END)
    z_end > h.z_f || error("the solution ends after redshift $LYA_Z_RED_END")
    lam = free_stream(h, h.z_i, z_end) / Mpc
    z_c, z_kd, coupled = find_kd(h, h.z_i, z_hi)
    coupled || return (lambda_fs=lam, x_c=NaN, x_kd=NaN, lambda_fs_pre=lam, lambda_fs_kd=lam, r_s=0.0, coupled=false)
    pre = free_stream(h, h.z_i, z_c) / Mpc
    post = z_kd < z_end ? free_stream_frozen(h, z_kd, z_end) / Mpc : 0.0
    return (lambda_fs=lam, x_c=exp(z_c), x_kd=z_kd >= z_hi ? Inf : exp(z_kd), lambda_fs_pre=pre,
            lambda_fs_kd=pre + post, r_s=sound_horizon(h, z_c, z_kd) / Mpc, coupled=true)
end

# --------------------------------------------------------------------------- per relic root
"""
    lyman_alpha_network(itp, tT_rel, m, g, s2)

Network solve at (m_N [GeV], g, sin^2 2theta) and its Lyman-alpha lengths:
`lyman_alpha_lengths` plus `omega_h2` and `f_N2` = N2/(N1 + N2) today. NaN lengths if the
solve fails.
"""
function lyman_alpha_network(itp, tT_rel, m, g, s2)
    p = table_params(itp, m, g, s2)
    net = Network(itp, p, tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    # with a splitting the conversions can run past x = 100: follow N2 to the end of the portal tables
    delta = table_delta(itp)
    sol = solve_network(net, dw, tT_rel; x_end=delta > 0 ? 1e3 : 100.0, maxiters=20_000)
    sol.retcode == DE.ReturnCode.Success || return failed_lengths()
    n, _ = state_densities(net, sol.u[end], sol.t[end])
    h = DarkHistory(net, sol)
    f2, x_fo = n[2] / (n[1] + n[2]), NaN
    delta > 0 && ((f2, x_fo) = f_N2_final(h, itp, p, p.m2; frozen_at_end=true, z_hi=log(m / tT_rel.T_nu_grid[end])))
    return (; lyman_alpha_lengths(h, tT_rel)..., omega_h2=network_omega_h2(net, sol.u[end], sol.t[end]),
            f_N2=f2, x_fo=x_fo)
end

"""As `lyman_alpha_network` for the fluid (`solve_relic`; f_N2 = 1/2 for equal masses, else at
the freeze-out of N1 N1 <-> N2 N2, `f_N2_final`)."""
function lyman_alpha_fluid(itp, tT_rel, m, g, s2; pan=FluidSolver(m, tT_rel, itp))
    sol, om, plateau = solve_relic(pan, tT_rel, g, s2)
    plateau || return failed_lengths()
    h = DarkHistory(pan, sol)
    f2, x_fo = 0.5, NaN
    table_delta(itp) > 0 && ((f2, x_fo) = f_N2_final(h, itp, fluid_rates!(pan), pan.N2.m; frozen_at_end=false,
                                                     z_hi=log(m / tT_rel.T_nu_grid[end])))
    return (; lyman_alpha_lengths(h, tT_rel)..., omega_h2=om, f_N2=f2, x_fo=x_fo)
end

failed_lengths() = (lambda_fs=NaN, x_c=NaN, x_kd=NaN, lambda_fs_pre=NaN, lambda_fs_kd=NaN, r_s=NaN,
                    coupled=false, omega_h2=NaN, f_N2=NaN, x_fo=NaN)

"""Columns of the Lyman-alpha CSVs; x_fo (freeze-out of N1 N1 <-> N2 N2, `f_N2_final`) only for
tables with a splitting."""
const LYMAN_ALPHA_HEADER = "m_N,y,sin2_2theta,omega_h2,lambda_fs_Mpc,x_c,x_kd,lambda_fs_pre_Mpc,lambda_fs_kd_Mpc,r_s_Mpc,coupled,f_N2"
lyman_alpha_header(delta) = delta > 0 ? LYMAN_ALPHA_HEADER * ",x_fo" : LYMAN_ALPHA_HEADER

"""One CSV row of `lyman_alpha_header(delta)`."""
lyman_alpha_row(m, g, s2, r; delta=0.0) =
    join((@sprintf("%.6g", v) for v in (m, g, s2, r.omega_h2, r.lambda_fs, r.x_c, r.x_kd,
          r.lambda_fs_pre, r.lambda_fs_kd, r.r_s, Int(r.coupled), r.f_N2, (delta > 0 ? (r.x_fo,) : ())...)), ",")
