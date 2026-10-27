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
    late::Any                 # nothing, or z -> (T, gap1, gap2, n1) after z_f (`with_conversions`)
end

function DarkHistory(bg, N1, m_A, g, sol, state; delta=0.0)
    z_f = sol.t[end]
    fin = state(sol.u[end], z_f)[1:4]
    return DarkHistory(bg, N1, state, sol, CkdTable(N1.m, m_A, g; delta=delta), sol.t[1], z_f,
                       scale_factor_z(bg, z_f), fin, nothing)
end

"""History of a fluid solution (fluid.jl): one T_N and one chemical potential for N1 and N2
(with a splitting the N2 gap is larger by (m_N2 - m_N1)/T_N). `delta` other than the solver's
(a per-point splitting on equal-mass tables, `mnu_splitting`) adds (delta m_N - (m_N2 - m_N1))/T_N
to the N2 gap, which keeps mu_N2 = mu_N1, and is the splitting of Gamma_kd."""
function DarkHistory(pan::FluidSolver, sol; delta=pan.N2.m / pan.N1.m - 1)
    dm_extra = delta * pan.N1.m - (pan.N2.m - pan.N1.m)
    function state(u, z)
        T, gap = N_temp_gap(pan, u)
        xi = xi_from_eta(pan, u[4], u[3])
        gA = gap_A_from_eta(u[4])
        return (T, gap, N2_gap(pan, u) + dm_extra / T, number_density(pan.N1, T, 0.0; gap=gap),
                pressure_dens(pan, T, xi; gap_A=gA), energy_dens(pan, T, xi; gap_A=gA))
    end
    return DarkHistory(pan, pan.N1, pan.A.m, pan.mp.y, sol, state; delta=delta)
end

"""History of a network solution (network.jl): MB species, separate N1 and N2 densities.
`delta` (default the solver's) is the splitting of Gamma_kd; the densities are the solver's."""
function DarkHistory(net::Network, sol; delta=splitting(net.p))
    function state(u, z)
        n, kappa = state_densities(net, u, z)
        T = dark_temperature(net, n, kappa)
        gap(k) = log(n_scaled_safe(net.masses[k], net.dofs[k], T) / n[k])
        return (T, gap(1), gap(2), n[1], sum(n) * T, sum(net.masses[k] * n[k] for k in 1:NSPECIES) + kappa)
    end
    return DarkHistory(net, Particle{Float64}(net.p.m1, 0, dof=2), net.masses[3], net.p.g, sol, state;
                       delta=delta)
end

"""(T, gap1, gap2, n1) at z: from the solution, then non-relativistic with conserved number
(and, with an N1-N2 splitting, the late conversions and their heat: `with_conversions`)."""
function dark_state(h::DarkHistory, z)
    z <= h.z_f && return h.state(h.sol(z)::Vector{Float64}, z)[1:4]
    h.late === nothing || return h.late(z)
    r = h.a_f / scale_factor_z(h.bg, z)
    T, g1, g2, n1 = h.fin
    return (T * r^2, g1, g2, n1 * r^3)
end

"""
ln R_r(m2)/R_r(m2 = m1) of N2 N2 -> N1 N1 (the table rate per (n/nh)^2) at T for a splitting
m2 - m1 << m1, from equal-mass tables: the same matrix element with the s-wave threshold factor
<p_f/p_i> = u e^u K_1(u) (u = (m2 - m1)/T; the reaction releases 2(m2 - m1)), i.e. 1 + u for
u << 1 and sqrt(pi u/2) for u >> 1. N1 N1 -> N2 N2 then follows from detailed balance,
R_f = R_r e^(-2u), as in tables with a splitting. Reproduces the delta = 1e-4 tables to 4e-4
in ln R over their whole grid (u <= 1e4; test_lyman_alpha.jl).
"""
function ln_conversion_shift(m1, m2, T)
    u = (m2 - m1) / T
    return 2 * log(n_scaled_safe(m2, 2, T) / n_scaled_safe(m1, 2, T)) + (u < 1e-8 ? u : log(u * besselkx_safe(1, u)))
end

"""
    with_conversions(h, itp, p, m2; frozen_at_end, z_hi)

The late N1 N1 <-> N2 N2 conversions for a splitting m_N2 = m2 > m_N1 (h.N1.m): the history
`h` with the dark sector after the solution continued with them and their heat, and
N2/(N1 + N2) today. With r = n_N2/n_N1, N = n_N1 + n_N2 ∝ a^-3 and the MB rates of the
network (tables at parameters `p`),

    dr/dz = (x/xdot) (1 + r) 2 (G_f - G_r)/n_N1,   G_f = R_f (n_N1/nh_N1)^2,  G_r = R_r (n_N2/nh_N2)^2,
    dT/dz = -2 T dln a/dz - w (2/3) (m2 - m1) (dr/dz)/(1 + r)^2,   w = Gamma_kd/(Gamma_kd + H),

nh the MB density at zero chemical potential: each conversion releases m2 - m1 of kinetic energy,
shared by the (non-relativistic) N while they scatter (Gamma_kd of `kd_ratio`); after kinetic
decoupling it stays with the converted N1 (a small hot component that is not followed). Beyond
the table grid (m/T_N > 1e8) the conversions are off. Before the end of the solution T is the
history's.
- `frozen_at_end = true` (the network, which follows N2 itself): from the history at its end
  (the conversions that remain, even slower than H, still deplete N2 by ~10%).
- `frozen_at_end = false` (the fluid, which holds N1 <-> N2 in equilibrium): r from the last z
  (after x = 1) where the relaxation rate lambda = 2 G_f/n_N1 + 2 G_r/n_N2 exceeds 1e3 H,
  where the equilibrium still holds. Up to the end of the solution the fluid's T, which has
  the heat of its equilibrium conversions; for g >= 1e-3 those are faster than H there.
Integrated to z_hi. Returns (history, f_N2, x where lambda last exceeds H; NaN if never).
"""
function with_conversions(h::DarkHistory, itp::RateInterpolants, p, m2; frozen_at_end, z_hi, dz=0.02)
    m1 = h.N1.m
    dm = m2 - m1
    # a splitting other than the tables' (per point at fixed m_nu, on equal-mass tables)
    per_point = !isapprox(dm / m1, table_delta(itp); rtol=1e-6, atol=1e-14)
    per_point && table_delta(itp) > 0 && error("with_conversions: splitting $(dm / m1) on tables with delta = $(table_delta(itp))")
    t = itp.dark[:N1N1_N2N2]
    lnR = ln_norm(reaction(:N1N1_N2N2), p, itp.p_ref)
    nh(m, T) = n_scaled_safe(m, 2, T)
    # (T, N, r) of the history at z (after z_f: adiabatic, without the conversions)
    function hist(z)
        T, g1, g2, n1 = dark_state(h, z)
        rh = nh(m2, T) / nh(m1, T) * exp(g1 - g2)
        return T, n1 * (1 + rh), rh
    end
    # (G_f/n1, G_r/n1): reactions per N1 and time, at (T, N, r); zero beyond the table grid
    # (m/T_N > 1e8), where the conversions are long frozen and the linear extrapolation of
    # ln R in ln x_N would grossly overstate the endothermic e^(-2 delta x_N)
    function rates(T, N, rr)
        n1 = N / (1 + rr)
        lx = log(m1 / T)
        lx > itp.lnxN[2] && return 0.0, 0.0
        lnRf, lnRr = t.lnRf(lx) + lnR, t.lnRr(lx) + lnR
        if per_point
            lnRr += ln_conversion_shift(m1, m2, T)
            lnRf = lnRr - 2dm / T
        end
        return exp(lnRf) * n1 / nh(m1, T)^2, exp(lnRr) * rr^2 * n1 / nh(m2, T)^2
    end
    # d ln r/dz without dividing by a vanishing r
    function dlnr(z, T, N, l)
        gf, gr = rates(T, N, exp(l))
        return dt_dz(h.bg, z) * (1 + exp(l)) * 2 * (gf * exp(min(-l, 700.0)) - gr * exp(-l))
    end
    lam_H(z, T, N, rr) = (g = rates(T, N, rr); 2 * (g[1] + g[2] / rr) / h.bg.H_interp_z(z))
    T_f, N_f, r_f = hist(h.z_f)
    alg = ODER.Rodas4P(autodiff=AutoFiniteDiff())
    if frozen_at_end
        l_f = log(r_f)
    else
        # equilibrium until lambda ~ 1e3 H, then the conversion equation with the fluid's T and N
        zs = collect(range(max(h.z_i, 0.0), h.z_f, length=max(2, ceil(Int, (h.z_f - max(h.z_i, 0.0)) / dz) + 1)))
        i0 = findlast(z -> lam_H(z, hist(z)...) >= 1e3, zs)
        z0 = i0 === nothing ? zs[1] : zs[i0]
        s1 = DE.solve(DE.ODEProblem((du, u, _, z) -> (hz = hist(z); du[1] = dlnr(z, hz[1], hz[2], u[1])),
                                    [log(hist(z0)[3])], (z0, h.z_f)), alg; reltol=1e-6, abstol=1e-8)
        l_f = s1.u[end][1]
    end
    # after the solution: r and T together, N ∝ a^-3
    lna(z) = log(scale_factor_z(h.bg, z))
    N_of(z) = N_f * exp(3 * (lna(h.z_f) - lna(z)))
    # the heat is shared by all N only while they scatter faster than H (Gamma_kd of
    # `kd_ratio`); after kinetic decoupling it stays with the few converted N1 (a small hot
    # component, not followed): weight Gamma_kd/(Gamma_kd + H)
    function rhs2!(du, u, _, z)
        T, N = exp(u[2]), N_of(z)
        du[1] = dlnr(z, T, N, u[1])
        rr = exp(u[1])
        n1 = N / (1 + rr)
        G = C_kd(h.tab, T, log(nh(m1, T) / n1); gap2=log(nh(m2, T) / (N - n1))) / n1
        w = G / (G + h.bg.H_interp_z(z))
        dlna = (lna(z + 1e-4) - lna(z - 1e-4)) / 2e-4
        du[2] = -2 * dlna - w * (2 / 3) * dm * rr * du[1] / (1 + rr)^2 / T
    end
    s2 = DE.solve(DE.ODEProblem(rhs2!, [l_f, log(T_f)], (h.z_f, z_hi)), alg; reltol=1e-6, abstol=1e-8)
    function late(z)
        l, lT = s2(min(z, z_hi))::Vector{Float64}
        T, N, rr = exp(lT), N_of(z), exp(l)
        n1 = N / (1 + rr)
        return (T, log(nh(m1, T) / n1), log(nh(m2, T) / (N - n1)), n1)
    end
    h2 = DarkHistory(h.bg, h.N1, h.state, h.sol, h.tab, h.z_i, h.z_f, h.a_f, h.fin, late)
    # (not `rr`: the closures above assign that name, and would share it)
    r_end = exp(s2.u[end][1])
    # freeze-out: the last z at which the conversions are faster than H
    function converting(z)
        T, g1, g2, n1 = dark_state(h2, z)
        n2 = n1 * nh(m2, T) / nh(m1, T) * exp(g1 - g2)
        return lam_H(z, T, n1 + n2, n2 / n1) >= 1
    end
    zs = range(h.z_i, z_hi, length=max(2, ceil(Int, (z_hi - h.z_i) / dz) + 1))
    i_fo = findlast(converting, zs)
    return h2, r_end / (1 + r_end), i_fo === nothing ? NaN : exp(zs[i_fo])
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
Splitting of the post-processing at (m_N, sin^2 2theta): the tables' if they have one, else
`mnu_splitting` at m_nu [GeV] (the solvers then run with equal masses: at m_nu = 0.1 eV the
splitting is far too small to change the relic abundance; notes/fixed_mnu.md of the PhD folder,
test/check_mnu_threshold.jl), or 0 for m_nu = nothing.
"""
function postproc_splitting(itp, m, s2, m_nu)
    table_delta(itp) > 0 && return table_delta(itp)
    return m_nu === nothing ? 0.0 : mnu_splitting(m, s2; m_nu=m_nu)
end

"""
    lyman_alpha_network(itp, tT_rel, m, g, s2; m_nu=nothing)

Network solve at (m_N [GeV], g, sin^2 2theta) and its Lyman-alpha lengths:
`lyman_alpha_lengths` plus `omega_h2`, `f_N2` = N2/(N1 + N2) today, and the splitting `delta`
(`postproc_splitting`). NaN lengths if the solve fails.
"""
function lyman_alpha_network(itp, tT_rel, m, g, s2; m_nu=nothing)
    p = table_params(itp, m, g, s2)
    net = Network(itp, p, tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    # with a splitting the late conversions and their heat are followed by `with_conversions`
    # from x = 100 on, as for the fluid: the network's single dark temperature would give the
    # bulk all of their heat even after kinetic decoupling
    delta = postproc_splitting(itp, m, s2, m_nu)
    sol = solve_network(net, dw, tT_rel; x_end=100.0, maxiters=20_000)
    sol.retcode == DE.ReturnCode.Success || return failed_lengths()
    n, _ = state_densities(net, sol.u[end], sol.t[end])
    h = DarkHistory(net, sol; delta=delta)
    f2, x_fo = n[2] / (n[1] + n[2]), NaN
    delta > 0 && ((h, f2, x_fo) = with_conversions(h, itp, p, m * (1 + delta); frozen_at_end=true,
                                                   z_hi=log(m / tT_rel.T_nu_grid[end])))
    return (; lyman_alpha_lengths(h, tT_rel)..., omega_h2=network_omega_h2(net, sol.u[end], sol.t[end]),
            f_N2=f2, x_fo=x_fo, delta=delta)
end

"""As `lyman_alpha_network` for the fluid (`solve_relic`; f_N2 = 1/2 for equal masses; with a
splitting the late conversions and their heat from `with_conversions`)."""
function lyman_alpha_fluid(itp, tT_rel, m, g, s2; pan=FluidSolver(m, tT_rel, itp), m_nu=nothing)
    sol, om, plateau = solve_relic(pan, tT_rel, g, s2)
    plateau || return failed_lengths()
    delta = postproc_splitting(itp, m, s2, m_nu)
    h = DarkHistory(pan, sol; delta=delta)
    f2, x_fo = 0.5, NaN
    delta > 0 && ((h, f2, x_fo) = with_conversions(h, itp, fluid_rates!(pan), m * (1 + delta); frozen_at_end=false,
                                                   z_hi=log(m / tT_rel.T_nu_grid[end])))
    return (; lyman_alpha_lengths(h, tT_rel)..., omega_h2=om, f_N2=f2, x_fo=x_fo, delta=delta)
end

failed_lengths() = (lambda_fs=NaN, x_c=NaN, x_kd=NaN, lambda_fs_pre=NaN, lambda_fs_kd=NaN, r_s=NaN,
                    coupled=false, omega_h2=NaN, f_N2=NaN, x_fo=NaN, delta=NaN)

"""Columns of the Lyman-alpha CSVs; x_fo (freeze-out of N1 N1 <-> N2 N2, `with_conversions`) only
with a splitting: for tables with one (`delta` > 0), and with `delta` (per point) as well for a
splitting at fixed m_nu (`per_point`)."""
const LYMAN_ALPHA_HEADER = "m_N,y,sin2_2theta,omega_h2,lambda_fs_Mpc,x_c,x_kd,lambda_fs_pre_Mpc,lambda_fs_kd_Mpc,r_s_Mpc,coupled,f_N2"
lyman_alpha_header(delta; per_point=false) =
    per_point ? LYMAN_ALPHA_HEADER * ",x_fo,delta" : delta > 0 ? LYMAN_ALPHA_HEADER * ",x_fo" : LYMAN_ALPHA_HEADER

"""One CSV row of `lyman_alpha_header(delta; per_point)`."""
lyman_alpha_row(m, g, s2, r; delta=0.0, per_point=false) =
    join((@sprintf("%.6g", v) for v in (m, g, s2, r.omega_h2, r.lambda_fs, r.x_c, r.x_kd,
          r.lambda_fs_pre, r.lambda_fs_kd, r.r_s, Int(r.coupled), r.f_N2,
          (per_point ? (r.x_fo, r.delta) : delta > 0 ? (r.x_fo,) : ())...)), ",")
