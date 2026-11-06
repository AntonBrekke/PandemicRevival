"""
network_reduced.jl

The network (network.jl) with A' and phi eliminated: A' <-> N1 N2 and phi <-> A'A' are taken
to be in equilibrium at all times, so that only the densities that survive all decays evolve.

STATE
=====
With the dressed densities (what each species count becomes after all decays)
    N1~ = n_N1 + n_A' + 2 n_phi,      N2~ = n_N2 + n_A' + 2 n_phi,
    kappa~ = rho - m_N1 N1~ - m_N2 N2~ = kappa + (m_A' - m_N1 - m_N2) n_A' + (m_phi - 2 m_N1 - 2 m_N2) n_phi,
u = [ln((N1~ + N2~)/s), ln(N2~/N1~), ln(kappa~/s^(4/3))] against z = ln x.

EQUATIONS
=========
    dNi~/dt + 3H Ni~ = sum_r (dN_i + dN_A' + 2 dN_phi)_r G_r,
    dkappa~/dt + 3H (kappa~ + P) = sum_r [K_r + (m_A' - m_N1 - m_N2) dN_A',r G_r + (m_phi - 2 m_N1 - 2 m_N2) dN_phi,r G_r],
with G_r, K_r the net rate and kinetic gain of network.jl and P = sum_i n_i T_N. The weight of
A' <-> N1 N2 and phi <-> A'A' vanishes in all three, so their rates (up to 1e20 H) never enter.
These equations are exact; the only approximation is the closure (`reduced_closure`),
    mu_A' = mu_N1 + mu_N2,      mu_phi = 2 mu_A',
which gives (n_N1, n_N2, n_A', n_phi, T_N) from the state. It requires that both decays are open
(m_A' > m_N1 + m_N2, m_phi > 2 m_A'): the constructor-free check is `reduced_applies`.

The N2~/N1~ equation is relaxed by N1 N1 <-> N2 N2 as in network.jl, including its cap
(LAMBDA_LO, LAMBDA_HI): those conversions are a real rate of this system, and not eliminated.
"""

isdefined(@__MODULE__, :Network) || include(joinpath(@__DIR__, "network.jl"))

"""True if both decays that the reduced network assumes are kinematically open."""
reduced_applies(p) = p.mA > p.m1 + p.m2 && p.mphi > 2p.mA

"""
(n_A', a) with a = n_A' + 2 n_phi, from the conserved Nt1, Nt2 and K_A = n_A'/(n_N1 n_N2),
K_phi = n_phi/n_A'^2: the root of n_A' = K_A (Nt1 - a)(Nt2 - a), which is unique on
[0, n_max] with n_max the n_A' where a = min(Nt1, Nt2). Newton, safeguarded by bisection.
"""
function reduced_partition(Nt1, Nt2, KA, Kphi)
    m = min(Nt1, Nt2)
    hi = 2m / (1 + sqrt(1 + 8Kphi * m))
    lo = 0.0
    x = min(hi, KA * Nt1 * Nt2 / (1 + KA * (Nt1 + Nt2)))
    for _ in 1:200
        a = x + 2Kphi * x^2
        f = x - KA * (Nt1 - a) * (Nt2 - a)
        f > 0 ? (hi = x) : (lo = x)
        step = f / (1 + KA * (1 + 4Kphi * x) * (Nt1 + Nt2 - 2a))
        xn = x - step
        (xn <= lo || xn >= hi) && (xn = (lo + hi) / 2)
        abs(xn - x) <= 1e-16 * xn && return xn, xn + 2Kphi * xn^2
        x = xn
    end
    return x, x + 2Kphi * x^2
end

"""
(n_N1, n_N2, ln n_A', ln n_phi) at T from the conserved Nt1, Nt2. Each of n_N1, n_N2 is computed
from the subtraction unless most of it has moved into A', where the equilibrium relation
keeps its relative precision.
"""
function reduced_densities_at(net::Network, Nt1, Nt2, T)
    lns = ntuple(k -> log(n_scaled_safe(net.masses[k], net.dofs[k], T)), 4)
    m1, m2, mA, mphi = net.masses
    lnKA = lns[3] - lns[1] - lns[2] - (mA - m1 - m2) / T
    lnKphi = lns[4] - 2lns[3] - (mphi - 2mA) / T
    KA, Kphi = exp(lnKA), exp(lnKphi)
    x, a = reduced_partition(Nt1, Nt2, KA, Kphi)
    n1 = Nt1 - a
    n2 = Nt2 - a
    if a > 0.5Nt2 && n1 > 0
        n2 = x / (KA * n1)
    elseif a > 0.5Nt1 && n2 > 0
        n1 = x / (KA * n2)
    end
    # in logs: a Boltzmann-suppressed A' or phi underflows any linear density
    lnA = max(lnKA + log(n1) + log(n2), log(n1) - 600)
    lnphi = max(lnKphi + 2lnA, log(n1) - 600)
    return n1, n2, lnA, lnphi
end

"""
    reduced_closure(net, Nt1, Nt2, kappat)

(n_N1, n_N2, ln n_A', ln n_phi, T_N) of the state, with mu_A' = mu_N1 + mu_N2 and mu_phi = 2 mu_A'.
T_N from kappat by the Illinois method in ln T: at fixed dressed numbers the energy rises
monotonically with T (entropy concave), and the bracket holds because every particle carries
at least 3T/2, and n_tot >= (Nt1 + Nt2)/4.
"""
function reduced_closure(net::Network, Nt1, Nt2, kappat)
    m1, m2, mA, mphi = net.masses
    dA, dphi = mA - m1 - m2, mphi - 2m1 - 2m2
    function excess(lnT)
        T = exp(lnT)
        n1, n2, lnA, lnphi = reduced_densities_at(net, Nt1, Nt2, T)
        nA, nphi = exp(lnA), exp(lnphi)
        return n1 * kinetic_per_particle(m1, T) + n2 * kinetic_per_particle(m2, T) +
               nA * (kinetic_per_particle(mA, T) + dA) + nphi * (kinetic_per_particle(mphi, T) + dphi) - kappat
    end
    Nt = Nt1 + Nt2
    # T_N within the tables' range, m_N/T_N in [1e-6, 1e8]; trial states of the stiff solver outside
    # it (or non-finite) give T = NaN, and reduced_rhs! rejects the step
    lo_T, hi_T = log(m1) - log(1e8), log(m1) + log(1e6)
    (isfinite(Nt) && isfinite(kappat) && Nt > 0 && kappat > 0) || return (NaN, NaN, NaN, NaN, NaN)
    b = clamp(log(8kappat / (3Nt)), lo_T, hi_T)
    a = clamp(log(kappat / (3Nt)), lo_T, hi_T)
    fa, fb = excess(a), excess(b)
    while fa > 0 && a > lo_T            # most of kappat stored in the rest mass of A', phi
        a = max(a - 1.0, lo_T)
        fa = excess(a)
    end
    (fa <= 0 && fb >= 0) || return (NaN, NaN, NaN, NaN, NaN)
    for _ in 1:100
        abs(fb) <= 1e-15 * kappat && (a = b; break)
        abs(fa) <= 1e-15 * kappat && (b = a; break)
        abs(b - a) < 1e-15 && break    # a, b swap places: not ordered
        c = b - fb * (b - a) / (fb - fa)
        fc = excess(c)
        if fc * fb < 0
            a, fa = b, fb
        else
            fa /= 2
        end
        b, fb = c, fc
    end
    T = exp((a + b) / 2)
    return (reduced_densities_at(net, Nt1, Nt2, T)..., T)
end

"""(Nt1, Nt2, kappat) of the state u at z."""
function reduced_state(net::Network, u, z)
    s = net.ent_interp_z(z)
    N = exp(u[1]) * s
    Nt1 = N / (1 + exp(u[2]))
    return Nt1, N - Nt1, exp(u[3]) * s^(4 / 3)
end

"""dN of the eliminated reactions, A' -> N1 N2 and phi -> A'A'."""
const ELIMINATED = ((1, 1, -1, 0), (0, 0, 2, -1))

"""
(C_Nt1, C_Nt2, C_Nt1 + C_Nt2, C_kappat, G_f, G_r of N1 N1 <-> N2 N2) from network.jl's unit rates.
The eliminated reactions have zero weight in all of these and are left out: their energy
weight is only zero to rounding (~1e-14 m), which their fluxes (up to 1e20 H) would amplify.
"""
function reduced_collision(net::Network, urs, lnA, lnphi)
    m1, m2, mA, mphi = net.masses
    dA, dphi = mA - m1 - m2, mphi - 2m1 - 2m2
    C1 = C2 = Ctot = CK = 0.0
    Gcf = Gcr = 0.0
    for ur in urs
        ur.dN in ELIMINATED && continue
        G, K = rate_and_kinetic(ur, lnA, lnphi)
        w = ur.dN[3] + 2ur.dN[4]
        C1 += (ur.dN[1] + w) * G
        C2 += (ur.dN[2] + w) * G
        Ctot += (ur.dN[1] + ur.dN[2] + 2w) * G
        CK += K + (dA * ur.dN[3] + dphi * ur.dN[4]) * G
        ur.dN == (-2, 2, 0, 0) && ((Gcf, Gcr) = (exp(ur.lnGf), exp(ur.lnGr)))
    end
    return C1, C2, Ctot, CK, Gcf, Gcr
end

function reduced_rhs!(du, u, net::Network, z)
    x = exp(z)
    T_nu = net.p.m1 / x
    H = net.H_interp_z(z)
    xfac = x / net.dx_dt_interp_z(z)
    Nt1, Nt2, kappat = reduced_state(net, u, z)
    n1, n2, lnA, lnphi, T_N = reduced_closure(net, Nt1, Nt2, kappat)
    if !(isfinite(T_N) && n1 > 0 && n2 > 0)
        du .= 1e10     # a trial state outside the closure: large but finite, the step is rejected
        return nothing
    end
    urs = unit_rates(net, T_N, T_nu, n1, n2)
    C1, C2, Ctot, CK, Gcf, Gcr = reduced_collision(net, urs, lnA, lnphi)
    du[1] = xfac * Ctot / (Nt1 + Nt2)
    du[2] = xfac * (C2 / Nt2 - C1 / Nt1)
    lamc = xfac * 2 * (Gcf + Gcr) * (1 / n1 + 1 / n2)
    if lamc > LAMBDA_LO
        w = min(1.0, log(lamc / LAMBDA_LO) / log(LAMBDA_HI / LAMBDA_LO))
        ueq = log(n_scaled_safe(net.masses[2], net.dofs[2], T_N) / n_scaled_safe(net.masses[1], net.dofs[1], T_N))
        du[2] = (1 - w) * du[2] - w * min(lamc, LAMBDA_HI) * (u[2] - ueq)
    end
    P = (n1 + n2 + exp(lnA) + exp(lnphi)) * T_N
    du[3] = xfac * (H * (kappat - 3P) + CK) / kappat
    return nothing
end

"""
Initial state: the DW number and energy of `network_initial_state`, with N2/N1 as seeded there.
A' and phi need no seed: they follow from the closure, which also removes the phi-seed
transient of the full network at large g (CLAUDE.md).
"""
function reduced_initial_state(net::Network, dw::DodelsonWidrow, tT_rel::TimeTempRelation; dz_seed=1e-3)
    u0, z0 = network_initial_state(net, dw, tT_rel; dz_seed=dz_seed)
    n, _ = state_densities(net, u0, z0)
    s = net.ent_interp_z(z0)
    r = n[2] / n[1]
    Nt1 = dw.n_ic / (1 + r)
    Nt2 = dw.n_ic - Nt1
    kappat = dw.rho_ic - net.masses[1] * Nt1 - net.masses[2] * Nt2
    return [log(dw.n_ic / s), log(r), log(kappat / s^(4 / 3))], z0
end

"""
    solve_reduced_network(net, dw, tT_rel; x_end=100, reltol=1e-6, abstol=1e-8, fmax_stop=0.5, dz_seed=1e-3)

As `solve_network`: Rodas4P with a finite-difference Jacobian, stopping (retcode Terminated) if
the dark sector becomes degenerate. Errors if an eliminated decay is closed (`reduced_applies`).
"""
function solve_reduced_network(net::Network, dw::DodelsonWidrow, tT_rel::TimeTempRelation;
                               x_end=100.0, reltol=1e-6, abstol=1e-8, fmax_stop=0.5, dz_seed=1e-3, kw...)
    reduced_applies(net.p) || error("reduced network needs m_A' > m_N1 + m_N2 and m_phi > 2 m_A'")
    u0, z0 = reduced_initial_state(net, dw, tT_rel; dz_seed=dz_seed)
    prob = DE.ODEProblem(reduced_rhs!, u0, (z0, log(x_end)), net)
    degenerate = DE.DiscreteCallback((u, t, integ) -> maximum(reduced_diagnostics(net, u, t).fmax) > fmax_stop,
                                     DE.terminate!; save_positions=(false, true))
    bad(u, p, t) = any(x -> !isfinite(x) || x > 700, u)
    return DE.solve(prob, ODER.Rodas4P(autodiff=AutoFiniteDiff()); reltol=reltol, abstol=abstol,
                    isoutofdomain=bad, callback=degenerate, kw...)
end

"""Omega h^2 of the state: every dressed N with its mass."""
function reduced_omega_h2(net::Network, u, z)
    Nt1, Nt2, _ = reduced_state(net, u, z)
    return (net.masses[1] * Nt1 + net.masses[2] * Nt2) / net.ent_interp_z(z) * s0 / rho_crit0_h2
end

"""As `network_diagnostics`: also the densities n = (N1, N2, A', phi)."""
function reduced_diagnostics(net::Network, u, z)
    Nt1, Nt2, kappat = reduced_state(net, u, z)
    n1, n2, lnA, lnphi, T_N = reduced_closure(net, Nt1, Nt2, kappat)
    n = (n1, n2, exp(lnA), exp(lnphi))
    fmax = ntuple(k -> n[k] / n_scaled_safe(net.masses[k], net.dofs[k], T_N), NSPECIES)
    return (z=z, x=exp(z), T_N=T_N, T_nu=net.p.m1 * exp(-z), n=n, fmax=fmax, omega_h2=reduced_omega_h2(net, u, z))
end
