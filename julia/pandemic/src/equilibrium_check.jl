"""
equilibrium_check.jl

Rates of every process of notes/tree_amplitudes.tex along a solved history,
per particle and divided by H, to check the assumptions of the solver
(notes/equilibrium_check.tex):

A1/A2/A3  one T_N for N1, N2, A' and mu_N1 = mu_N2 = mu_A'/2 need fast
          elastic and conversion processes inside the dark sector:
          `conv` N1N1 -> N2N2, `el12` N1N2 -> N1N2 (full Breit-Wigner, i.e.
          including N1N2 -> A' -> N1N2; `el12_offshell` without the s-channel), `A_dec` A' -> N1N2 and `A_inv`
          N1N2 -> A' (per N1), `NN_AA` and
          `AA_NN` (A'A' <-> NN; number-changing, in the solver);
A5        dark number enters only through N2 nu -> A' (`portal_in`, per N2);
          the neglected portal processes, per target N, are N1 nu -> N2N2,
          N2 nu -> N1N2 and N1 nu -> A'A' without their s-channel (`:nonres`;
          the on-shell parts are portal_in and N1 nu -> phi, the off-shell
          s-channel remainder is not counted, so these are lower bounds),
          N2 nu -> A' phi,
          N1 nu -> phi phi, N1 nu -> phi; each changes
          n~ = n_N1 + n_N2 + 2 n_A' by `DELTA_NTILDE`;
          N1 A' -> N1 A' (crossing of N1N1 -> A'A') is left out: its u-channel N2
          goes on shell inside the physical region (u reaches (m_A' - m_N)^2 >
          m_N^2), the singularity of scattering on an unstable particle; that
          on-shell part is A' -> N1 N2 followed by N1 N2 -> A', i.e. A_dec and
          A_inv, which are what keeps A' in kinetic contact with N here.
A6        phi is absent: `phi_prod` is the phi production rate per dark
          particle (N1, N2, A') from all processes with phi in the final state.
All with Maxwell-Boltzmann statistics (thermal_rates.jl), the dark sector at
(T_N, mu) from the solution and nu at T_nu with mu = 0.
"""

include(joinpath(@__DIR__, "kinetic_decoupling.jl"))
include(joinpath(@__DIR__, "thermal_rates.jl"))

"""Change of n~ = n_N1 + n_N2 + 2 n_A' per reaction of each neglected portal process."""
const DELTA_NTILDE = (N1nu_N2N2=1, N2nu_N1N2=1, N1nu_AA=3, N2nu_Aphi=1, N1nu_phiphi=-1, N1nu_phi=-1)

_q(name) = only(filter(q -> q.name === name, TREE_PROCESSES))

"""σ tables of all processes used below, for the parameters `p`, up to T_max."""
function equilibrium_tables(p; T_max)
    return (
        conv=SigmaTable(_q(:N1N1_N2N2), p; T_max=T_max, n=200),
        el12=SigmaTable(_q(:N1N2_N1N2), p; T_max=T_max, n=200),
        NN_AA=SigmaTable(_q(:N1N1_AA), p; T_max=T_max, n=200),
        AA_NN=SigmaTable(reversed_process(_q(:N1N1_AA)), p; T_max=T_max, n=200),
        N1nu_N2N2=SigmaTable(_q(:N1nu_N2N2), p; T_max=T_max, n=200),
        N2nu_N1N2=SigmaTable(_q(:N2nu_N1N2), p; T_max=T_max, n=200),
        N1nu_AA=SigmaTable(_q(:N1nu_AA), p; T_max=T_max, n=200),
        N2nu_Aphi=SigmaTable(_q(:N2nu_Aphi), p; T_max=T_max, n=200),
        N1nu_phiphi=SigmaTable(_q(:N1nu_phiphi), p; T_max=T_max, n=200),
        AA_phiphi=SigmaTable(_q(:AA_phiphi), p; T_max=T_max, n=200),
        N1N1_phiphi=SigmaTable(_q(:N1N1_phiphi), p; T_max=T_max, n=200),
        N1N2_Aphi=SigmaTable(_q(:N1N2_Aphi), p; T_max=T_max, n=200),
    )
end

"""
    rates_over_H(z, pan, sol, p, tabs)

NamedTuple of Gamma/H at z (see the header), plus T_N/T_nu, m/T_N, and the
fractions of N1 and A' in the dark-sector particle number.
"""
function rates_over_H(z, pan, sol, p, tabs)
    u = sol(z)::Vector{Float64}
    T_N, gap_N = N_temp_gap(pan, u)
    gap_A = gap_A_from_eta(u[4])
    T_nu = pan.N1.m * exp(-z)
    H = pan.H_interp_z(z)
    N1 = Species(p.m1, 2, T_N, gap_N)
    N2 = Species(p.m2, 2, T_N, gap_N)
    A = Species(p.mA, 3, T_N, gap_A)
    nu = Species(0.0, 2, T_nu, 0.0)
    # particle-number fractions, in logs (each density can underflow)
    wN = log(n_scaled(N1)) - gap_N
    wA = log(n_scaled(A)) - gap_A
    fA = 1 / (1 + 2 * exp(wN - wA))
    fN1 = (1 - fA) / 2
    r2(tab, a, b; prop=:ris) = rate_2to2(tab, a, b; propagator=prop)
    A_phi_inv = rate_inverse_decay(:phi_AA, p, A, A)[1]
    phi_prod = (r2(tabs.AA_phiphi, A, A)[1] * fA + r2(tabs.N1N1_phiphi, N1, N1)[1] * fN1 +
                r2(tabs.N1N2_Aphi, N1, N2)[1] * fN1 + r2(tabs.N2nu_Aphi, N2, nu)[1] * fN1 +
                2 * r2(tabs.N1nu_phiphi, N1, nu)[1] * fN1 + A_phi_inv * fA / 2 +
                rate_inverse_decay(:phi_N1N1, p, N1, N1)[1] * fN1 / 2 +
                rate_inverse_decay(:phi_N1nu, p, N1, nu)[1] * fN1)
    return (
        z=z, x=exp(z), TN_Tnu=T_N / T_nu, xd=p.m1 / T_N, fA=fA,
        H=H, dt_dz=dt_dz(pan, z), ln_nN=wN,
        conv=r2(tabs.conv, N1, N1)[1] / H,
        el12=r2(tabs.el12, N1, N2; prop=:bw)[1] / H,
        el12_offshell=r2(tabs.el12, N1, N2; prop=:nonres)[1] / H,
        A_dec=rate_decay(:A_N1N2, p, T_N) / H,
        A_inv=rate_inverse_decay(:A_N1N2, p, N1, N2)[1] / H,
        NN_AA=r2(tabs.NN_AA, N1, N1)[1] / H,
        AA_NN=r2(tabs.AA_NN, A, A)[1] / H,
        portal_in=rate_inverse_decay(:A_N2nu, p, N2, nu)[1] / H,
        A_to_N2nu=rate_decay(:A_N2nu, p, T_N) / H,
        N1nu_N2N2=r2(tabs.N1nu_N2N2, N1, nu)[1] / H,
        N2nu_N1N2=r2(tabs.N2nu_N1N2, N2, nu; prop=:nonres)[1] / H,
        N1nu_AA=r2(tabs.N1nu_AA, N1, nu; prop=:nonres)[1] / H,
        N2nu_Aphi=r2(tabs.N2nu_Aphi, N2, nu)[1] / H,
        N1nu_phiphi=r2(tabs.N1nu_phiphi, N1, nu)[1] / H,
        N1nu_phi=rate_inverse_decay(:phi_N1nu, p, N1, nu)[1] / H,
        phi_prod=phi_prod / H,
    )
end

"""
    equilibrium_history(pan, sol, p; n_z=60)

`rates_over_H` on a uniform grid in z over the solution.
"""
function equilibrium_history(pan, sol, p; n_z=60)
    z_i, z_f = sol.t[1], sol.t[end]
    T_max = max(pan.N1.m * exp(-z_i), maximum(N_temp_gap(pan, sol(z))[1] for z in range(z_i, z_f, length=50)))
    tabs = equilibrium_tables(p; T_max=T_max)
    return [rates_over_H(z, pan, sol, p, tabs) for z in range(z_i, z_f, length=n_z)]
end

"""
    equilibrium_summary(pan, sol, hist)

Per-point summary of a history from `equilibrium_history`:
- z_99: end of the production window, where ln Y_n is within 0.01 of its final value;
- min over the window of Gamma/H for each dark-sector process, and the value at the start;
- the neglected portal processes: max over the window of |Delta n~| Gamma_k
  relative to the included Gamma(N2 nu -> A'), and the e-folds of growth of n~
  each contributes over the window (`efolds_*`, see below), with those of the
  included process (`efolds_portal_in`);
- F_phi = ∫ Gamma_phi_prod dt over the whole solution (fraction of the dark
  particles that would have been converted into phi, without back-reaction).
"""
function equilibrium_summary(pan, sol, hist)
    lnY_end = sol.u[end][1]
    zs = [h.z for h in hist]
    i99 = something(findfirst(z -> sol(z)[1] >= lnY_end - 0.01, zs), length(zs))
    win = hist[1:max(i99, 1)]
    dz = zs[2] - zs[1]
    keys_in = (:conv, :el12, :el12_offshell, :A_dec, :A_inv, :NN_AA, :AA_NN)
    mins = NamedTuple{Symbol.(:min_, keys_in)}(Tuple(minimum(getfield(h, k) for h in win) for k in keys_in))
    starts = NamedTuple{Symbol.(:start_, keys_in)}(Tuple(getfield(hist[1], k) for k in keys_in))
    portal = keys(DELTA_NTILDE)
    rel(h, k) = abs(DELTA_NTILDE[k]) * getfield(h, k) / max(h.portal_in, 1e-300)
    maxrel = NamedTuple{Symbol.(:maxrel_, portal)}(Tuple(maximum(rel(h, k) for h in win) for k in portal))
    # e-folds of growth of n~ from each process over the window,
    # E_k = ∫ |Delta n~_k| Gamma_k (n_target/n~) dt, Gamma = (Gamma/H) H, with
    # n_target/n~ = n_N1/(n_N1 + n_N2 + 2 n_A') = f_N1/(1 + f_A). The pandemic
    # grows n~ exponentially, so these (not density-weighted integrals) measure
    # how much each process changes the final abundance: e^(E_k).
    ef(h, Gk) = Gk * h.H * h.dt_dz * dz * ((1 - h.fA) / 2) / (1 + h.fA)
    efolds_in = sum(ef(h, h.portal_in) for h in win)
    efolds = NamedTuple{Symbol.(:efolds_, portal)}(Tuple(
        sum(ef(h, abs(DELTA_NTILDE[k]) * getfield(h, k)) for h in win) for k in portal))
    F_phi = sum(h.phi_prod * h.H * h.dt_dz for h in hist) * dz    # ∫ Gamma_phi dt
    return merge((z_99=zs[i99], x_99=exp(zs[i99]), TN_Tnu_99=hist[i99].TN_Tnu, xd_99=hist[i99].xd,
                  F_phi=F_phi, max_phi_prod=maximum(h.phi_prod for h in hist)),
                 (efolds_portal_in=efolds_in,), mins, starts, maxrel, efolds)
end
