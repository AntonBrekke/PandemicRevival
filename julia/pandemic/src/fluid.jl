"""
fluid.jl

One-fluid solver of the dark sector, used for g >= 1e-3 (notes/model_limitations.tex,
Sec. "Large g"). Where every dark-internal rate is far above H, the dark sector is one
fluid in kinetic and chemical equilibrium,

    T_N1 = T_N2 = T_A' = T_phi = T_N,   mu_N1 = mu_N2 = mu,   mu_A' = 2 mu,   mu_phi = 4 mu,

with exact Fermi-Dirac / Bose-Einstein thermodynamics. The phi gap is
    (m_phi - mu_phi)/T_N = (m_phi - 2 m_A')/T_N + 2 (m_A' - mu_A')/T_N,
positive for m_phi > 2 m_A' whenever the A' gap is: no condensation.

STATE
=====
u = [ln Y_n, ln Y_rho, ln x_N, eta] against z = ln(m_N/T_nu), with Y_n = n~/s,
Y_rho = rho/s^(4/3), x_N = m_N/T_N and e^eta = (m_A' - mu_A')/T_N. The last two are
algebraic (mass matrix diag(1, 1, 0, 0)): (T_N, mu) such that the equilibrium n~ and rho
equal the state. n~ = n_N1 + n_N2 + 2 n_A' + 4 n_phi is the number of N after all decays;
A' -> N1 N2, phi -> A'A' and N1 N1 <-> N2 N2 do not change it. Per reaction,
Delta n~ = sum_F w - sum_I w with w = (1, 1, 2, 4) for (N1, N2, A', phi) (0 for nu).

    d ln Y_n/dz   = (x/xdot) C_n~ / n~,
    d ln Y_rho/dz = (x/xdot) [H (rho - 3P) + C_rho] / rho.

COLLISION TERMS
===============
Exact FD/BE quadratures (the processes that set the relic density at x ~ 1):
- A' <-> N2 nu (`coll_3_12`), the inverse decay that starts the pandemic;
- A'A' <-> N1 N1, N2 N2 (`coll_12_34_legendre`);
- N1 nu <-> phi (`coll_3_12`).
From the rate tables of the network (rate_tables.jl), MB in the dark sector:
- the portal 2 -> 2 processes N2 nu -> N1 N2 (RIS subtracted), N1 nu -> N2 N2,
  N1 nu -> A'A', N2 nu -> A' phi, N1 nu -> phi phi;
- the dark number-changing processes N1 N2 <-> A' phi, A'A' <-> phi phi,
  N1 N1 <-> phi phi, phi -> N1 N1.
The table rates are evaluated at the fluid's MB fugacities e^(-gap_i), gap_i =
(m_i - mu_i)/T_N: R_f prod_I e^(-gap) - R_r prod_F e^(-gap). That drops the
quantum-statistical factors of the dark species, an O(f) correction (N1 nu -> N2 N2 is
0.82-0.88 of the MB rate while T_N >> m_N; `portal_scale` rescales the portal 2 -> 2
rates for that sensitivity), and keeps the fixed points exactly: at T_N = T_nu and
sum_I mu = sum_F mu both directions are equal.

The solve starts at T_dw with the Dodelson-Widrow number and energy density, thermal.
"""

import NonlinearSolve as NLS
import DifferentialEquations as DE
import OrdinaryDiffEqRosenbrock as ODER
import LinearAlgebra as LA

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "time_temp_relation.jl"))
include(joinpath(@__DIR__, "dodelson_widrow.jl"))
include(joinpath(@__DIR__, "densities.jl"))
include(joinpath(@__DIR__, "coll_3_12.jl"))
include(joinpath(@__DIR__, "coll_12_34_legendre.jl"))
include(joinpath(@__DIR__, "interpolation_helpers.jl"))
include(joinpath(@__DIR__, "rate_tables.jl"))

"""Delta n~ per forward reaction (weights 1, 1, 2, 4 for N1, N2, A', phi)."""
delta_ntilde(r::Reaction) = sum((1, 1, 2, 4) .* delta_N(r))

# Reactions computed exactly (FD/BE) by the solver itself, or with Delta n~ = 0
# and no energy exchange with nu: never taken from the tables.
const FLUID_EXACT = (:N2nu_A, :N1nu_phi, :N1N1_AA, :N2N2_AA)

mutable struct FluidSolver{FdX, FEnt, FH}
    mp::ModelParams{Float64}      # g = mp.y, theta_1 = mp.theta

    N1::Particle{Float64}
    N2::Particle{Float64}
    A::Particle{Float64}
    nu::Particle{Float64}
    phi::Particle{Float64}

    # Bounds on T_N/T_nu for trial states, see `in_physical_domain`.
    T_ratio_min::Float64
    T_ratio_max::Float64
    # Wall-clock deadline for the current solve (Inf = none), see `check_deadline`.
    deadline::Float64

    # background interpolants keyed on z = log(m_N/T_nu)
    dx_dt_interp_z::FdX
    ent_interp_z::FEnt
    H_interp_z::FH

    itp::RateInterpolants
    # table reactions in use: (reaction, Delta n~)
    reactions::Vector{Tuple{Reaction,Int}}
    # rate_params and table normalisations, for the couplings in `rates_key`
    rates_key::Tuple{Float64,Float64}
    p_rates::Any
    norms::Dict{Symbol,Tuple{Float64,Float64}}
    # factor on the portal 2 -> 2 table rates (1: as tabulated)
    portal_scale::Float64
end

"""
    FluidSolver(m_N, tT_rel, itp; mA_over_mN=2.5, mphi_over_mN=100, portal_scale=1)

`itp` = RateInterpolants of the tables at these mass ratios. The couplings are set
through `pan.mp = ModelParams{Float64}(g, theta_1)`.
"""
function FluidSolver(m_N::Float64, tT_rel::TimeTempRelation{Float64}, itp::RateInterpolants;
                     mA_over_mN=2.5, mphi_over_mN=100.0, portal_scale=1.0, T_ratio_min=1e-6, T_ratio_max=1e3)
    N1 = Particle{Float64}(m_N, 1, dof=2)
    N2 = Particle{Float64}(m_N, 1, dof=2)
    A = Particle{Float64}(mA_over_mN * m_N, -1, dof=3)
    nu = Particle{Float64}(0.0, 1, dof=2)
    phi = Particle{Float64}(mphi_over_mN * m_N, -1, dof=1)
    A.m > N1.m + N2.m || error("FluidSolver assumes A' -> N1 N2 is open")
    phi.m > 2A.m || error("FluidSolver assumes phi -> A'A' is open (m_phi > 2 m_A')")
    isapprox(itp.p_ref.mA / itp.p_ref.m1, mA_over_mN; rtol=1e-12) &&
        isapprox(itp.p_ref.mphi / itp.p_ref.m1, mphi_over_mN; rtol=1e-12) ||
        error("rate tables are for other mass ratios")
    reactions = Tuple{Reaction,Int}[]
    for r in REACTIONS
        r.name in FLUID_EXACT && continue
        (haskey(itp.dark, r.name) || haskey(itp.portal, r.name)) || continue
        dn = delta_ntilde(r)
        # dark-only reactions that keep n~ change nothing in the fluid
        (dn == 0 && r.kind in (:dark2, :dark12)) && continue
        push!(reactions, (r, dn))
    end
    z_grid = log.(x_grid_of(m_N, tT_rel.T_nu_grid))
    dx_i = z_interpolation(z_grid, dx_dt_grid_of(m_N, tT_rel.T_nu_grid, tT_rel.dT_nu_dt_grid))
    ent_i = z_interpolation(z_grid, tT_rel.ent_grid)
    H_i = z_interpolation(z_grid, tT_rel.hubble_grid)
    return FluidSolver{typeof(dx_i),typeof(ent_i),typeof(H_i)}(
        ModelParams{Float64}(1e-3, 1e-8), N1, N2, A, nu, phi, T_ratio_min, T_ratio_max, Inf,
        dx_i, ent_i, H_i, itp, reactions, (NaN, NaN), nothing, Dict{Symbol,Tuple{Float64,Float64}}(), portal_scale)
end

"""rate_params and table normalisations for the current pan.mp (cached)."""
function fluid_rates!(pan::FluidSolver)
    key = (pan.mp.y, pan.mp.theta)
    if key != pan.rates_key
        s2 = sin(2 * pan.mp.theta)^2
        pan.p_rates = rate_params(pan.N1.m, pan.mp.y, s2; mA_over_mN=pan.A.m / pan.N1.m,
                                  mphi_over_mN=pan.phi.m / pan.N1.m)
        pan.norms = rate_norms(pan.itp, pan.p_rates)
        pan.rates_key = key
    end
    return pan.p_rates
end

# --------------------------------------------------------------------------- state
T_N_from_ln_x_N(pan, ln_x_N) = pan.N1.m / exp(ln_x_N)

"""
    gap_A_from_eta(eta)

(m_A - mu_A)/T_N = e^eta, floored at 1e-300. Pass it as `gap` wherever A' densities
are evaluated, instead of rebuilding it as m_A/T_N - xi_A: that difference rounds to
zero (or below) once e^eta is below the float spacing of m_A/T_N, which shows up as a
spurious Bose condensation.
"""
gap_A_from_eta(eta) = max(exp(eta), 1e-300)

"""xi_N = mu/T_N from (eta, ln x_N), with mu_A' = 2 mu."""
xi_from_eta(pan, eta, ln_x_N) = (pan.A.m / pan.N1.m * exp(ln_x_N) - gap_A_from_eta(eta)) / 2

"""(m_phi - mu_phi)/T_N with mu_phi = 2 mu_A'."""
gap_phi_of(pan::FluidSolver, T_N, gap_A) = (pan.phi.m - 2 * pan.A.m) / T_N + 2 * gap_A

"""(T_N, (m_N - mu)/T_N) of N1 (= N2) in the state u."""
function N_temp_gap(pan::FluidSolver, u)
    x_N = exp(u[3])
    gap = x_N * (1 - pan.A.m / (2 * pan.N1.m)) + gap_A_from_eta(u[4]) / 2
    return T_N_from_ln_x_N(pan, u[3]), gap
end

function num_dens(pan::FluidSolver, T_N, xi_N; gap_A)
    n = number_density(pan.N1, T_N, xi_N) + number_density(pan.N2, T_N, xi_N) +
        2 * number_density(pan.A, T_N, 2 * xi_N; gap=gap_A) +
        4 * number_density(pan.phi, T_N, 4 * xi_N; gap=gap_phi_of(pan, T_N, gap_A))
    return max(n, 1e-300)
end

function energy_dens(pan::FluidSolver, T_N, xi_N; gap_A)
    rho = energy_density(pan.N1, T_N, xi_N) + energy_density(pan.N2, T_N, xi_N) +
          energy_density(pan.A, T_N, 2 * xi_N; gap=gap_A) +
          energy_density(pan.phi, T_N, 4 * xi_N; gap=gap_phi_of(pan, T_N, gap_A))
    return max(rho, 1e-300)
end

function rho_3P(pan::FluidSolver, T_N, xi_N; gap_A)
    return rho_3P_diff(pan.N1, T_N, xi_N) + rho_3P_diff(pan.N2, T_N, xi_N) +
           rho_3P_diff(pan.A, T_N, 2 * xi_N; gap=gap_A) +
           rho_3P_diff(pan.phi, T_N, 4 * xi_N; gap=gap_phi_of(pan, T_N, gap_A))
end

"""Pressure of the fluid, (rho - (rho - 3P))/3."""
pressure_dens(pan::FluidSolver, T_N, xi_N; gap_A) =
    (energy_dens(pan, T_N, xi_N; gap_A=gap_A) - rho_3P(pan, T_N, xi_N; gap_A=gap_A)) / 3

"""Densities (n_N1, n_N2, n_A', n_phi), T_N, xi_N and the A' gap of a state u."""
function fluid_state(pan::FluidSolver, u)
    T_N = T_N_from_ln_x_N(pan, u[3])
    xi_N = xi_from_eta(pan, u[4], u[3])
    gap_A = gap_A_from_eta(u[4])
    n = (number_density(pan.N1, T_N, xi_N), number_density(pan.N2, T_N, xi_N),
         number_density(pan.A, T_N, 2xi_N; gap=gap_A),
         number_density(pan.phi, T_N, 4xi_N; gap=gap_phi_of(pan, T_N, gap_A)))
    return n, T_N, xi_N, gap_A
end

"""Omega h^2 of state u at z, counting the N that A' and phi decay into."""
function omega_h2_of(pan::FluidSolver, u, z)
    n, _ = fluid_state(pan, u)
    return pan.N1.m * (n[1] + n[2] + 2n[3] + 4n[4]) / pan.ent_interp_z(z) * s0 / rho_crit0_h2
end

"""
    in_physical_domain(pan, u, T_nu)

Whether u is a plausible dark-sector state at neutrino temperature T_nu. The
parametrisation keeps T_N > 0 and mu_A' < m_A' for any finite u, but trial stages of the
stiff solver can still jump to absurd values (T_N/T_nu from 1e-70 to 1e8 have been seen),
where the densities and collision terms break down. Required: all components finite;
T_N <= T_ratio_max T_nu (the dark sector only receives energy from the DW population and
the neutrinos); T_N >= T_ratio_min min(T_nu, T_nu^2/m_N) (a decoupled dark sector cools
at most like a^-2); e^eta >= 1e-300.
"""
function in_physical_domain(pan, u, T_nu)
    all(isfinite, u) || return false
    u[4] >= log(1e-300) || return false
    ln_x_nu = log(pan.N1.m / T_nu)
    ln_T_ratio = ln_x_nu - u[3]
    return log(pan.T_ratio_min) - max(zero(ln_x_nu), ln_x_nu) <= ln_T_ratio <= log(pan.T_ratio_max)
end

"""Thrown from the RHS once `pan.deadline` (a `time()`) has passed: abandons a pathologically slow solve."""
struct SolveDeadlineExceeded <: Exception end

@inline check_deadline(pan) = (time() > pan.deadline && throw(SolveDeadlineExceeded()); nothing)

# --------------------------------------------------------------------------- collision terms
"""
(net A' decay rate density, and its moments with weights E_N2 and E_A') of
A' <-> N2 nu, FD/BE, from one quadrature.
"""
function A_N2nu_moments(pan::FluidSolver, T_nu, T_N, xi_N; gap_A)
    T_nu = oftype(T_N, T_nu)
    return coll_3_12_moments(pan.mp, pan.N2, pan.nu, pan.A, (T_N, T_nu, T_N), (xi_N, zero(xi_N), 2 * xi_N);
                             sq_amp_func=coll_A_Nnu_sq_amp,
                             gaps=(pan.N2.m / T_N - xi_N, pan.nu.m / T_nu, gap_A))
end

"""
Quadrature settings of `coll_12_34_legendre` for A'A' <-> NN: accurate to 2.5e-6 over
typical states of the dark sector, at ~2.3x lower cost than its defaults. The nodes are
fixed, so the error is smooth in the state.
"""
const LEGENDRE_KW_AA = (L=16, n_c=24, n_E=12, n_s=8, n_theta=12)

"""
C_n~ of A'A' <-> N1 N1 and A'A' <-> N2 N2: -2 (C_N1N1 + C_N2N2), the 2 from Delta n~ (an
A' counts as two N), with C = `coll_12_34_legendre`/4 (the 1/4 for the two pairs of
identical particles, as in the original code). C_N2N2 = C_N1N1 for m_N1 = m_N2.
"""
function C_n_AA(pan::FluidSolver, T_N, xi_N; gap_A)
    gap_N = pan.N1.m / T_N - xi_N
    C_AA_N1N1 = coll_12_34_legendre(pan.mp, pan.A, pan.A, pan.N1, pan.N1, (T_N, T_N, T_N, T_N),
                                    (2 * xi_N, 2 * xi_N, xi_N, xi_N); gaps=(gap_A, gap_A, gap_N, gap_N),
                                    LEGENDRE_KW_AA...) / 4
    return -2 * C_AA_N1N1 - 2 * C_AA_N1N1
end

"""
`coll_3_12_moments` for phi <-> N1 nu (weights 1, E_N1, E_phi): net decay rate density of
phi and its energy moments, FD/BE.
"""
function phi_N1nu_moments(pan::FluidSolver, T_nu, T_N, xi_N, gap_A)
    msq = _tree(:phi_N1nu).msq(fluid_rates!(pan))
    T_nu = oftype(T_N, T_nu)
    return coll_3_12_moments(pan.mp, pan.N1, pan.nu, pan.phi, (T_N, T_nu, T_N), (xi_N, zero(xi_N), 4 * xi_N);
                             sq_amp_func=(mp, p1, p2, p3) -> msq,
                             gaps=(pan.N1.m / T_N - xi_N, 0.0, gap_phi_of(pan, T_N, gap_A)))
end

"""
(C_n~, C_rho) of the table reactions at the fluid state: reaction rates at the MB
fugacities e^(-gap_i) (see the header), times Delta n~; C_rho the energy the dark sector
receives from nu (E_nu-weighted portal rates). `breakdown`: a Dict that receives
(C_n~, C_rho) per reaction.
"""
function table_collision(pan::FluidSolver, T_N, T_nu, xi_N, gap_A; breakdown=nothing)
    fluid_rates!(pan)
    itp = pan.itp
    m = pan.N1.m
    lxN, lxnu, lr = log(m / T_N), log(m / T_nu), log(T_N / T_nu)
    gaps = (pan.N1.m / T_N - xi_N, pan.N2.m / T_N - xi_N, gap_A, gap_phi_of(pan, T_N, gap_A))
    side(fs) = -sum(gaps[dark_index(f)] for f in fs)
    Cn = 0.0
    Crho = 0.0
    for (r, dn) in pan.reactions
        lnR, lnQ = pan.norms[r.name]
        if r.kind in (:dark2, :dark12)
            # reverse from detailed balance, R_r = R_f e^((m_F - m_I)/T_N), exactly: the two
            # splines of the table satisfy it only to ~1e-4 between their nodes
            t = itp.dark[r.name]
            dm = (mass_sum(r.F, pan.p_rates) - mass_sum(r.I, pan.p_rates)) / T_N
            a = t.lnRf(lxN) + lnR + side(r.I)
            b = dm + side(r.F) - side(r.I)
            # expm1 near equilibrium; far from it exp(a) can underflow while e^b overflows
            G = abs(b) < 1 ? exp(a) * -expm1(b) : exp(a) - exp(a + b)
            Q = 0.0
        else
            t = itp.portal[r.name]
            G = exp(t.lnRf(lxnu, lr) + lnR + side(r.I)) * t.ratio_Rf(lxnu, lr) -
                exp(t.lnRr(lxnu, lr) + lnR + side(r.F)) * t.ratio_Rr(lxnu, lr)
            Q = exp(t.lnQf(lxnu, lr) + lnQ + side(r.I)) * t.ratio_Qf(lxnu, lr) -
                exp(t.lnQr(lxnu, lr) + lnQ + side(r.F)) * t.ratio_Qr(lxnu, lr)
            if r.kind === :portal2
                G *= pan.portal_scale
                Q *= pan.portal_scale
            end
        end
        Cn += dn * G
        Crho += Q
        breakdown === nothing || (breakdown[r.name] = (dn * G, Q))
    end
    return Cn, Crho
end

"""
(C_n~, C_rho): A' <-> N2 nu, A'A' <-> NN, phi <-> N1 nu (Delta n~ = -3 per decay) and the
table reactions. `breakdown`: a Dict that receives (C_n~, C_rho) per process.
"""
function collision_terms(pan::FluidSolver, T_nu, T_N, xi_N; gap_A, breakdown=nothing)
    c_1, c_EN2, c_EA = A_N2nu_moments(pan, T_nu, T_N, xi_N; gap_A=gap_A)
    C_AA = C_n_AA(pan, T_N, xi_N; gap_A=gap_A)
    d_1, d_EN1, d_Ephi = phi_N1nu_moments(pan, T_nu, T_N, xi_N, gap_A)
    Cn = -c_1 + C_AA - 3 * d_1
    Crho = (-c_EA + c_EN2) + (-d_Ephi + d_EN1)
    if breakdown !== nothing
        breakdown[:N2nu_A] = (-c_1, -c_EA + c_EN2)
        breakdown[:NN_AA] = (C_AA, 0.0)
        breakdown[:N1nu_phi] = (-3 * d_1, -d_Ephi + d_EN1)
    end
    tn, trho = table_collision(pan, T_N, T_nu, xi_N, gap_A; breakdown=breakdown)
    return Cn + tn, Crho + trho
end

# --------------------------------------------------------------------------- solver
"""RHS in z = ln x: d/dz = (x/xdot) d/dt; the last two components are the algebraic constraints."""
function fluid_rhs!(du, u, pan::FluidSolver, z)
    check_deadline(pan)
    x = exp(z)
    if !in_physical_domain(pan, u, pan.N1.m / x)
        # large but finite: the step's error estimate is large and the step is rejected
        # (NaN would make the step-size controller set dt = NaN)
        du .= 1e10
        return nothing
    end
    H = pan.H_interp_z(z)
    ent = pan.ent_interp_z(z)
    dx_dt = pan.dx_dt_interp_z(z)
    n = exp(u[1]) * ent
    rho = exp(u[2]) * ent^(4 / 3)
    T_nu = pan.N1.m / x
    T_N = T_N_from_ln_x_N(pan, u[3])
    xi_N = xi_from_eta(pan, u[4], u[3])
    gap_A = gap_A_from_eta(u[4])
    coll_n, coll_rho = collision_terms(pan, T_nu, T_N, xi_N; gap_A=gap_A)
    du[1] = x * coll_n / (n * dx_dt)
    du[2] = x * (H * rho_3P(pan, T_N, xi_N; gap_A=gap_A) + coll_rho) / (rho * dx_dt)
    du[3] = u[1] - log(num_dens(pan, T_N, xi_N; gap_A=gap_A) / ent)
    du[4] = u[2] - log(energy_dens(pan, T_N, xi_N; gap_A=gap_A) / ent^(4 / 3))
    return nothing
end

"""How far the equilibrium (n~, rho) at u = (ln x_N, eta) is from (n_ic, rho_ic), in logs."""
function n_rho_root(u, params)
    pan = params.pan
    check_deadline(pan)
    T_N = T_N_from_ln_x_N(pan, u[1])
    xi_N = xi_from_eta(pan, u[2], u[1])
    gap_A = gap_A_from_eta(u[2])
    return [log(num_dens(pan, T_N, xi_N; gap_A=gap_A) / params.n_ic),
            log(energy_dens(pan, T_N, xi_N; gap_A=gap_A) / params.rho_ic)]
end

"""Thermal start at T_dw with the DW number and energy density."""
function initial_conditions(tT_rel, dw, pan::FluidSolver)
    z_0 = log(pan.N1.m / tT_rel.T_nu_grid[dw.i_ic])
    ent0 = pan.ent_interp_z(z_0)
    sol = NLS.solve(NLS.NonlinearProblem(n_rho_root, [z_0, 0.0], (pan=pan, n_ic=dw.n_ic, rho_ic=dw.rho_ic)))
    return [log(dw.n_ic / ent0), log(dw.rho_ic / ent0^(4 / 3)), sol.u[1], sol.u[2]]
end

"""Solve from T_dw to x_end of `dw`."""
function solve_fluid(tT_rel::TimeTempRelation, dw::DodelsonWidrow, pan::FluidSolver; reltol=1e-4, kw...)
    fluid_rates!(pan)
    z_lim = (log(pan.N1.m / tT_rel.T_nu_grid[dw.i_ic]), log(pan.N1.m / tT_rel.T_nu_grid[dw.i_end+1]))
    f = DE.ODEFunction(fluid_rhs!, mass_matrix=LA.Diagonal([1.0, 1.0, 0.0, 0.0]))
    prob = DE.ODEProblem(f, initial_conditions(tT_rel, dw, pan), z_lim, pan)
    return DE.solve(prob, ODER.Rodas4P(autodiff=AutoFiniteDiff()); reltol=reltol,
                    isoutofdomain=(u, p, t) -> !in_physical_domain(p, u, p.N1.m / exp(t)), kw...)
end

"""Whether Omega h^2 has settled over the last `frac` of the solution, to relative spread `tol`."""
function check_plateau(pan::FluidSolver, sol; frac=0.05, tol=1e-3)
    n = length(sol.t)
    idx = (n - min(n, max(3, ceil(Int, frac * n))) + 1):n
    tail = [omega_h2_of(pan, sol.u[i], sol.t[i]) for i in idx]
    return (maximum(tail) - minimum(tail)) / abs(sum(tail) / length(tail)) < tol
end

"""
    solve_relic(pan, tT_rel, g, s2; reltol=1e-4, wall=900, plateau_tol=5e-3)

One solve at coupling g and sin^2 2theta = s2, to x = 100, or to 300 and 1000 where
Omega h^2 is still moving at x = 100 (at large g the reverse portal reactions deplete it
slowly). Returns (sol, Omega h^2, plateau); (nothing, NaN, false) if the solve failed or
exceeded `wall` seconds. Omega h^2 is the final value even if it has not settled.
"""
function solve_relic(pan::FluidSolver, tT_rel, g, s2; reltol=1e-4, wall=900.0, plateau_tol=5e-3)
    th = asin(sqrt(s2)) / 2
    pan.mp = ModelParams{Float64}(g, th)
    pan.deadline = time() + wall
    try
        om, last = NaN, nothing
        for x_end in (100.0, 300.0, 1000.0)
            dw = DodelsonWidrow{Float64}(pan.N1.m, th, tT_rel; x_end=x_end)
            sol = solve_fluid(tT_rel, dw, pan; reltol=reltol)
            DE.successful_retcode(sol) || return nothing, NaN, false
            om, last = omega_h2_of(pan, sol.u[end], sol.t[end]), sol
            # Omega h^2 wobbles at ~1e-3 at reltol 1e-4 with the table rates on
            check_plateau(pan, sol; tol=plateau_tol) && return sol, om, true
        end
        return last, om, false
    catch e
        e isa SolveDeadlineExceeded || @warn "fluid solve failed at m=$(pan.N1.m) g=$g s2=$s2" exception = e
        return nothing, NaN, false
    finally
        pan.deadline = Inf
    end
end
