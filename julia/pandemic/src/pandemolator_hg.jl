"""
pandemolator_hg.jl

One-fluid dark sector for large g (notes/model_limitations.tex, Sec. "Plan for
large g", step 1). Where every dark-internal rate is far above H, the dark sector
is one fluid in kinetic and chemical equilibrium,

    T_N1 = T_N2 = T_A' = T_phi = T_N,   mu_N1 = mu_N2 = mu,   mu_A' = 2 mu,   mu_phi = 4 mu,

with exact Fermi-Dirac / Bose-Einstein thermodynamics. This is the reduced solver
of pandemolator_z.jl (same state, same DAE, same z = ln(m_N/T_nu)) extended by

- phi in n~, rho and P (Bose-Einstein, 1 dof). Its gap is
      (m_phi - mu_phi)/T_N = (m_phi - 2 m_A')/T_N + 2 (m_A' - mu_A')/T_N,
  positive for m_phi > 2 m_A' whenever the A' gap is: no condensation;
- the inverse decay N1 nu <-> phi, exact FD/BE (`coll_3_12`, as A' <-> N2 nu);
- the portal 2 -> 2 processes N2 nu -> N1 N2 (RIS subtracted), N1 nu -> N2 N2,
  N1 nu -> A'A', N2 nu -> A' phi, N1 nu -> phi phi, and the dark number-changing
  processes N1 N2 <-> A' phi, A'A' <-> phi phi, N1 N1 <-> phi phi, phi -> N1 N1,
  from the rate tables of the network (rate_tables.jl).

n~ = n_N1 + n_N2 + 2 n_A' + 4 n_phi is the number of N after all decays; A' -> N1 N2,
phi -> A'A' and N1 N1 <-> N2 N2 do not change it. Per reaction, Delta n~ is
sum_F w - sum_I w with w = (1, 1, 2, 4) for (N1, N2, A', phi) (and 0 for nu).

THE TABLE RATES
===============
The tables are for MB dark species (FD nu, with Pauli blocking of a final nu).
Here they are evaluated at the fluid's MB fugacities e^(-gap_i), gap_i = (m_i -
mu_i)/T_N: a reaction rate is R_f prod_I e^(-gap) - R_r prod_F e^(-gap). That
drops the quantum-statistical factors of the dark species (f instead of 1 -+ f in
the final state, FD/BE instead of MB in the initial state), an O(f) correction to
the rates, while it keeps the fixed points exactly: at T_N = T_nu and
sum_I mu = sum_F mu both directions are equal, as with the exact factors.
The processes that set the relic density at x ~ 1 (A' <-> N2 nu, A'A' <-> NN) stay
exact, as in the reduced solver; `test/large_g_blocking.jl` estimates the error
of the others.

With `with_phi = false` and `with_tables = false` this is the reduced solver,
term by term (test/test_pandemolator_hg.jl).
"""

include(joinpath(@__DIR__, "pandemolator_z.jl"))
include(joinpath(@__DIR__, "rate_tables.jl"))

"""Delta n~ per forward reaction (weights 1, 1, 2, 4 for N1, N2, A', phi)."""
delta_ntilde(r::Reaction) = sum((1, 1, 2, 4) .* delta_N(r))

# Reactions computed exactly (FD/BE) by the solver itself, or with Delta n~ = 0
# and no energy exchange with nu: never taken from the tables.
const HG_EXACT = (:N2nu_A, :N1nu_phi, :N1N1_AA, :N2N2_AA)

mutable struct PandemolatorHG{T<:Real, FT, FdX, FEnt, FH}
    mp::ModelParams{T}

    N1::Particle{T}
    N2::Particle{T}
    A::Particle{T}
    nu::Particle{T}
    phi::Particle{T}

    fac_n_A::Int64
    verbose::Bool

    T_ratio_min::Float64
    T_ratio_max::Float64
    deadline::Float64

    t_interp_z::FT
    dx_dt_interp_z::FdX
    ent_interp_z::FEnt
    H_interp_z::FH

    with_phi::Bool
    with_tables::Bool
    itp::Union{Nothing,RateInterpolants}
    # table reactions in use: (reaction, Delta n~)
    reactions::Vector{Tuple{Reaction,Int}}
    # rate_params and table normalisations, for the couplings in `rates_key`
    rates_key::Tuple{Float64,Float64}
    p_rates::Any
    norms::Dict{Symbol,Tuple{Float64,Float64}}
    # factor on the portal 2 -> 2 table rates (1: as tabulated). test/large_g_blocking.jl finds
    # the exact FD/MB ratio of N1 nu -> N2 N2 at 0.82-0.88 while T_N >> m_N, mu ~ 0, and 1 later.
    portal_scale::Float64
end

"""
    PandemolatorHG(m_N, tT_rel, itp; mA_over_mN=2.5, mphi_over_mN=100, with_phi=true, with_tables=true)

`itp` = RateInterpolants of the tables at these mass ratios (`nothing` only with
with_tables = false). Couplings are set through `pan.mp` (g = mp.y, theta_1 = mp.theta),
as for PandemolatorZ.
"""
function PandemolatorHG(m_N::Float64, tT_rel::TimeTempRelation{Float64}, itp;
                        mA_over_mN=2.5, mphi_over_mN=100.0, with_phi=true, with_tables=true, verbose=false,
                        T_ratio_min=1e-6, T_ratio_max=1e3, portal_scale=1.0)
    N1 = Particle{Float64}(m_N, 1, dof=2)
    N2 = Particle{Float64}(m_N, 1, dof=2)
    A = Particle{Float64}(mA_over_mN * m_N, -1, dof=3)
    nu = Particle{Float64}(0.0, 1, dof=2)
    phi = Particle{Float64}(mphi_over_mN * m_N, -1, dof=1)
    A.m > N1.m + N2.m || error("PandemolatorHG assumes A' -> N1 N2 is open")
    (!with_phi || phi.m > 2A.m) || error("PandemolatorHG assumes phi -> A'A' is open (m_phi > 2 m_A')")
    if with_tables
        itp === nothing && error("with_tables needs the rate tables")
        isapprox(itp.p_ref.mA / itp.p_ref.m1, mA_over_mN; rtol=1e-12) &&
            isapprox(itp.p_ref.mphi / itp.p_ref.m1, mphi_over_mN; rtol=1e-12) ||
            error("rate tables are for other mass ratios")
    end
    reactions = Tuple{Reaction,Int}[]
    if with_tables
        for r in REACTIONS
            r.name in HG_EXACT && continue
            (haskey(itp.dark, r.name) || haskey(itp.portal, r.name)) || continue
            (!with_phi && (:phi in r.I || :phi in r.F)) && continue
            dn = delta_ntilde(r)
            # dark-only reactions that keep n~ change nothing in the fluid
            (dn == 0 && r.kind in (:dark2, :dark12)) && continue
            push!(reactions, (r, dn))
        end
    end
    z_grid = log.(x_grid_of(m_N, tT_rel.T_nu_grid))
    dx_dt_grid = dx_dt_grid_of(m_N, tT_rel.T_nu_grid, tT_rel.dT_nu_dt_grid)
    t_i = z_interpolation(z_grid, tT_rel.t_grid)
    dx_i = z_interpolation(z_grid, dx_dt_grid)
    ent_i = z_interpolation(z_grid, tT_rel.ent_grid)
    H_i = z_interpolation(z_grid, tT_rel.hubble_grid)
    return PandemolatorHG{Float64,typeof(t_i),typeof(dx_i),typeof(ent_i),typeof(H_i)}(
        ModelParams{Float64}(1e-3, 1e-8), N1, N2, A, nu, phi, 2, verbose, T_ratio_min, T_ratio_max, Inf,
        t_i, dx_i, ent_i, H_i, with_phi, with_tables, itp, reactions, (NaN, NaN), nothing,
        Dict{Symbol,Tuple{Float64,Float64}}(), portal_scale)
end

"""rate_params and table normalisations for the current pan.mp (cached)."""
function hg_rates!(pan::PandemolatorHG)
    key = (pan.mp.y, pan.mp.theta)
    if key != pan.rates_key
        s2 = sin(2 * pan.mp.theta)^2
        pan.p_rates = rate_params(pan.N1.m, pan.mp.y, s2; mA_over_mN=pan.A.m / pan.N1.m,
                                  mphi_over_mN=pan.phi.m / pan.N1.m)
        pan.with_tables && (pan.norms = rate_norms(pan.itp, pan.p_rates))
        pan.rates_key = key
    end
    return pan.p_rates
end

"""(m_phi - mu_phi)/T_N with mu_phi = 2 mu_A'."""
gap_phi_of(pan::PandemolatorHG, T_N, gap_A) = (pan.phi.m - 2 * pan.A.m) / T_N + 2 * gap_A

# --------------------------------------------------------------------------- thermodynamics
# These specialise the untyped functions of pandemolator_common.jl, so that
# n_rho_root (the initial condition) picks them up unchanged. gap_A is always given.

function num_dens(pan::PandemolatorHG, T_N, xi_N; debug=false, gap_A=nothing)
    n = number_density(pan.N1, T_N, xi_N) + number_density(pan.N2, T_N, xi_N) +
        2 * number_density(pan.A, T_N, 2 * xi_N; gap=gap_A)
    if pan.with_phi
        n += 4 * number_density(pan.phi, T_N, 4 * xi_N; gap=gap_phi_of(pan, T_N, gap_A))
    end
    return max(n, 1e-300)
end

function energy_dens(pan::PandemolatorHG, T_N, xi_N; gap_A=nothing)
    rho = energy_density(pan.N1, T_N, xi_N) + energy_density(pan.N2, T_N, xi_N) +
          energy_density(pan.A, T_N, 2 * xi_N; gap=gap_A)
    if pan.with_phi
        rho += energy_density(pan.phi, T_N, 4 * xi_N; gap=gap_phi_of(pan, T_N, gap_A))
    end
    return max(rho, 1e-300)
end

function rho_3P(pan::PandemolatorHG, T_N, xi_N; gap_A=nothing)
    r = rho_3P_diff(pan.N1, T_N, xi_N) + rho_3P_diff(pan.N2, T_N, xi_N) +
        rho_3P_diff(pan.A, T_N, 2 * xi_N; gap=gap_A)
    if pan.with_phi
        r += rho_3P_diff(pan.phi, T_N, 4 * xi_N; gap=gap_phi_of(pan, T_N, gap_A))
    end
    return r
end

# --------------------------------------------------------------------------- collision terms
"""
`coll_3_12_moments` for phi <-> N1 nu (weights 1, E_N1, E_phi): net decay rate
density of phi and its energy moments, FD/BE.
"""
function phi_N1nu_moments(pan::PandemolatorHG, T_nu, T_N, xi_N, gap_A)
    p = hg_rates!(pan)
    msq = _tree(:phi_N1nu).msq(p)
    T_nu = oftype(T_N, T_nu)
    return coll_3_12_moments(pan.mp, pan.N1, pan.nu, pan.phi, (T_N, T_nu, T_N), (xi_N, zero(xi_N), 4 * xi_N);
                             sq_amp_func=(mp, p1, p2, p3) -> msq,
                             gaps=(pan.N1.m / T_N - xi_N, 0.0, gap_phi_of(pan, T_N, gap_A)))
end

"""
(C_n~, C_rho) of the table reactions at the fluid state: reaction rates at the MB
fugacities e^(-gap_i) (see the header), times Delta n~; C_rho the energy the dark
sector receives from nu (E_nu-weighted portal rates). `breakdown`: a Dict that
receives (C_n~, C_rho) per reaction.
"""
function table_collision(pan::PandemolatorHG, T_N, T_nu, xi_N, gap_A; breakdown=nothing)
    pan.with_tables || return 0.0, 0.0
    hg_rates!(pan)
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
(C_n~, C_rho): the reduced solver's A' <-> N2 nu and A'A' <-> NN, plus phi <-> N1 nu
(Delta n~ = -3 per decay) and the table reactions.
"""
function collision_terms(pan::PandemolatorHG, T_nu, T_N, xi_N; gap_A=nothing, breakdown=nothing)
    c_1, c_EN2, c_EA = A_N2nu_moments(pan, T_nu, T_N, xi_N; gap_A=gap_A)
    C_AA = C_n_AA(pan, T_N, xi_N; gap_A=gap_A)
    Cn = -c_1 + C_AA
    Crho = -c_EA + c_EN2
    if breakdown !== nothing
        breakdown[:N2nu_A] = (-c_1, -c_EA + c_EN2)
        breakdown[:NN_AA] = (C_AA, 0.0)
    end
    if pan.with_phi
        d_1, d_EN1, d_Ephi = phi_N1nu_moments(pan, T_nu, T_N, xi_N, gap_A)
        Cn += -3 * d_1
        Crho += -d_Ephi + d_EN1
        breakdown === nothing || (breakdown[:N1nu_phi] = (-3 * d_1, -d_Ephi + d_EN1))
    end
    if pan.with_tables
        tn, trho = table_collision(pan, T_N, T_nu, xi_N, gap_A; breakdown=breakdown)
        Cn += tn
        Crho += trho
    end
    return Cn, Crho
end

# --------------------------------------------------------------------------- solver
"""RHS in z, as `mm_func_z!` (pandemolator_z.jl) with the PandemolatorHG physics."""
function mm_func_hg!(du, u, pan::PandemolatorHG, z)
    check_deadline(pan)
    x = exp(z)
    if !in_physical_domain(pan, u, pan.N1.m / x)
        reject_state!(du)
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

"""Thermal start at T_dw with the DW number and energy (as `initial_conditions_z`)."""
function initial_conditions_hg(tT_rel, dw, pan::PandemolatorHG)
    z_0 = log(pan.N1.m / tT_rel.T_nu_grid[dw.i_ic])
    ent0 = pan.ent_interp_z(z_0)
    sol = NLS.solve(NLS.NonlinearProblem(n_rho_root, [z_0, 0.0], (pan=pan, n_ic=dw.n_ic, rho_ic=dw.rho_ic)))
    return [log(dw.n_ic / ent0), log(dw.rho_ic / ent0^(4 / 3)), sol.u[1], sol.u[2]]
end

"""Solve from T_dw to x_end of `dw` (same interface as `pandemolate_z`)."""
function pandemolate_z(tT_rel::TimeTempRelation{T}, dw::DodelsonWidrow{T}, pan::PandemolatorHG{T};
                       reltol=nothing, abstol=nothing, kw...) where T <: Real
    hg_rates!(pan)
    z_lim = (log(pan.N1.m / tT_rel.T_nu_grid[dw.i_ic]), log(pan.N1.m / tT_rel.T_nu_grid[dw.i_end+1]))
    u0 = initial_conditions_hg(tT_rel, dw, pan)
    f = DE.ODEFunction(mm_func_hg!, mass_matrix=LA.Diagonal([1.0, 1.0, 0.0, 0.0]))
    prob = DE.ODEProblem(f, u0, z_lim, pan)
    opts = Dict{Symbol,Any}(kw)
    reltol === nothing || (opts[:reltol] = reltol)
    abstol === nothing || (opts[:abstol] = abstol)
    return DE.solve(prob, ODER.Rodas4P(autodiff=AutoFiniteDiff());
                    isoutofdomain=(u, p, t) -> !in_physical_domain(p, u, p.N1.m / exp(t)), opts...)
end

"""Densities (n_N1, n_N2, n_A', n_phi), T_N, xi_N of a solution state."""
function hg_state(pan::PandemolatorHG, u)
    T_N = T_N_from_ln_x_N(pan, u[3])
    xi_N = xi_from_eta(pan, u[4], u[3])
    gap_A = gap_A_from_eta(u[4])
    nN = number_density(pan.N1, T_N, xi_N)
    nA = number_density(pan.A, T_N, 2xi_N; gap=gap_A)
    nphi = pan.with_phi ? number_density(pan.phi, T_N, 4xi_N; gap=gap_phi_of(pan, T_N, gap_A)) : 0.0
    return (nN, number_density(pan.N2, T_N, xi_N), nA, nphi), T_N, xi_N, gap_A
end

"""Omega h^2 of state u at z, counting the N that A' and phi decay into."""
function omega_h2_of(pan::PandemolatorHG, u, z)
    n, _ = hg_state(pan, u)
    return pan.N1.m * (n[1] + n[2] + 2n[3] + 4n[4]) / pan.ent_interp_z(z) * s0 / rho_crit0_h2
end
