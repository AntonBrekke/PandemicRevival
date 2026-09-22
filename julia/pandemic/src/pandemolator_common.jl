include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "densities.jl"))
include(joinpath(@__DIR__, "coll_3_12.jl"))
# include(joinpath(@__DIR__, "coll_12_34.jl"))
include(joinpath(@__DIR__, "coll_12_34_legendre.jl"))

"""
pandemolator_common.jl

The dark-sector physics shared by every pandemolator parameterization
(pandemolator.jl, pandemolator_z.jl, ...): the algebraic-constraint side of
the DAE (T_N/xi_N from the state, number/energy densities, the equilibrium
root-finding objective) and the collision terms. None of it is specific to
which variable is integrated in or which coordinate the background
interpolants are keyed on -- every function here takes `pan` untyped (no
`Pandemolator`/`PandemolatorZ` annotation) and only touches fields common to
all of them (`N1`, `N2`, `A`, `nu`, `mp`, `fac_n_A`), so it works unchanged
for any parameterization that provides those fields.

Kept deliberately free of any particular solver's independent variable, ODE
setup, or interpolants -- that's what makes reuse across parameterizations
exact rather than approximate: the same functions are called, not
re-derived per file.
"""

function T_N_from_ln_x_N(pan, ln_x_N)
    return pan.N1.m / exp(ln_x_N)
end

"""
    gap_A_from_eta(eta)

(m_A - mu_A)/T_N = m_A/T_N - xi_A = e^eta, clamped away from zero as in
`xi_from_eta`. Pass it as `gap_A` wherever A' distributions or densities are
evaluated, instead of rebuilding it as m_A/T_N - xi_A: that difference rounds
to zero (or below) once e^eta is smaller than the float spacing of m_A/T_N,
which shows up as a spurious Bose condensation.
"""
gap_A_from_eta(eta) = max(exp(eta), 1e-300)

function xi_from_eta(pan, eta, ln_x_N)
    exp_eta = gap_A_from_eta(eta)
    return (pan.A.m / pan.N1.m * exp(ln_x_N) - exp_eta) / pan.fac_n_A
end

"""
    in_physical_domain(pan, u, T_nu)

Whether u = [ln Y_n, ln Y_rho, ln x_N, eta] is a plausible dark-sector state at
neutrino temperature T_nu. The parametrisation keeps T_N > 0 and xi_A < m_A/T_N
for any finite u, but trial stages of the stiff solver can still jump to
absurd values (T_N/T_nu from 1e-70 to 1e8 have been seen), where the
densities and collision terms break down. Required:
- all components finite;
- T_N <= pan.T_ratio_max * T_nu: the dark sector only receives energy from
  the Dodelson-Widrow population and the neutrinos;
- T_N >= pan.T_ratio_min * min(T_nu, T_nu^2 / m_N1): a decoupled dark sector
  cools at most like a^-2, i.e. T_N ∝ T_nu^2 once non-relativistic;
- e^eta >= 1e-300, the floor used in `gap_A_from_eta`.
"""
function in_physical_domain(pan, u, T_nu)
    all(isfinite, u) || return false
    u[4] >= log(1e-300) || return false
    ln_x_nu = log(pan.N1.m / T_nu)
    ln_T_ratio = ln_x_nu - u[3]
    return log(pan.T_ratio_min) - max(zero(ln_x_nu), ln_x_nu) <= ln_T_ratio <= log(pan.T_ratio_max)
end

"""
    check_deadline(pan)

Throws `SolveDeadlineExceeded` if `pan` has a `deadline` field (a wall-clock
time as returned by `time()`) that has passed. Called from the RHS and from the
initial-condition objective, so that a scan can abandon a pathologically slow
solve (e.g. at large mixing, where the steps can shrink without bound).
"""
struct SolveDeadlineExceeded <: Exception end

@inline function check_deadline(pan)
    if hasfield(typeof(pan), :deadline) && time() > pan.deadline
        throw(SolveDeadlineExceeded())
    end
    return nothing
end

"""
    reject_state!(du)

RHS for a state outside `in_physical_domain`: a large but finite derivative,
so that the step's error estimate is large and the step is rejected with a
smaller dt. NaN would make the PI step-size controller set dt = NaN, and
zeros would make the algebraic residuals look satisfied.
"""
function reject_state!(du)
    du .= 1e10
    return nothing
end

function rho_3P(pan, T_N, xi_N; gap_A=nothing)
    # Prepared for splitting N masses
    return (
        rho_3P_diff(pan.N1, T_N, xi_N)
        + rho_3P_diff(pan.N2, T_N, xi_N)
        + rho_3P_diff(pan.A, T_N, pan.fac_n_A * xi_N; gap=gap_A)
    )
end

function num_dens(pan, T_N, xi_N; debug=false, gap_A=nothing)
    n_N1 = number_density(pan.N1, T_N, xi_N, debug=debug)
    n_N2 = number_density(pan.N2, T_N, xi_N, debug=debug)
    n_A = number_density(pan.A, T_N, pan.fac_n_A * xi_N, debug=debug, gap=gap_A)
    if isnothing(n_A)
        number_density(pan.A, T_N, pan.fac_n_A * xi_N, debug=true, gap=gap_A)
    end
    if debug
        println("n_N1 = ", n_N1)
        println("n_N2 = ", n_N2)
        println("n_A = ", n_A)
    end
    return max(
        n_N1 + n_N2 + pan.fac_n_A * n_A,
        1e-300
    )
end

function energy_dens(pan, T_N, xi_N; gap_A=nothing)
    return max(
        (
            energy_density(pan.N1, T_N, xi_N)
            + energy_density(pan.N2, T_N, xi_N)
            + energy_density(pan.A, T_N, pan.fac_n_A * xi_N; gap=gap_A)
        ),
        1e-300
    )
end

"""
Objective for the initial-condition root-solve shared by
`initial_conditions`/`initial_conditions_z`: given (ln_x_N, eta), returns
how far (n, rho) implied by chemical equilibrium are from the
Dodelson-Widrow (n_ic, rho_ic), in log space.
"""
function n_rho_root(u, params)
    pan = params.pan
    check_deadline(pan)
    n_ic = params.n_ic
    rho_ic = params.rho_ic
    ln_x_N = u[1]
    eta = u[2]
    T_N = T_N_from_ln_x_N(pan, ln_x_N)
    xi_N = xi_from_eta(pan, eta, ln_x_N)
    gap_A = gap_A_from_eta(eta)
    n = num_dens(pan, T_N, xi_N; gap_A=gap_A)
    rho = energy_dens(pan, T_N, xi_N; gap_A=gap_A)
    if n / n_ic < 0
        if pan.verbose
            println("n/n_ic < 0 in n_rho_root")
            println("n/n_ic = ", n / n_ic)
            println("n = ", n)
            num_dens(pan, T_N, xi_N, debug=true, gap_A=gap_A)
        end
        return [log(1e-100), log(rho/rho_ic)]
    end
    if rho / rho_ic < 0
        if pan.verbose
            println("rho/rho_ic < 0 in n_rho_root")
            println("rho/rho_ic = ", rho / rho_ic)
        end
        return [log(n/n_ic), log(1e-100)]
    end
    return [log(n/n_ic), log(rho/rho_ic)]
end

# Dead code: not called from anywhere in the solver (n_rho_root is used
# instead, which solves for T_N and xi_N together). Kept as-is pending a
# decision on whether the single-variable (xi_N given T_N) special case
# they belong to is still wanted.
function n_root(xi_N, params)
    pan = params.pan
    n_ic = params.n_ic
    T_N = params.T_N
    n = num_dens(pan, T_N, xi_N)
    if n / n_ic < 0
        println("n/n_ic < 0 in n_root")
        return log(1e-100)
    end
    return log(n / n_ic)
end

function rho_root(xi_N, params)
    pan = params.pan
    n_ic = params.n_ic
    rho_ic = params.rho_ic
    T_N = params.T_N
    rho = energy_dens(pan, T_N, xi_N)
    if rho / rho_ic < 0
        println("rho/rho_ic < 0 in rho_root")
        return log(1e-100)
    end
    return log(rho / rho_ic)
end

"""
    collision_terms(pan, T_nu, T_N, xi_N)

(C_n, C_rho) by direct evaluation. This used to also dispatch to a
tabulated/interpolated fast path (a `CollisionTable` argument); that has
been removed for now -- see the design discussion on nondimensionalizing
the tabulation before reintroducing it (tabulate a dimensionless O(1)
residual against an analytic reference, not C_n itself).
"""
function collision_terms(pan, T_nu, T_N, xi_N; gap_A=nothing)
    # A' <-> N2 nu enters C_n with weight 1 and C_rho with weights E_A', E_N2;
    # all three come from a single quadrature (see `A_N2nu_moments`).
    c_1, c_EN2, c_EA = A_N2nu_moments(pan, T_nu, T_N, xi_N; gap_A=gap_A)
    return (-c_1 + C_n_AA(pan, T_N, xi_N; gap_A=gap_A), -c_EA + c_EN2)
end

"""
    A_N2nu_moments(pan, T_nu, T_N, xi_N; gap_A=nothing)

`coll_3_12` for A' <-> N2 nu with energy weights 1, E_N2 and E_A' (energy_type
Val(0), Val(1), Val(3)), from one quadrature via `coll_3_12_moments`. These
are the A' <-> N2 nu pieces of `C_n` (weight 1) and `C_rho` (E_A', E_N2).
"""
function A_N2nu_moments(pan, T_nu, T_N, xi_N; gap_A=nothing)
    T_nu = oftype(T_N, T_nu)
    temps_A_N2nu = (T_N, T_nu, T_N)
    xis_A_N2nu = (
        xi_N,
        zero(xi_N),
        pan.fac_n_A * xi_N,
    )
    return coll_3_12_moments(
        pan.mp,
        pan.N2,
        pan.nu,
        pan.A,
        temps_A_N2nu,
        xis_A_N2nu;
        sq_amp_func=coll_A_Nnu_sq_amp,
        gaps=gaps_A_N2nu(pan, T_nu, T_N, xi_N, gap_A),
    )
end

"""
Quadrature settings of `coll_12_34_legendre` used in the solver. Relative to a
high-order reference they are accurate to 2.5e-6 over typical states of the
dark sector (vs 1.5e-9 for the function's defaults), at ~2.3x lower cost.
Since the nodes are fixed, the error is smooth in the state, and far below
the ODE tolerances used.
"""
const LEGENDRE_KW_AA = (L=16, n_c=24, n_E=12, n_s=8, n_theta=12)

"""
    C_n_AA(pan, T_N, xi_N; gap_A=nothing)

Contribution of A'A' <-> N1 N1 and A'A' <-> N2 N2 to `C_n`.
"""
function C_n_AA(pan, T_N, xi_N; gap_A=nothing)
    # TODO: [01.06.26] Why divide by 4 and multiply by 4 in return statement? Symmetry factor (2*2)
    # TODO: [15.07.26] Double check sign of collision term
    temps_AA_NN = (T_N, T_N, T_N, T_N)
    xis_AA_NN = (
        pan.fac_n_A * xi_N,
        pan.fac_n_A * xi_N,
        xi_N,
        xi_N,
    )
    gaps_AA_N1N1 = isnothing(gap_A) ? nothing : (gap_A, gap_A, pan.N1.m / T_N - xi_N, pan.N1.m / T_N - xi_N)
    gaps_AA_N2N2 = isnothing(gap_A) ? nothing : (gap_A, gap_A, pan.N2.m / T_N - xi_N, pan.N2.m / T_N - xi_N)
    C_AA_N1N1 = coll_12_34_legendre(
        pan.mp,
        pan.A,
        pan.A,
        pan.N1,
        pan.N1,
        temps_AA_NN,
        xis_AA_NN;
        gaps=gaps_AA_N1N1,
        LEGENDRE_KW_AA...,
    ) / 4.
    # TODO: [16.09.27] Only valid for m_N1 = m_N2.
    C_AA_N2N2 = C_AA_N1N1
    # C_AA_N2N2 = coll_12_34_legendre(
    #     pan.mp,
    #     pan.A,
    #     pan.A,
    #     pan.N2,
    #     pan.N2,
    #     temps_AA_NN,
    #     xis_AA_NN;
    #     gaps=gaps_AA_N2N2,
    # ) / 4.
    # Factor 2 from number change in n_d = n_N1 + n_N2 + 2*n_A
    return - 2. * C_AA_N1N1 - 2. * C_AA_N2N2
end

"""
Gaps (m - mu)/T for the particles of A' <-> N2 nu, in the order used by
`coll_3_12` (N2, nu, A'), or `nothing` if `gap_A` is not known.
"""
function gaps_A_N2nu(pan, T_nu, T_N, xi_N, gap_A)
    isnothing(gap_A) && return nothing
    return (pan.N2.m / T_N - xi_N, pan.nu.m / T_nu, gap_A)
end

function C_n(pan, T_nu, T_N, xi_N; gap_A=nothing)
    """
    Anton: A lot of processes do not contriubute due to equilibrium or no change in particle number.

    Collision operator describing particle alpha:
    Cn[alpha]_{I_r -> F_r} = eps^alpha_r int dPI |M|^2 prod_{i in I_r} f_i * prod_{j in F_r} (1+k_j*f_j) / kappa_r

    kappa_r : symmetry factor
    eps^alpha_r : = -1 if alpha in F_r, = 1 if alpha in I_r

    From this, have Cn[alpha in I_r] = -Cn[beta in F_r], so
    Cn[alpha in I_r] + Cn[beta in F_r] = 0

    Code: C_n_3_12(type=0) = C[3]_{3<->12} = int dPI |M|^2 * [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]

    n = n1 + n2 + 2*nX
    Then for example the processes
    X -> 12:
    Cn[1]_{X->12} + Cn[2]_{X->12} + 2*Cn[X]_{X->12} = 0
    11 -> 22:
    2*Cn[1]_{11->22} + 2*Cn[2]_{11->22} = 0
    But for 11 <-> XX,
    2*Cn[1]_{11<->XX} + 4*Cn[X]_{11<->XX} = 2*Cn[X]_{11<->XX}
    and X <-> 1nu
    C[1]_{X<->1nu} + 2*C[X]_{X<->1nu} = C[X]_{X<->1nu}

    Thus in total, our Boltzmann equation is
    n + 3Hn = C[X]_{X<->1nu} + 2*C[X]_{11<->XX} + 2*C[X]_{22<->XX}
    """
    # th, m_Gamma_h2 do not matter anymore
    # as long as mN1 = mN2, xiN1 = xiN2, TN1 = TN2, do not need CX_XX_22 separately -- just add factor 2
    C_A_N2nu = A_N2nu_moments(pan, T_nu, T_N, xi_N; gap_A=gap_A)[1]
    return - C_A_N2nu + C_n_AA(pan, T_N, xi_N; gap_A=gap_A)
end

# rho = rho_N1 + rho_N2 + rho_A
function C_rho(pan, T_nu, T_N, xi_N; gap_A=nothing)
    """
    Anton: Internal processes of the DS does not contribute due to energy conservation.
    C1_X->12 + C2_X->12 + CX_X->12
    = int dPi * (2pi)^4 delta(E1+E2+E3) (E1 + E2 - E3)*fX*(1+k1*f1)*(1+k2*f2) = 0

    Hence the only contributions come from energy transer between the SM and the DS

    Code: C_rho_3_12(type) = int dPI E_type |M|^2 * [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]

    Trick to save calculation:
    C[1]_{3<->12} + C[3]_{3<->12}
    = int dPI (E3 - E1) |M|^2 [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]
    = int dPI E_2 |M|^2 [f1*f2*(1+k3*f3) - f3*(1+k1*f1)*(1+k2*f2)]
    = C_rho_3_12(type=2, ...)
    """
    _, CN2_A_N2nu, CA_A_N2nu = A_N2nu_moments(pan, T_nu, T_N, xi_N; gap_A=gap_A)
    # Cnu_A_N2nu = coll_3_12(
    #     pan.mp,
    #     pan.N2,
    #     pan.nu,
    #     pan.A,
    #     temps_A_N2nu,
    #     xis_A_N2nu;
    #     sq_amp_func=coll_A_Nnu_sq_amp,
    #     energy_type=Val(2)
    # )
    # TODO: [15.07.26] Double check sign
    # return Cnu_A_N2nu
    return - CA_A_N2nu + CN2_A_N2nu
end
