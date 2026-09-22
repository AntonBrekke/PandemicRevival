import Integrals
import QuadGK

include(joinpath(@__DIR__, "constants_functions.jl"))
include(joinpath(@__DIR__, "utils.jl"))

"""coll_n_3_12
Collision operator for decay of particle 3 into particles 1 and 2 (and inverse process).
"""

mutable struct Params_3_12{T<:Real, R<:Real, S<:Real, E}
    p1::Particle{T}
    p2::Particle{T}
    p3::Particle{T}
    temps::NTuple{3, R}
    xis::NTuple{3, S}
    energy_type::E
    # Optional (m - mu)/T per particle, see `dist`; `nothing` means m/T - xi.
    gaps::Union{Nothing, NTuple{3, S}}
    e1::Union{R, Nothing}
    e2::Union{R, Nothing}
    e3::Union{R, Nothing}
    mom1::Union{R, Nothing}
    mom2::Union{R, Nothing}
    mom3::Union{R, Nothing}
    debug::Bool

    function Params_3_12{T, R, S, E}(
            p1::Particle{T},
            p2::Particle{T},
            p3::Particle{T},
            temps::NTuple{3, R}, # = Vector{R}(undef, 3),
            xis::NTuple{3, S}; # = Vector{S}(undef, 3),
            energy_type::E = Val(0),
            gaps = nothing,
        ) where {T<:Real, R<:Real, S<:Real, E}
        new{T, R, S, E}(
            p1, p2, p3,
            temps,
            xis,
            energy_type,
            isnothing(gaps) ? nothing : NTuple{3, S}(gaps),
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            false,
        )
    end
end

@inline function energy_factor(
        p::Params_3_12{T, R, S, Val{0}}
    ) where {T<:Real, R<:Real, S<:Real}
    return one(R)
end
@inline function energy_factor(
        p::Params_3_12{T, R, S, Val{1}}
    ) where {T<:Real, R<:Real, S<:Real}
    return p.e1
end
@inline function energy_factor(
        p::Params_3_12{T, R, S, Val{2}}
    ) where {T<:Real, R<:Real, S<:Real}
    return p.e2
end
@inline function energy_factor(
        p::Params_3_12{T, R, S, Val{3}}
    ) where {T<:Real, R<:Real, S<:Real}
    return p.e3
end


function coll_A_Nnu_sq_amp(
        params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T}
    ) where T <: Real
    pre = 2. * params.y^2 * sin(params.theta)^2
    mass_dep = (p3.m-p1.m-p2.m) * (p3.m+p1.m+p2.m) * (2*p3.m^2+(p1.m-p2.m)^2)/p3.m^2
    return pre * mass_dep
end

function coll_3_12_e2_min(
        p::Params_3_12{T, R, S, E},
    ) where {T<:Real, R<:Real, S<:Real, E}
    mass_comb = p.p3.m^2 - p.p1.m^2 - p.p2.m^2
    e2_m = (p.e1 * mass_comb - p.mom1 * sqrt(mass_comb^2 - 4. * p.p1.m^2 * p.p2.m^2)) / (2. * p.p1.m^2)
    if e2_m < p.p2.m
        return p.p2.m
    else
        return e2_m
    end
    return e2_m
end

function coll_3_12_e2_max(
        p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    mass_comb = p.p3.m^2 - p.p1.m^2 - p.p2.m^2
    e2_p = (p.e1 * mass_comb + p.mom1 * sqrt(mass_comb^2 - 4. * p.p1.m^2 * p.p2.m^2)) / (2. * p.p1.m^2)
    return e2_p
end

function coll_3_12_ker(
        e2::R,
        p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    p.e2 = e2
    # p.mom2 = momentum(p.p2, p.e2)

    p.e3 = p.e1 + p.e2
    if p.e3 < p.p3.m
        println("E_3 = ", p.e3, " < E_1 + E_2 = ", p.e1 + p.e2)
        println("Set manually to E_3 = m_3")
        p.e3 = p.p3.m
    end
    # p.mom3 = momentum(p.p3, p.e3)

    gaps = isnothing(p.gaps) ? (nothing, nothing, nothing) : p.gaps
    f1 = dist(p.p1, p.temps[1], p.xis[1], p.e1; gap=gaps[1])
    f2 = dist(p.p2, p.temps[2], p.xis[2], p.e2; gap=gaps[2])
    f3 = dist(p.p3, p.temps[3], p.xis[3], p.e3; gap=gaps[3])

    # TODO: Double check sign in front of k and of total expression
    dist_fac_3_12 = (
        f3 * (1. - p.p1.k * f1) * (1. - p.p2.k * f2)
    )
    dist_fac_12_3 = (
        f1 * f2 * (1. - p.p3.k * f3)
    )
    dist_fac = dist_fac_3_12 - dist_fac_12_3

    # if p.debug
    #     println("dist_fac = ", dist_fac)
    #     if isnan(FD.partials(dist_fac)[1])
    #         dist(p.p3, p.temps[3], p.xis[3], p.e3; debug=true)
    #     end
    # end

    return energy_factor(p) * dist_fac
end

# Relative tolerances of the nested energy integrals in `coll_3_12`. There is
# deliberately no absolute tolerance, since the integrands are tiny in GeV
# units. The inner integral is converged more tightly than the outer one.
# Over typical states these give a relative error of ~1e-7 (1e-8/1e-10: ~6e-9
# at twice the cost), well below the ODE tolerances used.
const rtol_coll_3_12_outer = 1e-6
const rtol_coll_3_12_inner = 1e-8
# Energy cutoff in units of the relevant temperature, above the chemical potential.
const n_cut_coll_3_12 = 50.

"""(m - mu)/T of particle i (1, 2, 3) in `p`, taken from `p.gaps` if given."""
function gap_3_12(p::Params_3_12, i::Int)
    isnothing(p.gaps) || return p.gaps[i]
    part = (p.p1, p.p2, p.p3)[i]
    return part.m / p.temps[i] - p.xis[i]
end

"""
Integration range in e2 at the current `p.e1`: the kinematic range, cut where
f2(e2) (temperature T2) and f3(e1 + e2) (T3) have both dropped by e^-n_cut
from their values at the lower limit.
"""
function coll_3_12_e2_range(p::Params_3_12)
    e2_min = coll_3_12_e2_min(p)
    e2_max = coll_3_12_e2_max(p)
    e2_cut = e2_min + (n_cut_coll_3_12 + max(0., -gap_3_12(p, 2), -gap_3_12(p, 3))) * max(p.temps[2], p.temps[3])
    return e2_min, min(e2_max, e2_cut)
end

"""
Integration range in e1 and the thermal scale T13 = max(T1, T3): above e1_max,
f1(e1) (T1) and f3(e1 + e2) (T3) have both dropped by e^-n_cut.
"""
function coll_3_12_e1_range(p::Params_3_12)
    e1_min = p.p1.m
    T13 = max(p.temps[1], p.temps[3])
    e1_max = e1_min + (n_cut_coll_3_12 + max(0., -gap_3_12(p, 1), -gap_3_12(p, 3))) * T13
    return e1_min, e1_max, T13
end

function coll_3_12_int_e2(
        e1::R,
        p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    p.e1 = e1
    p.mom1 = momentum(p.p1, p.e1)

    e2_min, e2_hi = coll_3_12_e2_range(p)
    if !(e2_hi > e2_min)
        return zero(e1)
    end

    sol = QuadGK.quadgk(e2 -> coll_3_12_ker(e2, p), e2_min, e2_hi; rtol=rtol_coll_3_12_inner, atol=0)
    # if sol.u isa FD.Dual
    #     if isnan(FD.partials(sol.u)[1])
    #         println("found NaN")
    #         p.debug = true
    #         problem = Integrals.IntegralProblem(
    #             coll_3_12_ker,
    #             (e2_min, e2_max),
    #             p
    #         )
    #         sol = Integrals.solve(
    #             problem,
    #             Integrals.QuadGKJL(),
    #         )
    #         error()
    #     end
    # end
    return sol[1]
end

function coll_3_12_int_e1(p::Params_3_12{T, R, S, E}) where {T<:Real, R<:Real, S<:Real, E}
    # Breakpoints on the thermal scale resolve the peak near threshold.
    e1_min, e1_max, T13 = coll_3_12_e1_range(p)

    sol = try QuadGK.quadgk(
        e1 -> coll_3_12_int_e2(e1, p),
        e1_min, e1_min + T13, e1_min + 10 * T13, e1_max;
        rtol=rtol_coll_3_12_outer, atol=0,
    )
    catch e
        println(p.temps)
        println(p.xis)
        println("e_min = ", e1_min)
        println("e_max = ", e1_max)
        throw(e)
    end
    return sol[1]
end

function coll_3_12(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        temps::NTuple{3, R},
        xis::NTuple{3, S};
        energy_type::E = Val(0),
        sq_amp_func::F = coll_A_Nnu_sq_amp,
        gaps = nothing,
    ) where {T<:Real, R<:Real, S<:Real, E, F}
    params = Params_3_12{T, R, S, E}(
        p1, p2, p3,
        temps,
        xis;
        energy_type=energy_type,
        gaps=gaps,
    )
    integral_sol = coll_3_12_int_e1(params)

    pre = 1. / (2^5 * pi^3)
    sq_amp = sq_amp_func(model_params, p1, p2, p3)
    # TODO: Check prefactors!
    return pre * sq_amp * integral_sol
end

"""
    coll_3_12_moments(model_params, p1, p2, p3, temps, xis; sq_amp_func=coll_A_Nnu_sq_amp, gaps=nothing)

`coll_3_12` for energy_type = Val(0), Val(1) and Val(3) at once, i.e. the
integrals with weights 1, E1 and E3, as a tuple. All three come from one
nested quadrature over the same energies, which is about three times cheaper
than three calls and gives mutually consistent values. The weights E1, E3 are
divided by T13 = max(T1, T3) during integration, so that the norm-based error
control treats the three components alike.
"""
function coll_3_12_moments(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        temps::NTuple{3, R},
        xis::NTuple{3, S};
        sq_amp_func::F = coll_A_Nnu_sq_amp,
        gaps = nothing,
    ) where {T<:Real, R<:Real, S<:Real, F}
    p = Params_3_12{T, R, S, Val{0}}(p1, p2, p3, temps, xis; energy_type=Val(0), gaps=gaps)
    e1_min, e1_max, T13 = coll_3_12_e1_range(p)

    function outer(e1)
        p.e1 = e1
        p.mom1 = momentum(p.p1, e1)
        e2_min, e2_hi = coll_3_12_e2_range(p)
        e2_hi > e2_min || return zeros(R, 3)
        # [∫ de2 f, ∫ de2 f E3/T13]; coll_3_12_ker sets p.e3 = e1 + e2.
        I = QuadGK.quadgk(
            e2 -> (d = coll_3_12_ker(e2, p); [d, d * p.e3 / T13]),
            e2_min, e2_hi; rtol=rtol_coll_3_12_inner, atol=0,
        )[1]
        return [I[1], I[1] * e1 / T13, I[2]]
    end
    I = QuadGK.quadgk(outer, e1_min, e1_min + T13, e1_min + 10 * T13, e1_max; rtol=rtol_coll_3_12_outer, atol=0)[1]

    pre = sq_amp_func(model_params, p1, p2, p3) / (2^5 * pi^3)
    return (pre * I[1], pre * T13 * I[2], pre * T13 * I[3])
end



function coll_3_12_ker_log(
        y2::R,
        p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    e2 = exp(y2)
    return e2 * coll_3_12_ker(e2, p)
end

function coll_3_12_int_e2_log(
        y1::R, p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    p.e1 = exp(y1)
    p.mom1 = momentum(p.p1, p.e1)

    e2_min = coll_3_12_e2_min(p)
    e2_max = coll_3_12_e2_max(p)

    y2_min = log(e2_min)
    y2_max = log(e2_max)

    problem = Integrals.IntegralProblem(
        coll_3_12_ker_log,
        (y2_min, y2_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    res = p.e1 * sol.u
    return res
end

function coll_3_12_int_e1_log(
        p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    e1_min = p.p1.m
    e1_max = max(1e1*p.temps[1], 1e1*p.p1.m)
    # e1_max = Inf64

    y1_min = log(e1_min)
    y1_max = log(e1_max)

    problem = Integrals.IntegralProblem(
        coll_3_12_int_e2_log,
        (y1_min, y1_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol.u
end


function coll_3_12_log(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        temps::NTuple{3, R},
        xis::NTuple{3, S};
        energy_type::E = Val(0),
        sq_amp_func::F = coll_A_Nnu_sq_amp,
    ) where {T<:Real, R<:Real, S<:Real, E, F}
    params = Params_3_12{T, R, S, E}(
        p1, p2, p3,
        temps,
        xis;
        energy_type=energy_type
    )
    integral_sol = coll_3_12_int_e1_log(params)

    pre = 1. / (2^5 * pi^3)
    sq_amp = sq_amp_func(model_params, p1, p2, p3)
    # TODO: Check prefactors!
    return pre * sq_amp * integral_sol
end

### Dimensionless version, and a dimension-full wrapper (same interface as
### `coll_3_12`) built on top of it. See test/test_coll_3_12_dimless.jl for
### a direct numerical comparison against `coll_3_12`.
"""
Dimensionless version of `coll_3_12`. Reference scale `M = p1.m`: `xs = M
./ (T1, T2, T3)` replaces `temps` (so `xs[i] = p1.m/T_i`, matching the `x =
m/T` convention used elsewhere in this codebase), `xis` is unchanged
(already dimensionless by construction), and `u1, u2, u3` are `e1, e2, e3`
divided by `M`. Only depends on the masses through the ratios `p2.m/p1.m`,
`p3.m/p1.m` -- `coll_3_12_e2_min`/`_max` are already temperature-independent,
so nondimensionalizing them is just the substitution `e -> M*u`, `m_i ->
M*r_i`.

Deliberately kept as an independent implementation (not a thin wrapper
around `coll_3_12` evaluated at `p1.m=1`) so that comparing the two is an
actual test of the rescaling, not a tautology.
"""
mutable struct Params_3_12_dimless{T<:Real, R<:Real, S<:Real, E}
    p1::Particle{T}
    p2::Particle{T}
    p3::Particle{T}
    r2::T           # p2.m / p1.m
    r3::T           # p3.m / p1.m
    xs::NTuple{3, R}
    xis::NTuple{3, S}
    energy_type::E
    u1::Union{R, Nothing}
    u2::Union{R, Nothing}
    u3::Union{R, Nothing}
    mom1::Union{R, Nothing}
    mom2::Union{R, Nothing}
    mom3::Union{R, Nothing}
    debug::Bool

    function Params_3_12_dimless{T, R, S, E}(
            p1::Particle{T},
            p2::Particle{T},
            p3::Particle{T},
            xs::NTuple{3, R},
            xis::NTuple{3, S};
            energy_type::E = Val(0),
        ) where {T<:Real, R<:Real, S<:Real, E}
        new{T, R, S, E}(
            p1, p2, p3,
            p2.m / p1.m,
            p3.m / p1.m,
            xs,
            xis,
            energy_type,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            false,
        )
    end
end

@inline function energy_factor(
        p::Params_3_12_dimless{T, R, S, Val{0}}
    ) where {T<:Real, R<:Real, S<:Real}
    return one(R)
end
@inline function energy_factor(
        p::Params_3_12_dimless{T, R, S, Val{1}}
    ) where {T<:Real, R<:Real, S<:Real}
    return p.u1
end
@inline function energy_factor(
        p::Params_3_12_dimless{T, R, S, Val{2}}
    ) where {T<:Real, R<:Real, S<:Real}
    return p.u2
end
@inline function energy_factor(
        p::Params_3_12_dimless{T, R, S, Val{3}}
    ) where {T<:Real, R<:Real, S<:Real}
    return p.u3
end

function coll_3_12_u2_min_dimless(
        p::Params_3_12_dimless{T, R, S, E},
    ) where {T<:Real, R<:Real, S<:Real, E}
    mass_comb = p.r3^2 - one(p.r3) - p.r2^2
    u2_m = (p.u1 * mass_comb - p.mom1 * sqrt(mass_comb^2 - 4. * p.r2^2)) / 2.
    if u2_m < p.r2
        return p.r2
    else
        return u2_m
    end
end

function coll_3_12_u2_max_dimless(
        p::Params_3_12_dimless{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    mass_comb = p.r3^2 - one(p.r3) - p.r2^2
    u2_p = (p.u1 * mass_comb + p.mom1 * sqrt(mass_comb^2 - 4. * p.r2^2)) / 2.
    return u2_p
end

function coll_3_12_ker_dimless(
        u2::R,
        p::Params_3_12_dimless{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    p.u2 = u2

    p.u3 = p.u1 + p.u2
    if p.u3 < p.r3
        println("u_3 = ", p.u3, " < u_1 + u_2 = ", p.u1 + p.u2)
        println("Set manually to u_3 = r_3")
        p.u3 = p.r3
    end

    f1 = dist_sc(p.p1, p.xs[1], p.xis[1], p.u1)
    f2 = dist_sc(p.p2, p.xs[2], p.xis[2], p.u2)
    f3 = dist_sc(p.p3, p.xs[3], p.xis[3], p.u3)

    # TODO: Double check sign in front of k and of total expression
    dist_fac_3_12 = (
        f3 * (1. - p.p1.k * f1) * (1. - p.p2.k * f2)
    )
    dist_fac_12_3 = (
        f1 * f2 * (1. - p.p3.k * f3)
    )
    dist_fac = dist_fac_3_12 - dist_fac_12_3

    return energy_factor(p) * dist_fac
end

function coll_3_12_int_e2_dimless(
        u1::R,
        p::Params_3_12_dimless{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    p.u1 = u1
    # Mirrors `momentum(::Particle, e)`'s dual-safe clamp, for r1 = 1.
    u1_value = u1 isa FD.Dual ? FD.value(u1) : u1
    p.mom1 = u1_value <= one(u1) ? zero(u1) : sqrt(u1^2 - one(u1))

    u2_min = coll_3_12_u2_min_dimless(p)
    u2_max = coll_3_12_u2_max_dimless(p)

    problem = Integrals.IntegralProblem(
        coll_3_12_ker_dimless,
        (u2_min, u2_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol.u
end

function coll_3_12_int_e1_dimless(
        p::Params_3_12_dimless{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    u1_min = one(T)
    u1_max = max(1e1 / p.xs[1], 1e1)

    problem = Integrals.IntegralProblem(
        coll_3_12_int_e2_dimless,
        (u1_min, u1_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol.u
end

"""
    coll_3_12_dimless(p1, p2, p3, xs, xis; energy_type=Val(0))

Pure dimensionless collision-kernel integral: no coupling/`sq_amp` factor,
no overall mass scale. `xs = p1.m ./ (T1, T2, T3)`. Depends on the masses
only through the ratios `p2.m/p1.m`, `p3.m/p1.m`, so the same result is
valid for any (p1, p2, p3) with those ratios held fixed, regardless of the
absolute mass scale -- see `coll_3_12_scale` to get back to physical units.
"""
function coll_3_12_dimless(
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        xs::NTuple{3, R},
        xis::NTuple{3, S};
        energy_type::E = Val(0),
    ) where {T<:Real, R<:Real, S<:Real, E}
    params = Params_3_12_dimless{T, R, S, E}(
        p1, p2, p3,
        xs,
        xis;
        energy_type=energy_type,
    )
    return coll_3_12_int_e1_dimless(params)
end

# de1*de2 = M^2 du1*du2, plus one more power of M from `energy_factor` for
# energy_type = Val(1)/(2)/(3) (each picks up one explicit factor e_i = M*u_i).
_energy_type_power(::Val{0}) = 2
_energy_type_power(::Val{1}) = 3
_energy_type_power(::Val{2}) = 3
_energy_type_power(::Val{3}) = 3

"""
    coll_3_12_scale(model_params, p1, p2, p3, temps, xis; energy_type=Val(0), sq_amp_func=coll_A_Nnu_sq_amp)

Dimension-full wrapper with the same interface as `coll_3_12`, computed via
`coll_3_12_dimless`: nondimensionalize with `M = p1.m`, evaluate the
dimensionless integral, rescale back by `M^_energy_type_power(energy_type)`.
Should agree with `coll_3_12` up to quadrature error; a disagreement means
the nondimensionalization has a bug -- see test/test_coll_3_12_dimless.jl.
"""
function coll_3_12_scale(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        temps::NTuple{3, R},
        xis::NTuple{3, S};
        energy_type::E = Val(0),
        sq_amp_func::F = coll_A_Nnu_sq_amp,
    ) where {T<:Real, R<:Real, S<:Real, E, F}
    M = p1.m
    xs = M ./ temps
    shape = coll_3_12_dimless(p1, p2, p3, xs, xis; energy_type=energy_type)
    integral_sol = M^_energy_type_power(energy_type) * shape

    pre = 1. / (2^5 * pi^3)
    sq_amp = sq_amp_func(model_params, p1, p2, p3)
    return pre * sq_amp * integral_sol
end
