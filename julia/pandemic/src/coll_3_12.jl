import QuadGK

include(joinpath(@__DIR__, "constants_functions.jl"))
include(joinpath(@__DIR__, "utils.jl"))

"""
coll_3_12.jl

Collision term of the decay 3 -> 1 2 and its inverse with exact quantum statistics, as
three energy moments (`coll_3_12_moments`): used by the fluid solver for A' <-> N2 nu and
phi <-> N1 nu.
"""

mutable struct Params_3_12{T<:Real, R<:Real, S<:Real}
    p1::Particle{T}
    p2::Particle{T}
    p3::Particle{T}
    temps::NTuple{3, R}
    xis::NTuple{3, S}
    # Optional (m - mu)/T per particle, see `dist`; `nothing` means m/T - xi.
    gaps::Union{Nothing, NTuple{3, S}}
    e1::Union{R, Nothing}
    e2::Union{R, Nothing}
    e3::Union{R, Nothing}
    mom1::Union{R, Nothing}

    function Params_3_12{T, R, S}(p1::Particle{T}, p2::Particle{T}, p3::Particle{T}, temps::NTuple{3, R},
                                  xis::NTuple{3, S}; gaps=nothing) where {T<:Real, R<:Real, S<:Real}
        new{T, R, S}(p1, p2, p3, temps, xis, isnothing(gaps) ? nothing : NTuple{3, S}(gaps),
                     nothing, nothing, nothing, nothing)
    end
end

"""sum |M|^2 of A' -> N nu (theta_1 = params.theta, g = params.y)."""
function coll_A_Nnu_sq_amp(params::ModelParams{T}, p1::Particle{T}, p2::Particle{T}, p3::Particle{T}) where T <: Real
    pre = 2. * params.y^2 * sin(params.theta)^2
    mass_dep = (p3.m-p1.m-p2.m) * (p3.m+p1.m+p2.m) * (2*p3.m^2+(p1.m-p2.m)^2)/p3.m^2
    return pre * mass_dep
end

"""Kinematic limits of E2 at the current `p.e1` (lower limit at least m2)."""
function coll_3_12_e2_min(p::Params_3_12)
    mass_comb = p.p3.m^2 - p.p1.m^2 - p.p2.m^2
    e2_m = (p.e1 * mass_comb - p.mom1 * sqrt(mass_comb^2 - 4. * p.p1.m^2 * p.p2.m^2)) / (2. * p.p1.m^2)
    return max(e2_m, p.p2.m)
end

function coll_3_12_e2_max(p::Params_3_12)
    mass_comb = p.p3.m^2 - p.p1.m^2 - p.p2.m^2
    return (p.e1 * mass_comb + p.mom1 * sqrt(mass_comb^2 - 4. * p.p1.m^2 * p.p2.m^2)) / (2. * p.p1.m^2)
end

"""f3 (1 - k1 f1)(1 - k2 f2) - f1 f2 (1 - k3 f3) at (p.e1, e2), E3 = E1 + E2."""
function coll_3_12_ker(e2::R, p::Params_3_12{T, R, S}) where {T<:Real, R<:Real, S<:Real}
    p.e2 = e2
    p.e3 = max(p.e1 + p.e2, p.p3.m)
    gaps = isnothing(p.gaps) ? (nothing, nothing, nothing) : p.gaps
    f1 = dist(p.p1, p.temps[1], p.xis[1], p.e1; gap=gaps[1])
    f2 = dist(p.p2, p.temps[2], p.xis[2], p.e2; gap=gaps[2])
    f3 = dist(p.p3, p.temps[3], p.xis[3], p.e3; gap=gaps[3])
    return f3 * (1. - p.p1.k * f1) * (1. - p.p2.k * f2) - f1 * f2 * (1. - p.p3.k * f3)
end

# Relative tolerances of the nested energy integrals. There is deliberately no absolute
# tolerance, since the integrands are tiny in GeV units. The inner integral is converged
# more tightly than the outer one. Over typical states these give a relative error of
# ~1e-7 (1e-8/1e-10: ~6e-9 at twice the cost), well below the ODE tolerances used.
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

"""
    coll_3_12_moments(model_params, p1, p2, p3, temps, xis; sq_amp_func=coll_A_Nnu_sq_amp, gaps=nothing)

The net decay rate density of 3 -> 1 2 (3 -> 1 2 minus 1 2 -> 3) with the weights 1, E1
and E3, as a tuple:
    |M|^2/(32 pi^3) ∫ dE1 ∫ dE2 w [f3 (1 - k1 f1)(1 - k2 f2) - f1 f2 (1 - k3 f3)].
All three come from one nested quadrature over the same energies; the weights E1, E3 are
divided by T13 = max(T1, T3) during integration, so that the norm-based error control
treats the three components alike.
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
    p = Params_3_12{T, R, S}(p1, p2, p3, temps, xis; gaps=gaps)
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
