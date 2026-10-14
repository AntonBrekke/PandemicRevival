using Interpolations
import ForwardDiff as FD

mutable struct Particle{T <: Real}
    m::T
    k::Int64        # 1 for fermions, -1 for bosons, 0 for Maxwell-Boltzmann
    dof::Int64

    function Particle{T}(
            m::T,
            k::Int64;
            dof::Int64 = 1,
        ) where T <: Real
        new{T}(m, k, dof)
    end
end

struct ModelParams{T <: Real}
    y::T
    theta::T
end

"""
    occupation(p, arg)

Occupation number per dof, 1/(e^arg + k), for arg = (E - mu)/T. For bosons
e^arg - 1 is computed with expm1, so that it does not round to zero for
small arg.
"""
function occupation(p::Particle, arg)
    if p.k == -1
        return 1. / min(expm1(arg), 1e300)
    else
        return 1. / (min(exp(arg), 1e300) + p.k)
    end
end

"""
    dist(p, temp, xi, e; gap=nothing)

Occupation number per dof at energy e. gap = (m - mu)/T = m/T - xi: callers that know it
exactly pass it, since for large m/T the difference m/T - xi loses a small gap to
rounding. With it, the exponent (e - m)/T + gap = e/T - xi is exact for e close to m, and
expm1 keeps the boson denominator from rounding to zero.
"""
function dist(p::Particle, temp, xi, e; gap=nothing)
    g = isnothing(gap) ? p.m / temp - xi : gap
    res = occupation(p, (e - p.m) / temp + g)
    res < 0 && error("negative occupation number (Bose condensation): m/T = $(p.m / temp), xi = $xi, E = $e")
    return res
end

function momentum(p::Particle{T}, e::R) where {T<:Real, R<:Real}
    e_value = e isa FD.Dual ? FD.value(e) : e
    if e_value <= p.m
        return zero(e)
    end
    return sqrt(e^2 - p.m^2)
end
