mutable struct Particle{T <: Real}
    m::T
    k::T
    e::Union{T, Nothing}
    mom::Union{T, Nothing}
    xi::Union{T, Nothing}

    function Particle(
            m::T, k::T;
            e=nothing, mom=nothing, xi=nothing
        ) where T <: Real
        new{T}(m, k, e, mom, xi)
    end
end

struct ModelParams{T <: Real}
    y::T
    theta::T
end

function dist(temp, p::Particle{T}) where T <: Real
    if isnothing(p.xi)
        error("Chemical potential not set for particle with mass ", p.m)
    end # if
    return 1. / (exp(p.e/temp - p.xi) + p.k)
end

function momentum(p::Particle)
    if isnothing(p.e)
        error("Energy not set for particle with mass ", p.m)
    end # if
    return sqrt(p.e^2 - p.m^2)
end # function
