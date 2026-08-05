import DelimitedFiles as DF

mutable struct Particle{T <: Real}
    m::T
    k::Int64
    dof::Int64
    e::Union{T, Nothing}
    mom::Union{T, Nothing}

    function Particle{T}(
            m::T,
            k::Int64;
            dof::Int64 = 1,
            e::T = nothing, 
            mom::T = nothing,
        ) where T <: Real
        new{T}(m, k, dof, e, mom)
    end
end

struct ModelParams{T <: Real}
    y::T
    theta::T
end

function dist(p::Particle{T}, temp::T, xi::T) where T <: Real
    if isnothing(xi) || (isnothing(temp) || isnothing(p.e))
        error("Chemical potential, temperature or energy is not set for particle with mass ", p.m)
    end # if
    return 1. / (exp(p.e / temp - xi) + p.k)
end

function momentum(p::Particle{T}) where T <: Real
    if isnothing(p.e)
        error("Energy not set for particle with mass ", p.m)
    end # if
    if p.e < p.m
        if (p.e - p.m) / p.m > 1e-9
            error("Energy is smaller than the mass")
        else
            p.e = p.m
            return 0.
        end
    end
    return sqrt(p.e^2 - p.m^2)
end # function

function export_array_to_csv(array::Matrix{T}, filename::String) where T <: Real
    open(filename, "w") do io
        DF.writedlm(io, array, ',')
    end # open io
    return nothing
end