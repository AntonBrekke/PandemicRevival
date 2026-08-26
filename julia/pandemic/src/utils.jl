using Interpolations
# import DelimitedFiles as DF
import CSV

mutable struct Particle{T <: Real}
    m::T
    k::Int64        # 1 for fermions, -1 for bosons
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

function dist(
        p::Particle{T},
        temp::R,
        xi::R,
        e::R;
        debug=false
    ) where {T<:Real, R<:Real}
    if isnothing(xi) || (isnothing(temp) || isnothing(e))
        error("Chemical potential, temperature or energy is not set for particle with mass ", p.m)
    end # if
    res = p.dof / (exp(e / temp - xi) + p.k)
    if res < 0.
        println("Error: Negative distribution function.")
        println("x = ", p.m / temp, ", xi = ", xi, ", e = ", e)
        error()
    end
    return res
end

function momentum(p::Particle{T}, e::R) where {T<:Real, R<:Real}
    if e < p.m
        if (e - p.m) / p.m > 1e-9
            error("Energy is smaller than the mass")
        else
            return 0.
        end
    end
    return sqrt(e^2 - p.m^2)
end # function

function export_array_to_csv(
        array, # ::Matrix{T},
        filename::String
    ) # where T <: Real
    open(filename, "w") do io
        # DF.writedlm(io, array, ',')
        CSV.write(io, array)
    end # open io
    return nothing
end

function temp_interpolation(temp, u; neg=false)
    if neg
        u = - u
    end
    rev_temp = reverse(temp)
    rev_u = reverse(u)
    log_temp = log.(rev_temp)
    log_u = log.(rev_u)
    log_interp = linear_interpolation(log_temp, log_u, extrapolation_bc=Line())
    if neg
        return (temp_nu) -> - exp(log_interp(log(temp_nu)))
    else
        return (temp_nu) -> exp(log_interp(log(temp_nu)))
    end
end

function chem_pot_check(
        p::Particle{T},
        temp::R,
        xi::R,
    ) where {T<:Real, R<:Real}
    """Checks if the chemical potential has a valid value for a boson."""
    if p.k != -1
        return true
    end
    check = p.m / temp - xi
    if check < 0.
        return false
    else
        return true
    end
end
