using Interpolations
import ForwardDiff as FD
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

function dist(
        p::Particle{T},
        temp::R,
        xi::S,
        e::R;
        gap=nothing,
        debug=false
    ) where {T<:Real, R<:Real, S<:Real}
    if isnothing(xi) || (isnothing(temp) || isnothing(e))
        error("Chemical potential, temperature or energy is not set for particle with mass ", p.m)
    end # if
    # gap = (m - mu)/T = m/T - xi. Callers that know it exactly pass it: for
    # large m/T, m/T - xi loses a small gap to rounding. With it, the exponent
    # (e - m)/T + gap = e/T - xi is exact for e close to m, and expm1 keeps
    # the boson denominator exp(arg) - 1 from rounding to zero.
    g = isnothing(gap) ? p.m / temp - xi : gap
    res = occupation(p, (e - p.m) / temp + g)
    if res < 0.
        println("Error: Negative distribution function.")
        println("x = ", p.m / temp, ", xi = ", xi, ", e = ", e)
        error()
    end
    if debug
        println("res = ", res)
        println("e/T = ", e/temp)
        println("xi = ", xi)
        println(e/temp - xi)
        println(exp(e/temp - xi) + p.k)
    end
    return res
end

function momentum(p::Particle{T}, e::R) where {T<:Real, R<:Real}
    e_value = e isa FD.Dual ? FD.value(e) : e
    if e_value <= p.m
        return zero(e)
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


function dist_sc(
        p::Particle{T},
        x::R,
        xi::S,
        u::R;
        debug=false
    ) where {T<:Real, R<:Real, S<:Real}
    exp_val = min(
        exp(u * x - xi),
        1e300,
    )
    res = p.dof / (exp_val + p.k)
    if res < 0.
        println("Error: Negative distribution function.")
        println("x = ", x, ", xi = ", xi, ", u = ", u)
        error()
    end
    if debug
        println("res = ", res)
        println("u = ", u)
        println("x = ", x)
        println("e/T = ux = ", u*x)
        println("xi = ", xi)
        println(u*x - xi)
        println(exp(u*x - xi) + p.k)
    end
    return res
end
