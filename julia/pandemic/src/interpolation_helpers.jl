using Interpolations

"""
One-dimensional piecewise-linear interpolation with configurable constant
out-of-range values.
"""
struct LinearInterp{T <: Real}
    itp
    xmin::T
    xmax::T
    extrap_low::Union{Nothing, T}
    extrap_high::Union{Nothing, T}
end

function LinearInterp(
        x_grid::AbstractVector,
        y_grid::AbstractVector;
        extrap_low::Union{Nothing, T}=nothing,
        extrap_high::Union{Nothing, T}=nothing
    ) where T <: Real
    x = x_grid
    y = y_grid
    if length(x) < 2
        throw(ArgumentError("x must have length >= 2."))
    elseif x[end] <= x[1]
        throw(ArgumentError("x must be strictly increasing."))
    end

    itp = linear_interpolation(x, y; extrapolation_bc=Throw())
    return LinearInterp(
        itp,
        x[1],
        x[end],
        isnothing(extrap_low) ? nothing : extrap_low,
        isnothing(extrap_high) ? nothing : extrap_high,
    )
end

function (li::LinearInterp)(x::T) where T <: Real
    if x < li.xmin
        if isnothing(li.extrap_low)
            throw(DomainError(x, "x lies below interpolation range."))
        end
        return li.extrap_low
    end
    if x > li.xmax
        if isnothing(li.extrap_high)
            throw(DomainError(x, "x lies above interpolation range."))
        end
        return li.extrap_high
    end
    return li.itp(x)
end

"""
Log-log interpolation.

Extrapolation behavior matches the Python use cases in this repository:
- `extrap = nothing`: throw out-of-range
- `extrap = "b"`: clamp to boundary y-values
- `extrap = "c:min,max"`: constant values below/above range
"""
struct LogInterp{T <: Real}
    logbase::T
    itp::LinearInterp{T}
end

function LogInterp(
        x_grid::AbstractVector{T}, y_grid::AbstractVector{T};
        base::Real=Base.MathConstants.e,
        extrap::Union{Nothing, String}=nothing
    ) where T <: Real
    logbase = log(base)
    xlog = log.(x_grid) ./ logbase
    ylog = log.(y_grid) ./ logbase

    if length(xlog) < 2 || xlog[end] <= xlog[1]
        throw(ArgumentError("The values in x_grid need to be in ascending order."))
    end

    extrap_low = nothing
    extrap_high = nothing
    if !isnothing(extrap)
        parts = split(extrap, ':')
        mode = parts[1]
        if mode == "b"
            extrap_low = ylog[1]
            extrap_high = ylog[end]
        elseif mode == "c"
            if length(parts) != 2
                throw(ArgumentError("Invalid constant extrapolation format."))
            end
            vals = split(parts[2], ',')
            if length(vals) != 2
                throw(ArgumentError("Constant extrapolation must be c:min,max."))
            end
            extrap_low = log(parse(Float64, vals[1])) / logbase
            extrap_high = log(parse(Float64, vals[2])) / logbase
        else
            throw(ArgumentError("Unsupported extrapolation mode: $mode"))
        end
    end

    itp = LinearInterp(xlog, ylog; extrap_low=extrap_low, extrap_high=extrap_high)
    return LogInterp{T}(logbase, itp)
end

function (li::LogInterp)(x::T) where T <: Real
    xlog = log(x) / li.logbase
    return exp(li.logbase * li.itp(xlog))
end
