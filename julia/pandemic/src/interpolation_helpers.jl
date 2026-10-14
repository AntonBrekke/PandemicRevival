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

"""
    z_interpolation(z_grid, u)

Interpolation of `u` (log-linear) against the increasing z = log(m_N/T_nu) grid of the
background, so that a lookup costs one `exp`.
"""
function z_interpolation(z_grid, u)
    log_interp = linear_interpolation(z_grid, log.(u), extrapolation_bc=Line())
    return z -> exp(log_interp(z))
end

"""
    x_grid_of(m_ref, T_nu_grid)

x = m_ref / T_nu on the background grid (increasing). Used to build
z_grid = log.(x_grid_of(...)) and as an intermediate for `dx_dt_grid_of`.
"""
x_grid_of(m_ref, T_nu_grid) = m_ref ./ T_nu_grid

"""
    dx_dt_grid_of(m_ref, T_nu_grid, dT_nu_dt_grid)

dx/dt = d(m_ref/T)/dt = -(x^2/m_ref) dT/dt, tabulated on the background
grid. Since dT_nu/dt < 0 this is strictly positive, so it can be
log-interpolated directly, unlike dT_nu/dt itself.
"""
function dx_dt_grid_of(m_ref, T_nu_grid, dT_nu_dt_grid)
    x_grid = x_grid_of(m_ref, T_nu_grid)
    return -(x_grid .^ 2) .* dT_nu_dt_grid ./ m_ref
end
