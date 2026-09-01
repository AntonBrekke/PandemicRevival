using Interpolations
using Serialization

"""Tabulated collision terms for one fixed `Pandemolator` instance.

The temperature axes and the chemical-potential axis are stored in transformed
coordinates. Below `xi_split`, collision values use `log(-value)` scaling;
above it they use `asinh(value / scale)` scaling.
"""
struct CollisionTable{T<:Real, INL, INH, IRL, IRH}
    log_T_nu_grid::Vector{T}
    log_T_N_grid::Vector{T}
    xi_N_grid::Vector{T}
    xi_N_interp_grid::Vector{T}
    xi_N_scale::T
    xi_split::T
    xi_split_index::Int
    C_n_values::Array{T, 3}
    C_rho_values::Array{T, 3}
    C_n_interp_low::INL
    C_n_interp_high::INH
    C_rho_interp_low::IRL
    C_rho_interp_high::IRH
end

struct AsinhInterp{T<:Real, I}
    interpolation::I
    scale::T
end

function (interp::AsinhInterp)(x1, x2, x3)
    transformed = interp.interpolation(x1, x2, x3)
    return interp.scale * sinh(transformed)
end

struct NegativeLogInterp{I}
    interpolation::I
end

function (interp::NegativeLogInterp)(x1, x2, x3)
    return -exp(interp.interpolation(x1, x2, x3))
end

function _collision_interpolant(x1, x2, x3, values; value_scaling=:asinh)
    if any(!isfinite, values)
        throw(ArgumentError("collision values must be finite."))
    end
    if value_scaling == :negative_log && any(values .>= 0.)
        throw(ArgumentError("negative-log scaling requires strictly negative values."))
    end

    if value_scaling == :negative_log
        transformed_values = log.(-values)
    elseif value_scaling == :asinh
        scale = maximum(abs, values)
        if scale == 0.
            throw(ArgumentError("asinh-scaled collision values cannot be all zero."))
        end
        transformed_values = asinh.(values ./ scale)
    else
        throw(ArgumentError("Unknown collision-value scaling: $value_scaling"))
    end

    scale = maximum(abs, values)
    regular_interpolation = Interpolations.interpolate(
        transformed_values,
        Interpolations.BSpline(Interpolations.Linear()),
    )
    axes = (
        range(x1[1], x1[end], length=length(x1)),
        range(x2[1], x2[end], length=length(x2)),
        range(x3[1], x3[end], length=length(x3)),
    )
    interpolation = Interpolations.scale(regular_interpolation, axes)
    if value_scaling == :negative_log
        return NegativeLogInterp(interpolation)
    end
    return AsinhInterp(interpolation, scale)
end

"""Build a collision table by evaluating `C_n` and `C_rho` on every grid point.

`T_nu_min`, `T_nu_max`, `T_N_min`, and `T_N_max` are positive temperature
 bounds. The temperature axes are logarithmically spaced. The `xi_N` axis is
piecewise uniformly spaced after an `asinh` transform. The last point below
`xi_split` is shared by both interpolation branches, and the table must also keep the
bosonic distribution valid throughout the table, namely
`fac_n_A * xi_N < m_A / T_N` wherever `A` is a boson.
"""
function _collision_axes(
        T_nu_min,
        T_nu_max,
        n_T_nu,
        T_N_min,
        T_N_max,
        n_T_N,
        xi_N_min,
        xi_N_max,
        n_xi_N,
        xi_split=-0.1,
    )
    if T_nu_min <= 0. || T_nu_max <= T_nu_min
        throw(ArgumentError("Require 0 < T_nu_min < T_nu_max."))
    end
    if T_N_min <= 0. || T_N_max <= T_N_min
        throw(ArgumentError("Require 0 < T_N_min < T_N_max."))
    end
    if xi_N_max <= xi_N_min
        throw(ArgumentError("Require xi_N_min < xi_N_max."))
    end
    if xi_split <= xi_N_min || xi_split >= xi_N_max
        throw(ArgumentError("Require xi_N_min < xi_split < xi_N_max."))
    end
    if n_T_nu < 2 || n_T_N < 2 || n_xi_N < 2
        throw(ArgumentError("Each table axis must have at least two points."))
    end

    log_T_nu = collect(range(log(T_nu_min), log(T_nu_max), length=n_T_nu))
    log_T_N = collect(range(log(T_N_min), log(T_N_max), length=n_T_N))
    xi_N_scale = max(abs(xi_N_min), abs(xi_N_max))
    xi_N_interp_grid = collect(range(
        asinh(xi_N_min / xi_N_scale),
        asinh(xi_N_max / xi_N_scale),
        length=n_xi_N,
    ))
    xi_N = xi_N_scale .* sinh.(xi_N_interp_grid)
    xi_split_index = searchsortedlast(xi_N, xi_split)
    if xi_split_index < 2 || xi_split_index > n_xi_N - 2
        throw(ArgumentError("xi grid needs at least two points on each scaling branch."))
    end
    return (
        log_T_nu,
        log_T_N,
        exp.(log_T_nu),
        exp.(log_T_N),
        xi_N,
        xi_N_interp_grid,
        xi_N_scale,
        xi_split_index,
    )
end

function build_collision_table(
        pan,
        ; T_nu_min,
        T_nu_max,
        n_T_nu,
        T_N_min,
        T_N_max,
        n_T_N,
        xi_N_min,
        xi_N_max,
        n_xi_N,
        xi_split=-0.1,
        threaded=false,
    )
    log_T_nu, log_T_N, T_nu, T_N, xi_N, xi_N_interp_grid, xi_N_scale, xi_split_index = _collision_axes(
        T_nu_min,
        T_nu_max,
        n_T_nu,
        T_N_min,
        T_N_max,
        n_T_N,
        xi_N_min,
        xi_N_max,
        n_xi_N,
        xi_split,
    )

    if pan.A.k == -1
        xi_A_max = pan.A.m ./ T_N
        if any(pan.fac_n_A .* xi_N[end] .>= xi_A_max)
            println("xi_N_grid: ", xi_N[end])
            println("xi_A_max: ", xi_A_max)
            throw(ArgumentError(
                "xi_N_grid reaches the boson chemical-potential bound at some T_N."
            ))
        end
    end

    c_n = Array{Float64}(undef, n_T_nu, n_T_N, n_xi_N)
    c_rho = similar(c_n)
    indices = CartesianIndices(c_n)
    evaluate = function (index)
        i, j, k = Tuple(index)
        c_n[index] = C_n(pan, T_nu[i], T_N[j], xi_N[k])
        c_rho[index] = C_rho(pan, T_nu[i], T_N[j], xi_N[k])
    end
    if threaded
        Threads.@threads for index in indices
            evaluate(index)
        end
    else
        for index in indices
            evaluate(index)
        end
    end

    c_n_low = @view c_n[:, :, 1:xi_split_index-1]
    c_n_high = @view c_n[:, :, xi_split_index:end]
    c_rho_low = @view c_rho[:, :, 1:xi_split_index-1]
    c_rho_high = @view c_rho[:, :, xi_split_index:end]
    return CollisionTable(
        log_T_nu,
        log_T_N,
        xi_N,
        xi_N_interp_grid,
        xi_N_scale,
        xi_split,
        xi_split_index,
        c_n,
        c_rho,
        _collision_interpolant(log_T_nu, log_T_N, xi_N_interp_grid[1:xi_split_index-1], c_n_low; value_scaling=:negative_log),
        _collision_interpolant(log_T_nu, log_T_N, xi_N_interp_grid[xi_split_index:end], c_n_high; value_scaling=:asinh),
        _collision_interpolant(log_T_nu, log_T_N, xi_N_interp_grid[1:xi_split_index-1], c_rho_low; value_scaling=:negative_log),
        _collision_interpolant(log_T_nu, log_T_N, xi_N_interp_grid[xi_split_index:end], c_rho_high; value_scaling=:asinh),
    )
end

function _table_coordinate(value, grid, name)
    if value < grid[1] || value > grid[end]
        throw(DomainError(value, "$name is outside the collision-table range."))
    end
    return value
end

function collision_terms(table::CollisionTable, T_nu, T_N, xi_N)
    log_T_nu = _table_coordinate(log(T_nu), table.log_T_nu_grid, "T_nu")
    log_T_N = _table_coordinate(log(T_N), table.log_T_N_grid, "T_N")
    xi = _table_coordinate(xi_N, table.xi_N_grid, "xi_N")
    xi_interp = asinh(xi / table.xi_N_scale)
    xi_branch_split = table.xi_N_grid[table.xi_split_index]
    if xi < xi_branch_split
        return (
            table.C_n_interp_low(log_T_nu, log_T_N, xi_interp),
            table.C_rho_interp_low(log_T_nu, log_T_N, xi_interp),
        )
    end
    return (
        table.C_n_interp_high(log_T_nu, log_T_N, xi_interp),
        table.C_rho_interp_high(log_T_nu, log_T_N, xi_interp),
    )
end

collision_terms(pan, T_nu, T_N, xi_N) =
    (C_n(pan, T_nu, T_N, xi_N), C_rho(pan, T_nu, T_N, xi_N))

"""Save only portable table data; interpolation objects are rebuilt on load."""
function save_collision_table(table::CollisionTable, filename::AbstractString)
    open(filename, "w") do io
        serialize(io, (
            table.log_T_nu_grid,
            table.log_T_N_grid,
            table.xi_N_grid,
            table.xi_N_interp_grid,
            table.xi_N_scale,
            table.xi_split,
            table.xi_split_index,
            table.C_n_values,
            table.C_rho_values,
        ))
    end
    return nothing
end

function load_collision_table(filename::AbstractString)
    data = open(deserialize, filename)
    log_T_nu, log_T_N, xi_N, xi_N_interp_grid, xi_N_scale, xi_split, xi_split_index, c_n, c_rho = data
    c_n_low = @view c_n[:, :, 1:xi_split_index-1]
    c_n_high = @view c_n[:, :, xi_split_index:end]
    c_rho_low = @view c_rho[:, :, 1:xi_split_index-1]
    c_rho_high = @view c_rho[:, :, xi_split_index:end]
    return CollisionTable(
        log_T_nu,
        log_T_N,
        xi_N,
        xi_N_interp_grid,
        xi_N_scale,
        xi_split,
        xi_split_index,
        c_n,
        c_rho,
        _collision_interpolant(log_T_nu, log_T_N, xi_N_interp_grid[1:xi_split_index-1], c_n_low; value_scaling=:negative_log),
        _collision_interpolant(log_T_nu, log_T_N, xi_N_interp_grid[xi_split_index:end], c_n_high; value_scaling=:asinh),
        _collision_interpolant(log_T_nu, log_T_N, xi_N_interp_grid[1:xi_split_index-1], c_rho_low; value_scaling=:negative_log),
        _collision_interpolant(log_T_nu, log_T_N, xi_N_interp_grid[xi_split_index:end], c_rho_high; value_scaling=:asinh),
    )
end