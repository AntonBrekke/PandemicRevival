using DifferentialEquations

include(joinpath(@__DIR__, "constants_functions.jl"))

const t_max = 1e16 / hbar
const rtol_ode = 1e-6

function cumsimp_logspace(
        x_grid::AbstractVector{T},
        y_grid::AbstractVector{T}
    ) where T <: Real
    n = length(x_grid)
    if n != length(y_grid)
        throw(ArgumentError("x_grid and y_grid must have the same length."))
    end
    if n == 0
        return Float64[]
    elseif n == 1
        return [0.0]
    end

    x = Float64.(x_grid)
    y = Float64.(y_grid)

    delta_z = log(x[end] / x[1]) / (n - 1)
    g_grid = x .* y
    i_grid = zeros(Float64, n)

    last_even_int = 0.0
    for j_odd in 2:2:n
        i_grid[j_odd] = last_even_int + 0.5 * delta_z * (g_grid[j_odd - 1] + g_grid[j_odd])

        j_even = j_odd + 1
        if j_even <= n
            i_grid[j_even] = last_even_int + delta_z * (g_grid[j_even - 2] + 4.0 * g_grid[j_even - 1] + g_grid[j_even]) / 3.0
            last_even_int = i_grid[j_even]
        end
    end

    return i_grid
end

mutable struct TimeTempRelation{T<:Real}
    t_grid::Vector{T}
    sqrt_t_grid::Vector{T}
    T_SM_grid::Vector{T}
    T_nu_grid::Vector{T}
    hubble_grid::Vector{T}
    # hubble_cumsimp::Vector{Float64}
    # sf_grid::Vector{Float64}
    nu_dec_grid::Vector{Bool}
    dT_SM_dt_grid::Vector{T}
    dT_nu_dt_grid::Vector{T}
    ent_grid::Vector{T}

    function TimeTempRelation{T}(;
            T_start::T=1e8,
            t_end::T=t_max,
            t_gp_pd::Int64=1000,
        ) where T <: Real
        t_start = 1.0 / (2.0 * hubble_of_temps(T_start, T_start))
        grid_size_time = max(2, floor(Int, log10(t_end / t_start) * t_gp_pd))

        t_grid = 10.0 .^ range(log10(t_start), log10(t_end), length=grid_size_time)
        sqrt_t_grid = sqrt.(t_grid)

        u0 = [T_start * sqrt_t_grid[1], T_start * sqrt_t_grid[1]]
        u_prob = ODEProblem(
            u_der!,
            u0,
            (t_grid[1], t_grid[end]),
        )
        u_sol = solve(
            u_prob;
            reltol=rtol_ode,
            abstol=0.0,
            saveat=t_grid
        )

        T_SM_grid = Vector{T}(u_sol[1, :]) ./ sqrt_t_grid
        T_nu_grid = Vector{T}(u_sol[2, :]) ./ sqrt_t_grid

        hubble_grid = [hubble_of_temps(T_SM, T_nu) for (T_SM, T_nu) in zip(T_SM_grid, T_nu_grid)]
        # hubble_cumsimp = cumsimp_logspace(t_grid, hubble_grid)
        # Scale factor (a)
        # sf_grid = exp.(hubble_cumsimp)

        nu_dec_grid = (hubble_grid ./ (T_SM_grid .^ 5.0)) .> hubble_T5_nu_dec
        dT_SM_dt_grid = [dT_SM_dt(T_SM, hubble, nu_dec) for (T_SM, hubble, nu_dec) in zip(T_SM_grid, hubble_grid, nu_dec_grid)]
        dT_nu_dt_grid = [dT_nu_dt(T_nu, hubble, nu_dec) for (T_nu, hubble, nu_dec) in zip(T_nu_grid, hubble_grid, nu_dec_grid)]

        ent_grid = entropy.(T_SM_grid, T_nu_grid)

        new{T}(
            t_grid,
            sqrt_t_grid,
            T_SM_grid,
            T_nu_grid,
            hubble_grid,
            # hubble_cumsimp,
            # sf_grid,
            nu_dec_grid,
            dT_SM_dt_grid,
            dT_nu_dt_grid,
            ent_grid,
        )
    end
end

function rho(T_SM::T, T_nu::T) where T <: Real
    return rho_SM_no_nu(T_SM) + rho_nu(T_nu) + rho_m(T_SM, T_nu)
end

function hubble_of_temps(T_SM::T, T_nu::T) where T <: Real
    return sqrt(8.0 * pi * G * rho(T_SM, T_nu) / 3.0)
end

function dT_SM_dt(T_SM::T, hubble::T, nu_dec::Bool) where T <: Real
    if !nu_dec
        return -3.0 * hubble * (
            rho_SM_before_nu_dec(T_SM) + P_SM_before_nu_dec(T_SM)
        ) / (
            rho_der_SM_before_nu_dec(T_SM)
        )
    else
        return -3.0 * hubble * (
            rho_SM_no_nu(T_SM) + P_SM_no_nu(T_SM)
        ) / (
            rho_der_SM_no_nu(T_SM)
        )
    end
end

function dT_nu_dt(T_nu::T, hubble::T, nu_dec::Bool) where T <: Real
    if !nu_dec
        return dT_SM_dt(T_nu, hubble, nu_dec)
    else
        return - hubble * T_nu
    end
end

function entropy(T_SM::T, T_nu::T) where T <: Real
    return s_SM_no_nu(T_SM) + s_nu(T_nu)
end

function u_der!(du, u, p, t)
    sqrt_t = sqrt(t)
    T_SM = u[1] / sqrt_t
    T_nu = u[2] / sqrt_t

    hubble = hubble_of_temps(T_SM, T_nu)
    hubble_T5 = hubble / (T_SM^5.0)
    nu_dec = !isfinite(hubble_T5) || hubble_T5 > hubble_T5_nu_dec

    du[1] = T_SM / (2.0 * sqrt_t) + sqrt_t * dT_SM_dt(T_SM, hubble, nu_dec)
    du[2] = T_nu / (2.0 * sqrt_t) + sqrt_t * dT_nu_dt(T_nu, hubble, nu_dec)

    return nothing
end
