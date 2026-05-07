using DifferentialEquations

include(joinpath(@__DIR__, "constants_functions.jl"))

const t_max = 1e16 / hbar
const rtol_ode = 1e-6

function cumsimp_logspace(x_grid::AbstractVector{<:Real}, y_grid::AbstractVector{<:Real})
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

mutable struct TimeTempRelation
    psi_in_SM::Bool
    m_psi::Float64
    dof_psi::Float64
    k_psi::Int

    t_grid::Vector{Float64}
    sqrt_t_grid::Vector{Float64}
    T_SM_grid::Vector{Float64}
    T_nu_grid::Vector{Float64}
    hubble_grid::Vector{Float64}
    hubble_cumsimp::Vector{Float64}
    sf_grid::Vector{Float64}
    nu_dec_grid::Vector{Bool}
    dTSM_dt_grid::Vector{Float64}
    dTnu_dt_grid::Vector{Float64}
end

function rho_psi(rel::TimeTempRelation, T_SM::Real)
    if rel.psi_in_SM
        return 0.0
    end

    if rel.k_psi == -1
        return rho_boson(T_SM, rel.m_psi, rel.dof_psi)
    end
    return rho_fermion(T_SM, rel.m_psi, rel.dof_psi)
end

function P_psi(rel::TimeTempRelation, T_SM::Real)
    if rel.psi_in_SM
        return 0.0
    end

    if rel.k_psi == -1
        return P_boson(T_SM, rel.m_psi, rel.dof_psi)
    end
    return P_fermion(T_SM, rel.m_psi, rel.dof_psi)
end

function rho_der_psi(rel::TimeTempRelation, T_SM::Real)
    if rel.psi_in_SM
        return 0.0
    end

    if rel.k_psi == -1
        return rho_der_boson(T_SM, rel.m_psi, rel.dof_psi)
    end
    return rho_der_fermion(T_SM, rel.m_psi, rel.dof_psi)
end

rho(rel::TimeTempRelation, T_SM::Real, T_nu::Real) =
    rho_SM_no_nu(T_SM) + rho_nu(T_nu) + rho_m(T_SM, T_nu) + rho_psi(rel, T_SM)

hubble_of_temps(rel::TimeTempRelation, T_SM::Real, T_nu::Real) =
    sqrt(8.0 * pi * G * rho(rel, T_SM, T_nu) / 3.0)

function dTSM_dt(rel::TimeTempRelation, T_SM::Real, hubble::Real, nu_dec::Bool)
    if !nu_dec
        return -3.0 * hubble * (
            rho_SM_before_nu_dec(T_SM) + rho_psi(rel, T_SM) + P_SM_before_nu_dec(T_SM) + P_psi(rel, T_SM)
        ) / (
            rho_der_SM_before_nu_dec(T_SM) + rho_der_psi(rel, T_SM)
        )
    end

    return -3.0 * hubble * (
        rho_SM_no_nu(T_SM) + rho_psi(rel, T_SM) + P_SM_no_nu(T_SM) + P_psi(rel, T_SM)
    ) / (
        rho_der_SM_no_nu(T_SM) + rho_der_psi(rel, T_SM)
    )
end

function dTnu_dt(rel::TimeTempRelation, T_nu::Real, hubble::Real, nu_dec::Bool)
    if !nu_dec
        return dTSM_dt(rel, T_nu, hubble, nu_dec)
    end
    return -hubble * T_nu
end

function der!(du, u, rel::TimeTempRelation, t)
    sqrt_t = sqrt(t)
    T_SM = u[1] / sqrt_t
    T_nu = u[2] / sqrt_t

    hubble = hubble_of_temps(rel, T_SM, T_nu)
    hubble_T5 = hubble / (T_SM^5.0)
    nu_dec = !isfinite(hubble_T5) || hubble_T5 > hubble_T5_nu_dec

    du[1] = T_SM / (2.0 * sqrt_t) + sqrt_t * dTSM_dt(rel, T_SM, hubble, nu_dec)
    du[2] = T_nu / (2.0 * sqrt_t) + sqrt_t * dTnu_dt(rel, T_nu, hubble, nu_dec)

    return nothing
end

function TimeTempRelation(; T_start::Real=1e8, t_end::Real=t_max, t_gp_pd::Real=1e3,
    m_psi=nothing, dof_psi=nothing, k_psi=nothing)

    psi_in_SM = isnothing(m_psi)
    if !psi_in_SM && (isnothing(dof_psi) || isnothing(k_psi))
        throw(ArgumentError("When m_psi is provided, dof_psi and k_psi must also be provided."))
    end

    m_psi_val = psi_in_SM ? 0.0 : Float64(m_psi)
    dof_psi_val = psi_in_SM ? 0.0 : Float64(dof_psi)
    k_psi_val = psi_in_SM ? 0 : Int(k_psi)
    if !psi_in_SM && (k_psi_val != -1 && k_psi_val != 1)
        throw(ArgumentError("k_psi must be -1 (boson) or 1 (fermion)."))
    end

    rel = TimeTempRelation(
        psi_in_SM,
        m_psi_val,
        dof_psi_val,
        k_psi_val,
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Bool[],
        Float64[],
        Float64[],
    )

    t_start = 1.0 / (2.0 * hubble_of_temps(rel, T_start, T_start))
    grid_size_time = max(2, floor(Int, log10(t_end / t_start) * t_gp_pd))

    t_grid = 10.0 .^ range(log10(t_start), log10(t_end), length=grid_size_time)
    sqrt_t_grid = sqrt.(t_grid)

    u0 = [T_start * sqrt_t_grid[1], T_start * sqrt_t_grid[1]]
    prob = ODEProblem(der!, u0, (t_grid[1], t_grid[end]), rel)
    sol = solve(prob; reltol=rtol_ode, abstol=0.0, saveat=t_grid)

    T_SM_grid = Vector{Float64}(sol[1, :]) ./ sqrt_t_grid
    T_nu_grid = Vector{Float64}(sol[2, :]) ./ sqrt_t_grid

    hubble_grid = [hubble_of_temps(rel, T_SM, T_nu) for (T_SM, T_nu) in zip(T_SM_grid, T_nu_grid)]
    hubble_cumsimp = cumsimp_logspace(t_grid, hubble_grid)
    sf_grid = exp.(hubble_cumsimp)

    nu_dec_grid = (hubble_grid ./ (T_SM_grid .^ 5.0)) .> hubble_T5_nu_dec
    dTSM_dt_grid = [dTSM_dt(rel, T_SM, hubble, nu_dec) for (T_SM, hubble, nu_dec) in zip(T_SM_grid, hubble_grid, nu_dec_grid)]
    dTnu_dt_grid = [dTnu_dt(rel, T_nu, hubble, nu_dec) for (T_nu, hubble, nu_dec) in zip(T_nu_grid, hubble_grid, nu_dec_grid)]

    rel.t_grid = t_grid
    rel.sqrt_t_grid = sqrt_t_grid
    rel.T_SM_grid = T_SM_grid
    rel.T_nu_grid = T_nu_grid
    rel.hubble_grid = hubble_grid
    rel.hubble_cumsimp = hubble_cumsimp
    rel.sf_grid = sf_grid
    rel.nu_dec_grid = nu_dec_grid
    rel.dTSM_dt_grid = dTSM_dt_grid
    rel.dTnu_dt_grid = dTnu_dt_grid

    return rel
end
