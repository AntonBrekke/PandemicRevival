import Integrals

include(joinpath(@__DIR__, "constants_functions.jl"))
include(joinpath(@__DIR__, "utils.jl"))

"""coll_n_3_12
Collision operator for decay of particle 3 into particles 1 and 2 (and inverse process).
"""

mutable struct Params_3_12{T<:Real, R<:Real, E}
    model_params::ModelParams{T}
    p1::Particle{T}
    p2::Particle{T}
    p3::Particle{T}
    temps::Vector{R}
    xis::Vector{R}
    energy_type::E
    e1::Union{R, Nothing}
    e2::Union{R, Nothing}
    e3::Union{R, Nothing}
    mom1::Union{R, Nothing}
    mom2::Union{R, Nothing}
    mom3::Union{R, Nothing}

    function Params_3_12{T, R, E}(
            model_params::ModelParams{T},
            p1::Particle{T},
            p2::Particle{T},
            p3::Particle{T},
            temps::Vector{R}, # = Vector{R}(undef, 3),
            xis::Vector{R}; # = Vector{R}(undef, 3),
            energy_type::E = Val(0),
        ) where {T<:Real, R<:Real, E}
        new{T, R, E}(
            model_params,
            p1, p2, p3,
            temps,
            xis,
            energy_type,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
        )
    end
end

@inline function energy_factor(
        p::Params_3_12{T, R, Val{0}}
    ) where {T<:Real, R<:Real}
    return one(R)
end
@inline function energy_factor(
        p::Params_3_12{T, R, Val{1}}
    ) where {T<:Real, R<:Real}
    return p.e1
end
@inline function energy_factor(
        p::Params_3_12{T, R, Val{2}}
    ) where {T<:Real, R<:Real}
    return p.e2
end
@inline function energy_factor(
        p::Params_3_12{T, R, Val{3}}
    ) where {T<:Real, R<:Real}
    return p.e3
end


function coll_A_Nnu_sq_amp(
        params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T}
    ) where T <: Real
    pre = 2. * params.y^2 * sin(params.theta)^2
    mass_dep = (p3.m-p1.m-p2.m) * (p3.m+p1.m+p2.m) * (2*p3.m^2+(p1.m-p2.m)^2)/p3.m^2
    return pre * mass_dep
end

function coll_3_12_e2_min(
        p::Params_3_12{T, R, E},
    ) where {T<:Real, R<:Real, E}
    mass_comb = p.p3.m^2 - p.p1.m^2 - p.p2.m^2
    e2_m = (p.e1 * mass_comb - p.mom1 * sqrt(mass_comb^2 - 4. * p.p1.m^2 * p.p2.m^2)) / (2. * p.p1.m^2)
    if e2_m < p.p2.m
        return p.p2.m
    else
        return e2_m
    end
end

function coll_3_12_e2_max(p::Params_3_12{T, R, E}) where {T<:Real, R<:Real, E}
    mass_comb = p.p3.m^2 - p.p1.m^2 - p.p2.m^2
    e2_p = (p.e1 * mass_comb + p.mom1 * sqrt(mass_comb^2 - 4. * p.p1.m^2 * p.p2.m^2)) / (2. * p.p1.m^2)
    return e2_p
end

function coll_3_12_ker(
        e2::R,
        p::Params_3_12{T, R, E}
    ) where {T<:Real, R<:Real, E}
    p.e2 = e2
    p.mom2 = momentum(p.p2, p.e2)

    p.e3 = p.e1 + p.e2
    if p.e3 < p.p3.m
        println("E_3 = ", p.e3, " < E_1 + E_2 = ", p.e1 + p.e2)
        println("Set manually to E_3 = m_3")
        p.e3 = p.p3.m
    end
    p.mom3 = momentum(p.p3, p.e3)

    f1 = dist(p.p1, p.temps[1], p.xis[1], p.e1)
    f2 = dist(p.p2, p.temps[2], p.xis[2], p.e2)
    f3 = dist(p.p3, p.temps[3], p.xis[3], p.e3)

    # TODO: Double check sign in front of k and of total expression
    dist_fac = (
        + f3 * (1 - p.p1.k * f1) * (1 - p.p2.k * f2)
        - f1 * f2 * (1 - p.p3.k * f3)
    )
    return energy_factor(p) * dist_fac
end

function coll_3_12_int_e2(
        e1::R,
        p::Params_3_12{T, R, E}
    ) where {T<:Real, R<:Real, E}
    p.e1 = e1
    p.mom1 = momentum(p.p1, p.e1)

    e2_min = coll_3_12_e2_min(p)
    e2_max = coll_3_12_e2_max(p)

    problem = Integrals.IntegralProblem(
        coll_3_12_ker,
        (e2_min, e2_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol.u
end

function coll_3_12_int_e1(p::Params_3_12{T, R, E}) where {T<:Real, R<:Real, E}
    e1_min = p.p1.m
    e1_max = max(1e1*p.temps[1], 1e1*p.p1.m)
    # e1_max = Inf64

    problem = Integrals.IntegralProblem(
        coll_3_12_int_e2,
        (e1_min, e1_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol.u
end

function coll_3_12(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        temps::Vector{R},
        xis::Vector{R};
        energy_type::E = Val(0),
        sq_amp_func::F = coll_A_Nnu_sq_amp,
    ) where {T<:Real, R<:Real, E, F}
    params = Params_3_12{T, R, E}(
        model_params,
        p1, p2, p3,
        temps,
        xis;
        energy_type=energy_type
    )
    integral_sol = coll_3_12_int_e1(params)

    pre = 1. / (2^5 * pi^3)
    sq_amp = sq_amp_func(model_params, p1, p2, p3)
    # TODO: Check prefactors!
    return pre * sq_amp * integral_sol
end



function coll_3_12_ker_log(
        y2::R,
        p::Params_3_12{T, R, E}
    ) where {T<:Real, R<:Real, E}
    e2 = exp(y2)
    return e2 * coll_3_12_ker(e2, p)
end

function coll_3_12_int_e2_log(
        y1::R, p::Params_3_12{T, R, E}
    ) where {T<:Real, R<:Real, E}
    p.e1 = exp(y1)
    p.mom1 = momentum(p.p1, p.e1)

    e2_min = coll_3_12_e2_min(p)
    e2_max = coll_3_12_e2_max(p)

    y2_min = log(e2_min)
    y2_max = log(e2_max)

    problem = Integrals.IntegralProblem(
        coll_3_12_ker_log,
        (y2_min, y2_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    res = p.e1 * sol.u
    return res
end

function coll_3_12_int_e1_log(
        p::Params_3_12{T, R, E}
    ) where {T<:Real, R<:Real, E}
    e1_min = p.p1.m
    e1_max = max(1e1*p.temps[1], 1e1*p.p1.m)
    # e1_max = Inf64

    y1_min = log(e1_min)
    y1_max = log(e1_max)

    problem = Integrals.IntegralProblem(
        coll_3_12_int_e2_log,
        (y1_min, y1_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol.u
end


function coll_3_12_log(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        temps::Vector{R},
        xis::Vector{R};
        energy_type::E = Val(0),
        sq_amp_func::F = coll_A_Nnu_sq_amp,
    ) where {T<:Real, R<:Real, E, F}
    params = Params_3_12{T, R, E}(
        model_params,
        p1, p2, p3,
        temps,
        xis;
        energy_type=energy_type
    )
    integral_sol = coll_3_12_int_e1_log(params)

    pre = 1. / (2^5 * pi^3)
    sq_amp = sq_amp_func(model_params, p1, p2, p3)
    # TODO: Check prefactors!
    return pre * sq_amp * integral_sol
end
