import Integrals

include(joinpath(@__DIR__, "constants_functions.jl"))
include(joinpath(@__DIR__, "utils.jl"))

"""coll_n_3_12
Collision operator for decay of particle 3 into particles 1 and 2 (and inverse process).
"""

mutable struct Params_3_12{T<:Real, E}
    model_params::ModelParams{T}
    p1::Particle{T}
    p2::Particle{T}
    p3::Particle{T}
    temps::Vector{T}
    xis::Vector{T}
    energy_type::E

    function Params_3_12{T, E}(
            model_params::ModelParams{T},
            p1::Particle{T},
            p2::Particle{T},
            p3::Particle{T};
            temps::Vector{T} = Vector{T}(undef, 3),
            xis::Vector{T}= Vector{T}(undef, 3),
            energy_type::E = Val(0),
        ) where {T<:Real, E}
        new{T, E}(
            model_params,
            p1, p2, p3,
            temps,
            xis,
            energy_type,
        )
    end
end

@inline function energy_factor(
        ::Val{0},
        p::Params_3_12{T, E}
    ) where {T<:Real, E}
    return one(T)
end
@inline function energy_factor(
        ::Val{1},
        p::Params_3_12{T, E}
    ) where {T<:Real, E}
    return p.p1.e
end
@inline function energy_factor(
        ::Val{2},
        p::Params_3_12{T, E}
    ) where {T<:Real, E}
    return p.p2.e
end
@inline function energy_factor(
        ::Val{3},
        p::Params_3_12{T, E}
    ) where {T<:Real, E}
    return p.p3.e
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

function coll_3_12_e2_min(p::Params_3_12{T, E}) where {T<:Real, E}
    mass_comb = p.p3.m^2 - p.p1.m^2 - p.p2.m^2
    e2_m = (p.p1.e * mass_comb - p.p1.mom * sqrt(mass_comb^2 - 4. * p.p1.m^2 * p.p2.m^2)) / (2. * p.p1.m^2)
    if e2_m < p.p2.m
        return p.p2.m
    else
        return e2_m
    end
end

function coll_3_12_e2_max(p::Params_3_12{T, E}) where {T<:Real, E}
    mass_comb = p.p3.m^2 - p.p1.m^2 - p.p2.m^2
    e2_p = (p.p1.e * mass_comb + p.p1.mom * sqrt(mass_comb^2 - 4. * p.p1.m^2 * p.p2.m^2)) / (2. * p.p1.m^2)
    return e2_p
end

function coll_3_12_ker(e2::T, p::Params_3_12{T, E}) where {T<:Real, E}
    p.p2.e = e2
    p.p2.mom = momentum(p.p2)

    p.p3.e = p.p1.e + p.p2.e
    if p.p3.e < p.p3.m
        println("E_3 = ", p.p3.e, " < E_1 + E_2 = ", p.p1.e + p.p2.e)
        println("Set manually to E_3 = m_3")
        p.p3.e = p.p3.m
    end
    p.p3.mom = momentum(p.p3)

    f1 = dist(p.p1, p.temps[1], p.xis[1])
    f2 = dist(p.p2, p.temps[2], p.xis[2])
    f3 = dist(p.p3, p.temps[3], p.xis[3])

    # TODO: Double check sign in front of k and of total expression
    dist_fac = (
        f1 * f2 * (1 - p.p3.k * f3)
        - f3 * (1 - p.p1.k * f1) * (1 - p.p2.k * f2)
    )
    return energy_factor(p.energy_type, p) * dist_fac
end

function coll_3_12_int_e2(e1::T, p::Params_3_12{T, E}) where {T<:Real, E}
    p.p1.e = e1
    p.p1.mom = momentum(p.p1)

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

function coll_3_12_int_e1(p::Params_3_12{T, E}) where {T<:Real, E}
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
        temps::Vector{T},
        xis::Vector{T};
        energy_type::E = Val(0),
        sq_amp_func::F = coll_A_Nnu_sq_amp,
    ) where {T<:Real, E, F}
    params = Params_3_12{T, E}(
        model_params,
        p1, p2, p3;
        temps=temps,
        xis=xis,
        energy_type=energy_type
    )
    integral_sol = coll_3_12_int_e1(params)

    pre = 1. / (2. ^ 5 * pi^3)
    sq_amp = sq_amp_func(model_params, p1, p2, p3)
    # TODO: Check prefactors!
    return pre * sq_amp * integral_sol
end



function coll_3_12_ker_log(y2::T, p::Params_3_12{T, E}) where {T<:Real, E}
    e2 = exp(y2)
    return e2 * coll_3_12_ker(e2, p)
end

function coll_3_12_int_e2_log(y1::T, p::Params_3_12{T, E}) where {T<:Real, E}
    p.p1.e = exp(y1)
    p.p1.mom = momentum(p.p1)

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
    res = p.p1.e * sol.u
    return res
end

function coll_3_12_int_e1_log(p::Params_3_12{T, E}) where {T<:Real, E}
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
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        model_params::ModelParams{T};
        energy_type::E = Val(0),
        sq_amp_func::F = coll_A_Nnu_sq_amp,
    ) where {T<:Real, E, F}
    params = Params_3_12{T, E}(
        model_params,
        p1, p2, p3;
        energy_type=energy_type
    )
    integral_sol = coll_3_12_int_e1_log(params)

    pre = 1. / (2. ^ 5 * pi^3)
    sq_amp = sq_amp_func(model_params, p1, p2, p3)
    # TODO: Check prefactors!
    return pre * sq_amp * integral_sol
end
