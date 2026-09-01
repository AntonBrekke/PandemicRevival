import Integrals

include(joinpath(@__DIR__, "constants_functions.jl"))
include(joinpath(@__DIR__, "utils.jl"))

"""coll_n_3_12
Collision operator for decay of particle 3 into particles 1 and 2 (and inverse process).
"""

mutable struct Params_3_12{T<:Real, R<:Real, S<:Real, E}
    model_params::ModelParams{T}
    p1::Particle{T}
    p2::Particle{T}
    p3::Particle{T}
    temps::NTuple{3, R}
    xis::NTuple{3, S}
    energy_type::E
    e1::Union{R, Nothing}
    e2::Union{R, Nothing}
    e3::Union{R, Nothing}
    mom1::Union{R, Nothing}
    mom2::Union{R, Nothing}
    mom3::Union{R, Nothing}
    debug::Bool

    function Params_3_12{T, R, S, E}(
            model_params::ModelParams{T},
            p1::Particle{T},
            p2::Particle{T},
            p3::Particle{T},
            temps::NTuple{3, R}, # = Vector{R}(undef, 3),
            xis::NTuple{3, S}; # = Vector{S}(undef, 3),
            energy_type::E = Val(0),
        ) where {T<:Real, R<:Real, S<:Real, E}
        new{T, R, S, E}(
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
            false,
        )
    end
end

@inline function energy_factor(
        p::Params_3_12{T, R, S, Val{0}}
    ) where {T<:Real, R<:Real, S<:Real}
    return one(R)
end
@inline function energy_factor(
        p::Params_3_12{T, R, S, Val{1}}
    ) where {T<:Real, R<:Real, S<:Real}
    return p.e1
end
@inline function energy_factor(
        p::Params_3_12{T, R, S, Val{2}}
    ) where {T<:Real, R<:Real, S<:Real}
    return p.e2
end
@inline function energy_factor(
        p::Params_3_12{T, R, S, Val{3}}
    ) where {T<:Real, R<:Real, S<:Real}
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
        p::Params_3_12{T, R, S, E},
    ) where {T<:Real, R<:Real, S<:Real, E}
    mass_comb = p.p3.m^2 - p.p1.m^2 - p.p2.m^2
    e2_m = (p.e1 * mass_comb - p.mom1 * sqrt(mass_comb^2 - 4. * p.p1.m^2 * p.p2.m^2)) / (2. * p.p1.m^2)
    if e2_m < p.p2.m
        return p.p2.m
    else
        return e2_m
    end
    return e2_m
end

function coll_3_12_e2_max(
        p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    mass_comb = p.p3.m^2 - p.p1.m^2 - p.p2.m^2
    e2_p = (p.e1 * mass_comb + p.mom1 * sqrt(mass_comb^2 - 4. * p.p1.m^2 * p.p2.m^2)) / (2. * p.p1.m^2)
    return e2_p
end

function coll_3_12_ker(
        e2::R,
        p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    p.e2 = e2
    # p.mom2 = momentum(p.p2, p.e2)

    p.e3 = p.e1 + p.e2
    if p.e3 < p.p3.m
        println("E_3 = ", p.e3, " < E_1 + E_2 = ", p.e1 + p.e2)
        println("Set manually to E_3 = m_3")
        p.e3 = p.p3.m
    end
    # p.mom3 = momentum(p.p3, p.e3)

    f1 = dist(p.p1, p.temps[1], p.xis[1], p.e1)
    f2 = dist(p.p2, p.temps[2], p.xis[2], p.e2)
    f3 = dist(p.p3, p.temps[3], p.xis[3], p.e3)

    # TODO: Double check sign in front of k and of total expression
    dist_fac_3_12 = (
        f3 * (p.p1.dof - p.p1.k * f1) * (p.p2.dof - p.p2.k * f2)
    )
    dist_fac_12_3 = (
        f1 * f2 * (p.p3.dof - p.p3.k * f3)
    )
    dist_fac = dist_fac_3_12 - dist_fac_12_3

    # if p.debug
    #     println("dist_fac = ", dist_fac)
    #     if isnan(FD.partials(dist_fac)[1])
    #         dist(p.p3, p.temps[3], p.xis[3], p.e3; debug=true)
    #     end
    # end

    return energy_factor(p) * dist_fac
    # Used to compare to Python
    # return energy_factor(p) * dist_fac / (p.p1.dof * p.p2.dof * p.p3.dof)
end

function coll_3_12_int_e2(
        e1::R,
        p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    p.e1 = e1
    p.mom1 = momentum(p.p1, p.e1)

    e2_min = coll_3_12_e2_min(p)
    e2_max = coll_3_12_e2_max(p)
    # if !isfinite(e2_min) || !isfinite(e2_max) || e2_max <= e2_min
    #     return zero(R)
    # end

    problem = Integrals.IntegralProblem(
        coll_3_12_ker,
        (e2_min, e2_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    # if sol.u isa FD.Dual
    #     if isnan(FD.partials(sol.u)[1])
    #         println("found NaN")
    #         p.debug = true
    #         problem = Integrals.IntegralProblem(
    #             coll_3_12_ker,
    #             (e2_min, e2_max),
    #             p
    #         )
    #         sol = Integrals.solve(
    #             problem,
    #             Integrals.QuadGKJL(),
    #         )
    #         error()
    #     end
    # end
    return sol.u
end

function coll_3_12_int_e1(p::Params_3_12{T, R, S, E}) where {T<:Real, R<:Real, S<:Real, E}
    e1_min = p.p1.m
    e1_max = max(1e1*p.temps[1], 1e1*p.p1.m)
    # e1_max = Inf64

    problem = Integrals.IntegralProblem(
        coll_3_12_int_e2,
        (e1_min, e1_max),
        p
    )
    sol = try Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    catch e
        println(p.temps)
        println(p.xis)
        println("e_min = ", e1_min)
        println("e_max = ", e1_max)
        throw(e)
    end
    return sol.u
end

function coll_3_12(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        temps::NTuple{3, R},
        xis::NTuple{3, S};
        energy_type::E = Val(0),
        sq_amp_func::F = coll_A_Nnu_sq_amp,
    ) where {T<:Real, R<:Real, S<:Real, E, F}
    params = Params_3_12{T, R, S, E}(
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
        p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
    e2 = exp(y2)
    return e2 * coll_3_12_ker(e2, p)
end

function coll_3_12_int_e2_log(
        y1::R, p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
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
        p::Params_3_12{T, R, S, E}
    ) where {T<:Real, R<:Real, S<:Real, E}
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
        temps::NTuple{3, R},
        xis::NTuple{3, S};
        energy_type::E = Val(0),
        sq_amp_func::F = coll_A_Nnu_sq_amp,
    ) where {T<:Real, R<:Real, S<:Real, E, F}
    params = Params_3_12{T, R, S, E}(
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
