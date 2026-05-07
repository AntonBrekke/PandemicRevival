import Integrals

include(joinpath(@__DIR__, "constants_functions.jl"))
include(joinpath(@__DIR__, "utils.jl"))

"""coll_n_3_12
Collision operator for decay of particle 3 into particles 1 and 2 (and inverse process).
"""
function coll_A_Nnu_sq_amp(params::ModelParams, p1::Particle, p2::Particle, p3::Particle)
    pre = 2. * params.y^2 * sin(params.theta)^2
    mass_dep = (p3.m-p1.m-p2.m) * (p3.m+p1.m+p2.m) * (2*p3.m^2+(p1.m-p2.m)^2)/p3.m^2
    return pre * mass_dep
end

function coll_3_12_e2_min(e1, p1::Particle, p2::Particle, p3::Particle)
    mass_comb = p3.m^2 - p1.m^2 - p2.m^2
    mom1 = sqrt(e1^2 - p1.m^2)
    e2_m = (e1 * mass_comb - mom1 * sqrt(mass_comb^2 - 4. * p1.m^2 * p2.m^2)) / (2. * p1.m^2)
    if e2_m < p2.m
        return p2.m
    else
        return e2_m
    end
end

function coll_3_12_e2_max(e1, p1::Particle, p2::Particle, p3::Particle)
    mass_comb = p3.m^2 - p1.m^2 - p2.m^2
    mom1 = sqrt(e1^2 - p1.m^2)
    e2_p = (e1 * mass_comb + mom1 * sqrt(mass_comb^2 - 4. * p1.m^2 * p2.m^2)) / (2. * p1.m^2)
    return e2_p
end

function coll_3_12_ker(e2, p::NamedTuple)
    f1 = dist(p.temp, p.e1, p.p1)
    f2 = dist(p.temp, e2, p.p2)
    f3 = dist(p.temp, p.e1 + e2, p.p3)

    if p.energy_type == 0
        e = 1.
    elseif p.energy_type == 1
        e = p.e1
    elseif p.energy_type == 2
        e = e2
    elseif p.energy_type == 3
        e = p.e1 + e2
    else
        error("Invalid energy type")
    end
    # TODO: Double check sign in front of k and of total expression
    return  e * (
        f1 * f2 * (1 - p.p3.k * f3)
        - f3 * (1 - p.p1.k * f1) * (1 - p.p2.k * f2)
    )
end

function coll_3_12_int_e2(e1, p::NamedTuple)
    e2_m = coll_3_12_e2_min(e1, p.p1, p.p2, p.p3)
    e2_p = coll_3_12_e2_max(e1, p.p1, p.p2, p.p3)

    params = (
        e1=e1,
        temp=p.temp,
        p1=p.p1,
        p2=p.p2,
        p3=p.p3,
        energy_type=p.energy_type,
    )

    problem = Integrals.IntegralProblem(
        coll_3_12_ker,
        (e2_m, e2_p),
        params
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol
end

function coll_3_12_int_e1(
        temp,
        p1::Particle,
        p2::Particle,
        p3::Particle,
        energy_type::Int64,
    )
    params = (
        temp=temp,
        p1=p1,
        p2=p2,
        p3=p3,
        energy_type=energy_type,
    )
    e1_min = p1.m
    e1_max = max(1e1*temp, 1e1*p1.m)

    problem = Integrals.IntegralProblem(
        coll_3_12_int_e2,
        (e1_min, e1_max),
        params
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol
end

function coll_3_12(
        temp::Float64,
        p1::Particle,
        p2::Particle,
        p3::Particle,
        model_params::ModelParams,
        energy_type=0,
        sq_amp_func=coll_A_Nnu_sq_amp,
    )
    integral_sol = coll_3_12_int_e1(temp, p1, p2, p3, energy_type)

    pre = 1. / (2. ^ 5 * pi^3)
    sq_amp = sq_amp_func(model_params, p1, p2, p3)
    # TODO: Check prefactors!
    return pre * sq_amp * integral_sol.u[1]
end
