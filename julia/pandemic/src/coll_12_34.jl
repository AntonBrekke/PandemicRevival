using LaTeXStrings
ENV["GKSwstype"] = "nul"
import Plots as Plt
import Integrals

include(joinpath(@__DIR__, "utils.jl"))

mutable struct Params_12_34{T <: Real}
    model_params::ModelParams{T}
    p1::Particle{T}
    p2::Particle{T}
    p3::Particle{T}
    p4::Particle{T}
    temp::T
    s::Union{T, Nothing}
    # Only needed for numerical integration over t, not for analytical solution.
    t_min::Union{T, Nothing}
    t_max::Union{T, Nothing}
    a::Union{T, Nothing}

    function Params_12_34(
            model_params::ModelParams{T},
            p1::Particle{T},
            p2::Particle{T},
            p3::Particle{T},
            p4::Particle{T},
            temp::T;
            s=nothing,
            tm=nothing,
            tp=nothing,
            a=nothing,
        ) where T <: Real
        new{T}(model_params, p1, p2, p3, p4, temp, s, tm, tp, a)
    end
end

function delta(s::Real, pa::Particle, pb::Particle)
    return s + pa.m^2 - pb.m^2 - 2. * pa.e * (pa.e + pb.e)
end # function

function a_theta(p::Params_12_34)
    return 4. * p.p3.mom^2. * ((p.p1.e + p.p2.e)^2 - p.s)
end # function

function b_theta(p::Params_12_34)
    return - 2. * p.p3.mom / p.p1.mom * delta(p.s, p.p1, p.p2) * delta(p.s, p.p3, p.p4)
end # function

function c_theta(p::Params_12_34)
    return delta(p.s, p.p3, p.p4)^2 + p.p3.mom^2 / p.p1.mom^2 * (p.s - s_lim(1., p.p1, p.p2)) * (p.s - s_lim(-1., p.p1, p.p2))
end # function

# TODO: The square root can be negative. From notes it should be positive...
function cos_theta_lim(pm::Float64, p::Params_12_34)
    a = a_theta(p)
    b = b_theta(p)
    c = c_theta(p)
    insqrt = b^2 - 4. * a * c
    if insqrt < 0.
        println("Warning: b^2 - 4ac = ", insqrt, " < 0.")
        println("Set b^2 - 4ac = 0.")
        insqrt = 0.
    end # if
    c_th_lim = (- b + pm * sqrt(insqrt)) / (2. * a)
    if pm == -1. && c_th_lim < -1.
        println("cos(theta)_- = ", c_th_lim, " < -1.")
        println("cos(theta)_- is set to -1.")
        return -1.
    elseif pm == 1. && c_th_lim > 1.
        println("cos(theta)_+ = ", c_th_lim, " > 1.")
        println("cos(theta)_+ is set to 1.")
        return 1.
    end # if
    return c_th_lim
end # function

# TODO: Probably not needed.
function t_minmax(pm::Float64, p1::Particle, p3::Particle)
    return p1.m^2 + p3.m^2 - 2. * p1.e * p3.e + pm * 2. * p1.mom * p3.mom
end # function

function t_lim(pm::Float64, p::Params_12_34)
    cos_theta_lim_val = cos_theta_lim(pm, p)
    return p.p1.m^2 + p.p3.m^2 - 2. * p.p1.e * p.p3.e + 2. * p.p1.mom * p.p3.mom * cos_theta_lim_val
end # function

function s_lim(pm::Float64, pa::Particle, pb::Particle)
    return (pa.e + pb.e)^2 - (pa.mom - pm * pb.mom)^2
end # function

function s_min(p::Params_12_34)
    return max(s_lim(-1., p.p1, p.p2), s_lim(-1., p.p3, p.p4))
end

function s_max(p::Params_12_34)
    return min(s_lim(1., p.p1, p.p2), s_lim(1., p.p3, p.p4))
end

function coll_12_34_sq_amp(t, p::Params_12_34)
    mN = p.p3.m
    mA = p.p1.m
    pre = 8. * p.model_params.y^4
    denom = (mN^2 - t)^2 * (p.s + t - mN^2 - 2. * mA^2)^2
    nom = (
        - 2. * mN^8
        - 8. * mN^6 * (
            mA^2
            - t
        )
        - mN^4 * (
            30. * mA^4
            - 8. * mA^2 * (2. * p.s + 3. * t)
            + 3. * p.s^2 + 4. * p.s * t + 12. * t^2
        )
        + mN^2 * (
            - 28. * mA^6
            + 4. * mA^4 * (22. * p.s + 28. * t)
            - 2. * mA^2 * (3. * p.s^2 + 4. * p.s * t + 12. * t^2)
            + p.s^3 + 2. * p.s^2 * t + 8. * p.s * t^2 + 8. * t^3
        )
        - 4. * mA^8
        + 4. * mA^6 * (p.s + 3. * t)
        - mA^4 * (p.s^2 + 6. * p.s * t + 14. * t^2)
        + 2. * mA^2 * t * (p.s + 2. * t)^2
        - t * (p.s + t) * (p.s^2 + 2. * p.s * t + 2. * t^2)
    )
    return pre * nom / denom
    # return 1.
end # function


function coll_12_34_ker(t::Real, p::Params_12_34)
    amp_sq = coll_12_34_sq_amp(t, p)
    denom = p.a * (t - p.t_min) * (p.t_max - t)
    if denom < 0.
        error("Denominator in kernel is negative: ", denom)
    end # if
    return amp_sq / sqrt(denom)
end # function


function coll_12_34_int_t(s::Real, p::Params_12_34)
    p.s = s
    p.a = a_theta(p)
    p.t_min = t_lim(-1., p)
    p.t_max = t_lim(1., p)
    if p.t_min >= p.t_max
        return 0.
    end # if
    problem = Integrals.IntegralProblem(
        coll_12_34_ker,
        (p.t_min, p.t_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol[1]
end # function

function coll_12_34_int_s(e3::Real, p::Params_12_34)
    p.p3.e = e3
    p.p3.mom = momentum(p.p3)

    p.p4.e = p.p1.e + p.p2.e - e3
    if p.p4.e < p.p4.m
        println("e3 = ", e3, " gives")
        println("p4.e = ", p.p4.e, " < p4.m = ", p.p4.m)
        println("Set manually to p4.e = p4.m")
        p.p4.e = p.p4.m
    end # if
    p.p4.mom = momentum(p.p4)

    smin = s_min(p)
    smax = s_max(p)
    if smin >= smax
        return 0.
    end # if

    # reg = (smax - smin) / 1e5
    reg = 0.
    problem = Integrals.IntegralProblem(
        # coll_12_34_int_t,
        coll_12_34_int_t_anal,
        (smin + reg, smax - reg),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )

    f1 = dist(p.temp, p.p1)
    f2 = dist(p.temp, p.p2)
    f3 = dist(p.temp, p.p3)
    f4 = dist(p.temp, p.p4)

    # TODO: Study this relation
    dist_dep_12_34 = f1 * f2 * (1 - p.p3.k * f3) * (1 - p.p4.k * f4)
    dist_dep_34_12 = f3 * f4 * (1 - p.p1.k * f1) * (1 - p.p2.k * f2)
    return p.p3.mom * (dist_dep_12_34 - dist_dep_34_12) * sol[1]
    # return sol[1]
end # function

function coll_12_34_int_e3(e2::Real, p::Params_12_34)
    p.p2.e = e2
    p.p2.mom = momentum(p.p2)

    e3_min = p.p3.m
    e3_max = p.p1.e + e2 - p.p4.m
    if e3_min >= e3_max
        return 0.
    end # if
    problem = Integrals.IntegralProblem(
        coll_12_34_int_s,
        (e3_min, e3_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol[1]
end # function


function coll_12_34_int_e2(e1::Real, p::Params_12_34)
    p.p1.e = e1
    p.p1.mom = momentum(p.p1)

    e2_min = max(p.p2.m, p.p3.m + p.p4.m - e1)
    e2_max = max(1e2 * p.temp, 1e2 * p.p2.m)
    if e2_min >= e2_max
        return 0.
    end # if
    problem = Integrals.IntegralProblem(
        coll_12_34_int_e3,
        (e2_min, e2_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol
end # function

function coll_12_34_int_e1(p::Params_12_34)
    e1_min = p.p1.m
    e1_max = max(1e2 * p.temp, 1e2 * p.p1.m)

    problem = Integrals.IntegralProblem(
        coll_12_34_int_e2,
        (e1_min, e1_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol
end # function

function coll_12_34(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        p4::Particle{T},
        temp::T,
    ) where T <: Real
    params = Params_12_34(
        model_params,
        p1,
        p2,
        p3,
        p4,
        temp,
    )

    pre = 1.
    integral = coll_12_34_int_e1(params)
    return integral
end # function

function coll_12_34_int_t_anal(s::Real, p::Params_12_34)
    p.s = s
    mA = p.p1.m
    mN = p.p3.m
    p.a = a_theta(p)
    a = p.a
    # println("a = ", a)
    tm = t_lim(-1., p)
    tp = t_lim(1., p)
    pre = -8. * p.model_params.y^4
    # println(mN^2 - tm)
    # println(mN^2 - tp)
    denom1 = sqrt(a) * (s - 2*mA^2) * (mN^2 - tm)^(3/2) * (mN^2 - tp)^(3/2)
    num1 = (
        - 16 * mN^8
        + 16 * (tp+tm) * mN^6
        + 2 * mN^4 * (12*mA^4 + 4*(-s+tm+tp)*mA^2 + s^2 - 8*tm*tp - 2*s*(tm+tp))
        + 2 * mN^2 * (
            2*mA^6 - (s+8*(tm+tp))*mA^4 + 2*(s*(tm+tp) - 4*tm*tp)*mA^2 - s*(s*(tm+tp) - 4*tm*tp)
        )
        + 2*s^2*tm*tp - 2*mA^6*(tm+tp) + mA^4*(s*(tm+tp) + 8*tm*tp)
    )
    term1 = - num1 / denom1 * pi/2

    denom2 = sqrt(a) * (2*mA^2 - s) * (s + tm - mN^2 - 2*mA^2)^(3/2) * (s + tp - mN^2 - 2*mA^2)^(3/2)
    num2 = (
        16 * mN^8
        + 16 * mN^6 * (6*mA^2 - 3*s -tm -tp)
        + 2 * mN^4 * (
            84*mA^4 - 28*(3*s+tm+tp)*mA^2 
            + 19*s^2 + 8*tm*tp + 14*s*(tm+tp)
        )
        + 2 * mN^2 * (
            34*mA^6 -(57*s+16*(tm+tp))*mA^4 
            + 2*(12*s^2 + 9*(tm+tp)*s +4*tm*tp)*mA^2 
            - s*(2*s^2 + 3*(tm+tp)*s + 4*tm*tp)
        )
        - 24*mA^8 - 2*s^2*(s+tm)*(s+tp) + 4*mA^2*s^2*(2*s+tm+tp) 
        - mA^4*(14*s^2+7*(tm+tp)*s+8*tm*tp) + 2*mA^6*(12*s + 7*(tm+tp))
    )
    term2 = - num2/denom2 * pi/2

    term3 = 4 / sqrt(a) * pi/2

    return pre * (term1 + term2 + term3)
end

function coll_12_34_int_t_new(s::Real, p::Params_12_34)
    p.s = s
    mA = p.p1.m
    mN = p.p3.m
    p.a = a_theta(p)
    a = p.a
    tm = t_lim(-1., p)
    tp = t_lim(1., p)

    term1 = -(16*pi)/sqrt(a)

    term2 = (
        pi / (
            (2*mA^2-s)*(a*(mN^2-tm)*(mN^2-tp))^(3/2)
        )
        * 4 * a * (
            16 * mN^8
            - 16 * mN^6 * (tm+tp)
            - 2 * mN^4 * (
                12*mA^4+4*mA^2*(-s+tm+tp)+s^2-2*s*(tm+tp)-8*tm*tp
            )
            + 2 * mN^2 * (
                -2*mA^6+mA^4*(s+8*(tm+tp))-2*mA^2*(s*(tm+tp)-4*tm*tp)+s*(s*(tm+tp)-4*tm*tp)
            )
            + 2 * mA^6 * (tm+tp) - mA^4 * (s*(tm+tp) + 8*tm*tp)
            - 2 * s^2*tm*tp
        )
    )

    term3 = (
        - pi / (
            (2*mA^2-s)*(a*(mN^2+2*mA^2-s-tm)*(mN^2+2*mA^2-s-tp))^(3/2)
        )
        * 4 * a * (
            - 16 * mN^8
            + 16 * mN^6 * (-6*mA^2+3*s+tm+tp)
            - 2 * mN^4 * (
                84*mA^4-28*mA^2*(3*s+tm+tp)
                +19*s^2+14*s*(tm+tp)+8*tm*tp
            )
            + 2 * mN^2 * (
                -34*mA^6+mA^4*(57*s+16*(tm+tp))
                -2*mA^2*(12*s^2+9*s*(tm+tp)+4*tm*tp)
                +s*(2*s^2+3*s*(tm+tp)+4*tm*tp)
            )
            + 24 * mA^8
            - 2 * mA^6 * (12*s+7*(tm+tp))
            + mA^4 * (14*s^2+7*s*(tm+tp)+8*tm*tp)
            - 4 * mA^2 * s^2*(2*s+tm+tp)
            + 2 * s^2*(s+tm)*(s+tp)
        )
    )

    return p.model_params.y^4 * (term1 + term2 + term3)

end