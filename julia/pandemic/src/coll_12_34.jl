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
    temps::Vector{T}
    xis::Vector{T}
    s::Union{T, Nothing}
    # Only needed for numerical integration over t, not for analytical solution.
    t_min::Union{T, Nothing}
    t_max::Union{T, Nothing}
    a::Union{T, Nothing}

    function Params_12_34{T}(
            model_params::ModelParams{T},
            p1::Particle{T},
            p2::Particle{T},
            p3::Particle{T},
            p4::Particle{T};
            temps::Vector{T} = Vector{T}(undef, 4),
            xis::Vector{T} = Vector{T}(undef, 4),
            s=nothing,
            tm=nothing,
            tp=nothing,
            a=nothing,
        ) where T <: Real
        new{T}(
            model_params,
            p1, p2, p3, p4,
            temps,
            xis,
            s, tm, tp, a
        )
    end
end

function delta(s::T, pa::Particle{T}, pb::Particle{T}) where T <: Real
    del = s + pa.m^2 - pb.m^2 - 2. * pa.e * (pa.e + pb.e)
    if del > 0
        println("Delta = ", del)
        error("Should this happen?")
    end
    return del
end # function

function a_theta(p::Params_12_34{T}) where T <: Real
    return 4. * p.p3.mom^2. * ((p.p1.e + p.p2.e)^2 - p.s)
end # function

function b_theta(p::Params_12_34{T}) where T <: Real
    return - 2. * p.p3.mom / p.p1.mom * delta(p.s, p.p1, p.p2) * delta(p.s, p.p3, p.p4)
end # function

function c_theta(p::Params_12_34{T}) where T <: Real
    return delta(p.s, p.p3, p.p4)^2 + p.p3.mom^2 / p.p1.mom^2 * (p.s - s_lim(1., p.p1, p.p2)) * (p.s - s_lim(-1., p.p1, p.p2))
end # function

# TODO: The square root can be negative. From notes it should be positive. 
# Probably a numerical issue
function cos_theta_lim(pm::Float64, p::Params_12_34{T}) where T <: Real
    a = a_theta(p)
    b = b_theta(p)
    c = c_theta(p)
    insqrt = b^2 - 4. * a * c
    if insqrt < 0.
        # println("Warning: b^2 - 4ac = ", insqrt, " < 0.")
        # println("Set b^2 - 4ac = 0.")
        insqrt = 0.
    end # if
    c_th_lim = (- b + pm * sqrt(insqrt)) / (2. * a)
    if c_th_lim < -1.
        return -1.
    elseif c_th_lim > 1.
        return 1.
    end # if
    return c_th_lim
end # function

# TODO: Probably not needed.
function t_minmax(pm::Float64, p1::Particle{T}, p3::Particle{T}) where T <: Real
    return p1.m^2 + p3.m^2 - 2. * p1.e * p3.e + pm * 2. * p1.mom * p3.mom
end # function

function t_lim(pm::Float64, p::Params_12_34{T}) where T <: Real
    cos_theta_lim_val = cos_theta_lim(pm, p)
    return p.p1.m^2 + p.p3.m^2 - 2. * p.p1.e * p.p3.e + 2. * p.p1.mom * p.p3.mom * cos_theta_lim_val
end # function

function s_lim(pm::Float64, pa::Particle{T}, pb::Particle{T}) where T <: Real
    return (pa.e + pb.e)^2 - (pa.mom - pm * pb.mom)^2
end # function

function s_min(p::Params_12_34{T}) where T <: Real
    return max(s_lim(-1., p.p1, p.p2), s_lim(-1., p.p3, p.p4))
end

function s_max(p::Params_12_34{T}) where T <: Real
    return min(s_lim(1., p.p1, p.p2), s_lim(1., p.p3, p.p4))
end

function coll_12_34_sq_amp(t::T, p::Params_12_34{T}) where T <: Real
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


function coll_12_34_ker(t::T, p::Params_12_34{T}) where T <: Real
    amp_sq = coll_12_34_sq_amp(t, p)
    denom = p.a * (t - p.t_min) * (p.t_max - t)
    if denom <= 0.
        # println("Denominator in kernel is negative: ", denom)
        return zero(T)
    end # if
    return amp_sq / sqrt(denom)
end # function


function coll_12_34_int_t(s::T, p::Params_12_34{T}) where T <: Real
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
    return sol.u
end # function

function coll_12_34_int_s(e3::T, p::Params_12_34{T}) where T <: Real
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

    f1 = dist(p.p1, p.temps[1], p.xis[1])
    f2 = dist(p.p2, p.temps[2], p.xis[2])
    f3 = dist(p.p3, p.temps[3], p.xis[3])
    f4 = dist(p.p4, p.temps[4], p.xis[4])

    # TODO: Study this relation
    dist_dep_12_34 = f1 * f2 * (1 - p.p3.k * f3) * (1 - p.p4.k * f4)
    dist_dep_34_12 = f3 * f4 * (1 - p.p1.k * f1) * (1 - p.p2.k * f2)

    if (dist_dep_12_34 < 1e-30) && (dist_dep_34_12 < 1e-30)
        return 0.
    end

    if abs(dist_dep_12_34 - dist_dep_34_12) < 1e-60
        return 0.
    end

    # reg = (smax - smin) / 1e5
    reg = 0.
    problem = Integrals.IntegralProblem(
        # coll_12_34_int_t,
        coll_12_34_int_t_anal,
        (smin + reg, smax - reg),
        p
    )
    # if (p.temps[1] >= 10.) && (p.p1.e > 1000) && (p.p2.e > 1000)
    #     println("E3 = ", e3, ", E4 = ", p.p4.e)
    #     println(dist_dep_12_34)
    #     println(dist_dep_34_12)
    #     println(dist_dep_12_34 - dist_dep_34_12)
    #     println("smin = ", smin)
    #     println("smax = ", smax)
    # end

    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )

    # if (p.temps[1] >= 10.) && (p.p1.e > 1000) && (p.p2.e > 1000)
    #     println("Here?")
    # end

    if sol.u > 1.
        println("sol is big: ", sol.u)
    end

    return p.p3.mom * (dist_dep_12_34 - dist_dep_34_12) * sol.u
    # return sol[1]
end # function

function coll_12_34_int_e3(e2::T, p::Params_12_34{T}) where T <: Real
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
    return sol.u
end # function


function coll_12_34_int_e2(e1::T, p::Params_12_34{T}) where T <: Real
    p.p1.e = e1
    p.p1.mom = momentum(p.p1)

    e2_min = max(p.p2.m, p.p3.m + p.p4.m - e1)
    e2_max = max(1e2 * p.temps[2], 1e2 * p.p2.m)
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
    return sol.u
end # function

function coll_12_34_int_e1(p::Params_12_34{T}) where T <: Real
    e1_min = p.p1.m
    e1_max = max(1e2 * p.temps[1], 1e2 * p.p1.m)

    problem = Integrals.IntegralProblem(
        coll_12_34_int_e2,
        (e1_min, e1_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL(),
    )
    return sol.u
end # function

function coll_12_34(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        p4::Particle{T},
        temps::Vector{T},
        xis::Vector{T},
    ) where T <: Real
    params = Params_12_34{T}(
        model_params,
        p1,
        p2,
        p3,
        p4;
        temps=temps,
        xis=xis,
    )
    integral = coll_12_34_int_e1(params)
    return integral
end # function

function coll_12_34_int_t_anal(s::T, p::Params_12_34{T}) where T <: Real
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

    num1 = (
        - 16 * mN^8
        + 16 * (tp+tm) * mN^6
        + 2 * mN^4 * (12*mA^4 + 4*(-s+tm+tp)*mA^2 + s^2 - 8*tm*tp - 2*s*(tm+tp))
        + 2 * mN^2 * (
            2*mA^6 - (s+8*(tm+tp))*mA^4 + 2*(s*(tm+tp) - 4*tm*tp)*mA^2 - s*(s*(tm+tp) - 4*tm*tp)
        )
        + 2*s^2*tm*tp - 2*mA^6*(tm+tp) + mA^4*(s*(tm+tp) + 8*tm*tp)
    )

    problem1 = mN^2 - tp
    problem3 = mN^2 - tm
    if problem1 < 0
        term1 = 0
        if (problem1 / tp) > 1e-2
            println("Assumtion that factor is zero could be wrong.")
        end
    elseif problem3 < 0
        a = a_theta(p)
        b = b_theta(p)
        c = c_theta(p)
        println("m_N^2 = ", mN^2)
        println("tp = ", tp)
        println("tm = ", tm)
        println("problem 1 = ", problem1)
        println("problem 3 = ", problem3)
        println("cos_+ = ", cos_theta_lim(1., p))
        println("cos_- = ", cos_theta_lim(-1., p))
        println("s = ", s)
        println("s_min = ", s_min(p))
        println("s_max = ", s_max(p))
        println("s_12- = ", s_lim(-1., p.p1, p.p2))
        println("s_34- = ", s_lim(-1., p.p3, p.p4))
        println("s_12+ = ", s_lim(1., p.p1, p.p2))
        println("s_34+ = ", s_lim(1., p.p3, p.p4))
        println("a = ", a)
        println("b = ", b)
        println("c = ", c)
        println("b^2 = ", b_theta(p)^2)
        println("4ac = ", 4. * a_theta(p) * c_theta(p))
        println("Delta_12 = ", delta(p.s, p.p1, p.p2))
        println("Delta_34 = ", delta(p.s, p.p3, p.p4))
        println("m1 = ", p.p1.m)
        println("E1 = ", p.p1.e)
        println("p1 = ", p.p1.mom)
        println(sqrt(p.p1.e^2 - p.p1.mom^2))
        println((b^2 - 4 * a * c) / (4 * a^2))
        println(b^2 / (4 * a^2))
        error("Does this make any sense?")
    else
        denom1 = sqrt(a) * (s - 2*mA^2) * ((mN^2 - tm) * problem1)^(3/2)
        term1 = - num1 / denom1 * pi/2
    end

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

    # TODO: Maybe more clever way to do this? [HM: 09.07.26]
    problem2 = s + tm - mN^2 - 2*mA^2
    if problem2 < 0
        term2 = 0
        if (problem2 / s) > 1e-2
            error("Assumtion that factor is zero could be wrong.")
        end
    else
        denom2 = sqrt(a) * (2*mA^2 - s) * (problem2 * (s + tp - mN^2 - 2*mA^2))^(3/2)
        term2 = - num2/denom2 * pi/2
    end

    term3 = 4 / sqrt(a) * pi/2

    res = pre * (term1 + term2 + term3)

    # if (p.p1.e > 1000) && (p.p2.e > 1000) && (p.p3.e > 1000)
    #     println("s = ", s)
    #     println("res = ", res)
    # end

    return pre * (term1 + term2 + term3)
end

function coll_12_34_int_t_new(s::T, p::Params_12_34{T}) where T <: Real
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


##### Attempt to rewrite the t integral. Possibly slower. #####

function coll_12_34_ker_q(q::T, p::Params_12_34{T}) where T <: Real
    expq = exp(q)
    t = p.t_min + expq
    return expq * coll_12_34_ker(t, p)
end

function coll_12_34_ker_r(r::T, p::Params_12_34{T}) where T <: Real
    expr = exp(r)
    t = p.t_max - expr
    return expr * coll_12_34_ker(t, p)
end

function coll_12_34_int_t_qr(s::T, p::Params_12_34{T}) where T <: Real
    p.s = s
    p.a = a_theta(p)
    p.t_min = t_lim(-1., p)
    p.t_max = t_lim(1., p)
    if p.t_min >= p.t_max
        return 0.
    end # if

    t_0 = (p.t_max + p.t_min) / 2.
    q_max = log(t_0 - p.t_min)
    r_max = log(p.t_max - t_0)

    q_prob = Integrals.IntegralProblem(
        coll_12_34_ker_q,
        (-Inf64, q_max),
        p
    )
    r_prob = Integrals.IntegralProblem(
        coll_12_34_ker_r,
        (-Inf64, r_max),
        p
    )
    q_sol = Integrals.solve(
        q_prob,
        Integrals.QuadGKJL(),
    )
    r_sol = Integrals.solve(
        r_prob,
        Integrals.QuadGKJL(),
    )
    return q_sol.u + r_sol.u
end # function
