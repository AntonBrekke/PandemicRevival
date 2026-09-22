import Integrals

include(joinpath(@__DIR__, "utils.jl"))

"""
coll_12_34.jl

Collision integral for 1 2 <-> 3 4, set up for A'A' <-> N N (p1 = p2 = A',
p3 = p4 = N) with the amplitude of eq. (A9) of the paper (valid for
m_N1 = m_N2):

    coll_12_34(mp, p1, p2, p3, p4, temps, xis)
        = ∫ dΠ1 dΠ2 dΠ3 dΠ4 (2π)^4 δ^4(p1+p2-p3-p4) |M|^2
          × [f1 f2 (1 - k3 f3)(1 - k4 f4) - f3 f4 (1 - k1 f1)(1 - k2 f2)]
        = g^4/(256 π^6) ∫dE1 ∫dE2 ∫dE3 p3 [...] ∫ds ∫dt (|M|^2/g^4) / sqrt(-a (t - t_min)(t_max - t)),

with f_i = `dist(p_i, ...)` the occupation number per degree of freedom (no
dof factor), k_i = +1 (-1) for fermions (bosons), |M|^2 summed over all dof,
a = `a_theta` < 0, and t_min, t_max the t-limits at fixed (E1, E2, E3, s).
The parametrisation and its sign conventions follow Bringmann et al.,
arXiv:2206.10630, Appendix, Eqs. (17)-(24). No symmetry factors for identical
particles are included. The t-integral is done analytically in
`coll_12_34_int_t_new`.

See coll_12_34_legendre.jl for an independent implementation, and
test/test_coll_12_34_AA_NN.jl for the comparison between the two.
"""

mutable struct Params_12_34{T<:Real, R<:Real}
    p1::Particle{T}
    p2::Particle{T}
    p3::Particle{T}
    p4::Particle{T}
    temps::NTuple{4, R}
    xis::NTuple{4, R}
    e1::Union{R, Nothing}
    e2::Union{R, Nothing}
    e3::Union{R, Nothing}
    e4::Union{R, Nothing}
    mom1::Union{R, Nothing}
    mom2::Union{R, Nothing}
    mom3::Union{R, Nothing}
    mom4::Union{R, Nothing}
    s::Union{R, Nothing}
    # Only needed for numerical integration over t, not for analytical solution.
    t_min::Union{R, Nothing}
    t_max::Union{R, Nothing}
    a::Union{R, Nothing}
    # Tolerances for the nested integrals. The integrands carry physical units
    # (and are tiny in GeV), so a nonzero abstol easily makes them meaningless.
    reltol::Float64
    abstol::Float64

    function Params_12_34{T, R}(
            p1::Particle{T},
            p2::Particle{T},
            p3::Particle{T},
            p4::Particle{T},
            temps::NTuple{4, R}, # = Vector{R}(undef, 4),
            xis::NTuple{4, R}; # = Vector{R}(undef, 4),
            reltol::Real = 1e-6,
            abstol::Real = 0.,
        ) where {T<:Real, R<:Real}
        new{T, R}(
            p1, p2, p3, p4,
            temps,
            xis,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            reltol,
            abstol,
        )
    end
end

function delta(
        s::R,
        pa::Particle{T},
        pb::Particle{T},
        ea::R,
        eb::R
    ) where {T<:Real, R<:Real}
    del = s + pa.m^2 - pb.m^2 - 2. * ea * (ea + eb)
    return del
end # function

# The t-limits at fixed (E1, E2, E3, s) follow from a cos^2 + b cos + c = 0,
# with cos = cos(theta) and theta the lab angle between p1 and p3. The sign
# conventions are those of Bringmann et al., arXiv:2206.10630, Eqs. (17)-(20):
# a < 0, so that a cos^2 + b cos + c >= 0 between the roots
# c_{theta,+} <= c_{theta,-}. Geometrically, p1 and p3 make fixed angles with
# P = p1 + p2, with cosines -delta_12 / (2 p1 |P|) and -delta_34 / (2 p3 |P|),
# and c_{theta,+-} = cos(theta_1P +- theta_3P).
# Checked against explicit 4-vectors in test/test_coll_12_34_AA_NN.jl.
function a_theta(p::Params_12_34{T, R}) where {T <: Real, R <: Real}
    return - 4. * p.mom3^2 * ((p.e1 + p.e2)^2 - p.s)
end # function

function b_theta(p::Params_12_34{T, R}) where {T <: Real, R <: Real}
    return 2. * p.mom3 / p.mom1 * delta(p.s, p.p1, p.p2, p.e1, p.e2) * delta(p.s, p.p3, p.p4, p.e3, p.e4)
end # function

function c_theta(p::Params_12_34{T, R}) where {T <: Real, R <: Real}
    return (
        - delta(p.s, p.p3, p.p4, p.e3, p.e4)^2
        - p.mom3^2 / p.mom1^2
        * (p.s - s_lim(-1, p.e1, p.e2, p.mom1, p.mom2)) * (p.s - s_lim(1, p.e1, p.e2, p.mom1, p.mom2))
    )
end # function

# c_{theta,pm} = (-b pm sqrt(b^2 - 4ac)) / (2a). Since a < 0, pm = +1 gives
# the lower root and pm = -1 the upper root.
function cos_theta_lim(
        pm::Int64,
        p::Params_12_34{T, R}
    ) where {T <: Real, R <: Real}
    a = a_theta(p)
    b = b_theta(p)
    # b^2 - 4ac, factorized to avoid cancellation. Non-negative for s_min <= s <= s_max.
    disc = (
        4. * p.mom3^2 / p.mom1^2
        * (p.s - s_lim(-1, p.e1, p.e2, p.mom1, p.mom2)) * (p.s - s_lim(1, p.e1, p.e2, p.mom1, p.mom2))
        * (p.s - s_lim(-1, p.e3, p.e4, p.mom3, p.mom4)) * (p.s - s_lim(1, p.e3, p.e4, p.mom3, p.mom4))
    )
    c_th_lim = (- b + pm * sqrt(max(disc, zero(disc)))) / (2. * a)
    return clamp(c_th_lim, -one(c_th_lim), one(c_th_lim))
end # function

# t at cos(theta) = c_{theta,pm}: t_lim(1, p) = t_min, t_lim(-1, p) = t_max.
function t_lim(pm::Int64, p::Params_12_34{T, R}) where {T <: Real, R <: Real}
    cos_theta_lim_val = cos_theta_lim(pm, p)
    return p.p1.m^2 + p.p3.m^2 - 2. * p.e1 * p.e3 + 2. * p.mom1 * p.mom3 * cos_theta_lim_val
end # function

function s_lim(pm::Int64, ea::R, eb::R, moma::R, momb::R) where R <: Real
    return (ea + eb)^2 - (moma - pm * momb)^2
end # function

function s_min(p::Params_12_34{T, R}) where {T <: Real, R <: Real}
    return max(
        s_lim(-1, p.e1, p.e2, p.mom1, p.mom2),
        s_lim(-1, p.e3, p.e4, p.mom3, p.mom4)
    )
end

function s_max(p::Params_12_34{T, R}) where {T<:Real, R<:Real}
    return min(
        s_lim(1, p.e1, p.e2, p.mom1, p.mom2),
        s_lim(1, p.e3, p.e4, p.mom3, p.mom4)
    )
end

# |M|^2 for A'A' -> N N summed over all dof, eq. (A9). Valid for m_N1 = m_N2,
# with p1 = p2 = A', p3 = p4 = N and exchanged fermion mass p3.m.
# Coupling g^4 is moved out of the integral.
function coll_12_34_sq_amp(
        t::R,
        p::Params_12_34{T, R}
    ) where {T <: Real, R <: Real}
    mA = p.p1.m
    mN = p.p3.m
    u = 2. * mA^2 + 2. * mN^2 - p.s - t
    c2 = (mA^2 + 2. * mN^2)^2
    return 8. * (
        (4. * (mA^2 - mN^2)^2 - 16. * mN^4 + (2. * mN^2 + p.s)^2) / ((t - mN^2) * (u - mN^2))
        - c2 / (t - mN^2)^2
        - c2 / (u - mN^2)^2
        - 2.
    )
end # function

function coll_12_34_ker(
        t::R,
        p::Params_12_34{T, R}
    ) where {T <: Real, R <: Real}
    amp_sq = coll_12_34_sq_amp(t, p)
    denom = - p.a * (t - p.t_min) * (p.t_max - t)
    if denom < 0.
        println("Denominator in kernel is negative: ", denom)
        return zero(R)
    end # if
    return amp_sq / sqrt(denom)
end # function

# Numerical t-integral, used to test the closed form `coll_12_34_int_t_new`.
# t = t_mid - t_half cos(phi) removes the endpoint singularities:
# int dt |M|^2 / sqrt(-a (t - t_min)(t_max - t)) = 1/sqrt(-a) int_0^pi dphi |M|^2.
function coll_12_34_int_t(s::R, p::Params_12_34{T, R}) where {T<:Real, R<:Real}
    p.s = s
    p.a = a_theta(p)
    p.t_min = t_lim(1, p)
    p.t_max = t_lim(-1, p)
    t_mid = (p.t_max + p.t_min) / 2.
    t_half = (p.t_max - p.t_min) / 2.
    problem = Integrals.IntegralProblem(
        (phi, q) -> coll_12_34_sq_amp(t_mid - t_half * cos(phi), q),
        (0., pi),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL();
        reltol=1e-10,
        abstol=0.,
    )
    return sol.u / sqrt(-p.a)
end # function


function coll_12_34_int_s(e3::R, p::Params_12_34{T, R}) where {T<:Real, R<:Real}
    p.e3 = e3
    p.mom3 = momentum(p.p3, p.e3)

    p.e4 = p.e1 + p.e2 - p.e3
    if p.e4 < p.p4.m
        println("e3 = ", e3, " gives")
        println("p4.e = ", p.e4, " < p4.m = ", p.p4.m)
        println("Set manually to p4.e = p4.m")
        p.e4 = p.p4.m
    end # if
    p.mom4 = momentum(p.p4, p.e4)

    smin = s_min(p)
    smax = s_max(p)
    if smin >= smax
        return 0.
    end # if

    f1 = dist(p.p1, p.temps[1], p.xis[1], p.e1)
    f2 = dist(p.p2, p.temps[2], p.xis[2], p.e2)
    f3 = dist(p.p3, p.temps[3], p.xis[3], p.e3)
    f4 = dist(p.p4, p.temps[4], p.xis[4], p.e4)

    dist_dep_12_34 = (
        f1 * f2 * (1. - p.p3.k * f3) * (1. - p.p4.k * f4)
    )
    dist_dep_34_12 = (
        f3 * f4 * (1. - p.p1.k * f1) * (1. - p.p2.k * f2)
    )

    if (dist_dep_12_34 < 1e-30) && (dist_dep_34_12 < 1e-30)
        return 0.
    end

    # Relative, so that round-off at chemical equilibrium does not send the
    # adaptive quadrature (abstol = 0) chasing noise.
    if abs(dist_dep_12_34 - dist_dep_34_12) <= 1e-13 * max(dist_dep_12_34, dist_dep_34_12)
        return 0.
    end

    problem = Integrals.IntegralProblem(
        coll_12_34_int_t_new,
        (smin, smax),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL();
        reltol=p.reltol,
        abstol=p.abstol,
    )

    return p.mom3 * (dist_dep_12_34 - dist_dep_34_12) * sol.u
end # function

function coll_12_34_int_e3(
        e2::R,
        p::Params_12_34{T, R}
    ) where {T<:Real, R<:Real}
    p.e2 = e2
    p.mom2 = momentum(p.p2, p.e2)

    e3_min = p.p3.m
    e3_max = p.e1 + p.e2 - p.p4.m
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
        Integrals.QuadGKJL();
        reltol=p.reltol,
        abstol=p.abstol,
    )
    return sol.u
end # function


function coll_12_34_int_e2(
        e1::R,
        p::Params_12_34{T, R}
    ) where {T<:Real, R<:Real}
    p.e1 = e1
    p.mom1 = momentum(p.p1, p.e1)

    e2_min = max(p.p2.m, p.p3.m + p.p4.m - p.e1)
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
        Integrals.QuadGKJL();
        reltol=p.reltol,
        abstol=p.abstol,
    )
    return sol.u
end # function

function coll_12_34_int_e1(p::Params_12_34{T, R}) where {T<:Real, R<:Real}
    e1_min = p.p1.m
    e1_max = max(1e2 * p.temps[1], 1e2 * p.p1.m)

    problem = Integrals.IntegralProblem(
        coll_12_34_int_e2,
        (e1_min, e1_max),
        p
    )
    sol = Integrals.solve(
        problem,
        Integrals.QuadGKJL();
        reltol=p.reltol,
        abstol=p.abstol,
    )
    return sol.u
end # function

function coll_12_34(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        p4::Particle{T},
        temps::NTuple{4, R},
        xis::NTuple{4, R};
        reltol::Real = 1e-6,
        abstol::Real = 0.,
    ) where {T<:Real, R<:Real}
    params = Params_12_34{T, R}(
        p1,
        p2,
        p3,
        p4,
        temps,
        xis;
        reltol=reltol,
        abstol=abstol,
    )
    pre = 1. / (256. * pi^6)
    integral = pre * model_params.y^4 * coll_12_34_int_e1(params)
    return integral
end # function

# Closed form of int_{t_min}^{t_max} dt |M|^2 / sqrt(-a (t - t_min)(t_max - t))
# for the amplitude in `coll_12_34_sq_amp`. With t = t_mid + t_half cos(phi),
# t - mN^2 = -dt(phi) and u - mN^2 = -du(phi), where dt + du = s - 2 mA^2 and
#   int_0^pi dphi / dt       = pi / sqrt(dt(t_min) dt(t_max)),
#   int_0^pi dphi / dt^2     = pi (dt(t_min) + dt(t_max))/2 / (dt(t_min) dt(t_max))^(3/2),
#   1 / (dt du)              = (1/dt + 1/du) / (s - 2 mA^2),
# and likewise for du. All of dt, du are >= mN^2 > 0 in the physical region
# (t, u <= 0 when m1 = m2 and m3 = m4), so there are no poles.
function coll_12_34_int_t_new(
        s::R,
        p::Params_12_34{T, R}
    ) where {T<:Real, R<:Real}
    p.s = s
    mA = p.p1.m
    mN = p.p3.m
    p.a = a_theta(p)
    tm = t_lim(1, p)    # t_min
    tp = t_lim(-1, p)   # t_max

    dt_m = mN^2 - tm
    dt_p = mN^2 - tp
    du_m = s + tm - 2. * mA^2 - mN^2
    du_p = s + tp - 2. * mA^2 - mN^2
    prod_t = dt_m * dt_p
    prod_u = du_m * du_p

    num_tu = 4. * (mA^2 - mN^2)^2 - 16. * mN^4 + (2. * mN^2 + s)^2
    c2 = (mA^2 + 2. * mN^2)^2

    term_tu = num_tu / (s - 2. * mA^2) * (1. / sqrt(prod_t) + 1. / sqrt(prod_u))
    term_tt = c2 * (dt_m + dt_p) / 2. / prod_t^(3/2)
    term_uu = c2 * (du_m + du_p) / 2. / prod_u^(3/2)

    return 8. * pi * (term_tu - term_tt - term_uu - 2.) / sqrt(-p.a)
end


##### Attempt to rewrite the t integral. Possibly slower. #####

function coll_12_34_ker_q(
        q::R,
        p::Params_12_34{T, R}
    ) where {T <: Real, R <: Real}
    expq = exp(q)
    t = p.t_min + expq
    return expq * coll_12_34_ker(t, p)
end

function coll_12_34_ker_r(
        r::R,
        p::Params_12_34{T, R}
    ) where {T <: Real, R <: Real}
    expr = exp(r)
    t = p.t_max - expr
    return expr * coll_12_34_ker(t, p)
end

function coll_12_34_int_t_qr(
        s::R,
        p::Params_12_34{T, R}
    ) where {T <: Real, R <: Real}
    p.s = s
    p.a = a_theta(p)
    p.t_min = t_lim(1, p)
    p.t_max = t_lim(-1, p)
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
