import Integrals
import SpecialFunctions as SF
import Polylogarithms as PL
import PolyLog as PL2
import QuadGK
import ForwardDiff as FD

include(joinpath(@__DIR__, "constants_functions.jl"))

# TODO: [13.08.26] Test these functions for accuracy and speed.
const rtol_int = 1e-6
const atol_int = 1e-6

const m_T_r_MB = 1e2        # Maxwell-Boltzmann limit
const m_T_r_ur = 1e-3       # Ultra-relativistic limit
const m_T_r_nr = 6e2        # Non-relativistic limit
const max_f_arg = 1e2


QuadGK.kronrod(::Type{<:FD.Dual{T,V,N}}, n::Integer) where {T,V,N} = QuadGK.kronrod(V, n)


function number_density(
        p::Particle{T},
        temp::R,
        xi::S;
        debug=false
    ) where {T<:Real, R<:Real, S<:Real}
    x = p.m / temp
    x_xi_diff = x - xi
    if (p.k == 0) || (x_xi_diff >= m_T_r_MB)
        if (x > 1e-10) && (x < m_T_r_nr)
            return p.dof * exp(xi) * SF.besselk(2, x) * p.m^3 / (2. * pi^2 * x)
        elseif x >= m_T_r_nr
            return p.dof * exp(-x_xi_diff) * (p.m^2 / (2. * pi * x))^1.5 * (1. + 15. / (8. * x))
        end
        return p.dof * exp(xi) * temp^3 / pi^2
    else x_xi_diff < m_T_r_MB
        if debug
            println("n_xi")
        end
        return n_xi(p, temp, xi; debug=debug)
    end
end

function n_xi(
        p::Particle{T},
        temp::R,
        xi::S;
        debug=false
    ) where {T<:Real, R<:Real, S<:Real}
    x = p.m / temp
    x_xi_diff = x - xi
    if debug
        println("x = ", x)
        println("xi = ", xi)
        println("x_xi_diff = ", x_xi_diff)
    end
    if x < m_T_r_ur
        return - p.k * p.dof * temp^3 * PL2.reli3(- p.k * exp(xi)) / pi^2
    end

    if p.m == 0.
        log_E_min = log(1e-6 * temp)
    else 
        log_E_min = log(p.m)
    end
    E_max = (max_f_arg + xi) * temp
    if E_max <= p.m
        E_max = p.m + max_f_arg * temp
    end
    log_E_max = log(E_max)

    # if x > 200.
    #     epsabs = 0.
    # else
    #     # if p.k == 1
    #     #     epsabs = 1e-6 * 2. * pi^2 * n_fermion(p, temp, xi)
    #     # elseif p.k == -1
    #     #     epsabs = 1e-6 * 2. * pi^2 * n_boson(p, temp, xi)
    #     # end
    #     epsabs = 1e-15
    # end
    if p.k == -1
        if xi * temp >= p.m
            println("Error: xi * temp >= p.m in n_xi")
            println("xi = ", xi)
            println("temp = ", temp)
            println("p.m = ", p.m)
            throw(XiError)
            # return nothing
        end
    end

    params = (
        p = p,
        temp = temp,
        xi = xi
    )
    prob = Integrals.IntegralProblem(
        n_xi_integrand,
        (log_E_min, log_E_max),
        params,
    )
    sol = Integrals.solve(
        prob,
        Integrals.QuadGKJL(),
        abstol=atol_int,
        reltol=rtol_int
    )
    return 1. / (2. * pi^2) * sol.u
end

function n_xi_integrand(log_E, params)
    p = params.p
    temp = params.temp
    xi = params.xi
    E = exp(log_E)
    if (abs(E / temp - xi) > max_f_arg) || (E <= p.m)
        return 0.
    end
    return E^2 * sqrt((E - p.m) * (E + p.m)) * dist(p, temp, xi, E)
end


function energy_density(p, temp, xi)
    x = p.m / temp
    x_xi_diff = x - xi
    if (p.k == 0) || (x_xi_diff >= m_T_r_MB)
        if (x > 1e-10) && (x < m_T_r_nr)
            return p.dof * exp(xi) * p.m^4 * (SF.besselk(1, x) + 3. / x * SF.besselk(2, x)) / (2. * pi^2 * x)
        elseif x >= m_T_r_nr
            return p.m * (1. + 1.5 / x) * number_density(p, temp, xi)
        end
        return p.dof*3 * exp(xi) * (temp^4) / pi^2
    else
        return rho_xi(p, temp, xi)
    end
    return nothing
end

function rho_xi(p, temp, xi)
    x = p.m / temp
    x_xi_diff = x - xi
    if x < m_T_r_ur
        return - p.k * p.dof * 3. * temp^4 * PL2.reli4(- p.k * exp(xi)) / pi^2
    end
    if p.m == 0.
        log_E_min = log(1e-6 * temp)
    else
        log_E_min = log(p.m)
    end
    E_max = (max_f_arg + xi) * temp
    if E_max <= p.m
        E_max = p.m + max_f_arg * temp
    end
    log_E_max = log(E_max)
    # if x > 200.
    #     epsabs = 0.
    # else
    #     # if p.k == 1
    #     #     epsabs = 1e-6 * 2. * pi^2 * rho_fermion(p, temp, xi)
    #     # elseif p.k == -1
    #     #     epsabs = 1e-6 * 2. * pi^2 * rho_boson(p, temp, xi)
    #     # end
    #     epsabs = 1e-15
    # end
    params = (
        p = p,
        temp = temp,
        xi = xi
    )
    prob = Integrals.IntegralProblem(
        rho_xi_integrand,
        (log_E_min, log_E_max),
        params
    )
    sol = Integrals.solve(
        prob,
        Integrals.QuadGKJL(),
        abstol=atol_int,
        reltol=rtol_int
    )
    return 1. / (2. * pi^2) * sol.u
end

function rho_xi_integrand(log_E, params)
    p = params.p
    temp = params.temp
    xi = params.xi
    E = exp(log_E)
    if (abs(E / temp - xi) > max_f_arg) || (E <= p.m)
        return 0.
    end
    return E^3 * sqrt((E - p.m) * (E + p.m)) * dist(p, temp, xi, E)
end


function rho_3P_diff(p::Particle{T}, temp::R, xi::R) where {T<:Real, R<:Real}
    x = p.m / temp
    x_xi_diff = x - xi
    if p.k == 0 || (x_xi_diff >= m_T_r_MB)
        if x > 1e-10 && (x < m_T_r_nr)
            return p.dof * exp(xi) * SF.besselk(1, x) * p.m^4 / (2. * pi^2 * x)
        elseif x >= m_T_r_nr
            return (1 - 1.5 / x) * p.m * number_density(p, temp, xi)
        end
        return p.dof * exp(xi) * p.m^4 / (2. * pi^2 * x^2)
    else
        return rho_3P_diff_xi(p, temp, xi)
    end
    return nothing
end

function rho_3P_diff_xi(p::Particle{T}, temp::R, xi::R) where {T<:Real, R<:Real}
    x = p.m / temp
    x_xi_diff = x - xi
    if x < m_T_r_ur
        # TODO: [13.08.26] Should this just cancel out? See Brekke for details.
        return - p.k * p.dof * p.m^4 * PL2.reli2(- p.k * exp(xi)) / (2. * pi^2 * x^2)
    end
    if p.m == 0.
        log_E_min = log(1e-6 * temp)
    else 
        log_E_min = log(p.m)
    end
    E_max = (max_f_arg + xi) * temp
    if E_max <= p.m
        E_max = p.m + max_f_arg * temp
    end
    log_E_max = log(E_max)
    # if x > 200.
    #     epsabs = 0.
    # else
    #     # if p.k == 1
    #     #     epsabs = 1e-6 * 2. * pi^2 * rho_3P_diff_fermion(p, temp, xi)
    #     # elseif p.k == -1
    #     #     epsabs = 1e-6 * 2. * pi^2 * rho_3P_diff_boson(p, temp, xi)
    #     # end
    #     epsabs = 1e-15
    # end
    params = (
        p = p,
        temp = temp,
        xi = xi
    )
    prob = Integrals.IntegralProblem(
        rho_3P_diff_xi_integrand,
        (log_E_min, log_E_max),
        params
    )
    sol = Integrals.solve(
        prob,
        Integrals.QuadGKJL(),
        abstol=atol_int,
        reltol=rtol_int
    )
    return 1. / (2. * pi^2) * sol.u
end

function rho_3P_diff_xi_integrand(log_E, params)
    p = params.p
    temp = params.temp
    xi = params.xi
    E = exp(log_E)
    if (abs(E / temp - xi) > max_f_arg) || (E <= p.m)
        return 0.
    end
    mom = sqrt((E - p.m) * (E + p.m))
    return E * mom * p.m^2 * dist(p, temp, xi, E)
end
