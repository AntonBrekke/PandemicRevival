import Integrals
import SpecialFunctions as SF
import Polylogarithms as PL
import QuadGK
import ForwardDiff as FD

include(joinpath(@__DIR__, "constants_functions.jl"))

const m_T_r_MB = 1e2
const m_T_r_ur = 1e-3
const m_T_r_nr = 6e2
const max_f_arg = 1e2


function number_density(p::Particle{T}, temp::R, xi::R) where {T<:Real, R<:Real}
    x = p.m / temp
    x_xi_diff = x - xi
    if (p.k == 1.) && (x_xi_diff < m_T_r_MB)
        return n_fermion_xi(p, temp, xi)
    elseif (p.k == -1.) && (x_xi_diff < m_T_r_MB)
        return n_boson_xi(p, temp, xi)
    elseif (p.k == 0.) || (x_xi_diff >= m_T_r_MB)
        if (x > 1e-10) && (x < m_T_r_nr)
            return p.dof * exp(xi) * SF.besselk(2, x) * p.m^3 / (2. * pi^2 * x)
        elseif x >= m_T_r_nr
            return p.dof * exp(-x_xi_diff) * (p.m^2 / (2. * pi * x))^1.5 * (1. + 15. / (8. * x))
        end
        return p.dof * exp(xi) * temp^3 / pi^2
    end
    return nothing
end

function n_fermion_xi(p::Particle{T}, temp::R, xi::R) where {T<:Real, R<:Real}
    x = p.m / temp
    x_xi_diff = x - xi
    if x_xi_diff > m_T_r_MB
        return - p.dof * (m^2 / (2. * pi * x))^1.5 * real(PL.polylog(1.5, -exp(-x_xi_diff)))
    end
    if x < m_T_r_ur
        return - p.dof * temp^3 * real(PL.polylog(3, -exp(xi))) / pi^2
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

    if x > 200.
        epsabs = 0.
    else
        epsabs = 1e-6 * 2. * pi^2 * n_fermion(p, temp, xi)
    end

    params = (
        p = p,
        temp = temp,
        xi = xi
    )
    prob = Integrals.IntegralProblem(
        n_fermion_xi_integrand,
        (log_E_min, log_E_max),
        params
    )
    sol = Integrals.solve(
        prob,
        Integrals.QuadGKJL(),
    )
    return p.dof/(2. * pi^2) * sol.u
end

function n_fermion_xi_integrand(log_E, params)
    p = params.p
    temp = params.temp
    xi = params.xi
    E = exp(log_E)
    if (abs(E / temp - xi) > max_f_arg) || (E <= p.m)
        return 0.
    end
    return E^2 * sqrt((E - p.m) * (E + p.m)) / (exp(E / temp - xi) + 1.)
end

QuadGK.kronrod(::Type{<:FD.Dual{T,V,N}}, n::Integer) where {T,V,N} = QuadGK.kronrod(V, n)

function n_boson_xi(p, temp, xi)
    x = p.m / temp
    x_xi_diff = x - xi
    if x_xi_diff > m_T_r_MB
        return p.dof * (p.m^2 / (2. * pi * x))^1.5 * real(PL.polylog(1.5, exp(-x_xi_diff)))
    end
    if x < m_T_r_ur
        return p.dof * temp^3 * real(PL.polylog(3, exp(xi))) / pi^2
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

    if x > 200.
        epsabs = 0.
    else 
        epsabs = 1e-6 * 2. * pi^2 * n_boson(p, temp, xi)
    end
    if xi * temp >= p.m
        return nothing
    else
        params = (
            p = p,
            temp = temp,
            xi = xi
        )
        prob = Integrals.IntegralProblem(
            n_boson_xi_integrand,
            (log_E_min, log_E_max),
            params
        )
        sol = Integrals.solve(
            prob,
            Integrals.QuadGKJL(),
        )
        return p.dof / (2. * pi^2) * sol.u
    end
end

function n_boson_xi_integrand(log_E, params)
    p = params.p
    temp = params.temp
    xi = params.xi
    E = exp(log_E)
    if (abs(E / temp - xi) > max_f_arg) || (E <= p.m)
        return 0.
    end
    return E^2 * sqrt((E - p.m) * (E + p.m)) / (exp(E / temp - xi) - 1.)
end

function energy_density(p, temp, xi)
    x = p.m / temp
    x_xi_diff = x - xi
    if (p.k == 1.) && (x_xi_diff < m_T_r_MB)
        return rho_fermion_xi(p, temp, xi)
    elseif (p.k == -1.) && (x_xi_diff < m_T_r_MB)
        return rho_boson_xi(p, temp, xi)
    elseif (p.k == 0.) || (x_xi_diff >= m_T_r_MB)
        if (x > 1e-10) && (x < m_T_r_nr)
            return p.dof * exp(xi) * p.m^4 * (SF.besselk(1, x) + 3. / x * SF.besselk(2, x)) / (2. * pi^2 * x)
        elseif x >= m_T_r_nr
            return p.m * (1. + 1.5 / x) * number_density(p, temp, xi)
        end
        return p.dof*3 * exp(xi) * (temp^4) / pi^2
    end
    return nothing
end

function rho_fermion_xi(p, temp, xi)
    x = p.m / temp
    x_xi_diff = x - xi
    if x_xi_diff > m_T_r_MB
        return - p.dof * (p.m^2 / (2. * pi * x))^1.5 * (p.m * real(PL.polylog(1.5, -exp(-x_xi_diff))) + 1.5 * temp * real(PL.polylog(2.5, -exp(-x_xi_diff))))
    end
    if x < m_T_r_ur
        return - p.dof * 3. * temp^4 * real(PL.polylog(4, -exp(xi))) / pi^2
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
    if x > 200.
        epsabs = 0.
    else
        epsabs = 1e-6 * 2. * pi^2 * rho_fermion(p, temp, xi)
    end
    params = (
        p = p,
        temp = temp,
        xi = xi
    )
    prob = Integrals.IntegralProblem(
        rho_fermion_xi_integrand,
        (log_E_min, log_E_max),
        params
    )
    sol = Integrals.solve(
        prob,
        Integrals.QuadGKJL(),
    )
    return p.dof / (2. * pi^2) * sol.u
end

function rho_fermion_xi_integrand(log_E, params)
    p = params.p
    temp = params.temp
    xi = params.xi
    E = exp(log_E)
    if (abs(E / temp - xi) > max_f_arg) || (E <= p.m)
        return 0.
    end
    return E^2 * sqrt((E - p.m) * (E + p.m)) * E / (exp(E / temp - xi) + 1.)
end

function rho_boson_xi(p, temp, xi)
    x = p.m / temp
    x_xi_diff = x - xi
    if x_xi_diff > m_T_r_MB
        return p.dof * (p.m^2 / (2. * pi * x))^1.5 * (p.m * real(PL.polylog(1.5, exp(-x_xi_diff))) + 1.5 * temp * real(PL.polylog(2.5, exp(-x_xi_diff))))
    end
    if x < m_T_r_ur
        return p.dof * 3. * temp^4 * real(PL.polylog(4, exp(xi))) / p^2
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
    if x > 200.
        epsabs = 0.
    else
        epsabs = 1e-6 * 2. * pi^2 * rho_boson(p, temp, xi)
    end
    if xi * temp >= p.m
        return None
    else
        params = (
            p = p,
            temp = temp,
            xi = xi
        )
        prob = Integrals.IntegralProblem(
            rho_boson_xi_integrand,
            (log_E_min, log_E_max),
            params
        )
        sol = Integrals.solve(
            prob,
            Integrals.QuadGKJL(),
        )
        return p.dof / (2. * pi^2) * sol.u
    end
end

function rho_boson_xi_integrand(log_E, params)
    p = params.p
    temp = params.temp
    xi = params.xi
    E = exp(log_E)
    if (abs(E / temp - xi) > max_f_arg) || (E <= p.m)
        return 0.
    end
    return E^2 * sqrt((E - p.m) * (E + p.m)) * E / (exp(E / temp - xi) - 1.)
end

function rho_3P_diff(p::Particle{T}, temp::R, xi::R) where {T<:Real, R<:Real}
    x = p.m / temp
    x_xi_diff = x - xi
    if p.k == 1. && (x_xi_diff < m_T_r_MB)
        return rho_3P_diff_fermion_xi(p, temp, xi)
    elseif p.k == -1. && (x_xi_diff < m_T_r_MB)
        return rho_3P_diff_boson_xi(p, temp, xi)
    elseif p.k == 0. || (x_xi_diff >= m_T_r_MB)
        if x > 1e-10 && (x < m_T_r_nr)
            return p.dof * exp(xi) * SF.besselk(1, x) * p.m^4 / (2. * pi^2 * x)
        elseif x >= m_T_r_nr
            return (1 - 1.5 / x) * p.m * number_density(p, temp, xi)
        end
        return p.dof * exp(xi) * p.m^4 / (2. * pi^2 * x^2)
    end
    error("rho_3P_diff does not return a valid value. Double check logic.")
end

function rho_3P_diff_fermion_xi(p::Particle{T}, temp::R, xi::R) where {T<:Real, R<:Real}
    x = p.m / temp
    x_xi_diff = x - xi
    if x_xi_diff > m_T_r_MB
        return p.dof * exp(xi) * SF.besselk(1, x) * p.m^4 / (2. * pi^2 * x)
    end
    if x < m_T_r_ur
        return - p.dof * p.m^4 * real(PL.polylog(2, - exp(xi))) / (2. * pi^2 * x^2)
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
    if x > 200.
        epsabs = 0.
    else 
        epsabs = 1e-6 * 2. * pi^2 * rho_3P_diff_fermion(p, temp, xi)
    end
    params = (
        p = p,
        temp = temp,
        xi = xi
    )
    prob = Integrals.IntegralProblem(
        rho_3P_diff_fermion_xi_integrand,
        (log_E_min, log_E_max),
        params
    )
    sol = Integrals.solve(
        prob,
        Integrals.QuadGKJL(),
    )
    return p.dof / (2. * pi^2) * sol.u
end

function rho_3P_diff_fermion_xi_integrand(log_E, params)
    p = params.p
    temp = params.temp
    xi = params.xi
    E = exp(log_E)
    if (abs(E / temp - xi) > max_f_arg) || (E <= p.m)
        return 0.
    end
    mom = sqrt((E - p.m) * (E + p.m))
    return E * mom * p.m^2 / (exp(E / temp - xi) + 1.)
end

function rho_3P_diff_boson_xi(p::Particle{T}, temp::R, xi::R) where {T<:Real, R<:Real}
    x = p.m / temp
    x_xi_diff = x - xi
    if x_xi_diff > m_T_r_MB
        return p.dof * exp(xi) * SF.besslk(1, x) * p.m^4 / (2. * pi^2 * x)
    end
    if x < m_T_r_ur
        return p.dof * p.m^4 * real(PL.polylog(2, exp(xi))) / (2. * pi^2 * x^2)
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
    if x > 200.
        epsabs = 0.
    else
        epsabs = 1e-6 * 2. * pi^2 * rho_3P_diff_boson(p, temp, xi)
    end
    params = (
        p = p,
        temp = temp,
        xi = xi
    )
    prob = Integrals.IntegralProblem(
        rho_3P_diff_boson_xi_integrand,
        (log_E_min, log_E_max),
        params
    )
    sol = Integrals.solve(
        prob,
        Integrals.QuadGKJL(),
    )
    return p.dof/(2. * pi^2) * sol.u
end

function rho_3P_diff_boson_xi_integrand(
        log_E::T,
        params
    ) where T <: Real
    p = params.p
    temp = params.temp
    xi = params.xi
    E = exp(log_E)
    if (abs(E / temp - xi) > max_f_arg) || (E <= p.m)
        return 0.
    end
    mom = sqrt((E - p.m) * (E + p.m))
    return E * mom * p.m^2 / (exp(E / temp - xi) - 1.)
end

