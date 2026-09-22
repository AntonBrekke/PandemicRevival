import Integrals
import SpecialFunctions as SF
import Polylogarithms as PL
import PolyLog as PL2
import QuadGK
import ForwardDiff as FD

include(joinpath(@__DIR__, "constants_functions.jl"))

# Relative tolerance of the density integrals. There is deliberately no
# absolute tolerance: in GeV units the moments are ~1e-15 or smaller.
const rtol_int = 1e-10

# Maxwell-Boltzmann limit: for gap = (m - mu)/T >= m_T_r_MB, quantum
# statistics changes the densities by less than e^-m_T_r_MB ~ 6e-16.
const m_T_r_MB = 35.
# Integration cutoff of `thermal_moment`, relative to the largest value of f.
const max_f_arg = 1e2


QuadGK.kronrod(::Type{<:FD.Dual{T,V,N}}, n::Integer) where {T,V,N} = QuadGK.kronrod(V, n)

# All densities below accept an optional keyword `gap` = (m - mu)/T = m/T - xi.
# If it is not given, it is computed as m/T - xi. Callers that know it exactly
# (the A' gap is e^eta in the pandemolator) should pass it: for large m/T the
# difference m/T - xi loses a small gap to rounding, which for bosons turns
# into a spurious Bose condensation.
#
# Maxwell-Boltzmann particles (k = 0) and gap >= m_T_r_MB use the closed forms
# in terms of exponentially scaled Bessel functions,
#   exp(xi) K_nu(x) = exp(-gap) besselkx(nu, x),
# which do not underflow for any x. Otherwise the moments are integrated
# numerically by `thermal_moment`, which is accurate for all x (including
# x -> 0) and any gap > 0 for bosons.

"""
    thermal_moment(p, temp, gap, weight)

dof/(2 pi^2) ∫_m^∞ dE weight(E, |p|) f(E), with f the occupation number per
dof and gap = (m - mu)/T.

Integrated in t = (E - m)/T, where f = 1/(e^(t + gap) + k) is evaluated
without cancellation, up to t = max_f_arg + max(0, -gap): there f has dropped
by e^-max_f_arg below its largest value (for degenerate fermions, gap < 0,
the Fermi sea up to t = -gap is kept). Breakpoints are placed where the
integrand changes shape: t = x = m/T (momentum ~ mass), t = |gap| (Bose
enhancement, Fermi surface) and t = 1, 10, 50 (thermal tail).
"""
function thermal_moment(p::Particle, temp, gap, weight::F) where F
    x = p.m / temp
    t_max = max_f_arg + max(zero(gap), -gap)
    pts = [zero(t_max), t_max]
    for c in (x, gap, -gap, 1., 10., 50.)
        (c > 0 && c < t_max) && push!(pts, c)
    end
    sort!(pts)
    unique!(pts)
    integrand(t) = begin
        E = p.m + temp * t
        mom = sqrt(temp * t * (temp * t + 2 * p.m))
        temp * weight(E, mom) * occupation(p, t + gap)
    end
    val = QuadGK.quadgk(integrand, pts...; rtol=rtol_int, atol=0)[1]
    return p.dof / (2. * pi^2) * val
end

function number_density(
        p::Particle{T},
        temp::R,
        xi::S;
        gap=nothing,
        debug=false
    ) where {T<:Real, R<:Real, S<:Real}
    x = p.m / temp
    x_xi_diff = isnothing(gap) ? x - xi : gap
    if (p.k == 0) || (x_xi_diff >= m_T_r_MB)
        if x > 1e-10
            return p.dof * exp(-x_xi_diff) * SF.besselkx(2, x) * p.m^3 / (2. * pi^2 * x)
        end
        return p.dof * exp(-x_xi_diff) * temp^3 / pi^2
    else
        if debug
            println("n_xi")
        end
        return n_xi(p, temp, xi; gap=x_xi_diff, debug=debug)
    end
end

function n_xi(
        p::Particle{T},
        temp::R,
        xi::S;
        gap=nothing,
        debug=false
    ) where {T<:Real, R<:Real, S<:Real}
    x = p.m / temp
    x_xi_diff = isnothing(gap) ? x - xi : gap
    if debug
        println("x = ", x)
        println("xi = ", xi)
        println("x_xi_diff = ", x_xi_diff)
    end
    if p.k == -1 && x_xi_diff <= 0
        throw(DomainError(x_xi_diff, "m/T - xi <= 0 for a boson (Bose condensation) in n_xi: xi = $xi, temp = $temp, m = $(p.m)"))
    end
    return thermal_moment(p, temp, x_xi_diff, (E, mom) -> E * mom)
end


function energy_density(p, temp, xi; gap=nothing)
    x = p.m / temp
    x_xi_diff = isnothing(gap) ? x - xi : gap
    if (p.k == 0) || (x_xi_diff >= m_T_r_MB)
        if x > 1e-10
            return p.dof * exp(-x_xi_diff) * p.m^4 * (SF.besselkx(1, x) + 3. / x * SF.besselkx(2, x)) / (2. * pi^2 * x)
        end
        return p.dof * 3 * exp(-x_xi_diff) * (temp^4) / pi^2
    else
        return rho_xi(p, temp, xi; gap=x_xi_diff)
    end
end

function rho_xi(p, temp, xi; gap=nothing)
    x = p.m / temp
    x_xi_diff = isnothing(gap) ? x - xi : gap
    return thermal_moment(p, temp, x_xi_diff, (E, mom) -> E^2 * mom)
end


function rho_3P_diff(p::Particle{T}, temp::R, xi::R; gap=nothing) where {T<:Real, R<:Real}
    x = p.m / temp
    x_xi_diff = isnothing(gap) ? x - xi : gap
    if p.k == 0 || (x_xi_diff >= m_T_r_MB)
        if x > 1e-10
            return p.dof * exp(-x_xi_diff) * SF.besselkx(1, x) * p.m^4 / (2. * pi^2 * x)
        end
        return p.dof * exp(-x_xi_diff) * p.m^4 / (2. * pi^2 * x^2)
    else
        return rho_3P_diff_xi(p, temp, xi; gap=x_xi_diff)
    end
end

function rho_3P_diff_xi(p::Particle{T}, temp::R, xi::R; gap=nothing) where {T<:Real, R<:Real}
    x = p.m / temp
    x_xi_diff = isnothing(gap) ? x - xi : gap
    # rho - 3P = dof/(2 pi^2) ∫ dE |p| f (E^2 - |p|^2) = dof/(2 pi^2) ∫ dE m^2 |p| f
    return thermal_moment(p, temp, x_xi_diff, (E, mom) -> p.m^2 * mom)
end
