# src/thermal_rates.jl: Gondolo-Gelmini, detailed balance, resonance treatment.
using Test
import QuadGK
import SpecialFunctions as SF
include(joinpath(@__DIR__, "..", "src", "thermal_rates.jl"))

p = rate_params(1e-5, 1e-2, 1e-12)            # 10 keV, g = 0.01, sin^2 2theta = 1e-12
m, mA = p.m1, p.mA
q1 = only(filter(q -> q.name === :N1N1_N2N2, TREE_PROCESSES))
q3 = only(filter(q -> q.name === :N1N1_AA, TREE_PROCESSES))
q10 = only(filter(q -> q.name === :N2nu_N1N2, TREE_PROCESSES))

@testset "Gondolo-Gelmini at a common temperature" begin
    for T in (m, 0.1m, 0.02m)
        N = Species(m, 2, T, m / T)            # mu = 0
        tab = SigmaTable(q1, p; T_max=T)
        R = rate_2to2(tab, N, N)[3]
        # n^2 <sigma v>, <sigma v> = ∫ ds sigmabar (s - 4m^2) sqrt(s) K1(sqrt(s)/T) / (8 m^4 T K2^2)
        sigbar(s) = sigma_parts(q1, s, p)[3] / 4
        val = QuadGK.quadgk(w -> (rs = 2m + w * T; s = rs^2;
                    2rs * T * sigbar(s) * (s - 4m^2) * rs * SF.besselkx(1, rs / T) * exp(-w)), 0, 1, 5, 20, 80; rtol=1e-8)[1]
        # e^(-sqrt(s)/T) = e^(-2m/T) e^(-w); n^2 also carries e^(-2m/T): compare scaled
        n_s = n_scaled(N)
        sv_scaled = val / (8m^4 * T * SF.besselkx(2, m / T)^2)       # <sigma v> (e^(-2m/T) cancels)
        # 1e-3: linear interpolation of the tabulated sigma near threshold
        @test isapprox(R * exp(2m / T) / n_s^2, sv_scaled; rtol=1e-3)
    end
end

@testset "detailed balance" begin
    T = 0.3m
    xi_N = 0.4
    N1 = Species(m, 2, T, m / T - xi_N); N2 = Species(m, 2, T, m / T - xi_N)
    A = Species(mA, 3, T, mA / T - 2xi_N)     # mu_A = 2 mu_N
    nu = Species(0.0, 2, T, 0.0)
    # A' <-> N1 N2: n_A <Gamma> = R(N1 N2 -> A')
    R_inv = rate_inverse_decay(:A_N1N2, p, N1, N2)[3]
    @test isapprox(n_MB(A) * rate_decay(:A_N1N2, p, T), R_inv; rtol=1e-6)
    # A' <-> N2 nu with mu_A = mu_N2 (nu has mu = 0)
    A2 = Species(mA, 3, T, mA / T - xi_N)
    R_inv2 = rate_inverse_decay(:A_N2nu, p, N2, nu)[3]
    @test isapprox(n_MB(A2) * rate_decay(:A_N2nu, p, T), R_inv2; rtol=1e-6)
    # N1 N1 <-> A'A': rate densities equal in chemical equilibrium for this
    # process, mu_A = mu_N (with the solver's mu_A = 2 mu_N they differ by
    # e^(2 xi_A - 2 xi_N): the process drives mu to zero)
    A1 = Species(mA, 3, T, mA / T - xi_N)
    tab = SigmaTable(q3, p; T_max=T); tabr = SigmaTable(reversed_process(q3), p; T_max=T)
    @test isapprox(rate_2to2(tab, N1, N1)[3], rate_2to2(tabr, A1, A1)[3]; rtol=1e-3)
    @test isapprox(rate_2to2(tabr, A, A)[3] / rate_2to2(tab, N1, N1)[3], exp(2 * 2xi_N - 2xi_N); rtol=1e-3)
end

@testset "resonance: Breit-Wigner minus RIS = on-shell part" begin
    T = 0.4m
    N2 = Species(m, 2, T, m / T); nu = Species(0.0, 2, T, 0.0)
    tab = SigmaTable(q10, p; T_max=T)
    bw = rate_2to2(tab, N2, nu; propagator=:bw)[3]
    ris = rate_2to2(tab, N2, nu; propagator=:ris)[3]
    onshell = rate_inverse_decay(:A_N2nu, p, N2, nu)[3] * width(:A_N1N2, p) / total_width(:A, p)
    @test isapprox(bw - ris, onshell; rtol=2e-3)
end
