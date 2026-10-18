using Test
include(joinpath(@__DIR__, "..", "src", "late_decays.jl"))
import LinearAlgebra

"""
Light eigenvalue of the draft's mass matrix (nu_alpha, nu_+, nu_-, nu_0) with m_+ = m_- = m_M, in
BigFloat: Newton on det(M - lambda) from the see-saw value -m_alpha^2/m_0.
"""
function light_mass(m_alpha, m_D, m_M, m_0)
    M = BigFloat[0 0 0 m_alpha; 0 0 m_D m_M; 0 m_D 0 m_M; m_alpha m_M m_M m_0]
    d(l) = LinearAlgebra.det(M - l * LinearAlgebra.I)
    l = -m_alpha^2 / m_0
    for _ in 1:60
        h = abs(l) * big"1e-30"
        l -= d(l) / ((d(l + h) - d(l - h)) / (2h))
    end
    return l
end

@testset "SM decays of N1" begin
    # Boyarsky et al. normalisation of nu gamma, the draft's 3 nu rate, and their ratio 27 alpha/(8 pi)
    @test gamma_nugamma(1e-6, 1e-7) ≈ 1.38e-29 rtol = 0.02
    @test 1 / gamma_3nu(1e-6, 1e-8) ≈ 6e27 rtol = 0.05
    @test gamma_nugamma(1e-5, 1e-12) / gamma_3nu(1e-5, 1e-12) ≈ 27 * ALPHA_EM / (8pi) rtol = 1e-6
end

@testset "phi nu nu Yukawa = d m_nu / d v_phi" begin
    setprecision(BigFloat, 400) do
        for (m_nu, m_0, m_N, s2, g) in ((1e-10, 1e3, 1e-5, 1e-12, 1e-4), (1e-10, 1e5, 3e-5, 1e-15, 1e-2),
                                        (3e-11, 1e2, 1e-6, 1e-9, 1e-6))
            p = rate_params(m_N, g, s2; m_nu=m_nu, m_0=m_0)
            m_alpha = sqrt(big(m_nu) * m_0)
            th1 = sin(asin(sqrt(big(s2))) / 2)
            m_M = th1 * m_0 * m_N / (sqrt(big(2)) * m_alpha)
            v_phi = p.mA / g                     # m_M = y v_phi / sqrt2 scales with v_phi
            e = big"1e-20"
            dm = (light_mass(m_alpha, m_N, m_M * (1 + e), m_0) - light_mass(m_alpha, m_N, m_M * (1 - e), m_0)) / (2e)
            @test Float64(abs(dm) / v_phi) ≈ y_phi_nunu(p) rtol = 1e-4
        end
    end
end

@testset "dark decays" begin
    p = rate_params(1e-5, 1e-4, 1e-12)
    @test width(:N2_Anu, p) == 0.0                  # m_A' > m_N2: N2 has no decay
    @test width(:N1_phinu, p) == 0.0                # m_phi > m_N1
    q = rate_params(1e-5, 1e-4, 1e-12; mphi_over_mN=0.5)
    # L = -c phi N1 i gamma5 nu, c = g theta_1 m_N/m_A' (one contraction): c^2 m (1 - r^2)^2/(16 pi)
    c = q.g * q.th1 * q.m1 / q.mA
    @test gamma_tree(:N1_phinu, q) * hbar ≈ c^2 * q.m1 * (1 - 0.25)^2 / (16pi) rtol = 1e-6
end

@testset "cosmology" begin
    @test T_AGE0 / SEC_PER_GYR ≈ 13.8 rtol = 0.01
    @test T_REC / 3.15576e7 ≈ 3.75e5 rtol = 0.05
    @test time_of_a(a_of_time(1e9)) ≈ 1e9 rtol = 1e-8
    # radiation era: lambda = u0 a_d ln(a_e/a_d) / (H0 sqrt(Omega_r)) for u0 << 1, a_e << a_eq
    u0, a_d, a_e = 1e-3, 1e-9, 1e-6
    lam = u0 * a_d * log(a_e / a_d) / (H0_GEV * sqrt(OMEGA_R)) / Mpc
    @test free_stream_born(u0, a_d; z_end=1 / a_e - 1) ≈ lam rtol = 0.01
    # radiation era: <a> at decay = Gamma(3/2) a(tau)
    tau = 1e6
    @test delta_neff(1.0, tau, 1.0) ≈ OMEGA_C / (OMEGA_GAMMA * NU_PER_GAMMA) * sqrt(pi) / 2 * a_of_time(tau) rtol = 0.01
    @test ddm_bound(1.0, 1e12, 1.0) === :neff
    @test ddm_bound(1.0, 1e17, 1.0) === :cmb
    @test ddm_bound(1.0, 3e18, 1.0) === :late
    @test ddm_bound(1.0, 1e5, 1.0) === :allowed
    @test ddm_bound(0.01, 1e17, 1.0) === :allowed
end
