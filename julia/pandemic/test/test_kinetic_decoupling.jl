"""
Tests of src/kinetic_decoupling.jl against the old Python code. Run from julia/pandemic:
    julia --project=. test/test_kinetic_decoupling.jl

The reference values were computed with code/vector_mediator.py (`M2_gen`)
and code/C_res_vector_no_spin_stat.py (`C_dd_dd_gon_gel`, times 2/4 as in
`C_therm_kd` of code/sterile_caller.py) for g = 0.1, m = 10 keV, m_A = 2.5 m.
"""

using Test

include(joinpath(@__DIR__, "..", "src", "kinetic_decoupling.jl"))

const G, M = 0.1, 1e-5
const MA = 2.5 * M

@testset "N1 N1 -> N2 N2 amplitude vs vector_mediator.M2_gen" begin
    # (s, t, M2_gen(s, t, m, m, m, m, g^4, m_A^2, 0))
    ref = [(4.0001000000000005e-10, -8.499999999985029e-15, 0.00016385376345521625),
           (4.500000000000001e-10, -2.2500000000000016e-11, 0.00024862151512994676),
           (1.0000000000000003e-09, -3.0000000000000006e-11, 0.0031911725591719695),
           (4.000000000000001e-09, -2.3400000000000006e-09, 0.009711733040082033)]
    for (s, t, py) in ref
        @test isapprox(sq_amp_NN(s, t, M, MA, G), py; rtol=1e-6)
    end
end

@testset "threshold cross section vs draft sigma_NR" begin
    s = 4.00000001 * M^2
    # spin average 1/4, identical particles 1/2
    @test isapprox(sigma_NN(s, M, MA, G) / 8, G^4 * M^2 / (8pi * MA^4); rtol=1e-6)
end

@testset "C_kd vs old C_therm_kd (N1 N1 -> N2 N2 alone, old factor 1e3)" begin
    # (T, gap, 2 C_dd_dd_gon_gel / 4); the old sigma_gen_new returned 1e3 times the cross section
    ref = [(1.0000000000000002e-06, 15.0, 2.439227453379171e-42),
           (1.0000000000000001e-07, 20.0, 1.6922053549806285e-50),
           (3.3333333333333334e-09, 20.0, 1.0673413519199331e-55)]
    tab = CkdTable(M, MA, G)
    for (T, gap, py) in ref
        @test isapprox(1e3 * C_kd(tab, T, gap; gap2=Inf), py; rtol=2e-3)
    end
    # asymptotic e^x K_1(x) beyond the Amos limit
    @test isapprox(_besselk1x(1e4 - 1), SF.besselkx(1, 1e4 - 1); rtol=1e-12)
    @test isfinite(C_kd(tab, 1e-16, 20.0)) && C_kd(tab, 1e-16, 20.0) > 0
end

@testset "C_kd with N1 N2 -> N1 N2" begin
    # far below the A' pole (T << m_A - 2m) the extra channel is off-shell scattering.
    # sigma_NN has no identical-particle 1/2, so at threshold sigma_12/sigma_NN = 64.963/2
    # (notes/tree_amplitudes.tex, m_A = 2.5 m), and C_kd/C_kd(11 -> 22 alone) ->
    # 1 + 2 * 32.48 (up to O(T/m) velocity dependence: 2e-3 at T = 1e-3 m)
    tab = CkdTable(M, MA, G)
    T, gap = 1e-3 * M, 15.0
    s = 4M^2 * (1 + 1e-10)
    ratio12 = sigma_NN12(s, M, MA, G, width_A(M, MA, G)) / sigma_NN(s, M, MA, G)
    @test isapprox(ratio12, 64.963 / 2; rtol=1e-3)
    @test isapprox(C_kd(tab, T, gap) / C_kd(tab, T, gap; gap2=Inf), 1 + 2 * ratio12; rtol=5e-3)
    # near the pole the on-shell part (inverse decay and decay) dominates
    @test C_kd(tab, 0.3M, gap) / C_kd(tab, 0.3M, gap; gap2=Inf) > 1 + 2 * ratio12
end
