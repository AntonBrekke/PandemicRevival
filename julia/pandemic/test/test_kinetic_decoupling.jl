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

@testset "old amplitude port vs vector_mediator.M2_gen" begin
    # (s, t, M2_gen(s, t, m, m, m, m, g^4, m_A^2, 0))
    ref = [(4.0001000000000005e-10, -8.499999999985029e-15, 0.00016385376345521625),
           (4.500000000000001e-10, -2.2500000000000016e-11, 0.00024862151512994676),
           (1.0000000000000003e-09, -3.0000000000000006e-11, 0.0031911725591719695),
           (4.000000000000001e-09, -2.3400000000000006e-09, 0.009711733040082033)]
    for (s, t, py) in ref
        @test isapprox(sq_amp_NN_old(s, t, M, MA, G), py; rtol=1e-6)
    end
end

@testset "threshold cross section vs draft sigma_NR" begin
    s = 4.00000001 * M^2
    sig_nr = G^4 * M^2 / (8pi * MA^4)
    # spin average 1/4, identical particles 1/2
    @test isapprox(sigma_NN(s, M, MA, G; amp=:old) / 8, sig_nr; rtol=1e-6)
    @test isapprox(sigma_NN(s, M, MA, G; amp=:A6) / 8, 0.75 * sig_nr; rtol=1e-6)
end

@testset "C_kd vs old C_therm_kd (non-relativistic branch, with its factor 1e3)" begin
    # (T, gap, 2 C_dd_dd_gon_gel / 4)
    ref = [(1.0000000000000002e-06, 15.0, 2.439227453379171e-42),
           (1.0000000000000001e-07, 20.0, 1.6922053549806285e-50),
           (3.3333333333333334e-09, 20.0, 1.0673413519199331e-55)]
    for (T, gap, py) in ref
        @test isapprox(C_kd(T, gap, M, MA, G; amp=:old, sigma_scale=1e3), py; rtol=1e-6)
    end
    # asymptotic e^x K_1(x) beyond the Amos limit
    @test isapprox(_besselk1x(1e4 - 1), SF.besselkx(1, 1e4 - 1); rtol=1e-12)
    @test isfinite(C_kd(1e-16, 20.0, M, MA, G))
end
