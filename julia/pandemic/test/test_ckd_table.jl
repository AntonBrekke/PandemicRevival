"""
The tabulated momentum-exchange rate `C_kd(::CkdTable, ...)` of src/kinetic_decoupling.jl
against the adaptive `C_kd`, over the temperatures of a kinetic-decoupling search (T/m from
1e2 to 1e-10), for a small and a large coupling and unequal N1/N2 gaps. Run from
julia/pandemic:
    julia --project=. test/test_ckd_table.jl
"""

using Test

include(joinpath(@__DIR__, "..", "src", "kinetic_decoupling.jl"))

@testset "CkdTable vs C_kd, m = $(m * 1e6) keV, g = $g" for (m, g) in ((1.057e-5, 1e-3), (3.0e-6, 1e-5), (1e-5, 0.1))
    mA = 2.5m
    t = @elapsed tab = CkdTable(m, mA, g)
    println("table: ", length(tab.u), " points, ", round(t; digits=1), " s")
    for Tm in (1e2, 3.0, 1.0, 0.3, 0.1, 3e-2, 1e-3, 1e-6, 1e-10), (gap, gap2) in ((1.0, 1.0), (5.0, 9.0))
        T = Tm * m
        ref = C_kd(T, gap, m, mA, g; gap2=gap2)
        val = C_kd(tab, T, gap; gap2=gap2)
        @test isapprox(val, ref; rtol=2e-3)
        abs(val / ref - 1) > 2e-3 && println("  T/m = $Tm, gaps ($gap, $gap2): ", val / ref)
    end
end

# The narrow-width pole (used automatically below g ~ 1e-6, where the adaptive C_kd cannot
# resolve the A' width either), forced at a coupling where the exact integral still works.
@testset "narrow-width pole vs C_kd, g = $g" for g in (1e-5, 1e-4)
    m = 3.0e-6
    mA = 2.5m
    tab = CkdTable(m, mA, g; narrow_du=1.0)
    @test tab.res_area > 0
    for Tm in (3.0, 0.3, 0.1, 3e-2, 1e-3), (gap, gap2) in ((1.0, 1.0), (5.0, 9.0))
        T = Tm * m
        ref = C_kd(T, gap, m, mA, g; gap2=gap2)
        val = C_kd(tab, T, gap; gap2=gap2)
        @test isapprox(val, ref; rtol=2e-3)
        abs(val / ref - 1) > 2e-3 && println("  T/m = $Tm, gaps ($gap, $gap2): ", val / ref)
    end
end

# g = 1e-8, the smallest network coupling: the table must be finite and positive and scale
# as g^2 (pole-dominated) between 1e-8 and 1e-7 at T ~ m.
@testset "g = 1e-8 pole" begin
    m = 1e-5
    t8, t7 = CkdTable(m, 2.5m, 1e-8), CkdTable(m, 2.5m, 1e-7)
    @test t8.res_area > 0
    c8, c7 = C_kd(t8, 0.3m, 1.0), C_kd(t7, 0.3m, 1.0)
    @test isfinite(c8) && c8 > 0
    @test isapprox(c7 / c8, 100; rtol=1e-2)
end
