"""
The tabulated momentum-exchange rate `C_kd(::CkdTable, ...)` of src/kinetic_decoupling.jl
against nested adaptive quadrature (`C_kd_adaptive` below, ~2 s per call), over the
temperatures of a kinetic-decoupling search (T/m from 1e2 to 1e-10), for small and
large couplings and unequal N1/N2 gaps. ~2 min. Run from julia/pandemic:
    julia --project=. test/test_ckd_table.jl
"""

using Test

include(joinpath(@__DIR__, "..", "src", "kinetic_decoupling.jl"))

"""
C_kd by adaptive quadrature in w = (sqrt(s) - 2m)/T, with breakpoints around the A' pole
on the scale of its width. Resolves the pole only while its width is above the float
resolution of sqrt(s) (g >~ 1e-6).
"""
function C_kd_adaptive(T, gap, m, m_A, g; gap2=gap)
    Gamma_A = width_A(m, m_A, g)
    w12 = exp(gap - gap2)
    sig(s) = sigma_NN(s, m, m_A, g) / 2 + w12 * sigma_NN12(s, m, m_A, g, Gamma_A)
    integrand(w) = (rs = 2m + w * T; s = rs^2; 2 * rs * T * sig(s) * (s - 4m^2) * rs * _besselk1x(rs / T) * exp(-w))
    w_res, dw = (m_A - 2m) / T, Gamma_A / (2T)
    pts = [0.0, 1.0, 5.0, 20.0, 80.0]
    append!(pts, filter(>(0), w_res .+ dw .* (-1e4, -1e2, -10.0, -1.0, 0.0, 1.0, 10.0, 1e2, 1e4)))
    sort!(unique!(pts))
    return T / (32 * pi^4) * exp(-2 * gap) * QuadGK.quadgk(integrand, pts...; rtol=1e-6)[1]
end

@testset "CkdTable vs adaptive, m = $(m * 1e6) keV, g = $g" for (m, g) in ((1.057e-5, 1e-3), (3.0e-6, 1e-5), (1e-5, 0.1))
    mA = 2.5m
    tab = CkdTable(m, mA, g)
    for Tm in (1e2, 3.0, 1.0, 0.3, 0.1, 3e-2, 1e-3, 1e-6, 1e-10), (gap, gap2) in ((1.0, 1.0), (5.0, 9.0))
        T = Tm * m
        @test isapprox(C_kd(tab, T, gap; gap2=gap2), C_kd_adaptive(T, gap, m, mA, g; gap2=gap2); rtol=2e-3)
    end
end

# The narrow-width pole (used automatically below g ~ 1e-6, where the adaptive quadrature
# cannot resolve the A' width either), forced at couplings where it still works.
@testset "narrow-width pole vs adaptive, g = $g" for g in (1e-5, 1e-4)
    m = 3.0e-6
    mA = 2.5m
    tab = CkdTable(m, mA, g; narrow_du=1.0)
    @test tab.res_area > 0
    for Tm in (3.0, 0.3, 0.1, 3e-2, 1e-3), (gap, gap2) in ((1.0, 1.0), (5.0, 9.0))
        T = Tm * m
        @test isapprox(C_kd(tab, T, gap; gap2=gap2), C_kd_adaptive(T, gap, m, mA, g; gap2=gap2); rtol=2e-3)
    end
end

# g = 1e-8, the smallest network coupling: finite, positive, and pole-dominated (∝ g^2)
@testset "g = 1e-8 pole" begin
    m = 1e-5
    t8, t7 = CkdTable(m, 2.5m, 1e-8), CkdTable(m, 2.5m, 1e-7)
    @test t8.res_area > 0
    c8, c7 = C_kd(t8, 0.3m, 1.0), C_kd(t7, 0.3m, 1.0)
    @test isfinite(c8) && c8 > 0
    @test isapprox(c7 / c8, 100; rtol=1e-2)
end
