"""
Regression against the stored scans: re-solves relic roots of the network
(tmp/network_scan) and of the fluid solver (tmp/fluid_scan) and compares Omega h^2 and
the Lyman-alpha lengths with the stored values. ~10 min (the fluid solves dominate).
Run from julia/pandemic:
    julia --project=. test/test_regression.jl
"""

using Test

include(joinpath(@__DIR__, "..", "src", "lyman_alpha.jl"))

"""Rows of a CSV as Dicts keyed by (m_N [keV], y), rounded."""
function csv_rows(path)
    lines = readlines(path)
    hdr = split(lines[1], ',')
    rows = Dict{Tuple{Float64,Float64},Dict{String,String}}()
    for l in lines[2:end]
        d = Dict(zip(hdr, split(l, ',')))
        rows[(round(parse(Float64, d["m_N"]) * 1e6; sigdigits=4), round(parse(Float64, d["y"]); sigdigits=4))] = d
    end
    return rows
end
num(d, k) = parse(Float64, d[k])
"""isapprox, with NaN (an undefined length, e.g. x_kd of a never-coupled sector) equal to NaN."""
same(a, b; kw...) = (isnan(a) && isnan(b)) || isapprox(a, b; kw...)

const TMP = joinpath(@__DIR__, "..", "tmp")
const TT = TimeTempRelation{Float64}()
const ITP = RateInterpolants(load_rate_tables(rate_table_path(joinpath(TMP, "rate_tables"))))

@testset "network roots" begin
    roots = csv_rows(joinpath(TMP, "network_scan", "network_roots.csv"))
    lya = csv_rows(joinpath(TMP, "network_scan", "lyman_alpha.csv"))
    for key in ((3.724, 1e-8), (10.57, 1e-6), (10.57, 1e-4))
        r, l = roots[key], lya[key]
        res = lyman_alpha_network(ITP, TT, key[1] * 1e-6, key[2], num(r, "s2_network"))
        @test res.omega_h2 ≈ num(r, "omega_network") rtol = 2e-3
        @test res.f_N2 ≈ num(l, "f_N2") rtol = 1e-2
        @test res.lambda_fs ≈ num(l, "lambda_fs_Mpc") rtol = 2e-3
        @test res.lambda_fs_kd ≈ num(l, "lambda_fs_kd_Mpc") rtol = 2e-3
        @test res.r_s ≈ num(l, "r_s_Mpc") rtol = 2e-3 atol = 1e-6
        @test same(res.x_kd, num(l, "x_kd"); rtol=1e-2)
    end
end

@testset "fluid roots" begin
    roots = csv_rows(joinpath(TMP, "fluid_scan", "fluid_roots.csv"))
    lya = csv_rows(joinpath(TMP, "fluid_scan", "lyman_alpha.csv"))
    for key in ((30.0, 3.162e-3), (10.57, 0.01))
        r, l = roots[key], lya[key]
        res = lyman_alpha_fluid(ITP, TT, key[1] * 1e-6, key[2], num(r, "s2_fluid"))
        # the fluid solve at reltol 1e-4 reproduces Omega h^2 to ~1e-3 at these roots
        @test res.omega_h2 ≈ num(l, "omega_h2") rtol = 5e-3
        @test res.lambda_fs ≈ num(l, "lambda_fs_Mpc") rtol = 5e-3
        @test res.lambda_fs_kd ≈ num(l, "lambda_fs_kd_Mpc") rtol = 1e-2
        @test res.r_s ≈ num(l, "r_s_Mpc") rtol = 1e-2
        @test same(res.x_kd, num(l, "x_kd"); rtol=2e-2)
    end
end
