# network_root of run_network_scan.jl at given points: julia test/scan_subset.jl m_keV:y:s2_reduced,...
include(joinpath(@__DIR__, "..", "src", "run_network_scan.jl"))
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
tT_rel = TimeTempRelation{Float64}()
for spec in split(ARGS[1], ',')
    mk, y, s2r = parse.(Float64, split(spec, ':'))
    t = @elapsed (s2n, omn, st, ns) = network_root(itp, tT_rel, mk * 1e-6, y, s2r)
    @printf("%7.3f keV  y=%.3g  reduced %.4e  network %.4e (ratio %.3f)  Omega %.4f  %s  %d solves  %.1f s\n",
            mk, y, s2r, s2n, s2n / s2r, omn, st, ns, t)
    flush(stdout)
end
