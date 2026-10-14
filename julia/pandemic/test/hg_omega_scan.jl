# Omega h^2 of the high-g solver against sin^2 2theta at one (m_N, g): is it monotonic?
#   julia --project=. test/hg_omega_scan.jl m_keV:g s2,s2,...
include(joinpath(@__DIR__, "..", "src", "pandemolator_hg.jl"))
include(joinpath(@__DIR__, "..", "src", "relic_scan_z.jl"))
using Printf
mk, g = parse.(Float64, split(ARGS[1], ':'))
m = mk * 1e-6
tT_rel = TimeTempRelation{Float64}()
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
pan = PandemolatorHG(m, tT_rel, itp)
for s2 in parse.(Float64, split(ARGS[2], ','))
    th = asin(sqrt(s2)) / 2
    pan.mp = ModelParams{Float64}(g, th)
    t = @elapsed sol = pandemolate_z(tT_rel, DodelsonWidrow{Float64}(m, th, tT_rel), pan; reltol=1e-4)
    ok = DE.successful_retcode(sol)
    om = ok ? final_omega_h2_z(pan, sol) : NaN
    mx = ok ? maximum(omega_h2_of(pan, sol.u[i], sol.t[i]) for i in eachindex(sol.t)) : NaN
    @printf("s2 = %.3e: %s  Omega h2 = %.4g (max on the way %.4g), plateau %s  (%.0f s)\n", s2, sol.retcode, om, mx,
            ok && check_plateau_z(pan, sol; tol=5e-3), t)
    flush(stdout)
end
