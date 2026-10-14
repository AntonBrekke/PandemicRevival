# Smoke test of pandemolator_hg.jl: one point, reduced vs HG (off) vs HG (full); times each solve.
include(joinpath(@__DIR__, "..", "src", "pandemolator_hg.jl"))
include(joinpath(@__DIR__, "..", "src", "relic_scan_z.jl"))
using Printf
mk, g, s2 = length(ARGS) > 0 ? parse.(Float64, split(ARGS[1], ':')) : (10.57, 1e-3, 3.68159e-15)
m = mk * 1e-6
th = asin(sqrt(s2)) / 2
tT_rel = TimeTempRelation{Float64}()
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
dw = DodelsonWidrow{Float64}(m, th, tT_rel)
N1 = Particle{Float64}(m, 1, dof=2); N2 = Particle{Float64}(m, 1, dof=2)
A = Particle{Float64}(2.5m, -1, dof=3); nu = Particle{Float64}(0.0, 1, dof=2)
panz = PandemolatorZ{Float64}(ModelParams{Float64}(g, th), N1, N2, A, nu, tT_rel)
t = @elapsed solz = pandemolate_z(tT_rel, dw, panz; reltol=1e-4)
@printf("reduced:     %s  Omega h2 = %.5g  (%.1f s, %d steps)\n", solz.retcode, final_omega_h2_z(panz, solz), t, length(solz.t)); flush(stdout)
for (wp, wt) in ((false, false), (true, false), (true, true))
    pan = PandemolatorHG(m, tT_rel, itp; with_phi=wp, with_tables=wt)
    pan.mp = ModelParams{Float64}(g, th)
    t = @elapsed sol = pandemolate_z(tT_rel, dw, pan; reltol=1e-4)
    global pan_last, sol_last = pan, sol
    @printf("HG phi=%-5s tables=%-5s: %s  Omega h2 = %.5g  (%.1f s, %d steps)\n", wp, wt, sol.retcode,
            final_omega_h2_z(pan, sol), t, length(sol.t)); flush(stdout)
end
println("plateau (last HG solve): ", check_plateau_z(pan_last, sol_last))
zs = sol_last.t
for i in max(1, length(zs) - 12):length(zs)
    @printf("  x=%.4g Omega=%.6g\n", exp(zs[i]), omega_h2_of(pan_last, sol_last.u[i], zs[i]))
end
