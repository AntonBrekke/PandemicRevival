# Which reactions make n~ in the high-g solver: e-folds of n~ per reaction, int (x/xdot) C_r/n~ dz,
# along one solution; and the state (T_N/T_nu, mu/T_N, phi share) on the way.
#   julia --project=. test/hg_breakdown.jl m_keV:g:s2 [with_phi with_tables]
include(joinpath(@__DIR__, "..", "src", "pandemolator_hg.jl"))
include(joinpath(@__DIR__, "..", "src", "relic_scan_z.jl"))
using Printf
mk, g, s2 = parse.(Float64, split(ARGS[1], ':'))
wp = length(ARGS) > 1 ? parse(Bool, ARGS[2]) : true
wt = length(ARGS) > 2 ? parse(Bool, ARGS[3]) : true
m = mk * 1e-6; th = asin(sqrt(s2)) / 2
tT_rel = TimeTempRelation{Float64}()
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
dw = DodelsonWidrow{Float64}(m, th, tT_rel)
pan = PandemolatorHG(m, tT_rel, itp; with_phi=wp, with_tables=wt)
pan.mp = ModelParams{Float64}(g, th)
sol = pandemolate_z(tT_rel, dw, pan; reltol=1e-4)
@printf("%s, Omega h2 = %.4g\n", sol.retcode, final_omega_h2_z(pan, sol))
zs = range(sol.t[1], sol.t[end], length=600)
efold = Dict{Symbol,Float64}(); heat = Dict{Symbol,Float64}()
prev = nothing
for (i, z) in enumerate(zs)
    u = sol(z)
    n, T_N, xi, gA = hg_state(pan, u)
    x = exp(z); fac = x / pan.dx_dt_interp_z(z)
    nt = exp(u[1]) * pan.ent_interp_z(z); rho = exp(u[2]) * pan.ent_interp_z(z)^(4 / 3)
    bd = Dict{Symbol,Any}()
    collision_terms(pan, m / x, T_N, xi; gap_A=gA, breakdown=bd)
    dz = i == 1 ? 0.0 : step(zs)
    for (k, v) in bd
        efold[k] = get(efold, k, 0.0) + fac * v[1] / nt * dz
        heat[k] = get(heat, k, 0.0) + fac * v[2] / rho * dz
    end
    if i % 50 == 1
        @printf("x=%.3e TN/Tnu=%.4f TN/m=%.3g mu/T=%.3f  n_phi*4/n~=%.3g n_A*2/n~=%.3g Oh2=%.4g\n", x, T_N * x / m, T_N / m,
                xi, 4n[4] / nt, 2n[3] / nt, omega_h2_of(pan, u, z))
        println("    cumulative rho e-folds: ", join([@sprintf("%s=%.2f", k, heat[k]) for k in sort(collect(keys(heat))) if abs(heat[k]) > 1e-3], " "))
    end
end
println("e-folds of n~ and of rho (energy from nu) per reaction:")
for k in sort(collect(keys(efold)); by=k -> -abs(efold[k]))
    @printf("  %-12s %10.4f %10.4f\n", k, efold[k], heat[k])
end
