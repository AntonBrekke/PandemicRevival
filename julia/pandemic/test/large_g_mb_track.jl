# MB network at large g without the degeneracy stop: does mu run away above 0?
include(joinpath(@__DIR__, "..", "src", "network.jl"))
using Printf
tT_rel = TimeTempRelation{Float64}()
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
mk, g, s2, xe = parse.(Float64, split(ARGS[1], ':'))
m = mk * 1e-6
net = Network(itp, rate_params(m, g, s2), tT_rel)
dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
sol = solve_network(net, dw, tT_rel; x_end=xe, fmax_stop=Inf, maxiters=4000)
println(sol.retcode, " steps ", length(sol.t))
for z in range(sol.t[1], sol.t[end], length=25)
    d = network_diagnostics(net, sol(z), z)
    @printf("x=%.3e TN/Tnu=%.4f TN/m=%.3g  muN1/T=%.3f muA/T=%.3f fmaxN=%.3g Oh2=%.3g\n", d.x, d.T_N / d.T_nu, d.T_N / m,
            log(d.fmax[1]) + 1 / (d.T_N / m), log(d.fmax[3]) + 2.5 / (d.T_N / m), d.fmax[1], d.omega_h2)
end
