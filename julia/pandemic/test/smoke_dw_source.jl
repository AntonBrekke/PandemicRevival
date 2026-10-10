# DW production spread in temperature vs all at T_dw: F(T), and Omega at a relic point
include(joinpath(@__DIR__, "..", "src", "network.jl"))
tT_rel = TimeTempRelation{Float64}()
path = get(ARGS, 1, joinpath(@__DIR__, "..", "tmp", "rate_tables", "smoke_tables.jls"))
itp = RateInterpolants(load_rate_tables(path))
m, g, s2 = 10.57e-6, 1e-3, 3.6816e-15
dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
z_ic = log(m / tT_rel.T_nu_grid[dw.i_ic])
Y_final = dw.n_ic / Network(itp, rate_params(m, g, s2), tT_rel).ent(z_ic)
h = DWHistory(m, Y_final)
println("T_dw = ", T_d_dw(m), " GeV;  p/T_nu at T_dw = ", h.p0 * (Network(itp, rate_params(m, g, s2), tT_rel).ent(z_ic) / s0)^(1/3) / tT_rel.T_nu_grid[dw.i_ic], "  (reduced IC: ", dw.rho_ic / dw.n_ic / tT_rel.T_nu_grid[dw.i_ic], ")")
for T in (1.0, 0.5, T_d_dw(m), 0.2, 0.15, 0.1, 0.05, 0.02, 0.01)
    println("  T = ", round(T; sigdigits=3), " GeV:  F = ", round.(dw_fraction(h, T); sigdigits=4))
end
for (label, dwh) in (("all at T_dw", nothing), ("spread (Fig. 3)", h))
    net = Network(itp, rate_params(m, g, s2), tT_rel; dw=dwh)
    t = @elapsed sol = solve_network(net, dw, tT_rel; x_end=100.0)
    println(rpad(label, 18), sol.retcode, "  ", length(sol.t), " steps  ", round(t; digits=1), " s   Omega h2 = ",
            network_omega_h2(net, sol.u[end], sol.t[end]), "   x0 = ", exp(sol.t[1]))
end
