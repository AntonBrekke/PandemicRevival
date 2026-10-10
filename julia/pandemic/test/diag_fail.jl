# Why does a solve fail (not Terminated) at very large g?
include(joinpath(@__DIR__, "..", "src", "network.jl"))
tT_rel = TimeTempRelation{Float64}()
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
m, g, s2 = parse(Float64, ARGS[1]) * 1e-6, parse(Float64, ARGS[2]), parse(Float64, ARGS[3])
net = Network(itp, rate_params(m, g, s2), tT_rel)
dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
u0, z0 = network_initial_state(net, dw, tT_rel)
println("u0 = ", u0, "  finite: ", all(isfinite, u0))
d = network_diagnostics(net, u0, z0); println("initial fmax = ", d.fmax, "  T_N/T_nu = ", d.T_N / d.T_nu)
du = similar(u0); network_rhs!(du, u0, net, z0); println("du0 = ", du)
try
    sol = solve_network(net, dw, tT_rel; x_end=100.0, maxiters=20_000)
    println(sol.retcode, " at x = ", exp(sol.t[end]), " after ", length(sol.t), " steps")
catch e
    println("exception: ", sprint(showerror, e)[1:min(end, 400)])
end
