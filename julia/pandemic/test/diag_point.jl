# Where and why does the solve stall at a given point? progress, then Jacobian/noise at the stall
include(joinpath(@__DIR__, "..", "src", "network.jl"))
import LinearAlgebra as LA
tT_rel = TimeTempRelation{Float64}()
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
m, g, s2 = parse(Float64, ARGS[1]) * 1e-6, parse(Float64, ARGS[2]), parse(Float64, ARGS[3])
net = Network(itp, rate_params(m, g, s2), tT_rel)
dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
sol = solve_network(net, dw, tT_rel; x_end=100.0, maxiters=6000, progress=500)
println(sol.retcode, " at x = ", exp(sol.t[end])); flush(stdout)
u = sol.u[end]; z = sol.t[end]
d = network_diagnostics(net, u, z)
println("state: T_N/T_nu=", d.T_N / d.T_nu, "  n/s=", collect(state_densities(net, u, z)[1]) ./ net.ent(z), "  fmax=", d.fmax)
f(v) = (r = similar(v); network_rhs!(r, v, net, z); r)
du = f(u); println("du = ", du)
for j in 1:5
    for e in (1e-10, 1e-9, 1e-8, 1e-7)
        up = copy(u); up[j] += e
        print("  d du/du", j, " (e=", e, "): ", round.((f(up) - du) ./ e; sigdigits=4), "\n")
    end
end
flush(stdout)
