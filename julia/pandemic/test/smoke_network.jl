# Smoke test of network.jl on the coarse tables (tmp/rate_tables/smoke_tables.jls):
# one RHS evaluation (finite? how fast?), then a short integration.
include(joinpath(@__DIR__, "..", "src", "network.jl"))
path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "tmp", "rate_tables", "smoke_tables.jls")
x_end = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 0.01

t0 = time()
tT_rel = TimeTempRelation{Float64}()
println("background: ", round(time() - t0; digits=1), " s"); flush(stdout)

itp = RateInterpolants(load_rate_tables(path))
m, g, s2 = 1e-5, 1e-3, 3.554e-15            # a relic root of the reduced solver (10 keV, g = 1e-3)
p = rate_params(m, g, s2)
net = Network(itp, p, tT_rel)
theta = asin(sqrt(s2)) / 2
dw = DodelsonWidrow{Float64}(m, theta, tT_rel)
u0, z0 = network_initial_state(net, dw, tT_rel)
println("z0 = ", z0, "  x0 = ", exp(z0), "  u0 = ", u0); flush(stdout)
n, kappa = state_densities(net, u0, z0)
T_N = dark_temperature(net, n, kappa)
println("T_N/T_nu = ", T_N / (m * exp(-z0)), "   check kappa: ",
        sum(n[k] * kinetic_per_particle(net.masses[k], T_N) for k in 1:4) / kappa - 1); flush(stdout)
# QSS check: C_k at the root of its quadratic, through the full assembly
urs = unit_rates(net, T_N, m * exp(-z0), n[1], n[2])
for k in (3, 4)
    a0, a1, a2 = qss_coefficients(urs, k, n[3], n[4])
    nq, rate = qss_root(a0, a1, a2)
    nn = collect(n); nn[k] = nq
    Ck = assemble_collision(urs, nn[3], nn[4])[1][k]
    println("QSS k=", k, ": nq/n_k=", nq / n[k], "  lambda=", rate * exp(z0) / net.dx_dt(z0),
            "  C_k(nq)/a0=", Ck / a0, "  C_k(n) check: ", assemble_collision(urs, n[3], n[4])[1][k] /
            (a0 + a1 * n[k] + a2 * n[k]^2) - 1)
end
flush(stdout)
du = similar(u0)
network_rhs!(du, u0, net, z0)
println("du0 = ", du); flush(stdout)
t = @elapsed for _ in 1:100
    network_rhs!(du, u0, net, z0)
end
println("RHS: ", round(1e3 * t / 100; digits=3), " ms"); flush(stdout)

t = @elapsed sol = solve_network(net, dw, tT_rel; x_end=x_end, progress=2000)
println("solve to x = ", x_end, ": ", sol.retcode, ", ", length(sol.t), " steps, ", round(t; digits=1), " s"); flush(stdout)
for i in unique(round.(Int, range(1, length(sol.t), length=8)))
    d = network_diagnostics(net, sol.u[i], sol.t[i])
    println("  x=", round(d.x; sigdigits=3), "  T_N/T_nu=", round(d.T_N / d.T_nu; sigdigits=4),
            "  Y=", round.(collect(state_densities(net, sol.u[i], sol.t[i])[1]) ./ net.ent(sol.t[i]); sigdigits=3), "  fmax=", round.(d.fmax; sigdigits=2),
            "  Omega h2=", round(d.omega_h2; sigdigits=4))
end
flush(stdout)

# per-reaction breakdown at a few points of the solution
for i in unique(round.(Int, range(1, length(sol.t), length=4)))
    u, z = sol.u[i], sol.t[i]
    local n, kappa = state_densities(net, u, z)
    local T_N = dark_temperature(net, n, kappa)
    bd = Dict{Symbol,Any}()
    xfac = exp(z) / net.dx_dt(z)
    C, CK = network_collision(net, T_N, net.p.m1 * exp(-z), n; breakdown=bd)
    println("x=", round(exp(z); sigdigits=3), "  n=", round.(n; sigdigits=3), "  kappa=", round(kappa; sigdigits=3),
            "  T_N/T_nu=", round(T_N / (net.p.m1 * exp(-z)); sigdigits=3))
    for (k, v) in sort(collect(bd); by=kv -> -abs(kv[2].G))
        println("   ", rpad(k, 12), " G*xfac/n_tot=", round(v.G * xfac / sum(n); sigdigits=3),
                "   K*xfac/kappa=", round(v.K * xfac / kappa; sigdigits=3))
    end
end
flush(stdout)
