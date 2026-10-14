# Large-g step 0 (notes/model_limitations.tex, Sec. "Plan for large g"): the state at which
# the MB network stops (degenerate), and every reaction's contribution there.
#   julia --project=. test/large_g_stall.jl [m_keV:g:s2,...]
include(joinpath(@__DIR__, "..", "src", "network.jl"))
using Printf
import Serialization

tT_rel = TimeTempRelation{Float64}()
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
specs = length(ARGS) > 0 ? ARGS[1] : "10.57:0.005623:8.44755e-17,10.57:0.01:2.6761e-17,10.57:0.03162:2.78324e-18"
out = Dict{String,Any}()
for spec in split(specs, ',')
    mk, g, s2 = parse.(Float64, split(spec, ':'))
    m = mk * 1e-6
    net = Network(itp, rate_params(m, g, s2), tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    t0 = time()
    sol = solve_network(net, dw, tT_rel; x_end=100.0, maxiters=20000)
    u, z = sol.u[end], sol.t[end]
    d = network_diagnostics(net, u, z)
    n, kappa = state_densities(net, u, z)
    s = net.ent(z)
    xfac = exp(z) / net.dx_dt(z)
    H = net.H(z)
    @printf("\n=== %g keV  g = %g  s2 = %.4g: %s at x = %.3g after %d steps (%.1f s), x_start = %.3g\n",
            mk, g, s2, sol.retcode, exp(z), length(sol.t), time() - t0, exp(sol.t[1]))
    @printf("T_N/T_nu = %.3g  T_N/m = %.3g  T_nu/m = %.3g  Omega h2 = %.3g\n", d.T_N / d.T_nu, d.T_N / m, d.T_nu / m, d.omega_h2)
    for k in 1:4
        @printf("  %-4s n/s = %.3e  f_MB,max = %.3g  mu/T_N = %.3f\n", DARK_SPECIES[k], n[k] / s, d.fmax[k],
                log(d.fmax[k]) + net.masses[k] / d.T_N)
    end
    ntil = n[1] + n[2] + 2n[3] + 4n[4]
    bd = Dict{Symbol,Any}()
    C, CK = network_collision(net, d.T_N, d.T_nu, n; breakdown=bd)
    # contribution to d ln n~/dz and d ln kappa/dz per reaction
    @printf("  %-12s %12s %12s   (per unit z: d ln n~ and d ln kappa)\n", "reaction", "dln n~", "dln kappa")
    for r in REACTIONS
        haskey(bd, r.name) || continue
        dN = delta_N(r)
        dnt = dN[1] + dN[2] + 2dN[3] + 4dN[4]
        b = bd[r.name]
        @printf("  %-12s %12.3e %12.3e\n", r.name, xfac * dnt * b.G / ntil, xfac * b.K / kappa)
    end
    @printf("  Hubble: dln kappa/dz = %.3e\n", xfac * H * (kappa - 3sum(n) * d.T_N) / kappa)
    out[spec] = (z=z, u=u, T_N=d.T_N, T_nu=d.T_nu, n=n, kappa=kappa, z0=sol.t[1],
                 traj=[(t, collect(v)) for (t, v) in zip(sol.t, sol.u)])
    flush(stdout)
end
Serialization.serialize(joinpath(@__DIR__, "..", "tmp", "large_g_stall.jls"), out)
