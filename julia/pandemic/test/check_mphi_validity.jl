"""
Validity check of both solvers for a rate-table set with another m_phi, before scanning it: single
solves at the m_phi = 100 m_N roots (10.57 and 30 keV).

    julia --project=. test/check_mphi_validity.jl tmp/rate_tables/rate_tables_mA2.5_mphi10.0.jls

Network (g = 1e-7 ... 1e-2): retcode (Terminated = an MB occupation number exceeded 0.5),
Omega h^2, the largest fraction of the N number held in phi (w_phi n_phi / n~) along the solution
and at the end, the largest occupation number. Fluid (g = 1e-3 ... 0.1, only for m_phi > 2 m_A'):
Omega h^2 and whether it settles. At g = 1e-3 and 3.16e-3 both: the ratio of their Omega h^2 at
the same mixing (Omega h^2 is so steep in theta that 20% there is ~1% in the root).
"""

include(joinpath(@__DIR__, "..", "src", "lyman_alpha.jl"))
using Printf

const TMP = joinpath(@__DIR__, "..", "tmp")
itp = RateInterpolants(load_rate_tables(ARGS[1]))
rA, rphi = mass_ratios(itp)
tT = TimeTempRelation{Float64}()

function stored(file, col, mk, g)
    hdr = split(readline(file), ',')
    for l in readlines(file)[2:end]
        c = Dict(zip(hdr, split(l, ',')))
        isapprox(parse(Float64, c["m_N"]) * 1e6, mk; rtol=1e-3) && isapprox(parse(Float64, c["y"]), g; rtol=1e-3) &&
            c["status"] == "converged" && return parse(Float64, c[col])
    end
    return NaN
end
seed(mk, g) = (s = stored(joinpath(TMP, "network_scan", "network_roots.csv"), "s2_network", mk, g);
               isfinite(s) ? s : stored(joinpath(TMP, "fluid_scan", "fluid_roots.csv"), "s2_fluid", mk, g))

function network_check(mk, g, s2)
    m = mk * 1e-6
    net = Network(itp, table_params(itp, m, g, s2), tT)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT)
    t = @elapsed sol = solve_network(net, dw, tT; x_end=100.0, maxiters=20_000)
    fphi(u, z) = (n = state_densities(net, u, z)[1]; net.w_phi * n[4] / (n[1] + n[2] + 2n[3] + net.w_phi * n[4]))
    fmax = maximum(maximum(network_diagnostics(net, u, z).fmax) for (u, z) in zip(sol.u, sol.t))
    return (rc=sol.retcode, om=network_omega_h2(net, sol.u[end], sol.t[end]), x_end=exp(sol.t[end]),
            phimax=maximum(fphi(u, z) for (u, z) in zip(sol.u, sol.t)), phiend=fphi(sol.u[end], sol.t[end]),
            fmax=fmax, w=net.w_phi, t=t)
end

println("tables: m_A'/m_N = $rA, m_phi/m_N = $rphi")
p0 = table_params(itp, 1e-5, 1e-4, 1e-12)
println(@sprintf("phi lifetime at g = 1e-4: %.3g s, N per phi w_phi = %.4g", hbar / sum(width(n, p0) for n in (:phi_AA, :phi_N1N1, :phi_N1nu)), phi_weight(p0)))
om_net = Dict{Tuple{Float64,Float64},Float64}()
println("\nnetwork at the m_phi = 100 roots")
println("  m_keV      g    sin2_2theta  retcode     Omega_h2  x_end   max phi share  end   max occupation  time")
for mk in (10.57, 30.0), g in (1e-7, 1e-5, 1e-4, 1e-3, 3.162e-3, 1e-2)
    s2 = seed(mk, g)
    isfinite(s2) || continue
    r = network_check(mk, g, s2)
    r.rc == DE.ReturnCode.Success && (om_net[(mk, g)] = r.om)
    println(@sprintf("  %6.2f %9.3g  %10.4g  %-11s %9.4g  %6.3g  %10.3g  %8.3g  %10.3g  %6.1f s", mk, g, s2, r.rc, r.om, r.x_end,
                     r.phimax, r.phiend, r.fmax, r.t))
end
if rphi > 2rA
    println("\nfluid at the m_phi = 100 roots")
    println("  m_keV      g    sin2_2theta  Omega_h2  plateau  fluid/network  time")
    for mk in (10.57, 30.0), g in (1e-3, 3.162e-3, 1e-2, 3.162e-2, 0.1)
        s2 = seed(mk, g)
        isfinite(s2) || continue
        pan = FluidSolver(mk * 1e-6, tT, itp)
        t = @elapsed sol, om, plat = solve_relic(pan, tT, g, s2)
        rn = get(om_net, (mk, g), NaN)
        println(@sprintf("  %6.2f %9.3g  %10.4g  %9.4g  %-7s  %9.3g  %6.1f s", mk, g, s2, om, plat, om / rn, t))
    end
else
    println("\nfluid: not applicable (m_phi <= 2 m_A': phi <-> A'A' closed, mu_phi = 2 mu_A' does not hold)")
end
