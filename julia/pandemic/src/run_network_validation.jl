"""
Compare the species-resolved network with the reduced solver at relic roots
of the merged scan (tmp/relic_scan_theta_merged): Omega h^2 of the network at
the reduced solver's root, and the network's own root.

    julia --project=. src/run_network_validation.jl [table.jls] [m_keV:y,m_keV:y,...]

Default points lie inside the strip where the reduced solver's assumptions
hold (notes/equilibrium_check.tex), so the two should agree up to MB
statistics and the processes the reduced solver leaves out. Output:
tmp/network_validation.csv.
"""

include(joinpath(@__DIR__, "network.jl"))
using Printf

const SCAN_DIR = joinpath(@__DIR__, "..", "tmp", "relic_scan_theta_merged")

"""Reduced-solver root (sin^2 2theta, Omega h^2) at (m_keV, y) from the merged scan, or nothing."""
function reduced_root(m_keV, y)
    files = filter(f -> startswith(f, "m_N_") && endswith(f, "keV.csv"), readdir(SCAN_DIR))
    for f in files
        lines = readlines(joinpath(SCAN_DIR, f))
        for l in lines[2:end]
            c = split(l, ',')
            mk, yy = parse(Float64, c[1]) * 1e6, parse(Float64, c[2])
            isapprox(mk, m_keV; rtol=1e-3) && isapprox(yy, y; rtol=1e-3) || continue
            s2 = parse(Float64, c[3])
            return isnan(s2) ? nothing : (s2, parse(Float64, c[4]))
        end
    end
    return nothing
end

"""Network Omega h^2 at x_end for (m_N, g, sin^2 2theta), with the solve's retcode and step count."""
function network_omega(itp, tT_rel, m, g, s2; x_end=100.0)
    p = rate_params(m, g, s2)
    net = Network(itp, p, tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    sol = solve_network(net, dw, tT_rel; x_end=x_end, maxiters=20_000)
    # Terminated: the dark sector became degenerate (MB invalid), see solve_network
    om = sol.retcode == DE.ReturnCode.Success ? network_omega_h2(net, sol.u[end], sol.t[end]) : NaN
    return om, sol.retcode, length(sol.t)
end

"""
Root of log(Omega/0.12) in log sin^2 2theta by the secant method from the
reduced root: Omega grows like a power (DW seed) times an exponential
(pandemic) of sin^2 2theta, smooth in the logs.
"""
function network_root(itp, tT_rel, m, g, s2_0, om_0; tol=1e-3, maxit=12)
    f(om) = log(om / omega_d0)
    l0, f0 = log(s2_0), f(om_0)
    isfinite(f0) || return NaN, NaN
    l1 = l0 - 0.1 * sign(f0)
    om1, rc, nst = network_omega(itp, tT_rel, m, g, exp(l1))
    f1 = f(om1)
    for it in 1:maxit
        @printf("      iteration %d: sin^2 2theta = %.5e  Omega = %.5g  (%s, %d steps)\n", it, exp(l1), om1, rc, nst); flush(stdout)
        isfinite(f1) || return NaN, NaN
        abs(f1) < tol && return exp(l1), om1
        # secant step, at most a factor e^0.5 at a time (Omega is steep in sin^2 2theta)
        l2 = l1 - f1 * (l1 - l0) / (f1 - f0)
        l2 = clamp(l2, l1 - 0.5, l1 + 0.5)
        l0, f0 = l1, f1
        l1 = l2
        om1, rc, nst = network_omega(itp, tT_rel, m, g, exp(l1))
        f1 = f(om1)
    end
    return exp(l1), om1
end

function main(args)
    path = length(args) > 0 ? args[1] : rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))
    pts = length(args) > 1 ? [Tuple(parse.(Float64, split(s, ':'))) for s in split(args[2], ',')] :
          [(10.57, 1e-3), (2.21, 1e-3), (13.72, 5.623e-4), (37.77, 5.623e-4), (4.834, 3.162e-3), (10.57, 5.623e-3)]
    itp = RateInterpolants(load_rate_tables(path))
    tT_rel = TimeTempRelation{Float64}()
    out = joinpath(@__DIR__, "..", "tmp", length(args) > 2 ? args[3] : "network_validation.csv")
    open(out, "w") do io
        println(io, "m_keV,y,s2_reduced,omega_reduced,omega_network_at_reduced,s2_network,omega_network,retcode,steps,seconds")
        for (mk, y) in pts
            rr = reduced_root(mk, y)
            rr === nothing && (println("no reduced root at ", mk, " keV, y = ", y); flush(stdout); continue)
            s2r, omr = rr
            t = @elapsed (om, rc, nst) = network_omega(itp, tT_rel, mk * 1e-6, y, s2r)
            s2n, omn = network_root(itp, tT_rel, mk * 1e-6, y, s2r, om)
            line = @sprintf("%.4g,%.4g,%.6g,%.5g,%.5g,%.6g,%.5g,%s,%d,%.1f", mk, y, s2r, omr, om, s2n, omn, rc, nst, t)
            println(io, line); flush(io)
            @printf("%7.3f keV  y=%.3g  reduced root %.4e (Omega %.4f) | network Omega there %.4f (%s, %d steps, %.1f s) | network root %.4e (ratio %.3f)\n",
                    mk, y, s2r, omr, om, rc, nst, t, s2n, s2n / s2r)
            flush(stdout)
        end
    end
    println("written ", out)
end

main(ARGS)
