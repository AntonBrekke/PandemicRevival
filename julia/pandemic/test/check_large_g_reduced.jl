# Reduced network across sin^2 2theta at one mass, for several g: Omega h^2, outcome (Terminated =
# an MB occupation number above 0.5), the coldest T_N/T_nu, the largest occupation and where the
# solve ended. Quick physics picture before fluid runs.
#   julia --project=. test/check_large_g_reduced.jl <tables.jls> <m_keV> <g1,g2,...> <s2_lo> <s2_hi> [n]
include(joinpath(@__DIR__, "..", "src", "network_reduced.jl"))
using Printf

itp = RateInterpolants(load_rate_tables(ARGS[1]))
m = parse(Float64, ARGS[2]) * 1e-6
gs = parse.(Float64, split(ARGS[3], ','))
lo, hi = parse(Float64, ARGS[4]), parse(Float64, ARGS[5])
n = length(ARGS) > 5 ? parse(Int, ARGS[6]) : 13
tT = TimeTempRelation{Float64}()
println("m_A'/m_N, m_phi/m_N = ", mass_ratios(itp), ", m = $(m * 1e6) keV")
for g in gs
    println("g = $g")
    for s2 in exp.(range(log(lo), log(hi), length=n))
        net = Network(itp, table_params(itp, m, g, s2), tT)
        dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT)
        t = @elapsed sol = solve_reduced_network(net, dw, tT; maxiters=50_000)
        tmin, fmx = Inf, 0.0
        for (z, u) in zip(sol.t, sol.u)
            d = reduced_diagnostics(net, u, z)
            tmin = min(tmin, d.T_N / d.T_nu)
            fmx = max(fmx, maximum(d.fmax))
        end
        @printf("  s2=%.3e  %-10s x_end=%-8.3g Omega=%.3e  min T_N/T_nu=%.2e  max f=%.2e  steps=%d  %.1fs\n", s2, sol.retcode,
                exp(sol.t[end]), reduced_omega_h2(net, sol.u[end], sol.t[end]), tmin, fmx, length(sol.t), t)
        flush(stdout)
    end
end
