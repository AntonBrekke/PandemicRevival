# Fluid at large g for another table set: a ladder in sin^2 2theta at one (m, g), with Omega h^2,
# the outcome, the coldest T_N/T_nu and the smallest (most degenerate) N gap along the solution.
#   julia --project=. test/check_large_g_fluid.jl <tables.jls> <m_keV> <g> <s2_1,s2_2,...> [wall_s]
include(joinpath(@__DIR__, "..", "src", "fluid.jl"))
using Printf

itp = RateInterpolants(load_rate_tables(ARGS[1]))
m = parse(Float64, ARGS[2]) * 1e-6
g = parse(Float64, ARGS[3])
s2s = parse.(Float64, split(ARGS[4], ','))
wall = length(ARGS) > 4 ? parse(Float64, ARGS[5]) : 900.0
tT = TimeTempRelation{Float64}()
pan = FluidSolver(m, tT, itp)
println("m_A'/m_N, m_phi/m_N = ", mass_ratios(itp), ", m = $(m * 1e6) keV, g = $g")
for s2 in s2s
    pan.mp = ModelParams{Float64}(g, asin(sqrt(s2)) / 2)
    th = asin(sqrt(s2)) / 2
    pan.deadline = time() + wall
    t = @elapsed sol = try
        solve_fluid(tT, DodelsonWidrow{Float64}(m, th, tT; x_end=100.0), pan; reltol=1e-4)
    catch e
        e
    end
    pan.deadline = Inf
    if sol isa Exception
        @printf("s2=%.3e  exception %s after %.0f s\n", s2, typeof(sol), t)
    else
        tmin, gmin = Inf, Inf
        for (z, u) in zip(sol.t, sol.u)
            T_N, gap = N_temp_gap(pan, u)
            tmin = min(tmin, T_N * exp(z) / m)
            gmin = min(gmin, gap)
        end
        @printf("s2=%.3e  %s at x=%.3g, Omega h^2 there %.4g, steps %d, %.0f s, min T_N/T_nu %.3e, min gap_N %.3g\n", s2,
                sol.retcode, exp(sol.t[end]), omega_h2_of(pan, sol.u[end], sol.t[end]), length(sol.t), t, tmin, gmin)
    end
    flush(stdout)
end
