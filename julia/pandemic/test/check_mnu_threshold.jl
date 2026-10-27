# How small m_nu must be before the N1-N2 splitting it implies, delta = sin^2 theta_1 m_N/m_nu
# (mnu_splitting), changes the relic abundance. On the equal-mass tables m_nu enters production
# only through the couplings: the phi N1 N1 Yukawa y mu = (g/sqrt2)(m_N/m_A') delta (the masses
# stay equal, an O(delta) kinematic error). At the m_A' = 2.5 roots (10.57 keV): Omega h^2 at
# m_nu = M_NU (0.1 eV) and at the m_nu that gives delta = 1e-5 ... 1e-3, and the root shift
# -Delta ln Omega/(d ln Omega/d ln s2), the slope from s2 x 1.05.
#
#     julia --project=. -t 4 test/check_mnu_threshold.jl [m_keV] [out.csv]
include(joinpath(@__DIR__, "..", "src", "lyman_alpha.jl"))
using Printf

m_keV = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 10.57
out = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "..", "tmp", "mnu_threshold", "mnu_threshold_$(m_keV)keV.csv")
m = m_keV * 1e-6
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
tT = TimeTempRelation{Float64}()

"""The stored root of `solver` at (m, g) (tmp/<solver>_scan{,_fine})."""
function root(solver, g)
    col = solver == "network" ? "s2_network" : "s2_fluid"
    for dir in ("$(solver)_scan", "$(solver)_scan_fine")
        path = joinpath(@__DIR__, "..", "tmp", dir, "$(solver)_roots.csv")
        hdr = split(readline(path), ',')
        for l in readlines(path)[2:end]
            c = Dict(zip(hdr, split(l, ',')))
            isapprox(parse(Float64, c["m_N"]), m; rtol=1e-3) && isapprox(parse(Float64, c["y"]), g; rtol=1e-3) &&
                c["status"] == "converged" && return parse(Float64, c[col])
        end
    end
    return NaN
end

function omega(solver, g, s2, m_nu)
    if solver == "network"
        net = Network(itp, table_params(itp, m, g, s2; m_nu=m_nu), tT)
        dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT)
        sol = solve_network(net, dw, tT; maxiters=50_000)
        sol.retcode == DE.ReturnCode.Success || return NaN
        return network_omega_h2(net, sol.u[end], sol.t[end])
    end
    pan = FluidSolver(m, tT, itp; m_nu=m_nu)
    _, om, plateau = solve_relic(pan, tT, g, s2)
    return plateau ? om : NaN
end

deltas = [1e-5, 1e-4, 3e-4, 1e-3]
cases = [("network", g) for g in (1e-7, 1e-6, 1e-5, 1e-4, 3.162e-4, 1e-3)]
append!(cases, [("fluid", g) for g in (1e-3, 1e-2, 0.1)])
jobs = []
for (solver, g) in cases
    s2 = root(solver, g)
    isfinite(s2) || continue
    push!(jobs, (solver, g, s2, M_NU, 1.0))      # baseline
    push!(jobs, (solver, g, s2, M_NU, 1.05))     # slope
    for d in deltas
        th1 = sin(asin(sqrt(s2)) / 2)
        push!(jobs, (solver, g, s2, th1^2 * m * (1 + d) / d, 1.0))
    end
end
res = Vector{Float64}(undef, length(jobs))
Threads.@threads :dynamic for i in eachindex(jobs)
    solver, g, s2, m_nu, f = jobs[i]
    res[i] = try
        omega(solver, g, f * s2, m_nu)
    catch e
        @warn "failed" jobs[i] exception = e
        NaN
    end
    println("done ", jobs[i], " -> ", res[i]); flush(stdout)
end

mkpath(dirname(out))
open(out, "w") do io
    println(io, "solver,m_N,g,s2_root,m_nu_eV,delta,omega,omega_base,dlnO_dlns2,root_shift")
    for (solver, g) in cases
        idx = findall(j -> j[1] == solver && j[2] == g, jobs)
        isempty(idx) && continue
        base, up = res[idx[1]], res[idx[2]]
        slope = log(up / base) / log(1.05)
        s2 = jobs[idx[1]][3]
        for (k, d) in zip(idx[3:end], deltas)
            om = res[k]
            shift = -log(om / base) / slope
            @printf(io, "%s,%.6g,%.6g,%.6g,%.4g,%.3g,%.6g,%.6g,%.4g,%.4g\n", solver, m, g, s2, jobs[k][4] * 1e9, d, om, base, slope, shift)
            @printf("%-7s g=%-9.4g delta=%-6.0e m_nu=%-9.3g eV  Omega %.5f / %.5f  slope %.3g  root shift %+.3g%%\n",
                    solver, g, d, jobs[k][4] * 1e9, om, base, slope, 100 * shift)
        end
    end
end
println("written ", out)
