"""
Benchmark histories at relic roots, for either solver: what the solver computes along the
solution (network: 400 points in x = m_N/T_nu; fluid: the solver's accepted steps and 400
points of its dense output).

    julia --project=. src/run_benchmarks.jl solver:m_keV:g[:sin2_2theta],... [out_dir] [tables.jls]

solver is `network` or `fluid`; without sin2_2theta the root is taken from
tmp/<solver>_scan{,_fine}/<solver>_roots.csv (m_phi = 100 m_N). One CSV per point in <out_dir>
(default tmp/benchmarks), <solver>_m_<m>keV_g_<g>[_mphi<r>].csv (the suffix for tables other than
m_phi = 100 m_N), with the columns
- x, TN_Tnu, H_GeV, Y_N1, Y_N2, Y_A, Y_phi (n/s), omega_h2, gap_N1 = (m_N - mu_N1)/T_N;
- Gamma_kd_H: the momentum-exchange rate of N1 over H (lyman_alpha.jl; sets x_c and x_kd);
- network: `fwd:<reaction>`, the forward rate per particle of the reaction's first dark
  initial species over H, from the tables (resonance-subtracted rates can be negative: |.|);
- fluid: `net:<process>`, the process's collision term C_n~ over (n~ H), signed,
  n~ = n_N1 + n_N2 + 2 n_A' + 4 n_phi (the fluid evolves n~ and rho only), and `net_total`,
  the rate of change of n~ that the solver integrated, (dn~/dt + 3H n~)/(n~ H), from the
  solution. Where a process is fast (A'A' <-> NN at up to 1e12 H) the implicit solver keeps
  the fluid on its equilibrium, and the collision term at a stored state is the residual of
  a near-cancellation: only its size >> 1 is meaningful, while `net_total` is what moves n~.
  Also `fwd:<reaction>` as for the network, from the MB tables at the fluid's T_N and
  densities (A', phi at their equilibrium): smooth, and comparable with the network.
Plot with plot/benchmark.py.
"""

include(joinpath(@__DIR__, "lyman_alpha.jl"))
using Printf

"""The root of `solver` at (m_keV, g) from the stored scans."""
function stored_root(solver, m_keV, g)
    col = solver == "network" ? "s2_network" : "s2_fluid"
    for dir in ("$(solver)_scan", "$(solver)_scan_fine")
        path = joinpath(@__DIR__, "..", "tmp", dir, "$(solver)_roots.csv")
        isfile(path) || continue
        hdr = split(readline(path), ',')
        for l in readlines(path)[2:end]
            c = Dict(zip(hdr, split(l, ',')))
            isapprox(parse(Float64, c["m_N"]) * 1e6, m_keV; rtol=1e-3) && isapprox(parse(Float64, c["y"]), g; rtol=1e-3) &&
                c["status"] == "converged" && return parse(Float64, c[col])
        end
    end
    error("no converged $solver root at m = $m_keV keV, g = $g")
end

# 9 digits: the plots take derivatives of Y; x with 15: at large g the solver steps through the
# take-off in steps of d ln x < 1e-9
fmt(v) = @sprintf("%.9g", v)
row(x, rest) = join([@sprintf("%.15g", x); fmt.(rest)], ",")

function network_benchmark(itp, tT_rel, m, g, s2)
    net = Network(itp, table_params(itp, m, g, s2), tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    sol = solve_network(net, dw, tT_rel; x_end=100.0, maxiters=20_000)
    h = DarkHistory(net, sol)
    used = [r for r in REACTIONS if haskey(itp.dark, r.name) || haskey(itp.portal, r.name)]
    first_species = [dark_index(r.I[1]) for r in used]
    rows = String[]
    for z in range(sol.t[1], sol.t[end], length=400)
        u = sol(z)
        s = net.ent_interp_z(z)
        n, kappa = state_densities(net, u, z)
        T_N = dark_temperature(net, n, kappa)
        H = net.H_interp_z(z)
        urs = unit_rates(net, T_N, m * exp(-z), n[1], n[2])
        lnA, lnphi = log(n[3]), log(n[4])
        fwd = [abs(exp(ur.lnGf + ur.cI[1] * lnA + ur.cI[2] * lnphi) * ur.xf) / n[k] / H for (ur, k) in zip(urs, first_species)]
        push!(rows, row(exp(z), [T_N / (m * exp(-z)), H, (n ./ s)..., network_omega_h2(net, u, z),
                                 dark_state(h, z)[2], kd_ratio(h, z), fwd...]))
    end
    return ["fwd:$(r.name)" for r in used], rows, network_omega_h2(net, sol.u[end], sol.t[end])
end

function fluid_benchmark(itp, tT_rel, m, g, s2)
    pan = FluidSolver(m, tT_rel, itp)
    sol, om, plateau = solve_relic(pan, tT_rel, g, s2)
    plateau || error("fluid solve did not settle at m = $m, g = $g")
    h = DarkHistory(pan, sol)
    # forward rates from the network's MB unit rates at the fluid state
    net = Network(itp, table_params(itp, m, g, s2), tT_rel)
    used = [r for r in REACTIONS if haskey(itp.dark, r.name) || haskey(itp.portal, r.name)]
    first_species = [dark_index(r.I[1]) for r in used]
    names = nothing
    rows = String[]
    # the solver's steps and 400 points of its dense output: after the take-off the steps
    # alone leave decades with a handful of points
    zs = sort(unique([sol.t; range(sol.t[1], sol.t[end], length=400)]))
    dz = 1e-6
    for z in zs
        u = sol(z)
        z0, z1 = max(z - dz, sol.t[1]), min(z + dz, sol.t[end])
        total = (sol(z1)[1] - sol(z0)[1]) / (z1 - z0) * pan.dx_dt_interp_z(z) / exp(z) / pan.H_interp_z(z)
        n, T_N, xi_N, gap_A = fluid_state(pan, u)
        T_nu = m * exp(-z)
        H = pan.H_interp_z(z)
        br = Dict{Symbol,Tuple{Float64,Float64}}()
        collision_terms(pan, T_nu, T_N, xi_N; gap_A=gap_A, breakdown=br)
        names === nothing && (names = sort(collect(keys(br))))
        ntilde = n[1] + n[2] + 2n[3] + 4n[4]
        s = pan.ent_interp_z(z)
        # A' and phi floored as in the network (qss_blend): a Boltzmann-suppressed phi underflows
        nk = (n[1], n[2], max(n[3], n[1] * exp(-600)), max(n[4], n[1] * exp(-600)))
        urs = unit_rates(net, T_N, T_nu, nk[1], nk[2])
        lnA, lnphi = log(nk[3]), log(nk[4])
        fwd = [abs(exp(ur.lnGf + ur.cI[1] * lnA + ur.cI[2] * lnphi) * ur.xf) / nk[k] / H for (ur, k) in zip(urs, first_species)]
        push!(rows, row(exp(z), [T_N / T_nu, H, (n ./ s)..., omega_h2_of(pan, u, z), N_temp_gap(pan, u)[2],
                                 kd_ratio(h, z), total, (br[k][1] / (ntilde * H) for k in names)..., fwd...]))
    end
    return ["net_total"; ["net:$k" for k in names]; ["fwd:$(r.name)" for r in used]], rows, om
end

function main(args)
    out_dir = length(args) > 1 ? args[2] : joinpath(@__DIR__, "..", "tmp", "benchmarks")
    mkpath(out_dir)
    tables = length(args) > 2 ? args[3] : rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))
    itp = RateInterpolants(load_rate_tables(tables))
    rphi = mass_ratios(itp)[2]
    suffix = rphi == 100 ? "" : @sprintf("_mphi%g", rphi)
    tT_rel = TimeTempRelation{Float64}()
    for spec in split(args[1], ',')
        f = split(spec, ':')
        solver, m_keV, g = f[1], parse(Float64, f[2]), parse(Float64, f[3])
        s2 = length(f) > 3 ? parse(Float64, f[4]) : stored_root(solver, m_keV, g)
        t = @elapsed (rate_cols, rows, om) = (solver == "network" ? network_benchmark : fluid_benchmark)(itp, tT_rel, m_keV * 1e-6, g, s2)
        path = joinpath(out_dir, @sprintf("%s_m_%gkeV_g_%g%s.csv", solver, m_keV, g, suffix))
        open(path, "w") do io
            println(io, "# ", solver, " m_N = ", m_keV, " keV, g = ", g, ", sin^2 2theta = ", s2, ", m_phi = ", rphi, " m_N")
            println(io, "x,TN_Tnu,H_GeV,Y_N1,Y_N2,Y_A,Y_phi,omega_h2,gap_N1,Gamma_kd_H,", join(rate_cols, ","))
            foreach(r -> println(io, r), rows)
        end
        println("written ", path, @sprintf("  (Omega h2 = %.4f, %.0f s)", om, t)); flush(stdout)
    end
end

main(ARGS)
