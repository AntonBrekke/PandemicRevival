"""
Abundance history and reaction rates of the network at benchmark points.

    julia --project=. src/run_network_history.jl m_keV:g:sin2_2theta,... [out_dir]

One CSV per point in <out_dir> (default tmp/network_history): x, T_N/T_nu, H [GeV],
Y of N1, N2, A', phi, Omega h^2, and for every reaction the forward rate per
particle of its first dark initial species over H (Gamma_fwd/n_I1/H), from the
tables (resonance-subtracted rates can be negative; |.| is stored).
"""

include(joinpath(@__DIR__, "network.jl"))
using Printf

function history(itp, tT_rel, mk, g, s2, out_dir)
    m = mk * 1e-6
    net = Network(itp, rate_params(m, g, s2), tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    sol = solve_network(net, dw, tT_rel; x_end=100.0)
    names = [r.name for r in REACTIONS if haskey(itp.dark, r.name) || haskey(itp.portal, r.name)]
    first_species = [dark_index(r.I[1]) for r in REACTIONS if haskey(itp.dark, r.name) || haskey(itp.portal, r.name)]
    path = joinpath(out_dir, @sprintf("m_N_%gkeV_g_%g_sin22th_%.4g.csv", mk, g, s2))
    open(path, "w") do io
        println(io, "x,TN_Tnu,H_GeV,Y_N1,Y_N2,Y_A,Y_phi,omega_h2,", join(names, ","))
        for z in range(sol.t[1], sol.t[end], length=400)
            u = sol(z)
            s = net.ent(z)
            n, kappa = state_densities(net, u, z)
            T_N = dark_temperature(net, n, kappa)
            T_nu = m * exp(-z)
            H = net.H(z)
            urs = unit_rates(net, T_N, T_nu, n[1], n[2])
            lnA, lnphi = log(n[3]), log(n[4])
            rates = [abs(exp(ur.lnGf + ur.cI[1] * lnA + ur.cI[2] * lnphi) * ur.xf) / n[k] / H
                     for (ur, k) in zip(urs, first_species)]
            @printf(io, "%.6g,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%s\n", exp(z), T_N / T_nu, H, (n ./ s)...,
                    network_omega_h2(net, u, z), join((@sprintf("%.4g", r) for r in rates), ","))
        end
    end
    println("written ", path, "  (", sol.retcode, ", Omega h2 = ", round(network_omega_h2(net, sol.u[end], sol.t[end]); sigdigits=4), ")")
    flush(stdout)
end

itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
tT_rel = TimeTempRelation{Float64}()
out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__, "..", "tmp", "network_history")
mkpath(out_dir)
for spec in split(ARGS[1], ',')
    mk, g, s2 = parse.(Float64, split(spec, ':'))
    history(itp, tT_rel, mk, g, s2, out_dir)
end
