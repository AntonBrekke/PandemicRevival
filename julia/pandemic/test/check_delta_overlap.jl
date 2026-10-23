# Network vs fluid composition with an N1-N2 splitting, in the overlap of the two solvers.
#
#     julia --project=. test/check_delta_overlap.jl tables.jls m_keV:g:s2_network:s2_fluid,...
#
# Per point: (a) the network's own N2/(N1 + N2), followed to x = 1e3 (and continued by the
# conversion equation if still active: `f_N2_final(...; frozen_at_end=true)`); (b) the same
# equation started fluid-style on the network history (`frozen_at_end=false`), which tests
# the equation against the network; (c) the fluid at its own root (`lyman_alpha_fluid`).
include(joinpath(@__DIR__, "..", "src", "lyman_alpha.jl"))

itp = RateInterpolants(load_rate_tables(ARGS[1]))
tT_rel = TimeTempRelation{Float64}()
z_hi(m) = log(m / tT_rel.T_nu_grid[end])
println("delta = ", table_delta(itp))
for spec in split(ARGS[2], ',')
    mk, g, s2n, s2f = parse.(Float64, split(spec, ':'))
    m = mk * 1e-6
    p = table_params(itp, m, g, s2n)
    net = Network(itp, p, tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2n)) / 2, tT_rel)
    sol = solve_network(net, dw, tT_rel; x_end=1e3, maxiters=20_000)
    h = DarkHistory(net, sol)
    n, _ = state_densities(net, sol.u[end], sol.t[end])
    a = f_N2_final(h, itp, p, p.m2; frozen_at_end=true, z_hi=z_hi(m))
    b = f_N2_final(h, itp, p, p.m2; frozen_at_end=false, z_hi=z_hi(m))
    c = lyman_alpha_fluid(itp, tT_rel, m, g, s2f)
    println("m=$mk keV g=$g: network at x=1e3 $(round(n[2] / (n[1] + n[2]), sigdigits=3)), (a) $(round(a[1], sigdigits=3)) ",
            "x_fo=$(round(a[2], sigdigits=3)); (b) eq. on network history $(round(b[1], sigdigits=3)); ",
            "(c) fluid $(round(c.f_N2, sigdigits=3)) x_fo=$(round(c.x_fo, sigdigits=3)); ",
            "lambda_fs_kd net/fluid $(round(lyman_alpha_lengths(h, tT_rel).lambda_fs_kd, sigdigits=3))/$(round(c.lambda_fs_kd, sigdigits=3))")
    flush(stdout)
end
