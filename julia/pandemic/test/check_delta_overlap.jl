# Network vs fluid with an N1-N2 splitting, in the overlap of the two solvers.
#
#     julia --project=. test/check_delta_overlap.jl tables.jls m_keV:g:s2_network:s2_fluid,...
#
# Per point, N2/(N1 + N2) today and lambda_fs_kd [Mpc]:
#  (a) the network to x = 1e3 (its own N2 and heat), continued by `with_conversions` if still
#      converting there;
#  (b) the network to x = 1e3 with the conversions restarted fluid-style on its history
#      (`frozen_at_end=false`): tests the conversion equation against the network;
#  (c) the network to x = 100 only, continued by `with_conversions` (late conversions and heat
#      from the equation instead of the network): tests the continuation;
#  (d) the fluid at its own root (`lyman_alpha_fluid`).
include(joinpath(@__DIR__, "..", "src", "lyman_alpha.jl"))

itp = RateInterpolants(load_rate_tables(ARGS[1]))
tT_rel = TimeTempRelation{Float64}()
z_hi(m) = log(m / tT_rel.T_nu_grid[end])
fmt(v) = round(v, sigdigits=3)
println("delta = ", table_delta(itp))
for spec in split(ARGS[2], ',')
    mk, g, s2n, s2f = parse.(Float64, split(spec, ':'))
    m = mk * 1e-6
    p = table_params(itp, m, g, s2n)
    net = Network(itp, p, tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2n)) / 2, tT_rel)
    res = []
    for (x_end, frozen) in ((1e3, true), (1e3, false), (100.0, true))
        sol = solve_network(net, dw, tT_rel; x_end=x_end, maxiters=20_000)
        h, f2, x_fo = with_conversions(DarkHistory(net, sol), itp, p, p.m2; frozen_at_end=frozen, z_hi=z_hi(m))
        push!(res, (f2, lyman_alpha_lengths(h, tT_rel).lambda_fs_kd, x_fo))
    end
    c = lyman_alpha_fluid(itp, tT_rel, m, g, s2f)
    push!(res, (c.f_N2, c.lambda_fs_kd, c.x_fo))
    println("m=$mk keV g=$g  f_N2 / lambda_fs_kd / x_fo:  ",
            join(["($l) $(fmt(r[1])) / $(fmt(r[2])) / $(fmt(r[3]))" for (l, r) in zip("abcd", res)], "   "))
    flush(stdout)
end
