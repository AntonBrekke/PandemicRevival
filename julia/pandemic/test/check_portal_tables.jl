# Accuracy of the exact portal tables (portal_tables.jl) at random off-grid states in the range the
# fluid visits, against direct coll_portal calls; and of the MB tables at the same states.
#   julia --project=. -t 8 test/check_portal_tables.jl <portal_exact.jls> [n]
include(joinpath(@__DIR__, "..", "src", "rate_tables.jl"))
include(joinpath(@__DIR__, "..", "src", "portal_tables.jl"))
using Printf, Random

tabs = Serialization.deserialize(ARGS[1])
n = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 40
rA, rphi = tabs.mass_ratios
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables");
                                                        mA_over_mN=rA, mphi_over_mN=rphi)))
px = PortalExact(tabs)
p = itp.p_ref
m = p.m1
rng = MersenneTwister(1)
states = [(log(3e-5) + rand(rng) * (log(100.0) - log(3e-5)), log(1e-3) + rand(rng) * (log(2.0) - log(1e-3)),
           12 * rand(rng)) for _ in 1:n]
res = Vector{Any}(undef, n)
Threads.@threads for i in 1:n
    lxnu, lr, gap = states[i]
    out = []
    for name in PORTAL_EXACT
        T_nu = m * exp(-lxnu); T_N = exp(lr) * T_nu
        c = coll_portal(_tree(name), p, portal_parts(name, p), (T_N, T_nu, T_N, T_N), (gap, 0.0, gap, gap))
        sym = portal_sym(name)
        direct = (c.fwd[1], c.bwd[1]) .* sym
        tab = portal_rates(itp, px, name, lxnu, lr, gap)[1]
        mb = portal_rates(itp, nothing, name, lxnu, lr, gap)[1]
        scale = max(abs(direct[1]), abs(direct[2]))
        push!(out, (name, direct, tab, mb, scale))
    end
    res[i] = (states[i], out)
end
worst = Dict{Symbol,Float64}()
for (st, out) in res
    for (name, direct, tab, mb, scale) in out
        scale > 0 && isfinite(scale) || continue
        e_tab = abs(tab - (direct[1] - direct[2])) / scale
        e_mb = abs(mb - (direct[1] - direct[2])) / scale
        worst[name] = max(get(worst, name, 0.0), e_tab)
        @printf("x_nu=%8.3g r=%7.3g gap=%5.2f %-10s net error / max(fwd,bwd): exact table %.2e, MB table %.2e\n",
                exp(-st[1]) > 0 ? exp(st[1]) : NaN, exp(st[2]), st[3], name, e_tab, e_mb)
    end
end
println("largest error of the exact table: ", worst)
