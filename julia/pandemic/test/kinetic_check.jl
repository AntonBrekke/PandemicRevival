# Kinetic equilibrium along network solutions (the network assumes one T_N):
# elastic N1 N2 -> N1 N2 (incl. its on-shell A'), A' decay and conversions per
# particle over H, over the window in which Omega h^2 grows.
#     julia --project=. test/kinetic_check.jl m_keV:g:s2,...
include(joinpath(@__DIR__, "..", "src", "network.jl"))
using Printf
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
tT_rel = TimeTempRelation{Float64}()
for spec in split(ARGS[1], ',')
    mk, g, s2 = parse.(Float64, split(spec, ':'))
    m = mk * 1e-6
    p = rate_params(m, g, s2)
    net = Network(itp, p, tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    sol = solve_network(net, dw, tT_rel; x_end=100.0, saveat=range(log(1e-4 * mk / 10), log(30.0), length=80))
    zs, us = sol.t, sol.u
    om = [network_omega_h2(net, u, z) for (u, z) in zip(us, zs)]
    # production window: from 1% to 99% of the final Omega increase
    o0, o1 = om[1], om[end]
    i0 = findfirst(o -> o > o0 + 0.01 * (o1 - o0), om)
    i1 = findfirst(o -> o > o0 + 0.99 * (o1 - o0), om)
    T_max = maximum(network_diagnostics(net, u, z).T_N for (u, z) in zip(us, zs))
    tab_el = SigmaTable(_tree(:N1N2_N1N2), p; T_max=2T_max, n=600)
    @printf("%7.3f keV  g=%.3g  s2=%.4g  Omega=%.4f  window x = %.3g .. %.3g\n", mk, g, s2, o1, exp(zs[i0]), exp(zs[i1]))
    minel, minA, minconv = Inf, Inf, Inf
    for i in i0:i1
        u, z = us[i], zs[i]
        n, kappa = state_densities(net, u, z)
        T_N = dark_temperature(net, n, kappa)
        H = net.H(z)
        gap(k) = log(n_scaled_safe(net.masses[k], net.dofs[k], T_N) / n[k])
        N1 = Species(m, 2, T_N, gap(1)); N2 = Species(m, 2, T_N, gap(2))
        el = rate_2to2(tab_el, N1, N2; propagator=:bw)[1] / H                     # per N1
        Ad = rate_decay(:A_N1N2, p, T_N) / H                                       # per A'
        bd = Dict{Symbol,Any}()
        network_collision(net, T_N, m * exp(-z), n; breakdown=bd)
        urs = unit_rates(net, T_N, m * exp(-z), n[1], n[2])
        conv = 2 * exp(only(filter(ur -> ur.dN == (-2, 2, 0, 0), urs)).lnGf) / n[1] / H   # per N1
        minel, minA, minconv = min(minel, el), min(minA, Ad), min(minconv, conv)
        i % 8 == 0 && @printf("    x=%.3g  T_N/T_nu=%.3g  el/H=%.3g  Adec/H=%.3g  conv/H=%.3g\n",
                              exp(z), T_N / (m * exp(-z)), el, Ad, conv)
    end
    @printf("    min over window: elastic/H = %.3g, A' decay/H = %.3g, conversions/H = %.3g\n", minel, minA, minconv)
    flush(stdout)
end
