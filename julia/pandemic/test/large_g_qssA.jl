# Why is mu_A' far from mu_N1 + mu_N2 in the MB network at large g? Per-reaction C_A' at one state
# of the 10.57 keV, g = 5.6e-3 solve at x = ARGS[1] (e.g. 4e-5). Finding (2026-10-12): phi <-> A'A'
# (~1e12 per unit z, T_N > m_phi) dominates the A' balance, and the sequential A' / phi QSS roots are not
# a joint root; see notes/model_limitations.tex, Sec. "Large g: results".
include(joinpath(@__DIR__, "..", "src", "network.jl"))
using Printf
tT_rel = TimeTempRelation{Float64}()
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
mk, g, s2, xe = 10.57, 0.005623, 8.44755e-17, parse(Float64, ARGS[1])
m = mk * 1e-6
net = Network(itp, rate_params(m, g, s2), tT_rel)
dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
sol = solve_network(net, dw, tT_rel; x_end=xe, fmax_stop=Inf, maxiters=4000)
z = sol.t[end]; u = sol.u[end]
n, kappa = state_densities(net, u, z)
T_N = dark_temperature(net, n, kappa); T_nu = m / exp(z)
xfac = exp(z) / net.dx_dt(z)
urs = unit_rates(net, T_N, T_nu, n[1], n[2])
names = [r.name for r in REACTIONS if haskey(itp.dark, r.name) || haskey(itp.portal, r.name)]
nh = [n_scaled_safe(net.masses[k], net.dofs[k], T_N) for k in 1:4]
mu = [log(n[k] / nh[k]) + net.masses[k] / T_N for k in 1:4]
@printf("x=%.3e T_N/m=%.3g mu/T: N1 %.3f N2 %.3f A %.3f phi %.3f\n", exp(z), T_N / m, mu...)
nAeq = nh[3] * exp(mu[1] + mu[2] - net.masses[3] / T_N)
@printf("n_A = %.3e  n_A(mu_A = mu1+mu2) = %.3e\n", n[3], nAeq)
for (nA, lab) in ((n[3], "state"), (nAeq, "eq"))
    println("C_A per reaction at n_A = ", lab, " (per unit z, / n_A):")
    lnA, lnphi = log(nA), log(n[4])
    for (ur, nm) in zip(urs, names)
        ur.dN[3] == 0 && continue
        Gf = exp(ur.lnGf + ur.cI[1] * lnA + ur.cI[2] * lnphi) * ur.xf
        Gr = exp(ur.lnGr + ur.cF[1] * lnA + ur.cF[2] * lnphi) * ur.xr
        @printf("  %-12s dA=%2d  fwd %.3e  rev %.3e\n", nm, ur.dN[3], xfac * Gf / nA, xfac * Gr / nA)
    end
end
println("QSS coeffs: ", qss_coefficients(urs, 3, n[3], n[4]), "  root ", qss_root(qss_coefficients(urs, 3, n[3], n[4])...))
