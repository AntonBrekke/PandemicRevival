# Perturbativity and narrow-width measures of the dark sector against g, at fixed mass ratios:
#   julia --project=. test/check_perturbativity.jl [mA_over_mN] [mphi_over_mN]
# lambda = m_phi^2/(2 v_phi^2) = g^2 (m_phi/m_A')^2/2 (tree_amplitudes.tex), alpha = g^2/4pi,
# Gamma/M of A' and phi (all tree 1 -> 2 channels), the Born parameter of N N scattering through
# A' exchange, alpha m_N/m_A' (a Yukawa potential has no bound states and Born holds for << 1),
# and the one-loop size lambda/16pi^2, g^2/16pi^2.
include(joinpath(@__DIR__, "..", "src", "thermal_rates.jl"))
using Printf

rA = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 50.0
rphi = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 150.0
m = 1e-5
println("m_A'/m_N = $rA, m_phi/m_N = $rphi")
println("    g   lambda  lam/4pi  alpha   G_A'/m_A'  G_phi/m_phi  BR(phi->A'A')  alpha*mN/mA'  lam/16pi^2  g^2/16pi^2")
for g in (0.01, 0.03, 0.1, 0.178, 0.316, 0.5, 0.75, 1.0, 1.33, 1.78, 2.37, 3.16)
    p = rate_params(m, g, 1e-16; mA_over_mN=rA, mphi_over_mN=rphi)
    lam = g^2 * rphi^2 / rA^2 / 2
    GA, Gphi = total_width(:A, p), total_width(:phi, p)
    @printf("%6.3g  %7.3g  %6.3f  %6.3f  %9.3g  %10.3g  %10.6f  %11.3g  %10.3g  %10.3g\n", g, lam, lam / 4pi, g^2 / 4pi,
            GA / p.mA, Gphi / p.mphi, width(:phi_AA, p) / Gphi, g^2 / 4pi / rA, lam / 16pi^2, g^2 / 16pi^2)
end
