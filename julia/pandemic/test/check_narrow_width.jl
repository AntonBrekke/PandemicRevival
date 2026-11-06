# How good is the narrow-width treatment of the rate tables at large g? For the resonant portal
# processes (N2 nu -> N1 N2 via A', N1 nu -> A'A' via phi), with MB statistics throughout
# (thermal_rates.jl), the full Breit-Wigner rate at the physical width Gamma against what the tables
# use: the resonance-subtracted (RIS) rate in the narrow-width limit plus the on-shell part, the
# inverse decay times the branching ratio into the final state. The RIS formula of thermal_rates.jl
# does not converge for Gamma/M << 1e-3, so the narrow limit is approached from Gamma/2 and Gamma/4
# (their difference shows how far the RIS part still moves); only g >= 0.5 is checked.
#   julia --project=. test/check_narrow_width.jl [mA_over_mN] [mphi_over_mN]
include(joinpath(@__DIR__, "..", "src", "thermal_rates.jl"))
using Printf

rA = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 50.0
rphi = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 150.0
m = 1e-5
with_width(t, G) = SigmaTable(t.s_th, t.s_max, t.M, G, t.pole_in_range, t.interp)

# (process, resonance, inverse decay r -> a b, on-shell branching into the final state)
CASES = ((:N2nu_N1N2, :A, :A_N2nu, :A_N1N2), (:N1nu_AA, :phi, :phi_N1nu, :phi_AA))
println("m_A'/m_N = $rA, m_phi/m_N = $rphi; NWA / full Breit-Wigner (MB, T_N = T_nu = T); RIS(G/4)/RIS(G/2) - 1")
for (name, res, invdec, final) in CASES
    q = _proc(name)
    for g in (0.5, 1.0, 1.33)
        p = rate_params(m, g, 1e-16; mA_over_mN=rA, mphi_over_mN=rphi)
        M = field_mass(res, p)
        G = total_width(res, p)
        br = width(final, p) / G
        line = @sprintf("%-10s g=%-5g Gamma/M=%.2e |", name, g, G / M)
        for T in (0.05M, 0.15M)
            tab = SigmaTable(q, p; T_max=T, n=250)
            a = Species(m, 2, T, m / T)
            b = Species(0.0, 2, T, 0.0)
            full = rate_2to2(tab, a, b; propagator=:bw)[3]
            r2 = rate_2to2(with_width(tab, G / 2), a, b; propagator=:ris)[3]
            r4 = rate_2to2(with_width(tab, G / 4), a, b; propagator=:ris)[3]
            on = br * rate_inverse_decay(invdec, p, a, b)[3]
            line *= @sprintf("  T=%.2fM: NWA/BW %.4f, RIS share %.3f, RIS(G/4)/RIS(G/2)-1 %.1e", T / M, (r4 + on) / full,
                             r4 / (r4 + on), r4 / r2 - 1)
        end
        println(line)
        flush(stdout)
    end
end
