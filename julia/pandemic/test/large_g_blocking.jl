# Quantum statistics of the dark species in the portal 2 -> 2 rates (large-g step 0.2,
# notes/model_limitations.tex): N1 nu -> N2 N2 integrated directly with Fermi-Dirac N
# (Pauli blocking of both final N2, FD initial N1), against the MB-fugacity form that
# pandemolator_hg.jl takes from the tables, at states of the 10.57 keV, g = 1e-2 history.
# The MB version is first checked against the table itself.
#   julia --project=. -t 5 test/large_g_blocking.jl
include(joinpath(@__DIR__, "..", "src", "pandemolator_hg.jl"))
using Printf
import QuadGK

const MK, G, S2 = 10.57, 1e-2, 2.6761e-17
p = rate_params(MK * 1e-6, G, S2)
m = p.m1
q = _proc(:N1nu_N2N2)

const GLN, GLW = QuadGK.gauss(8)

"""Gauss-Legendre nodes and weights on consecutive intervals of `edges`."""
function panels(edges)
    xs, ws = Float64[], Float64[]
    for i in 1:length(edges)-1
        a, b = edges[i], edges[i+1]
        b > a || continue
        for j in eachindex(GLN)
            push!(xs, (a + b) / 2 + (b - a) / 2 * GLN[j]); push!(ws, (b - a) / 2 * GLW[j])
        end
    end
    return xs, ws
end

"""
(forward, reverse, forward E_nu-weighted, reverse E_nu-weighted) reaction rate densities of
N1 nu <-> N2 N2 at (T_N, T_nu, mu/T_N = xi), quantum (FD N) or MB-fugacity N; nu FD in both.
"""
function rates(T_N, T_nu, xi; quantum)
    mu = xi * T_N
    fN(E) = quantum ? 1 / (exp((E - mu) / T_N) + 1) : exp(-(E - mu) / T_N)
    bN(E) = quantum ? 1 - fN(E) : 1.0
    fnu(E) = 1 / (exp(E / T_nu) + 1)
    # E_a - m and E_nu on log panels
    ka, wka = panels(exp.(range(log(1e-4T_N), log(60T_N + max(mu, 0.0)), length=12)))
    Tlo = min(T_N, T_nu)
    en, wen = panels(exp.(range(log(1e-4Tlo), log(60T_nu), length=16)))
    cs, wcs = panels([-1.0, 0.0, 0.9, 0.99, 0.999, 1.0])
    phs, wphs = panels([0.0, pi / 2, pi])
    out = zeros(4)
    for (k, wk) in zip(ka, wka)
        Ea = m + k; pa = sqrt(k * (k + 2m))
        fa = fN(Ea)
        for (Ev, wv) in zip(en, wen)
            fv = fnu(Ev)
            # neither direction contributes here (forward f_a f_nu, reverse needs E_nu <~ 60 T_N)
            (fa * fv < 1e-40 && Ev > 80T_N + 4abs(mu) + 3m) && continue
            for (c, wc) in zip(cs, wcs)
                s = m^2 + 2Ev * (Ea - pa * c)
                s > 4m^2 * (1 + 1e-12) || continue
                rs = sqrt(s)
                Et = Ea + Ev
                Pz = sqrt(max(pa^2 + Ev^2 + 2pa * Ev * c, 0.0))
                beta, gam = Pz / Et, Et / rs
                ps = sqrt(s / 4 - m^2)                 # final |p*|
                pas = (s - m^2) / (2rs); Eas = (s + m^2) / (2rs)
                # component of p_a along P in the CM frame
                pa_par = Pz > 0 ? (pa * (pa + Ev * c) / Pz) : pa
                cal = clamp(gam * (pa_par - beta * Ea) / pas, -1.0, 1.0)
                sal = sqrt(1 - cal^2)
                # CM angle panels graded towards both peaks (t and u channel)
                thp = min(pi / 2, max(m, 1e-8rs) / sqrt(pas * ps))
                edges = [0.0]
                h = thp / 8
                while edges[end] < pi / 2
                    push!(edges, min(edges[end] + h, pi / 2)); h *= 2
                end
                edges = vcat(edges, reverse(pi .- edges[1:end-1]))
                ths, wths = panels(edges)
                for (th, wt) in zip(ths, wths)
                    ct, st = cos(th), sin(th)
                    t = 2m^2 - 2(Eas * rs / 2 - pas * ps * ct)
                    M2 = q.msq(s, t, p)
                    acc = 0.0; accE = 0.0; accr = 0.0
                    for (ph, wp) in zip(phs, wphs)
                        cz = ct * cal + st * sal * cos(ph)
                        Ec = gam * (rs / 2 + beta * ps * cz)
                        Ed = Et - Ec
                        Ed > m || continue
                        acc += wp * bN(Ec) * bN(Ed)
                        accr += wp * fN(Ec) * fN(Ed)
                    end
                    # d3p_a/(2E_a) = 2 pi p_a dE_a, d3p_nu/(2E_nu) = pi E_nu dE_nu dcos (azimuth of nu
                    # about p_a done), dPhi_2 = p*/(16 pi^2 rs) sin th dth dph * 2
                    # (phi in [0, pi], symmetric), 1/2 for identical N2
                    jac = wk * wv * wc * wt * (2pi * pa) * (pi * Ev) / (2pi)^6 * ps / (16pi^2 * rs) * st * 2 * 0.5 * M2
                    fw = jac * fa * fv * acc
                    rv = jac * accr * (quantum ? 1 - fa : 1.0) * (1 - fv)
                    out[1] += fw; out[2] += rv; out[3] += fw * Ev; out[4] += rv * Ev
                end
            end
        end
    end
    return out
end

"""The table's (forward, reverse, Qf, Qr) at the same state and MB fugacities."""
function table_rates(T_N, T_nu, xi)
    itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
    lnR, lnQ = rate_norms(itp, p)[:N1nu_N2N2]
    t = itp.portal[:N1nu_N2N2]
    lxnu, lr = log(m / T_nu), log(T_N / T_nu)
    g = m / T_N - xi
    return (exp(t.lnRf(lxnu, lr) + lnR - g) * t.ratio_Rf(lxnu, lr), exp(t.lnRr(lxnu, lr) + lnR - 2g) * t.ratio_Rr(lxnu, lr),
            exp(t.lnQf(lxnu, lr) + lnQ - g) * t.ratio_Qf(lxnu, lr), exp(t.lnQr(lxnu, lr) + lnQ - 2g) * t.ratio_Qr(lxnu, lr))
end

# (label, T_N/m, T_N/T_nu, mu/T_N): states of the g = 1e-2 history (test/hg_breakdown.jl)
states = [("x=4.3e-4 (thermalised, injection)", 19.4, 0.0084, 0.0),
          ("x=1.5e-3", 11.1, 0.0164, 0.0),
          ("x=1.8e-2", 1.4, 0.0248, 0.0),
          ("x=8.6 (late)", 0.147, 1.27, 0.0),
          ("x=30 (late, mu > 0)", 0.0556, 1.65, 8.9)]
res = Vector{Any}(undef, length(states))
Threads.@threads for i in eachindex(states)
    lab, tn, r, xi = states[i]
    T_N = tn * m; T_nu = T_N / r
    t0 = time()
    mb = rates(T_N, T_nu, xi; quantum=false)
    qu = rates(T_N, T_nu, xi; quantum=true)
    res[i] = (lab, mb, qu, time() - t0)
end
for (i, (lab, tn, r, xi)) in enumerate(states)
    _, mb, qu, dt = res[i]
    tb = table_rates(tn * m, tn * m / r, xi)
    @printf("%s  (%.0f s)\n", lab, dt)
    @printf("   MB direct / table:  fwd %.4f  rev %.4f  Qf %.4f  Qr %.4f\n", (mb ./ tb)...)
    @printf("   FD / MB:            fwd %.4f  rev %.4f  Qf %.4f  Qr %.4f   net n %.4f  net E %.4f\n", (qu ./ mb)...,
            (qu[1] - qu[2]) / (mb[1] - mb[2]), (qu[3] - qu[4]) / (mb[3] - mb[4]))
end
