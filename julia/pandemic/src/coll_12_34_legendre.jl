import QuadGK
import LinearAlgebra as LA

include(joinpath(@__DIR__, "utils.jl"))

"""
coll_12_34_legendre.jl

Collision integral for a 2 <-> 2 process 1 2 <-> 3 4, with the same meaning
and signature as `coll_12_34` (src/coll_12_34.jl):

    coll_12_34_legendre(mp, p1, p2, p3, p4, temps, xis)
        = ∫ dΠ1 dΠ2 dΠ3 dΠ4 (2π)^4 δ^4(p1+p2-p3-p4) |M|^2
          × [f1 f2 (1 - k3 f3)(1 - k4 f4) - f3 f4 (1 - k1 f1)(1 - k2 f2)],

with f_i = `dist(p_i, T_i, xi_i, E_i)` the occupation number per degree of
freedom (no dof factor), k_i = +1 (-1) for fermions (bosons), and |M|^2 the
squared amplitude summed over all dof, including couplings. No symmetry factors for identical particles are included.

Method
------
For fixed s and total energy E = E1 + E2 = E3 + E4, go to the CM frame, which
moves with |P| = sqrt(E^2 - s) along the z-axis. Let c1 (c3) be the cosine of
the CM angle between p1* (p3*) and z, and c* = n1·n3 the cosine of the CM
scattering angle. The lab energies are linear in c1, c3,

    E1 = [E (s + m1^2 - m2^2) + c1 |P| λ12^(1/2)] / (2s),   E2 = E - E1,
    E3 = [E (s + m3^2 - m4^2) + c3 |P| λ34^(1/2)] / (2s),   E4 = E - E3,

while |M|^2 depends only on (s, c*). In these variables

    C = 1/(512 π^6) ∫ds ∫dE |P| λ12^(1/2) λ34^(1/2) / (4 s^2)
        × (1/4π) ∫dΩ1 ∫dΩ3 |M|^2(s, n1·n3) F(n1·z, n3·z).

At fixed (s, E) the distribution factor is a sum of products,
F = a(c1) b(c3) - a'(c1) b'(c3), and the Funk-Hecke theorem does both
solid-angle integrals exactly,

    (1/4π) ∫dΩ1 ∫dΩ3 |M|^2 a b = (π/2) Σ_l (2l+1) M_l(s) â_l b̂_l,
    M_l(s) = ∫dc* |M|^2(s, c*) P_l(c*),   â_l = ∫dc a(c) P_l(c),

so that

    C = 1/(4096 π^5) ∫ds λ12^(1/2) λ34^(1/2) / s^2 Σ_l (2l+1) M_l(s)
        × ∫_{√s}^∞ dE |P| [â_l b̂_l - â'_l b̂'_l].

In the Maxwell-Boltzmann limit a, b are constant, only l = 0 survives, and
this reduces to the textbook T/(512 π^5) ∫ds K1(√s/T)/√s ∫dt |M|^2.

Truncated Legendre expansions of the angular dependence of 2 <-> 2 collision
kernels are standard in neutrino transport. For a closely related example
(the pair-annihilation kernel nu nubar <-> e+ e-, including the effect of the
truncation order), see J. A. Pons, J. A. Miralles, J. M. Ibáñez, Astron.
Astrophys. Suppl. Ser. 129 (1998) 343, arXiv:astro-ph/9802333. A proof of
the Funk-Hecke formula is given in e.g. C. Müller, Analysis of Spherical
Symmetries in Euclidean Spaces (Springer, 1998).

Why this form:
- The amplitude is only evaluated as a function of the CM scattering angle,
  so t/u-channel forward/backward peaks sit at known places (θ* = 0, π) and
  are resolved with panels graded towards them. Any |M|^2(s, t, u) can be
  plugged in through `sq_amp_func`, with no analytic t-integration.
- No clipped integration limits or 1/sqrt endpoint singularities: after
  w = √s = w_th + σ^2 and E = √s + τ^2 all integrands are smooth.
- The l-sum converges quickly since at least one of â_l, b̂_l is smooth.
- For equal temperatures, detailed balance gives a' b' = e^{Δξ} a b pointwise
  with Δξ = ξ3 + ξ4 - ξ1 - ξ2, so the forward-backward difference is taken
  analytically as -expm1(Δξ), free of cancellation near chemical equilibrium.

Quadrature knobs (keyword arguments): `L` (highest Legendre order), `n_c`
(Gauss-Legendre nodes in c1, c3), `n_E` (nodes per τ-panel, 2 panels),
`n_s` (nodes per σ-panel), `n_theta` (nodes per θ*-panel), `n_cut` (energy
cutoff in units of the largest temperature, above chemical potentials).

Accuracy of the defaults (m_A = 2.5 m_N, ξ_A = 2 ξ_N, 1e-3 <= x <= 50,
see test/test_coll_12_34_AA_NN.jl): ~1e-11 relative, ~3 ms per call. The
exception is a boson close to condensation while relativistic
(m_A/T - ξ_A ≲ 1e-2 with T ≫ m_A), where f_A is sharply peaked in c1 and
the error grows to ~1e-3; there `n_c = 128` brings it back to ~1e-5.
"""

"""
    sq_amp_AA_NN(model_params, p1, p2, p3, p4, s, t, u)

|M|^2 for A'A' -> N_i N_i summed over all dof, eq. (A9) of the paper. Valid
for m_N1 = m_N2: p1 = p2 = A', p3 = p4 = N, and the exchanged fermion has mass
p3.m. The gauge coupling is g = model_params.y. `t` and `u` are passed
separately so that callers can compute both without cancellation.
"""
function sq_amp_AA_NN(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        p4::Particle{T},
        s::R,
        t::R,
        u::R,
    ) where {T<:Real, R<:Real}
    mA2 = p1.m^2
    mN2 = p3.m^2
    tt = t - mN2
    uu = u - mN2
    c2 = (mA2 + 2 * mN2)^2
    return 8 * model_params.y^4 * (
        (4 * (mA2 - mN2)^2 - 16 * mN2^2 + (2 * mN2 + s)^2) / (tt * uu)
        - c2 / tt^2
        - c2 / uu^2
        - 2
    )
end

"""sqrt of the Källén function λ(w^2, ma^2, mb^2), factorized in w = √s."""
@inline function kallen_sqrt_w(w, ma, mb)
    dm = abs(ma - mb)
    return sqrt(max(zero(w), (w - ma - mb) * (w + ma + mb) * (w - dm) * (w + dm)))
end

"""Fill P[l+1] = P_l(x) for l = 0, ..., length(P) - 1."""
function legendre_all!(P::AbstractVector, x)
    L = length(P) - 1
    P[1] = one(x)
    L >= 1 && (P[2] = x)
    for l in 1:L-1
        P[l+2] = ((2 * l + 1) * x * P[l+1] - l * P[l]) / (l + 1)
    end
    return P
end

struct LegendreRules12_34
    L::Int
    c::Vector{Float64}          # Gauss-Legendre nodes in c1, c3
    Pw::Matrix{Float64}         # Pw[l+1, i] = w_i P_l(c_i)
    x_E::Vector{Float64}
    w_E::Vector{Float64}
    x_s::Vector{Float64}
    w_s::Vector{Float64}
    x_th::Vector{Float64}
    w_th::Vector{Float64}
end

if !@isdefined(_LEGENDRE_RULES_12_34)
    const _LEGENDRE_RULES_12_34 = Dict{NTuple{5, Int}, LegendreRules12_34}()
end

function legendre_rules_12_34(L::Int, n_c::Int, n_E::Int, n_s::Int, n_theta::Int)
    return get!(_LEGENDRE_RULES_12_34, (L, n_c, n_E, n_s, n_theta)) do
        c, wc = QuadGK.gauss(n_c)
        Pw = Matrix{Float64}(undef, L + 1, n_c)
        P = Vector{Float64}(undef, L + 1)
        for i in 1:n_c
            legendre_all!(P, c[i])
            Pw[:, i] .= wc[i] .* P
        end
        x_E, w_E = QuadGK.gauss(n_E)
        x_s, w_s = QuadGK.gauss(n_s)
        x_th, w_th = QuadGK.gauss(n_theta)
        LegendreRules12_34(L, c, Pw, x_E, w_E, x_s, w_s, x_th, w_th)
    end
end

"""
Legendre moments M[l+1] = ∫_{-1}^{1} dc* |M|^2(s, c*) P_l(c*), integrated in
θ* with panels graded geometrically towards θ* = 0 (t-channel peak) and
θ* = π (u-channel peak), down to a fraction of the peak width.
"""
function amp_legendre_moments!(
        M::AbstractVector,
        P::AbstractVector,
        model_params,
        p1, p2, p3, p4,
        w,
        rules::LegendreRules12_34,
        sq_amp_func::F,
    ) where F
    s = w^2
    m1, m2, m3, m4 = p1.m, p2.m, p3.m, p4.m
    # CM energies and momenta; |p1*| = |p2*| = k1, |p3*| = |p4*| = k3.
    e1 = (s + m1^2 - m2^2) / (2 * w)
    e3 = (s + m3^2 - m4^2) / (2 * w)
    e4 = w - e3
    k1 = kallen_sqrt_w(w, m1, m2) / (2 * w)
    k3 = kallen_sqrt_w(w, m3, m4) / (2 * w)
    # t = t0 - 4 k1 k3 sin^2(θ*/2), u = u0 - 4 k1 k3 cos^2(θ*/2), where
    # t0 = m1^2 + m3^2 - 2(e1 e3 - k1 k3) is evaluated without cancellation.
    t0 = m1^2 + m3^2 - 2 * (m1^2 * e3^2 + m3^2 * k1^2) / (e1 * e3 + k1 * k3)
    u0 = m1^2 + m4^2 - 2 * (m1^2 * e4^2 + m4^2 * k1^2) / (e1 * e4 + k1 * k3)
    kk = 4 * k1 * k3

    fill!(M, zero(eltype(M)))
    m_min = max(min(m1, m2, m3, m4), 1e-8 * w)
    th_peak = min(pi / 2, m_min / sqrt(k1 * k3))
    lo = zero(w)
    hi = th_peak / 8
    while lo < pi / 2
        hi = min(hi, pi / 2)
        half = (hi - lo) / 2
        mid = (hi + lo) / 2
        for j in eachindex(rules.x_th)
            th = mid + half * rules.x_th[j]
            sin_h2 = sin(th / 2)^2
            cos_h2 = cos(th / 2)^2
            # θ* = th and its mirror θ* = π - th share sin θ* and P_l up to (-1)^l.
            amp = sq_amp_func(model_params, p1, p2, p3, p4, s, t0 - kk * sin_h2, u0 - kk * cos_h2)
            amp_mirror = sq_amp_func(model_params, p1, p2, p3, p4, s, t0 - kk * cos_h2, u0 - kk * sin_h2)
            wt = half * rules.w_th[j] * sin(th)
            legendre_all!(P, cos(th))
            for l in 0:rules.L
                M[l+1] += wt * (iseven(l) ? amp + amp_mirror : amp - amp_mirror) * P[l+1]
            end
        end
        lo = hi
        hi = 2 * hi
    end
    return M
end

function coll_12_34_legendre(
        model_params::ModelParams{T},
        p1::Particle{T},
        p2::Particle{T},
        p3::Particle{T},
        p4::Particle{T},
        temps::NTuple{4, R},
        xis::NTuple{4, R};
        sq_amp_func::F = sq_amp_AA_NN,
        L::Int = 24,
        n_c::Int = 32,
        n_E::Int = 16,
        n_s::Int = 10,
        n_theta::Int = 16,
        n_cut::Real = 50.,
        gaps = nothing,
    ) where {T<:Real, R<:Real, F}
    rules = legendre_rules_12_34(L, n_c, n_E, n_s, n_theta)
    V = promote_type(T, R)
    # Stiff ODE solvers can evaluate trial states with non-finite T or xi.
    # Return NaN so that the step is rejected, instead of throwing further down.
    if !(all(isfinite, temps) && all(isfinite, xis) && all(>(zero(R)), temps) &&
            (isnothing(gaps) || all(isfinite, gaps)))
        return V(NaN)
    end
    # Optional (m - mu)/T per particle, passed on to `dist` (see there).
    g1, g2, g3, g4 = isnothing(gaps) ? (nothing, nothing, nothing, nothing) : gaps
    m1, m2, m3, m4 = p1.m, p2.m, p3.m, p4.m
    A12 = m1^2 - m2^2
    A34 = m3^2 - m4^2

    # Detailed balance for equal temperatures: backward = exp(dxi) * forward.
    equal_temps = all(==(temps[1]), temps)
    fb_fac = -expm1(xis[3] + xis[4] - xis[1] - xis[2])

    # Integration ranges: w = √s = w_th + σ^2, E = w + τ^2.
    w_th = max(m1 + m2, m3 + m4)
    T_max = maximum(temps)
    mu_max = max(zero(V), xis[1] * temps[1] + xis[2] * temps[2], xis[3] * temps[3] + xis[4] * temps[4])
    z_cut = n_cut * T_max + mu_max
    sig_max = sqrt(z_cut)
    tau_max = sqrt(z_cut)
    # σ-panels graded geometrically from the threshold scale to the cutoff.
    sig_lo = sqrt(min(m1, m2, m3, m4, minimum(temps))) / 4
    n_sig_pan = max(1, ceil(Int, log2(sig_max / sig_lo)))

    n_nodes = length(rules.c)
    M = Vector{V}(undef, L + 1)
    P = Vector{V}(undef, L + 1)
    a_fw = Vector{V}(undef, n_nodes); b_fw = similar(a_fw)
    a_bw = similar(a_fw); b_bw = similar(a_fw)
    ah_fw = Vector{V}(undef, L + 1); bh_fw = similar(ah_fw)
    ah_bw = similar(ah_fw); bh_bw = similar(ah_fw)

    total = zero(V)
    for i_pan in 1:n_sig_pan
        sig_a = i_pan == 1 ? zero(V) : sig_max / 2^(n_sig_pan - i_pan + 1)
        sig_b = sig_max / 2^(n_sig_pan - i_pan)
        sig_half = (sig_b - sig_a) / 2
        sig_mid = (sig_b + sig_a) / 2
        for i_s in eachindex(rules.x_s)
            sig = sig_mid + sig_half * rules.x_s[i_s]
            w = w_th + sig^2
            s = w^2
            pref = kallen_sqrt_w(w, m1, m2) * kallen_sqrt_w(w, m3, m4) / s^2
            pref == 0 && continue
            amp_legendre_moments!(M, P, model_params, p1, p2, p3, p4, w, rules, sq_amp_func)
            for l in 0:L
                M[l+1] *= 2 * l + 1
            end
            rl12 = kallen_sqrt_w(w, m1, m2)
            rl34 = kallen_sqrt_w(w, m3, m4)

            e_sum = zero(V)
            for i_tpan in 1:2
                tau_a = (i_tpan - 1) * tau_max / 2
                tau_half = tau_max / 4
                tau_mid = tau_a + tau_half
                for i_E in eachindex(rules.x_E)
                    tau = tau_mid + tau_half * rules.x_E[i_E]
                    z = tau^2
                    E = w + z
                    mom_P = tau * sqrt(z + 2 * w)          # |P| = sqrt(z (z + 2w))
                    # |P| dE = 2 τ |P| dτ
                    jac_E = 2 * tau * mom_P
                    for i in 1:n_nodes
                        c = rules.c[i]
                        e1 = max(m1, (E * (s + A12) + c * mom_P * rl12) / (2 * s))
                        e2 = max(m2, E - e1)
                        e3 = max(m3, (E * (s + A34) + c * mom_P * rl34) / (2 * s))
                        e4 = max(m4, E - e3)
                        f1 = dist(p1, temps[1], xis[1], e1; gap=g1)
                        f2 = dist(p2, temps[2], xis[2], e2; gap=g2)
                        f3 = dist(p3, temps[3], xis[3], e3; gap=g3)
                        f4 = dist(p4, temps[4], xis[4], e4; gap=g4)
                        a_fw[i] = f1 * f2
                        b_fw[i] = (1. - p3.k * f3) * (1. - p4.k * f4)
                        if !equal_temps
                            a_bw[i] = (1. - p1.k * f1) * (1. - p2.k * f2)
                            b_bw[i] = f3 * f4
                        end
                    end
                    LA.mul!(ah_fw, rules.Pw, a_fw)
                    LA.mul!(bh_fw, rules.Pw, b_fw)
                    acc = zero(V)
                    if equal_temps
                        for l in 1:L+1
                            acc += M[l] * ah_fw[l] * bh_fw[l]
                        end
                        acc *= fb_fac
                    else
                        LA.mul!(ah_bw, rules.Pw, a_bw)
                        LA.mul!(bh_bw, rules.Pw, b_bw)
                        for l in 1:L+1
                            acc += M[l] * (ah_fw[l] * bh_fw[l] - ah_bw[l] * bh_bw[l])
                        end
                    end
                    e_sum += tau_half * rules.w_E[i_E] * jac_E * acc
                end
            end
            # ds = 2w dw = 4 w σ dσ
            total += sig_half * rules.w_s[i_s] * 4 * w * sig * pref * e_sum
        end
    end
    return total / (4096 * pi^5)
end
