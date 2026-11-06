"""
coll_portal.jl

Exact collision integrals, with FD/BE statistics of all four particles (Pauli blocking of the
final N), for the portal 2 <-> 2 processes 1 2 <-> 3 4 with a massless neutrino, as used by the
fluid instead of the MB tables:

    coll_portal(proc, p, parts, temps, xis; gaps, e_index)
        = (C, C_E),
    C   = ∫ dΠ1..dΠ4 (2π)^4 δ^4 |M|^2 [f1 f2 (1 - k3 f3)(1 - k4 f4) - f3 f4 (1 - k1 f1)(1 - k2 f2)],
    C_E = the same with the energy E_e of particle `e_index` in the integrand

(no symmetry factors; f per dof; |M|^2 summed over all dof). `proc` is an entry of
TreeAmplitudes.TREE_PROCESSES (or crossed_process), `p` its couplings (rate_params).

Method: as coll_12_34_legendre.jl (CM variables s, E, c1, c3 and the Funk-Hecke theorem), with
two changes for a hot neutrino meeting cold, possibly degenerate N:
- the Legendre projections â_l = ∫dc a(c) P_l(c) use Gauss panels in c whose breakpoints sit where
  the occupation numbers change: at fixed (s, E) the lab energies are linear in c, so a cold N
  (T_N << T_nu) occupies a narrow band of c, and a degenerate one has a Fermi surface there;
- the energy integral E = √s + z is graded on all temperatures.
A resonant s-channel (proc.resonance, sum|M|^2 = X |D|^2 + 2 Y Re D + Z) is taken in the narrow-width
limit as in rate_tables.jl `nw_rate_density`: the Hadamard finite part of X/(s - M^2)^2 and the
principal value of Y/(s - M^2), which do not depend on the width; the on-shell part is the inverse
decay, a separate process of the fluid. The outer s integral is adaptive (QuadGK).
"""

include(joinpath(@__DIR__, "coll_12_34_legendre.jl"))
isdefined(@__MODULE__, :TreeAmplitudes) || include(joinpath(@__DIR__, "tree_amplitudes.jl"))

const PORTAL_L = 24          # highest Legendre order
const PORTAL_NG = 8          # Gauss nodes per c-panel
const PORTAL_NE = 8          # Gauss nodes per E-panel
const PORTAL_NTH = 16        # nodes per theta*-panel of the amplitude moments
const PORTAL_NCUT = 50.0     # energy cutoff in units of the largest temperature (above mu)
const PORTAL_RTOL = 1e-7     # outer s integral

const _GL_PORTAL = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
gl_nodes(n) = get!(() -> QuadGK.gauss(n), _GL_PORTAL, n)

"""
Occupation scales of particle j: the energies max(m, mu) + k T and, if degenerate (mu > m), the
Fermi surface mu - k T, from which a panel boundary is placed.
"""
function _escales!(out, m, T, gap)
    mu = m - gap * T
    base = max(m, mu)
    for k in (0.0, 0.3, 1.0, 3.0, 10.0, 30.0)
        push!(out, base + k * T)
    end
    if mu > m
        for k in (0.3, 1.0, 3.0, 10.0)
            mu - k * T > m && push!(out, mu - k * T)
        end
    end
    return out
end

"""
Gauss nodes and weights in c ∈ [-1, 1] for one side (particles a, b with E_a = α + β c, E_b = E - E_a),
with panel boundaries at the c where E_a or E_b crosses one of their occupation scales.
"""
function side_nodes(α, β, E, ma, Ta, ga, mb, Tb, gb; ng=PORTAL_NG, L=PORTAL_L)
    # uniform base panels, enough nodes for P_l up to l = L (else the projections alias), plus the
    # occupation scales
    nbase = cld(3 * (L + 1), 2 * ng)
    br = Float64[-1.0 + 2k / nbase for k in 0:nbase]
    if abs(β) > 0
        ea = _escales!(Float64[], ma, Ta, ga)
        eb = _escales!(Float64[], mb, Tb, gb)
        for e in ea
            c = (e - α) / β
            -1 < c < 1 && push!(br, c)
        end
        for e in eb
            c = (E - e - α) / β
            -1 < c < 1 && push!(br, c)
        end
    end
    sort!(br)
    unique!(br)
    x, w = gl_nodes(ng)
    cs, ws = Float64[], Float64[]
    for i in 1:length(br)-1
        a, b = br[i], br[i+1]
        b - a < 1e-14 && continue
        h, mid = (b - a) / 2, (b + a) / 2
        for j in eachindex(x)
            push!(cs, mid + h * x[j])
            push!(ws, h * w[j])
        end
    end
    return cs, ws
end

"""
Ms[l+1, k] = (2l + 1) ∫dc* A_k(s, c*) P_l(c*) for the parts A_k = amp(s, t)[k] (a tuple), as
amp_legendre_moments! (coll_12_34_legendre.jl) but with every part from one call per node and the
forward/backward panels graded by the smallest nonzero mass (a massless neutrino has no peak).
"""
function amp_moments_multi!(Ms, P, p1, p2, p3, p4, w, rules, amp)
    s = w^2
    m1, m2, m3, m4 = p1.m, p2.m, p3.m, p4.m
    e1 = (s + m1^2 - m2^2) / (2w)
    e3 = (s + m3^2 - m4^2) / (2w)
    e4 = w - e3
    k1 = kallen_sqrt_w(w, m1, m2) / (2w)
    k3 = kallen_sqrt_w(w, m3, m4) / (2w)
    t0 = m1^2 + m3^2 - 2 * (m1^2 * e3^2 + m3^2 * k1^2) / (e1 * e3 + k1 * k3)
    kk = 4k1 * k3
    fill!(Ms, 0.0)
    nz = filter(>(0), (m1, m2, m3, m4))
    m_min = max(isempty(nz) ? 0.0 : minimum(nz), 1e-8w)
    th_peak = min(pi / 2, m_min / sqrt(k1 * k3))
    L = size(Ms, 1) - 1
    lo, hi = 0.0, th_peak / 8
    while lo < pi / 2
        hi = min(hi, pi / 2)
        half, mid = (hi - lo) / 2, (hi + lo) / 2
        for j in eachindex(rules.x_th)
            th = mid + half * rules.x_th[j]
            sh, ch = sin(th / 2)^2, cos(th / 2)^2
            a = amp(s, t0 - kk * sh)          # θ* = th
            b = amp(s, t0 - kk * ch)          # θ* = π - th
            wt = half * rules.w_th[j] * sin(th)
            legendre_all!(P, cos(th))
            for k in eachindex(a), l in 0:L
                Ms[l+1, k] += wt * (iseven(l) ? a[k] + b[k] : a[k] - b[k]) * P[l+1]
            end
        end
        lo = hi
        hi = 2hi
    end
    for k in 1:size(Ms, 2), l in 0:L
        Ms[l+1, k] *= 2l + 1
    end
    return Ms
end

"""âl[l+1] = Σ_i w_i v_i P_l(c_i) for every vector v in vs (written into the columns of out)."""
function project!(out::AbstractMatrix, P::AbstractVector, cs, ws, vs...)
    fill!(out, 0.0)
    for i in eachindex(cs)
        legendre_all!(P, cs[i])
        for (k, v) in enumerate(vs)
            wv = ws[i] * v[i]
            wv == 0 && continue
            for l in eachindex(P)
                out[l, k] += wv * P[l]
            end
        end
    end
    return out
end

struct PortalCtx{F}
    p::Any                       # rate_params (couplings, masses)
    parts::NTuple{4,Particle{Float64}}
    temps::NTuple{4,Float64}
    gaps::NTuple{4,Float64}      # (m - mu)/T per particle
    amps::F                      # (s, t) -> tuple: (|M|^2,), or (X, Y, Z) of a resonant process
    namps::Int
    e_index::Int                 # particle whose energy weights C_E (0: none)
    terms::Symbol                # :net, :fwd or :bwd (diagnostics)
    L::Int
    ng::Int
    ne::Int
    z_cut::Float64
    zpts::Vector{Float64}        # breakpoints of z = E - √s
end

"""
At √s = w: the integrand per unit s of every amplitude in ctx.amps, (number, energy) pairs flattened,
(n_1, e_1, n_2, e_2, ...), i.e. λ12^(1/2) λ34^(1/2)/s^2 Σ_l (2l+1) M_l ∫dE |P| [â b̂ - â' b̂'] (without
the 1/(4096 π^5) of the total).
"""
function portal_s_integrand(w, ctx::PortalCtx)
    p1, p2, p3, p4 = ctx.parts
    m1, m2, m3, m4 = p1.m, p2.m, p3.m, p4.m
    nA = ctx.namps
    s = w^2
    rl12 = kallen_sqrt_w(w, m1, m2)
    rl34 = kallen_sqrt_w(w, m3, m4)
    pref = rl12 * rl34 / s^2
    out = zeros(2nA)
    pref == 0 && return out
    L = ctx.L
    rules = legendre_rules_12_34(L, 8, 8, 8, PORTAL_NTH)
    Ms = Matrix{Float64}(undef, L + 1, nA)
    P = Vector{Float64}(undef, L + 1)
    amp_moments_multi!(Ms, P, p1, p2, p3, p4, w, rules, ctx.amps)
    A12, A34 = m1^2 - m2^2, m3^2 - m4^2
    T1, T2, T3, T4 = ctx.temps
    g1, g2, g3, g4 = ctx.gaps
    ei = ctx.e_index
    xE, wE = gl_nodes(ctx.ne)
    ah = Matrix{Float64}(undef, L + 1, 4)       # fw, bw, fw*E, bw*E
    bh = Matrix{Float64}(undef, L + 1, 4)
    zs = [0.0; filter(z -> 0 < z < ctx.z_cut, ctx.zpts); ctx.z_cut]
    for ip in 1:length(zs)-1
        za, zb = zs[ip], zs[ip+1]
        # z = E - w on each panel through τ = √z (smooth at z = 0, where |P| ∝ √z)
        ta, tb = sqrt(za), sqrt(zb)
        th, tm = (tb - ta) / 2, (tb + ta) / 2
        for j in eachindex(xE)
            tau = tm + th * xE[j]
            z = tau^2
            E = w + z
            momP = tau * sqrt(z + 2w)
            jac = 2tau * momP * th * wE[j]               # |P| dE = 2 τ |P| dτ
            α1, β1 = E * (s + A12) / (2s), momP * rl12 / (2s)
            α3, β3 = E * (s + A34) / (2s), momP * rl34 / (2s)
            c1, w1 = side_nodes(α1, β1, E, m1, T1, g1, m2, T2, g2; ng=ctx.ng, L=L)
            c3, w3 = side_nodes(α3, β3, E, m3, T3, g3, m4, T4, g4; ng=ctx.ng, L=L)
            n1, n3 = length(c1), length(c3)
            afw, abw, afwE, abwE = zeros(n1), zeros(n1), zeros(n1), zeros(n1)
            for i in 1:n1
                e1 = max(m1, α1 + β1 * c1[i]); e2 = max(m2, E - e1)
                f1 = dist(p1, T1, 0.0, e1; gap=g1); f2 = dist(p2, T2, 0.0, e2; gap=g2)
                afw[i] = f1 * f2
                abw[i] = (1 - p1.k * f1) * (1 - p2.k * f2)
                we = ei == 1 ? e1 : ei == 2 ? e2 : 1.0
                afwE[i] = afw[i] * we; abwE[i] = abw[i] * we
            end
            bfw, bbw, bfwE, bbwE = zeros(n3), zeros(n3), zeros(n3), zeros(n3)
            for i in 1:n3
                e3 = max(m3, α3 + β3 * c3[i]); e4 = max(m4, E - e3)
                f3 = dist(p3, T3, 0.0, e3; gap=g3); f4 = dist(p4, T4, 0.0, e4; gap=g4)
                bfw[i] = (1 - p3.k * f3) * (1 - p4.k * f4)
                bbw[i] = f3 * f4
                we = ei == 3 ? e3 : ei == 4 ? e4 : 1.0
                bfwE[i] = bfw[i] * we; bbwE[i] = bbw[i] * we
            end
            project!(ah, P, c1, w1, afw, abw, afwE, abwE)
            project!(bh, P, c3, w3, bfw, bbw, bfwE, bbwE)
            eon1 = ei in (1, 2)      # the weighted side
            cf = ctx.terms === :bwd ? 0.0 : 1.0
            cb = ctx.terms === :fwd ? 0.0 : 1.0
            for k in 1:nA
                accn = 0.0; acce = 0.0
                for l in 1:L+1
                    Mk = Ms[l, k]
                    accn += Mk * (cf * ah[l, 1] * bh[l, 1] - cb * ah[l, 2] * bh[l, 2])
                    ei == 0 && continue
                    acce += Mk * (eon1 ? (cf * ah[l, 3] * bh[l, 1] - cb * ah[l, 4] * bh[l, 2]) :
                                         (cf * ah[l, 1] * bh[l, 3] - cb * ah[l, 2] * bh[l, 4]))
                end
                out[2k-1] += jac * accn
                out[2k] += jac * acce
            end
        end
    end
    return pref .* out
end

"""
    coll_portal(proc, p, parts, temps, gaps; e_index=2)

(C, C_E) [GeV^4, GeV^5] of the 2 <-> 2 process `proc` (TREE_PROCESSES entry) with parameters `p`,
particles `parts` (Particle, k = 1 fermion, -1 boson; the massless one is the neutrino), their
temperatures and gaps (m - mu)/T (for the neutrino -mu/T). See the header.
"""
function coll_portal(proc, p, parts, temps, gaps; e_index=2, rtol=PORTAL_RTOL, terms=:net, zonly=false,
                     L=PORTAL_L, ng=PORTAL_NG, ne=PORTAL_NE, ncut=PORTAL_NCUT)
    all(isfinite, temps) && all(>(0), temps) && all(isfinite, gaps) || return (NaN, NaN)
    m1, m2, m3, m4 = (q.m for q in parts)
    T_max = maximum(temps)
    mus = ntuple(i -> parts[i].m - gaps[i] * temps[i], 4)
    mu_max = max(0.0, mus[1] + mus[2] - m1 - m2, mus[3] + mus[4] - m3 - m4)
    z_cut = ncut * T_max + mu_max
    zpts = sort!([T * f for T in temps for f in (0.03, 0.3, 1.0, 3.0, 10.0, 30.0)])
    w_th = max(m1 + m2, m3 + m4)
    w_max = w_th + z_cut
    resonant = proc.resonance !== nothing && !zonly
    M_r = resonant ? field_mass(proc.resonance, p) : 0.0
    amps = zonly ? ((s, t) -> (proc.xyz(s, t, p)[3],)) : resonant ? ((s, t) -> proc.xyz(s, t, p)) :
           ((s, t) -> (proc.msq(s, t, p),))
    ctx = PortalCtx(p, parts, temps, gaps, amps, (resonant ? 3 : 1), e_index, terms, L, ng, ne, z_cut, zpts)
    S(w) = portal_s_integrand(w, ctx)
    # breakpoints in q = w - w_th on the scales of all temperatures (and the threshold)
    qk = sort!([T * f for T in temps for f in (0.01, 0.1, 1.0, 5.0, 20.0, 80.0)])
    qpts(lo, hi) = sort!([lo; filter(x -> lo < x < hi, qk); hi])
    quadv(f, pts) = QuadGK.quadgk(f, pts...; rtol=rtol, atol=0.0)[1]
    if !resonant || !(w_th < M_r < w_max)
        # zero-width form (a pole outside the range is regular there)
        comb(v, s) = resonant ? [v[1] / (s - M_r^2)^2 + 2v[3] / (s - M_r^2) + v[5],
                                 v[2] / (s - M_r^2)^2 + 2v[4] / (s - M_r^2) + v[6]] : v
        G(q) = (w = w_th + q; 2w .* comb(S(w), w^2))
        r = quadv(G, qpts(0.0, z_cut))
        return (r[1] / (4096pi^5), r[2] / (4096pi^5))
    end
    s0 = M_r^2
    s_th = w_th^2
    Lw = min(s0 - s_th, 0.5s0, (w_max^2 - s0) * 0.999)
    zw(s, v) = (d = s - s0; [v[1] / d^2 + 2v[3] / d + v[5], v[2] / d^2 + 2v[4] / d + v[6]])
    Gq(q) = (w = w_th + q; 2w .* zw(w^2, S(w)))
    below = quadv(Gq, qpts(0.0, sqrt(s0 - Lw) - w_th))
    above = quadv(Gq, qpts(sqrt(s0 + Lw) - w_th, z_cut))
    # window |s - s0| < Lw: S is per unit s; FP of X/d^2 and PV of 2Y/d paired at s0 ± d
    v0 = S(sqrt(s0))
    h0 = (v0[1], v0[2])
    function fx(d)
        vp, vm = S(sqrt(s0 + d)), S(sqrt(s0 - d))
        return [(vp[1] + vm[1] - 2h0[1]) / d^2 + 2(vp[3] - vm[3]) / d + vp[5] + vm[5],
                (vp[2] + vm[2] - 2h0[2]) / d^2 + 2(vp[4] - vm[4]) / d + vp[6] + vm[6]]
    end
    delta = 1e-4 * Lw
    dk = sort!(unique!([delta; filter(d -> delta < d < Lw, [[abs((w_th + q)^2 - s0) for q in qk]; [Lw * 10.0^-j for j in 3:-1:1]]); Lw]))
    J = fx(delta) .* delta .+ quadv(fx, dk) .- [2h0[1] / Lw, 2h0[2] / Lw]
    r = below .+ J .+ above
    return (r[1] / (4096pi^5), r[2] / (4096pi^5))
end
