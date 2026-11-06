"""
coll_portal.jl

Exact collision integrals, with FD/BE statistics of all four particles (Pauli blocking of the
final N), for the portal 2 <-> 2 processes 1 2 <-> 3 4 with a massless neutrino:

    coll_portal(proc, p, parts, temps, gaps; e_index) = (fwd = (R, Q), bwd = (R, Q)[, fwdZ, bwdZ]),
    R_fwd = ∫ dΠ1..dΠ4 (2π)^4 δ^4 |M|^2 f1 f2 (1 - k3 f3)(1 - k4 f4),
    R_bwd = the same with f3 f4 (1 - k1 f1)(1 - k2 f2),
    Q     = the same with the energy E_e of particle `e_index` in the integrand,

so that the net rate density is R_fwd - R_bwd (no symmetry factors; f per dof; |M|^2 summed over
all dof). `proc` is an entry of TreeAmplitudes.TREE_PROCESSES, `p` its couplings (rate_params). For
a resonant s-channel fwdZ, bwdZ are the non-resonant part Z alone (positive; the tables' reference).

Method: as coll_12_34_legendre.jl (CM variables s, E, c1, c3 and the Funk-Hecke theorem), with
changes for a hot neutrino meeting cold, possibly degenerate N:
- the Legendre projections â_l = ∫dc a(c) P_l(c) use Gauss panels in c with breakpoints where the
  occupation numbers change (at fixed (s, E) the lab energies are linear in c, so a cold N, T_N << T_nu,
  occupies a narrow band of c, and a degenerate one has a Fermi surface there), on top of uniform
  panels with enough nodes for P_l up to L;
- the energy E = √s + z and the outer integral in √s are graded on all temperatures; all quadratures
  have fixed nodes (smooth in the state: the result is tabulated, portal_tables.jl).
A resonant s-channel (proc.resonance, sum|M|^2 = X |D|^2 + 2 Y Re D + Z) is taken in the narrow-width
limit as in rate_tables.jl `nw_rate_density`: the Hadamard finite part of X/(s - M^2)^2 and the
principal value of Y/(s - M^2), which do not depend on the width; the on-shell part is the inverse
decay, a separate process of the fluid.
"""

isdefined(@__MODULE__, :coll_12_34_legendre) || include(joinpath(@__DIR__, "coll_12_34_legendre.jl"))
isdefined(@__MODULE__, :TreeAmplitudes) || include(joinpath(@__DIR__, "tree_amplitudes.jl"))

const PORTAL_L = 24          # highest Legendre order
const PORTAL_NG = 8          # Gauss nodes per c-panel
const PORTAL_NE = 8          # Gauss nodes per E-panel
const PORTAL_NS = 8          # Gauss nodes per panel of the outer s integral
const PORTAL_NTH = 16        # nodes per theta*-panel of the amplitude moments
const PORTAL_NCUT = 50.0     # energy cutoff in units of the largest temperature (above mu)

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
        # every breakpoint is kept, clamped into [-1, 1]: a panel then shrinks continuously to zero
        # width instead of appearing or vanishing as (s, E) change, so the integral is smooth in s
        # (the narrow-width finite part takes second differences of it at |s - M^2| down to 1e-4 L)
        for e in ea
            push!(br, clamp((e - α) / β, -1.0, 1.0))
        end
        for e in eb
            push!(br, clamp((E - e - α) / β, -1.0, 1.0))
        end
    end
    sort!(br)
    x, w = gl_nodes(ng)
    cs, ws = Float64[], Float64[]
    for i in 1:length(br)-1
        a, b = br[i], br[i+1]
        b == a && continue
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
    parts::NTuple{4,Particle{Float64}}
    temps::NTuple{4,Float64}
    gaps::NTuple{4,Float64}      # (m - mu)/T per particle
    amps::F                      # (s, t) -> tuple: (|M|^2,), or (X, Y, Z) of a resonant process
    namps::Int
    e_index::Int                 # particle whose energy weights Q (0: none)
    L::Int
    ng::Int
    ne::Int
    z_cut::Float64
    zpts::Vector{Float64}        # breakpoints of z = E - √s
end

"""
Total energies E at which the lab energy E_j(c = ±1) = (E e* ± |P| p*)/√s of a particle with CM energy
e*, momentum p* and mass mj equals k/√s (k = E_j √s), |P| = √(E^2 - s): the roots of
(E e* - k)^2 = p*^2 (E^2 - s), i.e. mj^2 E^2 - 2 e* k E + k^2 + p*^2 s = 0 (linear for mj = 0).
"""
function _edges_E!(out, k, estar, pstar, mj, s)
    if mj == 0
        estar > 0 && push!(out, (k^2 + pstar^2 * s) / (2estar * k))
        return out
    end
    disc = (estar * k)^2 - mj^2 * (k^2 + pstar^2 * s)
    disc < 0 && return out
    r = sqrt(disc)
    push!(out, (estar * k + r) / mj^2, (estar * k - r) / mj^2)
    return out
end

"""Columns l = 0..L of P_l at the nodes cs, as a matrix (L+1) x n."""
function legendre_matrix(cs, L)
    Pm = Matrix{Float64}(undef, L + 1, length(cs))
    for (i, c) in enumerate(cs)
        legendre_all!(view(Pm, :, i), c)
    end
    return Pm
end

"""
At √s = w: the integrand per unit s, λ12^(1/2) λ34^(1/2)/s^2 Σ_l (2l+1) M_l ∫dE |P| â_l b̂_l (without
the 1/(4096 π^5) of the total), for every amplitude part k: v[4(k-1) + j], j = 1 forward number,
2 forward energy, 3 backward number, 4 backward energy.
"""
function portal_s_integrand(w, ctx::PortalCtx)
    p1, p2, p3, p4 = ctx.parts
    m1, m2, m3, m4 = p1.m, p2.m, p3.m, p4.m
    nA = ctx.namps
    s = w^2
    rl12 = kallen_sqrt_w(w, m1, m2)
    rl34 = kallen_sqrt_w(w, m3, m4)
    pref = rl12 * rl34 / s^2
    out = zeros(4nA)
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
    # z breakpoints: the temperature scales, and the kinematic edges where a particle's occupation
    # scale enters the c-range (a cold particle makes the integrand step there, at an E that moves
    # with s; without these the integral jumps as s changes, which the finite part amplifies)
    edges = Float64[]
    es = ((s + m1^2 - m2^2) / (2w), rl12 / (2w), (s + m3^2 - m4^2) / (2w), rl34 / (2w))
    cm = ((es[1], es[2]), (w - es[1], es[2]), (es[3], es[4]), (w - es[3], es[4]))   # (e*, p*) of 1..4
    for (j, (mj, Tj, gj)) in enumerate(((m1, T1, g1), (m2, T2, g2), (m3, T3, g3), (m4, T4, g4)))
        for e in _escales!(Float64[], mj, Tj, gj)
            e > mj || continue
            _edges_E!(edges, e * w, cm[j]..., mj, s)
        end
    end
    zs = sort!(unique!([0.0; filter(z -> 0 < z < ctx.z_cut, [ctx.zpts; edges .- w]); ctx.z_cut]))
    for ip in 1:length(zs)-1
        # z = E - w through τ = √z on each panel (smooth at z = 0, where |P| ∝ √z)
        ta, tb = sqrt(zs[ip]), sqrt(zs[ip+1])
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
            # weighted values: columns fw, bw, fw*E, bw*E (E of the side carrying e_index, else 1)
            V1 = Matrix{Float64}(undef, n1, 4)
            for i in 1:n1
                e1 = max(m1, α1 + β1 * c1[i]); e2 = max(m2, E - e1)
                f1 = dist(p1, T1, 0.0, e1; gap=g1); f2 = dist(p2, T2, 0.0, e2; gap=g2)
                we = ei == 1 ? e1 : ei == 2 ? e2 : 1.0
                a, ab = f1 * f2, (1 - p1.k * f1) * (1 - p2.k * f2)
                V1[i, 1] = w1[i] * a; V1[i, 2] = w1[i] * ab
                V1[i, 3] = w1[i] * a * we; V1[i, 4] = w1[i] * ab * we
            end
            V3 = Matrix{Float64}(undef, n3, 4)
            for i in 1:n3
                e3 = max(m3, α3 + β3 * c3[i]); e4 = max(m4, E - e3)
                f3 = dist(p3, T3, 0.0, e3; gap=g3); f4 = dist(p4, T4, 0.0, e4; gap=g4)
                we = ei == 3 ? e3 : ei == 4 ? e4 : 1.0
                b, bb = (1 - p3.k * f3) * (1 - p4.k * f4), f3 * f4
                V3[i, 1] = w3[i] * b; V3[i, 2] = w3[i] * bb
                V3[i, 3] = w3[i] * b * we; V3[i, 4] = w3[i] * bb * we
            end
            ah = legendre_matrix(c1, L) * V1        # (L+1) x 4
            bh = legendre_matrix(c3, L) * V3
            eon1 = ei in (1, 2)
            for k in 1:nA
                fn = fe = bn = be = 0.0
                for l in 1:L+1
                    Mk = Ms[l, k]
                    fn += Mk * ah[l, 1] * bh[l, 1]
                    bn += Mk * ah[l, 2] * bh[l, 2]
                    if eon1
                        fe += Mk * ah[l, 3] * bh[l, 1]; be += Mk * ah[l, 4] * bh[l, 2]
                    else
                        fe += Mk * ah[l, 1] * bh[l, 3]; be += Mk * ah[l, 2] * bh[l, 4]
                    end
                end
                o = 4(k - 1)
                out[o+1] += jac * fn; out[o+2] += jac * fe
                out[o+3] += jac * bn; out[o+4] += jac * be
            end
        end
    end
    return pref .* out
end

"""Σ over the panels between the sorted points `pts` of an n-point Gauss rule for the vector-valued f."""
function gauss_panels(f, pts; n=PORTAL_NS)
    x, wq = gl_nodes(n)
    acc = nothing
    for i in 1:length(pts)-1
        a, b = pts[i], pts[i+1]
        b > a || continue
        h, mid = (b - a) / 2, (b + a) / 2
        for j in eachindex(x)
            v = (h * wq[j]) .* f(mid + h * x[j])
            acc = acc === nothing ? v : acc .+ v
        end
    end
    return acc
end

"""
    coll_portal(proc, p, parts, temps, gaps; e_index=2)

(fwd = (R, Q), bwd = (R, Q)) [GeV^4, GeV^5] of the 2 <-> 2 process `proc` (TREE_PROCESSES entry) with
parameters `p`, particles `parts` (Particle, k = 1 fermion, -1 boson; the massless one is the
neutrino), their temperatures and gaps (m - mu)/T (for the neutrino -mu/T); for a resonant process
also fwdZ, bwdZ (the non-resonant part Z alone). See the header.
"""
function coll_portal(proc, p, parts, temps, gaps; e_index=2, L=PORTAL_L, ng=PORTAL_NG, ne=PORTAL_NE,
                     ncut=PORTAL_NCUT)
    bad = (fwd=(NaN, NaN), bwd=(NaN, NaN), fwdZ=(NaN, NaN), bwdZ=(NaN, NaN))
    (all(isfinite, temps) && all(>(0), temps) && all(isfinite, gaps)) || return bad
    m1, m2, m3, m4 = (q.m for q in parts)
    T_max = maximum(temps)
    mus = ntuple(i -> parts[i].m - gaps[i] * temps[i], 4)
    mu_max = max(0.0, mus[1] + mus[2] - m1 - m2, mus[3] + mus[4] - m3 - m4)
    z_cut = ncut * T_max + mu_max
    zpts = sort!(unique!([T * f for T in temps for f in (0.03, 0.3, 1.0, 3.0, 10.0, 30.0)]))
    w_th = max(m1 + m2, m3 + m4)
    w_max = w_th + z_cut
    resonant = proc.resonance !== nothing
    M_r = resonant ? field_mass(proc.resonance, p) : 0.0
    amps = resonant ? ((s, t) -> proc.xyz(s, t, p)) : ((s, t) -> (proc.msq(s, t, p),))
    ctx = PortalCtx(parts, temps, gaps, amps, (resonant ? 3 : 1), e_index, L, ng, ne, z_cut, zpts)
    S(w) = portal_s_integrand(w, ctx)
    norm = 1 / (4096pi^5)
    # outer variable σ = √(w - w_th) (the threshold behaves like √(w - w_th)), breakpoints on the scales
    # of all temperatures
    qk = sort!(unique!([T * f for T in temps for f in (0.01, 0.1, 1.0, 5.0, 20.0, 80.0)]))
    σpts(qlo, qhi) = sqrt.(sort!([qlo; filter(x -> qlo < x < qhi, qk); qhi]))
    # the integrand per dσ of the parts in `sel` combined with propagator weights `cw(s)` (per part)
    function outer(qlo, qhi, comb)
        g(σ) = (w = w_th + σ^2; 2σ * 2w .* comb(S(w), w^2))     # ds = 2w dw, dw = 2σ dσ
        return gauss_panels(g, σpts(qlo, qhi))
    end
    pick(v, k) = v[4(k-1)+1:4k]
    if !resonant
        r = outer(0.0, z_cut, (v, s) -> v) .* norm
        return (fwd=(r[1], r[2]), bwd=(r[3], r[4]))
    end
    s0 = M_r^2
    full_zw(v, s) = (d = s - s0; [pick(v, 1) ./ d^2 .+ 2 .* pick(v, 2) ./ d .+ pick(v, 3); pick(v, 3)])
    if !(w_th < M_r < w_max)
        r = outer(0.0, z_cut, full_zw) .* norm      # the pole outside the range: zero-width form
        return (fwd=(r[1], r[2]), bwd=(r[3], r[4]), fwdZ=(r[5], r[6]), bwdZ=(r[7], r[8]))
    end
    s_th = w_th^2
    Lw = min(s0 - s_th, 0.5s0, (w_max^2 - s0) * 0.999)
    below = outer(0.0, sqrt(s0 - Lw) - w_th, full_zw)
    above = outer(sqrt(s0 + Lw) - w_th, z_cut, full_zw)
    # window |s - s0| < Lw: FP of X/d^2 and PV of 2Y/d paired at s0 ± d, Z regular
    h0 = pick(S(sqrt(s0)), 1)
    function fx(d)
        vp, vm = S(sqrt(s0 + d)), S(sqrt(s0 - d))
        zz = pick(vp, 3) .+ pick(vm, 3)
        return [(pick(vp, 1) .+ pick(vm, 1) .- 2 .* h0) ./ d^2 .+ 2 .* (pick(vp, 2) .- pick(vm, 2)) ./ d .+ zz; zz]
    end
    delta = 1e-4 * Lw
    dq = [abs((w_th + q)^2 - s0) for q in qk]
    dk = sort!(unique!([delta; filter(d -> delta < d < Lw, [dq; [Lw * 10.0^-j for j in 3:-1:1]]); Lw]))
    J = fx(delta) .* delta .+ gauss_panels(fx, dk) .- [2 .* h0 ./ Lw; zeros(4)]
    r = (below .+ J .+ above) .* norm
    return (fwd=(r[1], r[2]), bwd=(r[3], r[4]), fwdZ=(r[5], r[6]), bwdZ=(r[7], r[8]))
end

"""Net (R, Q) of a coll_portal result: forward minus backward."""
portal_net(c) = (c.fwd[1] - c.bwd[1], c.fwd[2] - c.bwd[2])
