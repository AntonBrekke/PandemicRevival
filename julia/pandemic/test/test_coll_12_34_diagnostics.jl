include(joinpath(@__DIR__, "../src/coll_12_34.jl"))
import Integrals

"""
Diagnostics for `coll_12_34` (the A A <-> N N channel), which currently does
not produce usable results. Each check REPORTS rather than asserts, so the
suite runs to completion and prints a summary -- an analysis tool, not a
pass/fail regression test.

Process convention follows the call in pandemolator.jl:
    coll_12_34(mp, A, A, N1, N1, temps, xis)
so p1 = p2 = A, p3 = p4 = N, and the squared amplitude uses mA = p1.m,
mN = p3.m.

NOTE on the t-measure: `coll_12_34_ker` is amp_sq / sqrt(a*(t-t_min)*(t_max-t)).
Since  int_{tm}^{tp} dt / sqrt((t-tm)(tp-t)) = pi  regardless of tm, tp, a
limit-INDEPENDENT term such as `term1 = -16*pi/sqrt(a)` is legitimate here,
and a nonzero result on a zero-width range is NOT by itself a bug. The
substitution t = tm + (tp-tm)*sin^2(theta) removes both endpoint
singularities and gives the reliable numerical reference used below:
    int ker dt = (2/sqrt(a)) * int_0^{pi/2} amp_sq(t(theta)) dtheta
"""

const M_N = 1e-5
const M_A = 2.5 * M_N

function _setup()
    A = Particle{Float64}(M_A, -1, dof=3)
    N = Particle{Float64}(M_N, 1, dof=2)
    xi_N = -1.0
    xi_A = 2.0 * xi_N
    T = M_N                       # x = m_N/T = 1
    p = Params_12_34{Float64,Float64}(
        A, A, N, N, (T, T, T, T), (xi_A, xi_A, xi_N, xi_N),
    )
    p.e1 = 1.5 * M_A; p.mom1 = momentum(A, p.e1)
    p.e2 = 1.8 * M_A; p.mom2 = momentum(A, p.e2)
    p.e3 = 1.2 * M_A; p.mom3 = momentum(N, p.e3)
    p.e4 = p.e1 + p.e2 - p.e3; p.mom4 = momentum(N, p.e4)
    return p, A, N
end

"""Reliable numerical value of the t-integral over an explicit [tm, tp],
via t = tm + (tp-tm)sin^2(theta) (removes the 1/sqrt endpoint singularities)."""
function t_int_numeric(s, p, tm, tp)
    p.s = s
    p.a = a_theta(p)
    f(theta, q) = coll_12_34_sq_amp(tm + (tp - tm) * sin(theta)^2, q)
    sol = Integrals.solve(
        Integrals.IntegralProblem(f, (0.0, pi / 2), p),
        Integrals.QuadGKJL(); reltol=1e-10, abstol=1e-80,
    )
    return 2 / sqrt(p.a) * sol.u
end

"""`cos_theta_lim` uses the discriminant `b^2 + 4ac` ([coll_12_34.jl:99]);
the standard quadratic discriminant is `b^2 - 4ac`. Because a*c < 0 here,
`b^2 - 4ac` is guaranteed positive (always-real roots, as the kinematics
require) while `b^2 + 4ac` goes negative and is clamped to zero."""
function diag_discriminant(p; n=200)
    smin, smax = s_min(p), s_max(p)
    n_ac, n_plus, n_minus = 0, 0, 0
    for i in 1:n
        p.s = smin + (i - 0.5) / n * (smax - smin)
        a, b, c = a_theta(p), b_theta(p), c_theta(p)
        a * c < 0 && (n_ac += 1)
        b^2 + 4a * c < 0 && (n_plus += 1)
        b^2 - 4a * c < 0 && (n_minus += 1)
    end
    println("[1] Discriminant sign in cos_theta_lim")
    println("    a*c < 0                 : $n_ac / $n")
    println("    current  b^2 + 4ac < 0  : $n_plus / $n   (clamped to 0 -> degenerate roots)")
    println("    standard b^2 - 4ac < 0  : $n_minus / $n")
end

"""A usable definite integral needs t_min < t_max."""
function diag_t_range(p; n=200)
    smin, smax = s_min(p), s_max(p)
    npos, nzero, nneg = 0, 0, 0
    for i in 1:n
        p.s = smin + (i - 0.5) / n * (smax - smin); p.a = a_theta(p)
        w = t_lim(1, p) - t_lim(-1, p)
        w == 0 ? (nzero += 1) : (w < 0 ? (nneg += 1) : (npos += 1))
    end
    println("[2] t-range width, t_lim(+1) - t_lim(-1)")
    println("    > 0 (usable)   : $npos / $n")
    println("    == 0 (empty)   : $nzero / $n")
    println("    < 0 (inverted) : $nneg / $n")
end

"""`a_theta = 4*mom3^2*((e1+e2)^2 - s)` vanishes at mom3 = 0, i.e. exactly at
the lower limit `e3_min = p3.m` of the e3 integral. `term1 = -16pi/sqrt(a)`
then diverges; Inf - Inf gives the NaN that aborts the full evaluation."""
function diag_a_endpoint(p, N)
    println("[3] a_theta -> 0 at the e3 integration endpoint (Inf/NaN source)")
    e3_min = p.p3.m
    saved = p.e3
    for fac in (1.0, 1.0 + 1e-12, 1.0 + 1e-6, 1.1)
        p.e3 = fac * e3_min; p.mom3 = momentum(N, p.e3)
        p.e4 = p.e1 + p.e2 - p.e3; p.mom4 = momentum(N, p.e4)
        p.s = 0.5 * (s_min(p) + s_max(p))
        a = a_theta(p)
        t1 = a > 0 ? -(16pi) / sqrt(a) : -Inf
        println("    e3 = $fac * m_N : mom3 = $(p.mom3)   a = $a   term1 = $t1")
    end
    p.e3 = saved; p.mom3 = momentum(N, p.e3)
    p.e4 = p.e1 + p.e2 - p.e3; p.mom4 = momentum(N, p.e4)
end

"""The squared amplitude has double poles at t = mN^2 (t-channel) and
t = mN^2 + 2mA^2 - s (u-channel): on-shell N exchange. A double pole inside
the domain is not integrable and needs a width / real-intermediate-state
subtraction. Evaluated on the CORRECTED range."""
function diag_poles(p; n=200)
    smin, smax = s_min(p), s_max(p)
    nt, nu, nvalid = 0, 0, 0
    for i in 1:n
        p.s = smin + (i - 0.5) / n * (smax - smin); p.a = a_theta(p)
        tm, tp = t_range_fixed(p)
        (isnan(tm) || isnan(tp) || tp <= tm) && continue
        nvalid += 1
        tm <= M_N^2 <= tp && (nt += 1)
        tm <= (M_N^2 + 2M_A^2 - p.s) <= tp && (nu += 1)
    end
    println("[4] Double poles inside the corrected t-range ($nvalid valid points)")
    println("    t-channel  t = mN^2           inside : $nt")
    println("    u-channel  t = mN^2+2mA^2-s   inside : $nu")
end

"""Corrected-discriminant t-range (source untouched)."""
function cos_theta_lim_fixed(pm, q)
    a, b, c = a_theta(q), b_theta(q), c_theta(q)
    D = b^2 - 4a * c
    D < 0 && return NaN
    return clamp((-b + pm * sqrt(D)) / (-2a), -1.0, 1.0)
end

function t_range_fixed(q)
    base = q.p1.m^2 + q.p3.m^2 - 2 * q.e1 * q.e3
    t1 = base + 2 * q.mom1 * q.mom3 * cos_theta_lim_fixed(-1, q)
    t2 = base + 2 * q.mom1 * q.mom3 * cos_theta_lim_fixed(+1, q)
    return min(t1, t2), max(t1, t2)
end

"""The decisive check: with the discriminant sign FIXED (so the t-range is
non-degenerate), does the closed-form `coll_12_34_int_t_new` reproduce direct
numerical integration of the same kernel over the same range? Restricted to
pole-free s so the integral is well defined.

`cos_theta_lim` is redefined below so that `t_lim` -- and hence
`coll_12_34_int_t_new`'s own internal range -- uses the corrected
discriminant, making this a like-for-like comparison."""
function diag_analytic_vs_numeric(p; n=60, nshow=8)
    smin, smax = s_min(p), s_max(p)
    println("[5] closed form vs numeric t-integral, corrected range, pole-free s")
    shown, ncmp = 0, 0
    worst = 0.0
    for i in 1:n
        s = smin + (i - 0.5) / n * (smax - smin)
        p.s = s; p.a = a_theta(p)
        tm, tp = t_range_fixed(p)
        (isnan(tm) || isnan(tp) || tp <= tm) && continue
        upole = M_N^2 + 2M_A^2 - s
        (tm <= upole <= tp || tm <= M_N^2 <= tp) && continue
        num = try t_int_numeric(s, p, tm, tp) catch; NaN end
        ana = try coll_12_34_int_t_new(s, p) catch; NaN end
        (isnan(num) || isnan(ana)) && continue
        ncmp += 1
        rel = abs(num - ana) / max(abs(num), abs(ana), 1e-300)
        rel > worst && (worst = rel)
        if shown < nshow
            shown += 1
            println("    s=$s")
            println("       numeric   = $num")
            println("       closed fm = $ana     rel.diff = $rel")
        end
    end
    println("    compared $ncmp pole-free points; worst relative difference = $worst")
    return worst
end

function diag_end_to_end()
    println("[6] End-to-end coll_12_34")
    A = Particle{Float64}(M_A, -1, dof=3)
    N = Particle{Float64}(M_N, 1, dof=2)
    mp = ModelParams(1e-5, asin(sqrt(1e-11)) / 2)
    xi_N = -1.0; xi_A = 2xi_N
    for x in (1e-1, 1e0, 1e1)
        T = M_N / x
        res = try
            v = coll_12_34(mp, A, A, N, N, (T, T, T, T), (xi_A, xi_A, xi_N, xi_N))
            isfinite(v) ? "$v" : "NON-FINITE ($v)"
        catch e
            "THREW $(typeof(e))"
        end
        println("    x = $x : $res")
    end
end

function test_coll_12_34_diagnostics()
    p, A, N = _setup()
    println("="^72)
    println("coll_12_34 diagnostics  (m_N=$M_N, m_A=$M_A, process A A <-> N N)")
    println("kinematics: e1=$(p.e1) e2=$(p.e2) e3=$(p.e3) e4=$(p.e4)")
    println("="^72, "\n")
    diag_discriminant(p);  println()
    diag_t_range(p);       println()
    diag_a_endpoint(p, N); println()
    diag_poles(p);         println()
    diag_end_to_end();     println()

    # Corrected discriminant (b^2 - 4ac) AND corrected root ordering, so that
    # t_lim(-1) < t_lim(+1). With the `-pm` convention below the ordering comes
    # out right; with `+pm` the range is inverted at every sampled point.
    println("-- redefining cos_theta_lim: discriminant b^2-4ac, ordered roots --\n")
    @eval function cos_theta_lim(pm::Int64, p::Params_12_34{T,R}) where {T<:Real,R<:Real}
        a, b, c = a_theta(p), b_theta(p), c_theta(p)
        D = b^2 - 4a * c
        D < 0 && (D = 0.0)
        return clamp((-b - pm * sqrt(D)) / (-2a), -1.0, 1.0)
    end
    Base.invokelatest(diag_t_range, p);              println()
    Base.invokelatest(diag_analytic_vs_numeric, p);  println()
    Base.invokelatest(diag_end_to_end);              println()
    println("="^72)
    return nothing
end

test_coll_12_34_diagnostics()
