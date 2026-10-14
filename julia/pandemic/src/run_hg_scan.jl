"""
Relic roots of the high-g one-fluid solver (pandemolator_hg.jl) at every
(m_N, y) of the merged reduced-solver scan with y >= y_min, starting from the
reduced root. Distributed over worker processes (one TimeTempRelation and one
table set per worker).

    julia --project=. src/run_hg_scan.jl [--nproc 8] [--ymin 1e-3] [--out tmp/hg_scan] [--limit n] [--points network_roots.csv]
                                         [--only m_keV:y,...] [--portal-scale 1.0]

Output: <out>/hg_roots.csv, one row per point: the reduced root, the high-g root
and Omega h^2 there, status (converged / failed), number of solves, seconds, and
diagnostics of the solution at the root: the largest T_N/T_nu, and the smallest
ratio over x <= 3 of the slowest dark-internal rate (A'A' <-> NN per N, at
mu = 0) to H, which must be >> 1 for the one-fluid assumption.
Rows are appended as points finish (`hg_roots_partial.csv`), so an interrupted
run keeps its results.
"""

using Distributed
using Printf

function parse_opts(args)
    o = Dict("nproc" => "8", "ymin" => "1e-3", "out" => joinpath(@__DIR__, "..", "tmp", "hg_scan"), "limit" => "0",
             "portal-scale" => "1.0", "only" => "",
             "points" => joinpath(@__DIR__, "..", "tmp", "network_scan", "network_roots.csv"))
    i = 1
    while i <= length(args)
        k = args[i][3:end]
        haskey(o, k) || error("unknown option $(args[i])")
        o[k] = args[i+1]
        i += 2
    end
    return o
end

const OPTS = parse_opts(ARGS)
addprocs(parse(Int, OPTS["nproc"]); exeflags="--project=$(Base.active_project())")

@everywhere begin
    include(joinpath(@__DIR__, "pandemolator_hg.jl"))
    include(joinpath(@__DIR__, "relic_scan_z.jl"))
    const TT_REL = TimeTempRelation{Float64}()
    const ITP = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
    const PANS = Dict{Float64,Any}()
    const PORTAL_SCALE = Ref(1.0)
    hg_pan(m) = get!(() -> PandemolatorHG(m, TT_REL, ITP; portal_scale=PORTAL_SCALE[]), PANS, m)

    """
    (Omega h^2, solution, plateau) of one high-g solve. Omega h^2 is the final value even if it
    has not settled (the root search uses its sign); NaN if the solve failed.
    """
    function hg_omega(m, g, s2; reltol=1e-4, wall=900.0)
        pan = hg_pan(m)
        th = asin(sqrt(s2)) / 2
        pan.mp = ModelParams{Float64}(g, th)
        pan.deadline = time() + wall
        try
            # as theta_point: a longer solve where Omega h^2 is still moving at x = 100 (at
            # large g the reverse portal reactions deplete it slowly)
            om, last = NaN, nothing
            for x_end in (100.0, 300.0, 1000.0)
                dw = DodelsonWidrow{Float64}(m, th, TT_REL; x_end=x_end)
                sol = pandemolate_z(TT_REL, dw, pan; reltol=reltol)
                DE.successful_retcode(sol) || return NaN, nothing, false
                om, last = final_omega_h2_z(pan, sol), sol
                # Omega h^2 wobbles at ~1e-3 at reltol 1e-4 once the table rates are on (root tolerance 1e-2)
                check_plateau_z(pan, sol; tol=5e-3) && return om, sol, true
            end
            return om, last, false
        catch e
            e isa SolveDeadlineExceeded || @warn "hg solve failed at m=$m g=$g s2=$s2" exception = e
            return NaN, nothing, false
        finally
            pan.deadline = Inf
        end
    end

    """Diagnostics of a solution: max T_N/T_nu, min over x <= 3 of Gamma(A'A' <-> NN per N at mu = 0)/H."""
    function hg_diagnostics(m, sol)
        pan = hg_pan(m)
        tmax, rmin = 0.0, Inf
        for z in range(sol.t[1], sol.t[end], length=200)
            u = sol(z)
            _, T_N, _, _ = hg_state(pan, u)
            tmax = max(tmax, T_N * exp(z) / m)
            if exp(z) <= 3
                # reaction rate of N N -> A'A' per N at the actual T_N and mu = 0, from the tables
                hg_rates!(pan)
                lnR, _ = pan.norms[:N1N1_AA]
                t = pan.itp.dark[:N1N1_AA]
                g1 = pan.N1.m / T_N
                Rf = exp(t.lnRf(log(m / T_N)) + lnR - 2g1)
                nN = n_scaled_safe(pan.N1.m, 2, T_N) * exp(-g1)
                rmin = min(rmin, 2Rf / nN / pan.H_interp_z(z))
            end
        end
        return tmax, rmin
    end

    """
    Freeze-in root (the smallest sin^2 2theta at which Omega h^2 rises through 0.12) from s2_0.
    At large g Omega h^2 is not monotonic in theta: the reverse portal reactions (N N -> N nu)
    deplete it more at larger mixing, so above its maximum it falls again. The search therefore
    only moves down from a point with Omega > 0.12 and up from one with Omega < 0.12, keeps the
    bracket lo < hi with f(lo) < 0 < f(hi), and refines it by false position in ln s2.
    Status: converged (|ln(Omega/0.12)| < tol with Omega settled), bracketed (Omega jumps across
    the target: bracket < 1e-3 in ln s2), noplateau (|f| < tol, Omega still moving at x = 1000),
    failed.
    """
    function hg_root(m, g, s2_0; tol=1e-2, maxit=25)
        lnf(om) = log(om / omega_d0)
        lo = (-Inf, NaN, NaN)          # (ln s2, f, Omega), f < 0
        hi = (Inf, NaN, NaN)           # f > 0
        l = log(s2_0)
        nsolve = 0
        for _ in 1:maxit
            om, sol, plat = hg_omega(m, g, exp(l))
            nsolve += 1
            isfinite(om) || return exp(l), NaN, :failed, nsolve, nothing
            f = lnf(om)
            abs(f) < tol && return exp(l), om, plat ? :converged : :noplateau, nsolve, sol
            if f > 0
                l < hi[1] && (hi = (l, f, om))
            else
                (l > lo[1] && l < hi[1]) && (lo = (l, f, om))
            end
            if isfinite(lo[1]) && isfinite(hi[1])
                w = hi[1] - lo[1]
                w < 1e-3 && return exp((lo[1] + hi[1]) / 2), (lo[3] + hi[3]) / 2, :bracketed, nsolve, nothing
                l = clamp(lo[1] - lo[2] * w / (hi[2] - lo[2]), lo[1] + 0.05w, hi[1] - 0.05w)
            elseif isfinite(hi[1])
                # d ln Omega / d ln s2 ~ 5 on the freeze-in branch
                l = hi[1] - clamp(hi[2] / 5, 0.3, 3.0)
            else
                l = lo[1] + clamp(-lo[2] / 5, 0.3, 3.0)
            end
        end
        return exp(l), NaN, :failed, nsolve, nothing
    end

    function hg_point(pt)
        m, y, s2r, omr, s2net, stnet = pt
        t = @elapsed (s2, om, st, ns, sol) = hg_root(m, y, s2r)
        tmax, rmin = sol === nothing ? (NaN, NaN) : hg_diagnostics(m, sol)
        return (m, y, s2r, omr, s2net, stnet, s2, om, st, ns, t, tmax, rmin)
    end
end

"""
Reduced roots (m, y, s2, Omega) with the network root and status, from --points
(default tmp/network_scan/network_roots.csv; plot/fine_g_seeds.py writes seeds in its format).
"""
function scan_points(ymin)
    pts = []
    for l in readlines(OPTS["points"])[2:end]
        c = split(l, ',')
        m, y, s2r, omr, s2n = parse.(Float64, c[1:5])
        (y >= ymin * (1 - 1e-3) && isfinite(s2r)) || continue
        push!(pts, (m, y, s2r, omr, s2n, c[7]))
    end
    return pts
end

function main()
    out = OPTS["out"]
    mkpath(out)
    pts = scan_points(parse(Float64, OPTS["ymin"]))
    # largest couplings and masses first: the slowest points start early
    sort!(pts; by=p -> (-p[2], -p[1]))
    # --only m_keV:y,...: just these points
    if !isempty(OPTS["only"])
        want = [parse.(Float64, split(w, ':')) for w in split(OPTS["only"], ',')]
        pts = filter(p -> any(w -> isapprox(p[1] * 1e6, w[1]; rtol=1e-3) && isapprox(p[2], w[2]; rtol=1e-3), want), pts)
    end
    ps = parse(Float64, OPTS["portal-scale"])
    @everywhere PORTAL_SCALE[] = $ps
    n = parse(Int, OPTS["limit"])
    n > 0 && (pts = pts[round.(Int, range(1, length(pts), length=n))])
    println(length(pts), " points on ", nworkers(), " workers"); flush(stdout)
    header = "m_N,y,s2_reduced,omega_reduced,s2_network,status_network,s2_hg,omega_hg,status,n_solves,seconds,TN_Tnu_max,rate_AA_min_over_H"
    partial = joinpath(out, "hg_roots_partial.csv")
    open(io -> println(io, header), partial, "w")
    t0 = time()
    fmt(r) = @sprintf("%.6g,%.6g,%.6g,%.5g,%.6g,%s,%.6g,%.5g,%s,%d,%.1f,%.4g,%.4g", r...)
    res = []
    next = Ref(0)
    @sync for w in workers()
        @async while true
            i = (next[] += 1)          # tasks on the main process are cooperative: no race
            i > length(pts) && break
            r = remotecall_fetch(hg_point, w, pts[i])
            push!(res, r)
            open(io -> println(io, fmt(r)), partial, "a")
            @printf("%d/%d  m=%.4g keV y=%.4g: %s, s2 %.4g -> %.4g (%d solves, %.0f s), %.1f min\n", length(res), length(pts),
                    r[1] * 1e6, r[2], r[9], r[3], r[7], r[10], r[11], (time() - t0) / 60)
            flush(stdout)
        end
    end
    open(joinpath(out, "hg_roots.csv"), "w") do io
        println(io, header)
        for r in sort(res; by=r -> (r[1], r[2]))
            println(io, fmt(r))
        end
    end
    println("done after ", round((time() - t0) / 60; digits=1), " min"); flush(stdout)
end

main()
