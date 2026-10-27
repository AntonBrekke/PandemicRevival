"""
Relic roots of the one-fluid solver (fluid.jl) at the points of a seed file with
y >= ymin, starting from the seed's sin^2 2theta. Distributed over worker processes
(one TimeTempRelation and one table set per worker).

    julia --project=. src/run_fluid_scan.jl [--nproc 8] [--ymin 1e-3] [--out tmp/fluid_scan]
        [--points tmp/network_scan/network_roots.csv] [--only m_keV:y,...] [--portal-scale 1.0]
        [--tables tmp/rate_tables/rate_tables_mA2.5_mphi100.0.jls] [--min-step 0.3]

--min-step: smallest step in ln sin^2 2theta before the root is bracketed (`fluid_root`);
smaller than the default 0.3 for seeds that are already close (e.g. 0.03 for seeds within ~5%).
--points: CSV with columns m_N,y,s2_reduced,omega_reduced,s2_network,..,status (the
network scan's output, whose s2_reduced is the reduced-solver root; plot/fine_g_seeds.py
writes seeds in the same format). Output: <out>/fluid_roots.csv, one row per point: the
seed, the network root and status, the fluid root and Omega h^2 there, status, number of
solves, seconds, and diagnostics of the solution at the root: the largest T_N/T_nu, and
the smallest ratio over x <= 3 of the A'A' <-> NN rate per N (at mu = 0) to H. That
process is in the collision term with its actual rate; the ratio says whether it can hold
the fluid at mu = 0 (number-changing equilibrium), not whether the one-fluid assumptions
(kinetic equilibrium, mu_N1 = mu_N2, fast A' and phi decays) hold. Rows are also appended to
<out>/fluid_roots_partial.csv as points finish.
"""

using Distributed
using Printf

rate_table_path_default() = joinpath(@__DIR__, "..", "tmp", "rate_tables", "rate_tables_mA2.5_mphi100.0.jls")

function parse_opts(args)
    o = Dict("nproc" => "8", "ymin" => "1e-3", "out" => joinpath(@__DIR__, "..", "tmp", "fluid_scan"),
             "portal-scale" => "1.0", "only" => "", "min-step" => "0.3",
             "tables" => rate_table_path_default(),
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
@eval @everywhere const TABLES = $(OPTS["tables"])

@everywhere begin
    include(joinpath(@__DIR__, "fluid.jl"))
    const TT_REL = TimeTempRelation{Float64}()
    const ITP = RateInterpolants(load_rate_tables(TABLES))
    const PANS = Dict{Float64,Any}()
    const PORTAL_SCALE = Ref(1.0)
    const MIN_STEP = Ref(0.3)
    fluid_pan(m) = get!(() -> FluidSolver(m, TT_REL, ITP; portal_scale=PORTAL_SCALE[]), PANS, m)

    """Diagnostics of a solution: max T_N/T_nu, min over x <= 3 of Gamma(A'A' <-> NN per N at mu = 0)/H (see the header)."""
    function fluid_diagnostics(pan, sol)
        m = pan.N1.m
        tmax, rmin = 0.0, Inf
        for z in range(sol.t[1], sol.t[end], length=200)
            _, T_N, _, _ = fluid_state(pan, sol(z))
            tmax = max(tmax, T_N * exp(z) / m)
            if exp(z) <= 3
                # reaction rate of N N -> A'A' per N at the actual T_N and mu = 0, from the tables
                fluid_rates!(pan)
                lnR, _ = pan.norms[:N1N1_AA]
                g1 = m / T_N
                Rf = exp(pan.itp.dark[:N1N1_AA].lnRf(log(m / T_N)) + lnR - 2g1)
                nN = n_scaled_safe(m, 2, T_N) * exp(-g1)
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
    A failed solve (exception or the wall limit, in practice strongly overproducing histories)
    counts as above the target, so the search steps down from a seed that overproduces by orders
    of magnitude (another m_phi); after 3 failures in a row it gives up.
    Status: converged (|ln(Omega/0.12)| < tol with Omega settled), bracketed (Omega jumps across
    the target: bracket < 1e-3 in ln s2), noplateau (|f| < tol, Omega still moving at x = 1000),
    failed.
    """
    function fluid_root(pan, g, s2_0; tol=1e-2, maxit=25)
        lnf(om) = log(om / omega_d0)
        lo = (-Inf, NaN, NaN)          # (ln s2, f, Omega), f < 0
        hi = (Inf, NaN, NaN)           # f > 0
        l = log(s2_0)
        nsolve = 0
        nfail = 0
        for _ in 1:maxit
            sol, om, plat = solve_relic(pan, TT_REL, g, exp(l))
            nsolve += 1
            nfail = isfinite(om) ? 0 : nfail + 1
            nfail >= 3 && return exp(l), NaN, :failed, nsolve, nothing
            f = isfinite(om) ? lnf(om) : Inf
            abs(f) < tol && return exp(l), om, plat ? :converged : :noplateau, nsolve, sol
            if f > 0
                l < hi[1] && (hi = (l, f, om))
            else
                (l > lo[1] && l < hi[1]) && (lo = (l, f, om))
            end
            if isfinite(lo[1]) && isfinite(hi[1])
                w = hi[1] - lo[1]
                if w < 1e-3
                    isfinite(hi[3]) || return exp(hi[1]), NaN, :failed, nsolve, nothing
                    return exp((lo[1] + hi[1]) / 2), (lo[3] + hi[3]) / 2, :bracketed, nsolve, nothing
                end
                l = isfinite(hi[2]) ? clamp(lo[1] - lo[2] * w / (hi[2] - lo[2]), lo[1] + 0.05w, hi[1] - 0.05w) :
                    (lo[1] + hi[1]) / 2
            elseif isfinite(hi[1])
                # d ln Omega / d ln s2 ~ 5 on the freeze-in branch (a failed solve: the largest step)
                l = hi[1] - clamp(hi[2] / 5, MIN_STEP[], 3.0)
            else
                l = lo[1] + clamp(-lo[2] / 5, MIN_STEP[], 3.0)
            end
        end
        return exp(l), NaN, :failed, nsolve, nothing
    end

    function fluid_point(pt)
        m, y, s2r, omr, s2net, stnet = pt
        pan = fluid_pan(m)
        t = @elapsed (s2, om, st, ns, sol) = fluid_root(pan, y, s2r)
        tmax, rmin = sol === nothing ? (NaN, NaN) : fluid_diagnostics(pan, sol)
        return (m, y, s2r, omr, s2net, stnet, s2, om, st, ns, t, tmax, rmin)
    end
end

"""Seeds (m, y, s2, Omega) with the network root and status, from --points."""
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
    if !isempty(OPTS["only"])
        want = [parse.(Float64, split(w, ':')) for w in split(OPTS["only"], ',')]
        pts = filter(p -> any(w -> isapprox(p[1] * 1e6, w[1]; rtol=1e-3) && isapprox(p[2], w[2]; rtol=1e-3), want), pts)
    end
    ps = parse(Float64, OPTS["portal-scale"])
    @everywhere PORTAL_SCALE[] = $ps
    mstep = parse(Float64, OPTS["min-step"])
    @everywhere MIN_STEP[] = $mstep
    println(length(pts), " points on ", nworkers(), " workers"); flush(stdout)
    header = "m_N,y,s2_reduced,omega_reduced,s2_network,status_network,s2_fluid,omega_fluid,status,n_solves,seconds,TN_Tnu_max,rate_AA_min_over_H"
    partial = joinpath(out, "fluid_roots_partial.csv")
    open(io -> println(io, header), partial, "w")
    t0 = time()
    fmt(r) = @sprintf("%.6g,%.6g,%.6g,%.5g,%.6g,%s,%.6g,%.5g,%s,%d,%.1f,%.4g,%.4g", r...)
    res = []
    next = Ref(0)
    @sync for w in workers()
        @async while true
            i = (next[] += 1)          # tasks on the main process are cooperative: no race
            i > length(pts) && break
            r = remotecall_fetch(fluid_point, w, pts[i])
            push!(res, r)
            open(io -> println(io, fmt(r)), partial, "a")
            @printf("%d/%d  m=%.4g keV y=%.4g: %s, s2 %.4g -> %.4g (%d solves, %.0f s), %.1f min\n", length(res), length(pts),
                    r[1] * 1e6, r[2], r[9], r[3], r[7], r[10], r[11], (time() - t0) / 60)
            flush(stdout)
        end
    end
    open(joinpath(out, "fluid_roots.csv"), "w") do io
        println(io, header)
        foreach(r -> println(io, fmt(r)), sort(res; by=r -> (r[1], r[2])))
    end
    rm(partial)
    println("done after ", round((time() - t0) / 60; digits=1), " min"); flush(stdout)
end

main()
