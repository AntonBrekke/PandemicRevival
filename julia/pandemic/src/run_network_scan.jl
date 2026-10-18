"""
Relic roots of the species-resolved network at every (m_N, y) of a set of seed roots,
starting from the seed's sin^2 2theta. Threaded over points.

    julia --project=. -t 8 src/run_network_scan.jl [tables.jls] [out_dir] [seeds.csv]

Default seeds: the roots of the original reduced-solver scan, stored in
tmp/reduced_roots (one m_N_<keV>keV.csv per mass: m_N,y,sin2_2theta,omega_h2,...;
the reduced solver itself was removed on 2026-10-14). seeds.csv (the same columns, e.g.
from plot/fine_g_seeds.py) replaces them. The seed appears in the s2_reduced column.

Output: <out_dir>/network_roots.csv (default tmp/network_scan), one row per
point: the seed, the network root and Omega h^2 there, the status
(converged; bracketed = Omega jumps across the target, root bracketed to 1e-6
in ln sin^2 2theta; degenerate = the dark sector became degenerate at the
seed and the MB network stopped; failed) and the number of solves.
"""

include(joinpath(@__DIR__, "network.jl"))
using Printf

const SCAN_DIR = joinpath(@__DIR__, "..", "tmp", "reduced_roots")

"""All (m_N [GeV], y, sin^2 2theta, Omega h^2) with a finite seed root in `files`."""
function seed_roots(files=[joinpath(SCAN_DIR, f) for f in readdir(SCAN_DIR) if startswith(f, "m_N_") && endswith(f, "keV.csv")])
    pts = Tuple{Float64,Float64,Float64,Float64}[]
    for f in files
        for l in readlines(f)[2:end]
            c = split(l, ',')
            s2 = parse(Float64, c[3])
            isnan(s2) || push!(pts, (parse(Float64, c[1]), parse(Float64, c[2]), s2, parse(Float64, c[4])))
        end
    end
    return pts
end

"""(Omega h^2, status) of one network solve."""
function solve_omega(itp, tT_rel, m, g, s2)
    net = Network(itp, table_params(itp, m, g, s2), tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    sol = try
        solve_network(net, dw, tT_rel; x_end=100.0, maxiters=20_000)
    catch e
        return NaN, :failed
    end
    sol.retcode == DE.ReturnCode.Success && return network_omega_h2(net, sol.u[end], sol.t[end]), :ok
    return NaN, sol.retcode == DE.ReturnCode.Terminated ? :degenerate : :failed
end

"""
Root of ln(Omega/0.12) in ln sin^2 2theta from the seed s2_0: secant steps (at most a factor
e^0.5) until the root is bracketed, then safeguarded false position, or bisection when the upper
end is not a regular solve. A degenerate or failed solve (MaxIters in a strongly overproducing
history) counts as above the target, so the search steps down from a seed that overproduces by
orders of magnitude (another m_phi); after 6 such solves in a row (3 e-folds in sin^2 2theta,
enough for any overproducing seed) it returns their status, since the network then does not
apply at any mixing nearby. Only a successful solve within `tol` of the target is :converged. Omega can jump across the target when the pandemic takes off; the root is then
returned with status :bracketed once the bracket is narrower than 1e-6 in ln sin^2 2theta, or
:degenerate (:failed) if the upper end of that bracket is a degenerate (failed) solve, i.e. the
root lies where the network does not apply. Returns (s2, Omega, status, number of solves).
"""
function network_root(itp, tT_rel, m, g, s2_0; tol=1e-3, maxit=30)
    lnf(om) = log(om / omega_d0)
    lo = (-Inf, -Inf, NaN, :ok)       # (ln s2, f, Omega, status) with f < 0
    hi = (Inf, Inf, NaN, :ok)         # f > 0, or a degenerate / failed solve (f = Inf)
    l, lp, fp = log(s2_0), NaN, NaN
    nbad = 0
    for nsolve in 1:maxit+1
        om, st = solve_omega(itp, tT_rel, m, g, exp(l))
        nbad = st === :ok ? 0 : nbad + 1
        nbad >= 6 && return exp(l), NaN, st, nsolve
        f = st === :ok ? lnf(om) : Inf
        abs(f) < tol && return exp(l), om, :converged, nsolve
        f < 0 ? (lo = (l, f, om, st)) : (hi = (l, f, om, st))
        if isfinite(lo[1]) && isfinite(hi[1])
            w = hi[1] - lo[1]
            if w < 1e-6
                hi[4] === :ok || return exp(hi[1]), NaN, hi[4], nsolve
                return exp((lo[1] + hi[1]) / 2), (lo[3] + hi[3]) / 2, :bracketed, nsolve
            end
            ln = isfinite(hi[2]) ? lo[1] - lo[2] * w / (hi[2] - lo[2]) : (lo[1] + hi[1]) / 2
            ln = clamp(ln, lo[1] + 0.1w, hi[1] - 0.1w)       # keep shrinking the bracket
        elseif isfinite(f) && isfinite(fp)
            ln = clamp(l - f * (l - lp) / (f - fp), l - 0.5, l + 0.5)
        elseif nsolve == 1 && isfinite(f)
            ln = l - 0.1 * sign(f)                            # first step from the seed
        else
            ln = l - 0.5 * sign(f)                            # above the target without a lower end: step down
        end
        lp, fp, l = l, f, ln
    end
    return exp(l), NaN, :failed, maxit + 1
end

function main(args)
    path = length(args) > 0 ? args[1] : rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))
    out_dir = length(args) > 1 ? args[2] : joinpath(@__DIR__, "..", "tmp", "network_scan")
    mkpath(out_dir)
    itp = RateInterpolants(load_rate_tables(path))
    tT_rel = TimeTempRelation{Float64}()
    pts = length(args) > 2 ? seed_roots([args[3]]) : seed_roots()
    # cheap points first is irrelevant; interleave masses so progress is representative
    sort!(pts; by=p -> (p[2], p[1]))
    # NETWORK_SCAN_LIMIT=n: every k-th point only, n in total (smoke tests)
    if haskey(ENV, "NETWORK_SCAN_LIMIT")
        n = parse(Int, ENV["NETWORK_SCAN_LIMIT"])
        pts = pts[round.(Int, range(1, length(pts), length=n))]
    end
    println(length(pts), " points on ", Threads.nthreads(), " threads"); flush(stdout)
    res = Vector{Any}(undef, length(pts))
    done = Threads.Atomic{Int}(0)
    t0 = time()
    Threads.@threads :dynamic for i in eachindex(pts)
        m, y, s2r, omr = pts[i]
        t = @elapsed (s2n, omn, st, ns) = network_root(itp, tT_rel, m, y, s2r)
        res[i] = (m, y, s2r, omr, s2n, omn, st, ns, t)
        k = Threads.atomic_add!(done, 1) + 1
        k % 20 == 0 && (@printf("%d/%d done, %.1f min\n", k, length(pts), (time() - t0) / 60); flush(stdout))
    end
    out = joinpath(out_dir, "network_roots.csv")
    open(out, "w") do io
        println(io, "m_N,y,s2_reduced,omega_reduced,s2_network,omega_network,status,n_solves,seconds")
        for r in sort(res; by=r -> (r[1], r[2]))
            @printf(io, "%.6g,%.6g,%.6g,%.5g,%.6g,%.5g,%s,%d,%.1f\n", r...)
        end
    end
    println("written ", out, " after ", round((time() - t0) / 60; digits=1), " min")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
