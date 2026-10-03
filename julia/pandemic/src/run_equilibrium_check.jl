"""
run_equilibrium_check.jl

Rates/H of every process along solved relic points (src/equilibrium_check.jl,
notes/equilibrium_check.tex).

    julia --project=. src/run_equilibrium_check.jl --benchmarks [--out DIR] [--nproc N]
        fixed benchmark points, for m_phi = 3, 10, 100 m_N: one CSV per point
        and m_phi with the full history (default DIR tmp/equilibrium)
    julia --project=. src/run_equilibrium_check.jl --scan DIR [--nproc N]
        every freeze-in root of the theta scan in DIR (as run_lyman_alpha.jl),
        m_phi = 100 m_N: one summary row per root in DIR/equilibrium.csv
        (resumable: rows already present are skipped)

Parameters beyond (m_N, g, sin^2 2theta): m_nu = 0.1 eV, m_0 = 1 TeV, m_A' = 2.5 m_N
(rate_params), as decided on 2026-10-03.
"""

using Distributed

function parse_opts(args)
    o = Dict{String, String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--benchmarks"
            o["mode"] = "benchmarks"; i += 1
        elseif a == "--scan"
            o["mode"] = "scan"; o["scan"] = args[i+1]; i += 2
        elseif a in ("--out", "--nproc")
            o[a[3:end]] = args[i+1]; i += 2
        else
            error("unknown argument $a")
        end
    end
    haskey(o, "mode") || error("give --benchmarks or --scan DIR")
    return o
end

read_rows(path) = (ls = filter(!isempty, strip.(readlines(path))); cols = split(ls[1], ",");
                   [Dict(zip(cols, split(l, ","))) for l in ls[2:end]])

"Freeze-in roots (m_N [GeV], y, sin2_2theta) as in run_lyman_alpha.jl."
function scan_roots(dir)
    roots = Dict{Tuple{Float64,Float64}, Float64}()
    for f in filter(f -> occursin(r"^m_N_.*keV\.csv$", f), readdir(dir))
        for r in read_rows(joinpath(dir, f))
            (r["converged"] == "true" && r["plateau_ok"] == "true" && parse(Int, r["slope"]) > 0) || continue
            m, y, s = parse(Float64, r["m_N"]), parse(Float64, r["y"]), parse(Float64, r["sin2_2theta"])
            isfinite(s) || continue
            k = (m, y)
            (!haskey(roots, k) || s < roots[k]) && (roots[k] = s)
        end
    end
    return sort([(k[1], k[2], v) for (k, v) in roots])
end

const BENCH_MASSES_KEV = (1.703, 10.57, 119.4)
const BENCH_Y = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
const BENCH_MPHI = (3.0, 10.0, 100.0)

opts = parse_opts(ARGS)
nproc = parse(Int, get(opts, "nproc", "12"))
if opts["mode"] == "benchmarks"
    scan = joinpath(@__DIR__, "../tmp/relic_scan_theta_merged")
    roots = scan_roots(scan)
    pick = [(m, y, s) for (m, y, s) in roots
            if any(mk -> isapprox(m * 1e6, mk; rtol=1e-3), BENCH_MASSES_KEV) && any(yb -> isapprox(y, yb; rtol=1e-6), BENCH_Y)
               && (isapprox(m * 1e6, 10.57; rtol=1e-3) || isapprox(y, 1e-3; rtol=1e-6))]
    jobs = [(m, y, s, r) for (m, y, s) in pick for r in BENCH_MPHI]
    outdir = get(opts, "out", joinpath(@__DIR__, "../tmp/equilibrium"))
    mkpath(outdir)
else
    roots = scan_roots(opts["scan"])
    out = joinpath(opts["scan"], "equilibrium.csv")
    done = Set{Tuple{Float64,Float64}}()
    if isfile(out)
        for r in read_rows(out)
            push!(done, (parse(Float64, r["m_N"]), parse(Float64, r["y"])))
        end
    end
    jobs = [(m, y, s, 100.0) for (m, y, s) in roots if !((m, y) in done)]
end
println("$(length(jobs)) jobs ($(opts["mode"]))"); flush(stdout)
isempty(jobs) && exit(0)

addprocs(min(nproc, length(jobs)); exeflags=["--project=$(Base.active_project())", "--threads=1"])
const CODE = joinpath(@__DIR__, "equilibrium_check.jl")
@everywhere begin
    include($CODE)
    LA.BLAS.set_num_threads(1)
    const _TT = Ref{Any}(nothing)
    function eq_job(m_N, y, s, mphi_ratio, mode)
        _TT[] === nothing && (_TT[] = TimeTempRelation{Float64}())
        pan = PandemolatorZ{Float64}(ModelParams{Float64}(1e-4, 1e-6),
            Particle{Float64}(m_N, 1, dof=2), Particle{Float64}(m_N, 1, dof=2),
            Particle{Float64}(2.5 * m_N, -1, dof=3), Particle{Float64}(0.0, 1, dof=2), _TT[])
        sol, omega, conv, rc = solve_relic_point(pan, _TT[], y, asin(sqrt(s)) / 2; reltol=1e-4)
        sol === nothing && return nothing
        p = rate_params(m_N, y, s; mphi_over_mN=mphi_ratio)
        hist = equilibrium_history(pan, sol, p; n_z=mode == "benchmarks" ? 120 : 50)
        return (omega=omega, hist=hist, summary=equilibrium_summary(pan, sol, hist))
    end
end

fmt(x) = x isa Real ? string(Float64(x)) : string(x)
t0 = time()
results = Dict{Any, Any}()
lk = ReentrantLock()
todo = copy(jobs)
@sync for w in workers()
    @async while true
        job = lock(lk) do
            isempty(todo) ? nothing : popfirst!(todo)
        end
        job === nothing && break
        m, y, s, r = job
        res = try
            remotecall_fetch(eq_job, w, m, y, s, r, opts["mode"])
        catch e
            println("m_N = $(m * 1e6) keV, y = $y FAILED: ", sprint(showerror, e)); flush(stdout)
            continue
        end
        res === nothing && (println("m_N = $(m * 1e6) keV, y = $y: no solution"); continue)
        lock(lk) do
            if opts["mode"] == "benchmarks"
                f = joinpath(outdir, "m_N_$(round(m * 1e6, sigdigits=4))keV_y_$(y)_mphi_$(Int(r))mN.csv")
                ks = keys(res.hist[1])
                open(f, "w") do io
                    println(io, "# m_N=$m y=$y sin2_2theta=$s mphi_over_mN=$r omega_h2=$(res.omega)")
                    println(io, join(ks, ","))
                    for h in res.hist
                        println(io, join((fmt(getfield(h, k)) for k in ks), ","))
                    end
                end
                open(f * ".summary", "w") do io
                    for (k, v) in pairs(res.summary)
                        println(io, k, ",", fmt(v))
                    end
                end
            else
                ks = keys(res.summary)
                new = !isfile(out)
                open(out, "a") do io
                    new && println(io, "m_N,y,sin2_2theta,omega_h2,", join(ks, ","))
                    println(io, "$m,$y,$s,$(res.omega),", join((fmt(getfield(res.summary, k)) for k in ks), ","))
                end
            end
            results[job] = true
            println("[$(length(results))/$(length(jobs)), $(round((time() - t0) / 60, digits=1)) min] m_N = $(round(m * 1e6, sigdigits=4)) keV, y = $y, m_phi = $(r) m_N"); flush(stdout)
        end
    end
end
println("done in $(round((time() - t0) / 60, digits=1)) min")
