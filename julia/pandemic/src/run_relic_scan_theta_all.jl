"""
run_relic_scan_theta_all.jl

Parallel driver for relic_scan_theta_z.jl: for every mass m_N and coupling y,
root-find sin^2(2 theta) for Omega h^2 = 0.12. Results are written per mass to
<out>/m_N_<m_N in keV>keV.csv (the format plot/relic_scan_theta.py reads), with
the solver output in m_N_<m_N>keV.log (m_N_<m_N>keV_chunk<j>.log with
--y-chunks > 1) next to it.

Usage (from julia/pandemic):
    julia --project=. src/run_relic_scan_theta_all.jl [options]

Options: see USAGE below (or --help).

Example:
    julia --project=. src/run_relic_scan_theta_all.jl --masses 1.5:250:12 --ys 1e-6:1e-2:9 --nproc 12

PARALLELISATION
===============
Worker processes (Distributed), not threads: every worker loads the code once,
builds its own TimeTempRelation once and keeps its own lazily-initialised
caches, so nothing is shared between solves running at the same time.

The unit of work is (mass, chunk of couplings). Within a chunk the couplings
are scanned in increasing order, since each search starts from a root
predicted from the previous couplings. The first coupling of every chunk has no
prediction and falls back to the coarse grid, which costs more solves, so
--y-chunks > 1 only pays off if there are more workers than masses.

Every finished point is printed and written to the CSV of its mass right away,
so the progress can be followed and partial results survive an interrupted run.
"""

using Distributed

const USAGE = """
Usage: julia --project=. src/run_relic_scan_theta_all.jl [options]
    --masses LIST     masses in keV: "1.5,4,10" or "min:max:n" (log-spaced)
    --ys LIST         couplings y:   "1e-6,1e-5" or "min:max:n" (log-spaced)
    --nproc N         number of worker processes (default: number of CPU threads)
    --y-chunks K      split the couplings of each mass into K parallel chunks (default 1)
    --reltol R        ODE rtol for the whole search (default: ScanConfigThetaZ / RELIC_ODE_RELTOL)
    --out DIR         output directory (default tmp/relic_scan_theta)
"""

const DEFAULT_MASSES_KEV = [1.5, 2.5, 4, 6.5, 10, 16, 25, 40, 65, 100, 160, 250]
const DEFAULT_YS = [1e-6, 3.16e-6, 1e-5, 3.16e-5, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2]

"""
"a,b,c" -> [a, b, c]; "min:max:n" -> n log-spaced values (rounded to 4 significant
digits, so that file names stay readable).
"""
function parse_grid(s::AbstractString)
    if occursin(":", s)
        parts = split(s, ":")
        length(parts) == 3 || error("expected min:max:n, got \"$s\"")
        lo, hi, n = parse(Float64, parts[1]), parse(Float64, parts[2]), parse(Int, parts[3])
        (lo > 0 && hi > 0 && n >= 1) || error("need min, max > 0 and n >= 1 in \"$s\"")
        vals = n == 1 ? [lo] : exp10.(range(log10(lo), log10(hi), length=n))
        return round.(vals, sigdigits=4)
    end
    return parse.(Float64, split(s, ","))
end

function parse_args(args)
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        key = args[i]
        if key in ("-h", "--help")
            print(USAGE)
            exit(0)
        end
        (startswith(key, "--") && i < length(args)) || error("unexpected argument \"$key\" (see --help)")
        opts[key[3:end]] = args[i+1]
        i += 2
    end
    unknown = setdiff(keys(opts), ["masses", "ys", "nproc", "y-chunks", "reltol", "out"])
    isempty(unknown) || error("unknown option(s): $(join("--" .* unknown, ", "))")

    reltol = haskey(opts, "reltol") ? parse(Float64, opts["reltol"]) :
             haskey(ENV, "RELIC_ODE_RELTOL") ? parse(Float64, ENV["RELIC_ODE_RELTOL"]) : nothing
    return (
        masses=sort(unique(haskey(opts, "masses") ? parse_grid(opts["masses"]) : Float64.(DEFAULT_MASSES_KEV))),
        ys=sort(unique(haskey(opts, "ys") ? parse_grid(opts["ys"]) : DEFAULT_YS)),
        nproc=haskey(opts, "nproc") ? parse(Int, opts["nproc"]) : Sys.CPU_THREADS,
        y_chunks=haskey(opts, "y-chunks") ? parse(Int, opts["y-chunks"]) : 1,
        reltol=reltol,
        out=get(opts, "out", joinpath(@__DIR__, "../tmp/relic_scan_theta")),
    )
end

"Split sorted `ys` into at most `k` contiguous chunks of (nearly) equal length."
function split_chunks(ys, k)
    k = clamp(k, 1, length(ys))
    bounds = round.(Int, range(0, length(ys), length=k + 1))
    return [ys[bounds[j]+1:bounds[j+1]] for j in 1:k]
end

csv_path(out, m_keV) = joinpath(out, "m_N_$(m_keV)keV.csv")
# One log per job: concurrent appends from several processes to one file overwrite each other.
log_path(out, m_keV, j, n_chunks) =
    joinpath(out, n_chunks == 1 ? "m_N_$(m_keV)keV.log" : "m_N_$(m_keV)keV_chunk$(j).log")

"""
Runs all jobs on the workers. Workers send every finished point through
`progress`; for each, the main process prints a line and rewrites the CSV of
that mass.
"""
function run_jobs(opts, chunks, units)
    results = Dict(m => RelicPointThetaZ[] for m in opts.masses)
    n_done = Dict(m => 0 for m in opts.masses)
    n_total = length(opts.masses) * length(opts.ys)
    queue = Channel{eltype(units)}(length(units))
    foreach(u -> put!(queue, u), units)
    close(queue)
    progress = RemoteChannel(() -> Channel{Any}(Inf))

    t_start = time()
    elapsed() = "$(round((time() - t_start) / 60, digits=1)) min"

    printer = @async while true
        msg = take!(progress)
        msg === nothing && break
        m, points, n_solves, t = msg
        append!(results[m], points)
        sort!(results[m], by=q -> (q.y, q.branch))
        save_results_theta_z(copy(results[m]), csv_path(opts.out, m))
        n_done[m] += 1
        roots = points[1].retcode == :NoRoot ? "no root" :
            join(["sin2_2theta = $(round(q.sin2_2theta, sigdigits=4)) (Omega h^2 = $(round(q.omega_h2, sigdigits=4))" *
                  (q.converged ? "" : ", not converged") * ")" for q in points], ", ")
        println("[$(sum(values(n_done)))/$n_total, $(elapsed())] m_N = $m keV, y = $(points[1].y): $roots, " *
                "$n_solves solves, $(round(t / 60, digits=1)) min" *
                (n_done[m] == length(opts.ys) ? " (mass done)" : ""))
        flush(stdout)
    end

    @sync for p in workers()
        @async for (m, j, ys) in queue
            try
                remotecall_fetch(scan_job, p, m, ys, opts.reltol, log_path(opts.out, m, j, length(chunks)), progress)
            catch e
                println("m_N = $m keV, y-chunk $j/$(length(chunks)) FAILED on worker $p: " * sprint(showerror, e))
                flush(stdout)
                # A dead worker would fail every job it is handed: leave the queue to the others.
                e isa ProcessExitedException && break
            end
        end
    end
    put!(progress, nothing)
    wait(printer)

    incomplete = [m for m in opts.masses if n_done[m] < length(opts.ys)]
    println("all jobs finished in $(elapsed())" *
            (isempty(incomplete) ? "" : "; incomplete masses [keV]: $incomplete"))
end

# --- setup at top level, so that run_jobs sees the code loaded by @everywhere ---
opts = parse_args(ARGS)
chunks = split_chunks(opts.ys, opts.y_chunks)
units = [(m, j, c) for m in opts.masses for (j, c) in enumerate(chunks)]
n_workers = clamp(opts.nproc, 1, length(units))

println("masses [keV]: $(opts.masses)")
println("couplings y:  $(opts.ys)")
println("$(length(units)) jobs ($(length(chunks)) y-chunk(s) per mass) on $n_workers worker(s) -> $(opts.out)")
flush(stdout)

mkpath(opts.out)

addprocs(n_workers; exeflags=["--project=$(Base.active_project())", "--threads=1"])
const CODE = joinpath(@__DIR__, "relic_scan_theta_z.jl")
@everywhere begin
    using Logging
    include($CODE)
    LA.BLAS.set_num_threads(1)

    const _TT_REL = Ref{Any}(nothing)

    "Scans the couplings `ys` at one mass on this worker, logging to `log` and sending each point to `progress`."
    function scan_job(m_N_keV::Float64, ys::Vector{Float64}, reltol, log::String, progress)
        cfg = reltol === nothing ? ScanConfigThetaZ() :
              ScanConfigThetaZ(ode_reltol=reltol, ode_reltol_coarse=max(reltol, 1e-4))
        _TT_REL[] === nothing && (_TT_REL[] = TimeTempRelation{Float64}())
        open(log, "w") do io
            with_logger(SimpleLogger(io)) do
                run_scan_theta_z(m_N_keV * 1e-6, ys, cfg; tT_rel=_TT_REL[], io=io,
                    on_point=(points, n_solves, t) -> put!(progress, (m_N_keV, points, n_solves, t)))
            end
        end
    end
end

run_jobs(opts, chunks, units)
rmprocs(workers())
