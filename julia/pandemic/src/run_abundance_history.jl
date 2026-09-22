"""
run_abundance_history.jl

Abundance history (as in test/test_pandemolate.jl) for given (m_N, y,
sin^2 2theta), solved with the z-parameterized solver that the relic scan uses.
Each point is written to <out>/m_N_<m_N>keV_y_<y>_sin22th_<sin^2 2theta>.csv
with the columns of `transform_sol`; plot them with plot/abundance_history.py.

Usage (from julia/pandemic):
    julia --project=. src/run_abundance_history.jl <m_N in keV> <y> <sin^2 2theta> [options]
    julia --project=. src/run_abundance_history.jl --scan [DIR] [options]

With --scan, every converged freeze-in root in DIR/m_N_*keV.csv (default
tmp/relic_scan_theta, the output of run_relic_scan_theta_all.jl) is run.
"""

using Distributed

const USAGE = """
Usage: julia --project=. src/run_abundance_history.jl <m_N in keV> <y> <sin^2 2theta> [options]
       julia --project=. src/run_abundance_history.jl --scan [DIR] [options]
    --scan [DIR]   run all converged freeze-in roots of the relic scan in DIR
                   (default tmp/relic_scan_theta)
    --nproc N      number of worker processes (default: number of CPU threads, at most one per point)
    --reltol R     ODE rtol (default 1e-4; at the 1e-3 of the relic scan, single solves can be off by tens of percent)
    --out DIR      output directory (default tmp/abundance_history)
"""

const COLUMNS = [:x_nu, :x_N, :hubble, :ent, :y_n, :y_rho, :xi_N, :xi_A,
                 :y_N1, :y_N2, :y_A, :coll_n, :coll_A_N2nu, :coll_AA_NN]

function parse_args(args)
    positional = String[]
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help")
            print(USAGE)
            exit(0)
        elseif a == "--scan"
            has_dir = i < length(args) && !startswith(args[i+1], "--")
            opts["scan"] = has_dir ? args[i+1] : normpath(joinpath(@__DIR__, "../tmp/relic_scan_theta"))
            i += has_dir ? 2 : 1
        elseif startswith(a, "--")
            (a[3:end] in ("nproc", "reltol", "out") && i < length(args)) || error("bad option \"$a\" (see --help)")
            opts[a[3:end]] = args[i+1]
            i += 2
        else
            push!(positional, a)
            i += 1
        end
    end
    if haskey(opts, "scan")
        isempty(positional) || error("give either --scan or <m_N> <y> <sin^2 2theta>, not both")
        points = scan_points(opts["scan"])
    else
        length(positional) == 3 || error("expected <m_N in keV> <y> <sin^2 2theta> (see --help)")
        m_keV, y, s = parse.(Float64, positional)
        points = [(m_N=m_keV * 1e-6, y=y, sin2_2theta=s, omega_h2=NaN)]
    end
    return (
        points=points,
        nproc=haskey(opts, "nproc") ? parse(Int, opts["nproc"]) : Sys.CPU_THREADS,
        reltol=haskey(opts, "reltol") ? parse(Float64, opts["reltol"]) : 1e-4,
        out=get(opts, "out", normpath(joinpath(@__DIR__, "../tmp/abundance_history"))),
    )
end

"""
Converged freeze-in roots (slope > 0, as in plot/relic_scan_theta.py) of all
m_N_*keV.csv files in `dir`, sorted by (m_N, y).
"""
function scan_points(dir)
    points = NamedTuple{(:m_N, :y, :sin2_2theta, :omega_h2), NTuple{4, Float64}}[]
    for f in filter(f -> occursin(r"^m_N_.*keV\.csv$", f), readdir(dir))
        lines = readlines(joinpath(dir, f))
        header = split(lines[1], ",")
        col(name) = findfirst(==(name), header)
        for line in lines[2:end]
            r = split(line, ",")
            (r[col("converged")] == "true" && r[col("plateau_ok")] == "true" &&
             parse(Int, r[col("slope")]) > 0) || continue
            push!(points, (m_N=parse(Float64, r[col("m_N")]), y=parse(Float64, r[col("y")]),
                           sin2_2theta=parse(Float64, r[col("sin2_2theta")]),
                           omega_h2=parse(Float64, r[col("omega_h2")])))
        end
    end
    isempty(points) && error("no converged freeze-in roots in $dir")
    return sort(points, by=p -> (p.m_N, p.y))
end

fmt(v) = string(round(v, sigdigits=4))
point_name(p) = "m_N_$(fmt(p.m_N * 1e6))keV_y_$(fmt(p.y))_sin22th_$(fmt(p.sin2_2theta))"

"Solves all points on the workers and prints a line as each one finishes."
function run_points(opts)
    queue = Channel{Any}(length(opts.points))
    foreach(p -> put!(queue, p), opts.points)
    close(queue)
    n_done, n_failed = Ref(0), Ref(0)
    t_start = time()

    @sync for w in workers()
        @async for p in queue
            name = point_name(p)
            try
                omega_h2, t = remotecall_fetch(history_job, w, p.m_N, p.y, p.sin2_2theta, opts.reltol,
                                               joinpath(opts.out, name * ".csv"))
                n_done[] += 1
                scan = isnan(p.omega_h2) ? "" : " (scan: $(round(p.omega_h2, sigdigits=4)))"
                println("[$(n_done[] + n_failed[])/$(length(opts.points)), $(round((time() - t_start) / 60, digits=1)) min] " *
                        "$name: Omega h^2 = $(round(omega_h2, sigdigits=4))$scan, $(round(t, digits=1)) s")
            catch e
                n_failed[] += 1
                println("[$(n_done[] + n_failed[])/$(length(opts.points))] $name FAILED on worker $w: " * sprint(showerror, e))
                e isa ProcessExitedException && break
            finally
                flush(stdout)
            end
        end
    end
    println("done: $(n_done[]) point(s) written to $(opts.out)" * (n_failed[] > 0 ? ", $(n_failed[]) failed" : ""))
end

# --- setup at top level, so that run_points sees the code loaded by @everywhere ---
opts = parse_args(ARGS)
mkpath(opts.out)
n_workers = clamp(opts.nproc, 1, length(opts.points))
println("$(length(opts.points)) point(s) on $n_workers process(es) -> $(opts.out)")
flush(stdout)

n_workers > 1 && addprocs(n_workers; exeflags=["--project=$(Base.active_project())", "--threads=1"])
const CODE = joinpath(@__DIR__, "relic_scan_z.jl")
const COLS = COLUMNS
@everywhere begin
    import CSV
    include($CODE)
    LA.BLAS.set_num_threads(1)

    const _TT_REL = Ref{Any}(nothing)

    "Solves one point, writes its history to `path`, returns (Omega h^2, wall time)."
    function history_job(m_N, y, sin2_2theta, reltol, path)
        t = @elapsed begin
            _TT_REL[] === nothing && (_TT_REL[] = TimeTempRelation{Float64}())
            tT_rel = _TT_REL[]
            theta = asin(sqrt(sin2_2theta)) / 2
            N1 = Particle{Float64}(m_N, 1, dof=2)
            N2 = Particle{Float64}(m_N, 1, dof=2)
            A = Particle{Float64}(2.5 * m_N, -1, dof=3)
            nu = Particle{Float64}(0.0, 1, dof=2)
            pan = PandemolatorZ{Float64}(ModelParams{Float64}(y, theta), N1, N2, A, nu, tT_rel)
            dw = DodelsonWidrow{Float64}(m_N, theta, tT_rel)

            sol = pandemolate_z(tT_rel, dw, pan; reltol=reltol)
            DE.successful_retcode(sol) || error("solver returned $(sol.retcode)")
            omega_h2 = final_omega_h2_z(pan, sol)
            results = transform_sol_z(pan, sol)
            CSV.write(path, NamedTuple{Tuple($COLS)}(Tuple(eachcol(results))))
        end
        return omega_h2, t
    end
end

run_points(opts)
n_workers > 1 && rmprocs(workers())
