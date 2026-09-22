"""
run_lyman_alpha.jl

Post-processing of the theta scan (run_relic_scan_theta_all.jl): for every
freeze-in root in <scan>/m_N_*keV.csv, re-solve once at the root and compute
the Lyman-alpha lengths of N: the free-streaming length without kinetic
decoupling (src/lyman_alpha.jl), and kinetic decoupling with the free-streaming
length after it and the sound horizon before it (src/kinetic_decoupling.jl),
with the (A6) amplitude and, for comparison, as the old code computed them
(its amplitude and its factor 1e3). The Lyman-alpha
quantities are thus computed once per relic point, not in every solve of the
root search.

Roots are selected as in plot/relic_scan_theta.py: converged, plateau_ok,
slope > 0, and per (m_N, y) the smallest such sin^2 2theta.

Output: <out> (default <scan>/lyman_alpha.csv), rewritten after every point, with
    m_N,y,sin2_2theta,omega_h2_scan,omega_h2,converged,retcode,
    lambda_fs_Mpc,lambda_fs_eq_Mpc,lambda_fs_free_Mpc,v_end,
    x_kd,lambda_fs_kd_Mpc,r_s_Mpc,x_kd3,lambda_fs_kd3_Mpc,r_s3_Mpc,
    x_kd_old,lambda_fs_kd_old_Mpc,r_s_old_Mpc
lambda_fs_* without "kd": no kinetic decoupling (`free_streaming_length`);
x_kd, lambda_fs_kd, r_s: `lyman_alpha_lengths` with (A6) at Gamma_kd = H, "3"
at 3H; "_old": old amplitude with the old code's factor 1e3, at H. Points
already in <out> with a finite lambda_fs_kd are skipped, so an interrupted run
can be resumed (a file with another header is recomputed).

Usage (from julia/pandemic):
    julia --project=. src/run_lyman_alpha.jl [--scan DIR] [--out FILE] [--nproc N] [--reltol R] [--masses LIST]
"""

using Distributed

const USAGE = """
Usage: julia --project=. src/run_lyman_alpha.jl [options]
    --scan DIR       directory with the scan CSVs (default tmp/relic_scan_theta)
    --out FILE       output CSV (default <scan>/lyman_alpha.csv)
    --nproc N        number of worker processes (default: number of CPU threads)
    --reltol R       ODE rtol of the re-solve (default 1e-4, as ScanConfigThetaZ)
    --masses LIST    only these masses in keV, e.g. "1.5,10" (default: all)
"""

const HEADER = "m_N,y,sin2_2theta,omega_h2_scan,omega_h2,converged,retcode,lambda_fs_Mpc,lambda_fs_eq_Mpc,lambda_fs_free_Mpc,v_end," *
               "x_kd,lambda_fs_kd_Mpc,r_s_Mpc,x_kd3,lambda_fs_kd3_Mpc,r_s3_Mpc,x_kd_old,lambda_fs_kd_old_Mpc,r_s_old_Mpc"
const N_COLS = length(split(HEADER, ","))
const COL_LAMBDA_KD = findfirst(==("lambda_fs_kd_Mpc"), split(HEADER, ","))

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
    unknown = setdiff(keys(opts), ["scan", "out", "nproc", "reltol", "masses"])
    isempty(unknown) || error("unknown option(s): $(join("--" .* unknown, ", "))")
    scan = get(opts, "scan", joinpath(@__DIR__, "../tmp/relic_scan_theta"))
    return (
        scan=scan,
        out=get(opts, "out", joinpath(scan, "lyman_alpha.csv")),
        nproc=haskey(opts, "nproc") ? parse(Int, opts["nproc"]) : Sys.CPU_THREADS,
        reltol=haskey(opts, "reltol") ? parse(Float64, opts["reltol"]) : 1e-4,
        masses=haskey(opts, "masses") ? parse.(Float64, split(opts["masses"], ",")) : nothing,
    )
end

"Reads a CSV with a header line into a Vector of Dict(column => string)."
function read_csv(path)
    lines = filter(!isempty, strip.(readlines(path)))
    isempty(lines) && return Dict{String,String}[]
    cols = split(lines[1], ",")
    return [Dict(zip(cols, split(l, ","))) for l in lines[2:end]]
end

"Freeze-in roots (m_N [GeV], y, sin2_2theta, omega_h2) of all scan CSVs, as in plot/relic_scan_theta.py."
function freeze_in_roots(scan_dir, masses_keV)
    roots = Dict{Tuple{Float64,Float64}, Tuple{Float64,Float64}}()
    for f in sort(filter(f -> occursin(r"^m_N_.*keV\.csv$", f), readdir(scan_dir)))
        for r in read_csv(joinpath(scan_dir, f))
            (r["converged"] == "true" && r["plateau_ok"] == "true" && parse(Int, r["slope"]) > 0) || continue
            m, y, s = parse(Float64, r["m_N"]), parse(Float64, r["y"]), parse(Float64, r["sin2_2theta"])
            isfinite(s) || continue
            masses_keV === nothing || any(mk -> isapprox(m * 1e6, mk; rtol=1e-6), masses_keV) || continue
            key = (m, y)
            (!haskey(roots, key) || s < roots[key][1]) && (roots[key] = (s, parse(Float64, r["omega_h2"])))
        end
    end
    return sort([(k[1], k[2], v[1], v[2]) for (k, v) in roots])
end

"Rows of a previous run, keyed on (m_N, y, sin2_2theta); only those with a finite lambda_fs."
function previous_rows(out)
    isfile(out) || return Dict{NTuple{3,Float64}, String}()
    lines = filter(!isempty, strip.(readlines(out)))
    rows = Dict{NTuple{3,Float64}, String}()
    (isempty(lines) || lines[1] != HEADER) && return rows
    for l in lines[2:end]
        v = split(l, ",")
        length(v) == N_COLS && isfinite(parse(Float64, v[COL_LAMBDA_KD])) &&
            (rows[(parse(Float64, v[1]), parse(Float64, v[2]), parse(Float64, v[3]))] = l)
    end
    return rows
end

function write_rows(out, rows)
    mkpath(dirname(abspath(out)))
    tmp = out * ".tmp"
    open(tmp, "w") do io
        println(io, HEADER)
        for k in sort(collect(keys(rows)))
            println(io, rows[k])
        end
    end
    mv(tmp, out; force=true)
end

# --- setup at top level, so that the workers see the code loaded by @everywhere ---
opts = parse_args(ARGS)
points = freeze_in_roots(opts.scan, opts.masses)
rows = previous_rows(opts.out)
todo = [p for p in points if !haskey(rows, p[1:3])]
println("$(length(points)) freeze-in roots in $(opts.scan), $(length(points) - length(todo)) already done -> $(opts.out)")
flush(stdout)
isempty(todo) && exit(0)

n_workers = clamp(opts.nproc, 1, length(todo))
addprocs(n_workers; exeflags=["--project=$(Base.active_project())", "--threads=1"])
const CODE = joinpath(@__DIR__, "kinetic_decoupling.jl")
@everywhere begin
    include($CODE)
    LA.BLAS.set_num_threads(1)

    const _TT_REL = Ref{Any}(nothing)
    const _PANS = Dict{Float64, Any}()

    "PandemolatorZ at mass m_N (GeV), m_A = 2.5 m_N as in the theta scan, cached per worker."
    function pan_at(m_N)
        _TT_REL[] === nothing && (_TT_REL[] = TimeTempRelation{Float64}())
        get!(_PANS, m_N) do
            PandemolatorZ{Float64}(
                ModelParams{Float64}(1e-4, 1e-6),
                Particle{Float64}(m_N, 1, dof=2), Particle{Float64}(m_N, 1, dof=2),
                Particle{Float64}(2.5 * m_N, -1, dof=3), Particle{Float64}(0.0, 1, dof=2),
                _TT_REL[],
            )
        end
    end

    "Re-solves one root and returns its CSV row."
    function lya_job(m_N, y, s, omega_scan, reltol)
        pan = pan_at(m_N)
        theta = asin(sqrt(s)) / 2
        sol, omega, converged, retcode = solve_relic_point(pan, _TT_REL[], y, theta; reltol=reltol)
        fs = (lambda_fs=NaN, lambda_fs_eq=NaN, lambda_fs_free=NaN, v_end=NaN)
        nan6 = (x_kd=NaN, lambda_fs=NaN, r_s=NaN, x_kd3=NaN, lambda_fs3=NaN, r_s3=NaN)
        kd, kd_old = nan6, nan6
        if sol !== nothing
            try
                fs = free_streaming_length(pan, _TT_REL[], sol)
                kd = lyman_alpha_lengths(pan, _TT_REL[], sol)
                kd_old = lyman_alpha_lengths(pan, _TT_REL[], sol; amp=:old, sigma_scale=1e3)
            catch e
                @warn "Lyman-alpha lengths failed at m_N=$m_N, y=$y" exception = (e, catch_backtrace())
            end
        end
        return "$m_N,$y,$s,$omega_scan,$omega,$converged,$retcode," *
               "$(fs.lambda_fs),$(fs.lambda_fs_eq),$(fs.lambda_fs_free),$(fs.v_end)," *
               "$(kd.x_kd),$(kd.lambda_fs),$(kd.r_s),$(kd.x_kd3),$(kd.lambda_fs3),$(kd.r_s3)," *
               "$(kd_old.x_kd),$(kd_old.lambda_fs),$(kd_old.r_s)"
    end
end

t_start = time()
lock_rows = ReentrantLock()
n_done = Ref(0)
@sync for p in workers()
    @async while true
        job = lock(lock_rows) do
            isempty(todo) ? nothing : popfirst!(todo)
        end
        job === nothing && break
        m, y, s, om = job
        row = try
            remotecall_fetch(lya_job, p, m, y, s, om, opts.reltol)
        catch e
            println("m_N = $(m * 1e6) keV, y = $y FAILED on worker $p: " * sprint(showerror, e))
            flush(stdout)
            e isa ProcessExitedException && break
            continue
        end
        lock(lock_rows) do
            rows[(m, y, s)] = row
            write_rows(opts.out, rows)
            n_done[] += 1
            v = split(row, ",")
            f(i) = round(parse(Float64, v[i]), sigdigits=3)
            println("[$(n_done[]), $(round((time() - t_start) / 60, digits=1)) min] m_N = $(round(m * 1e6, sigdigits=4)) keV, " *
                    "y = $y: x_kd = $(f(12)), lambda_fs = $(f(13)) Mpc, r_s = $(f(14)) Mpc (no kd: lambda_fs = $(f(8))), Omega h^2 = $(f(5))")
            flush(stdout)
        end
    end
end
rmprocs(workers())
println("done in $(round((time() - t_start) / 60, digits=1)) min -> $(opts.out)")
