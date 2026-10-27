"""
Lyman-alpha lengths at the fluid solver's relic roots (lyman_alpha.jl): re-solves each
converged root and writes one row per root (`LYMAN_ALPHA_HEADER`). Distributed over
worker processes.

    julia --project=. src/run_fluid_lyman_alpha.jl [--nproc 8] [--roots tmp/fluid_scan/fluid_roots.csv] [--out tmp/fluid_scan/lyman_alpha.csv]
        [--tables tmp/rate_tables/rate_tables_mA2.5_mphi100.0.jls] [--m-nu 0.1]

--m-nu [eV]: with equal-mass tables the late N1 N1 <-> N2 N2 conversions are followed with the
splitting at fixed m_nu (`mnu_splitting`, default 0.1 eV = M_NU; 0 = none), which adds the
columns x_fo, delta.
"""

using Distributed

function parse_opts(args)
    dir = joinpath(@__DIR__, "..", "tmp", "fluid_scan")
    o = Dict("nproc" => "8", "roots" => joinpath(dir, "fluid_roots.csv"), "out" => joinpath(dir, "lyman_alpha.csv"),
             "tables" => joinpath(@__DIR__, "..", "tmp", "rate_tables", "rate_tables_mA2.5_mphi100.0.jls"),
             "m-nu" => "0.1")
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

hdr = split(readline(OPTS["roots"]), ',')
col(name) = findfirst(==(name), hdr)
todo = [Tuple(parse(Float64, c[col(k)]) for k in ("m_N", "y", "s2_fluid"))
        for c in (split(l, ',') for l in readlines(OPTS["roots"])[2:end]) if c[col("status")] == "converged"]
println(length(todo), " converged roots"); flush(stdout)

addprocs(min(parse(Int, OPTS["nproc"]), length(todo)); exeflags=["--project=$(Base.active_project())", "--threads=1"])
@eval @everywhere const TABLES = $(OPTS["tables"])
@eval @everywhere const M_NU_EV = $(parse(Float64, OPTS["m-nu"]))
@everywhere begin
    include(joinpath(@__DIR__, "lyman_alpha.jl"))
    LA.BLAS.set_num_threads(1)
    const TT_REL = TimeTempRelation{Float64}()
    const ITP = RateInterpolants(load_rate_tables(TABLES))
    const PANS = Dict{Float64,Any}()
    const PER_POINT = M_NU_EV > 0 && table_delta(ITP) == 0

    function lya_job(m, g, s2)
        pan = get!(() -> FluidSolver(m, TT_REL, ITP), PANS, m)
        r = try
            lyman_alpha_fluid(ITP, TT_REL, m, g, s2; pan=pan, m_nu=PER_POINT ? M_NU_EV * 1e-9 : nothing)
        catch e
            @warn "Lyman-alpha failed at m=$m g=$g" exception = (e, catch_backtrace())
            failed_lengths()
        end
        return lyman_alpha_row(m, g, s2, r; delta=table_delta(ITP), per_point=PER_POINT)
    end
end

out_rows = String[]
next = Ref(0)
t0 = time()
@sync for w in workers()
    @async while true
        i = (next[] += 1)
        i > length(todo) && break
        push!(out_rows, remotecall_fetch(lya_job, w, todo[i]...))
        open(OPTS["out"], "w") do io
            println(io, lyman_alpha_header(table_delta(ITP); per_point=PER_POINT))
            foreach(l -> println(io, l), sort(out_rows; by=l -> parse.(Float64, split(l, ',')[1:2])))
        end
        println("[$(length(out_rows))/$(length(todo)), $(round((time() - t0) / 60; digits=1)) min] ", out_rows[end]); flush(stdout)
    end
end
println("done -> ", OPTS["out"])
