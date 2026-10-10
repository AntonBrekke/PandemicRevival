"""
Build the rate tables of src/rate_tables.jl for one (m_A'/m_N, m_phi/m_N) and
save them to tmp/rate_tables/.

    julia --project=. -t 8 src/run_build_rate_tables.jl [--mA 2.5] [--mphi 100] [--only name1,name2]

The tables do not depend on m_N or the couplings. Threads parallelise over
grid points; 8 is a sensible default on a laptop (one table point needs little
memory). `--only` rebuilds a subset of reactions into a separate file (for
checks; the network needs all of them).
"""

include(joinpath(@__DIR__, "rate_tables.jl"))

function parse_args(args)
    opts = Dict("mA" => "2.5", "mphi" => "100", "only" => "")
    i = 1
    while i <= length(args)
        key = lstrip(args[i], '-')
        haskey(opts, key) || error("unknown option $(args[i])")
        opts[key] = args[i+1]
        i += 2
    end
    return opts
end

opts = parse_args(ARGS)
mA, mphi = parse(Float64, opts["mA"]), parse(Float64, opts["mphi"])
reactions = isempty(opts["only"]) ? REACTIONS : [reaction(Symbol(n)) for n in split(opts["only"], ',')]
println("threads: ", Threads.nthreads(), "   m_A'/m_N = ", mA, "   m_phi/m_N = ", mphi)
println("reactions: ", join((r.name for r in reactions), ", "))
flush(stdout)

t0 = time()
tabs = build_rate_tables(; mA_over_mN=mA, mphi_over_mN=mphi, reactions=reactions)
dir = joinpath(@__DIR__, "..", "tmp", "rate_tables")
path = rate_table_path(dir; mA_over_mN=mA, mphi_over_mN=mphi)
isempty(opts["only"]) || (path = replace(path, ".jls" => "_partial.jls"))
save_rate_tables(path, tabs)
println("saved ", path, " after ", round((time() - t0) / 60; digits=1), " min")
