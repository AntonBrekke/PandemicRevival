"""
Exact (Pauli-blocked) portal 2 <-> 2 correction tables (portal_tables.jl) for one MB table set.

    julia --project=. -t 16 src/run_build_portal_tables.jl [--mA 3] [--mphi 100] [--coarse]

Reads tmp/rate_tables/rate_tables_mA<mA>_mphi<mphi>.jls, writes portal_exact_mA<mA>_mphi<mphi>.jls next to it.
--coarse: every other point of each grid (a quick test, written to <...>_coarse.jls).
"""

include(joinpath(@__DIR__, "rate_tables.jl"))
include(joinpath(@__DIR__, "portal_tables.jl"))

function main(args)
    opts = Dict("mA" => "3", "mphi" => "100")
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--coarse"
            opts["coarse"] = "1"; i += 1
        else
            opts[lstrip(a, '-')] = args[i+1]; i += 2
        end
    end
    mA, mphi = parse(Float64, opts["mA"]), parse(Float64, opts["mphi"])
    dir = joinpath(@__DIR__, "..", "tmp", "rate_tables")
    itp = RateInterpolants(load_rate_tables(rate_table_path(dir; mA_over_mN=mA, mphi_over_mN=mphi)))
    coarse = haskey(opts, "coarse")
    kw = coarse ? (lnxnu=PX_LNXNU[1:2:end], lnr=PX_LNR[1:2:end], gaps=PX_GAP[1:2:end]) : NamedTuple()
    println("threads: ", Threads.nthreads(), "   m_A'/m_N = $mA   m_phi/m_N = $mphi", coarse ? "   (coarse)" : "")
    t0 = time()
    tabs = build_portal_tables(itp.p_ref, mass_ratios(itp); kw...)
    path = portal_table_path(dir; mA_over_mN=mA, mphi_over_mN=mphi)
    coarse && (path = replace(path, ".jls" => "_coarse.jls"))
    Serialization.serialize(path, tabs)
    println("saved ", path, " after ", round((time() - t0) / 60; digits=1), " min")
end

main(ARGS)
