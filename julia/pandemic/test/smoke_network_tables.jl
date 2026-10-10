# Coarse tables for smoke-testing network.jl (not for physics). Skips the slow
# resonant portal reactions unless ARGS[1] == "all".
include(joinpath(@__DIR__, "..", "src", "rate_tables.jl"))
skip = get(ARGS, 1, "") == "all" ? Symbol[] : [:N2nu_N1N2, :N1nu_AA]
t0 = time()
tabs = build_rate_tables(; reactions=filter(r -> !(r.name in skip), REACTIONS),
                         lnxN=range(log(1e-6), log(1e6), length=49),
                         lnxnu=range(log(3e-6), log(1e2), length=31), lnr=range(log(1e-4), log(3.0), length=11))
save_rate_tables(joinpath(@__DIR__, "..", "tmp", "rate_tables", "smoke_tables.jls"), tabs)
println("done in ", round(time() - t0; digits=1), " s"); flush(stdout)
