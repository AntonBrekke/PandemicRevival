include(joinpath(@__DIR__, "..", "src", "rate_tables.jl"))
t = load_rate_tables(ARGS[1])
for (k, v) in sort(collect(t.dark); by=first)
    println(rpad(k, 13), " dark   non-finite: ", count(!isfinite, v.lnRf), "/", length(v.lnRf), "  range lnRf ", extrema(filter(isfinite, v.lnRf)))
end
for (k, v) in sort(collect(t.portal); by=first)
    nf = [count(!isfinite, getfield(v, f)) for f in keys(v)]
    rr = extrema(v.ratio_Rf)
    println(rpad(k, 13), " portal non-finite per field: ", nf, "  ratio_Rf range ", round.(rr; sigdigits=3))
end
