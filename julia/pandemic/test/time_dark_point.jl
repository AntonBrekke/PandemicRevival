include(joinpath(@__DIR__, "..", "src", "rate_tables.jl"))
p = reference_params(); m = p.m1
T_max = parse(Float64, get(ARGS, 1, "100")) * m
for r in filter(r -> r.kind === :dark2, REACTIONS)
    t = @elapsed tab = reaction_sigma_table(r, p; T_max=T_max, n=800)
    print(rpad(r.name, 12), " table ", round(t; digits=1), " s |")
    for lx in (8.0, 6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0)
        T = m * 10.0^(-lx)
        T > T_max && continue
        t = @elapsed v = dark_point(r, p, tab, T)
        print(" 1e", Int(lx), ":", round(v[1]; sigdigits=5), " (", round(t; digits=2), "s)")
    end
    println(); flush(stdout)
end
