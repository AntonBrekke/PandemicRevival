include(joinpath(@__DIR__, "..", "src", "rate_tables.jl"))
p = reference_params(); m = p.m1
names = length(ARGS) > 0 ? Symbol.(split(ARGS[1], ',')) : [r.name for r in REACTIONS if r.kind in (:portal2, :portal12)]
T_max = 3.4e6m
for name in names
    r = reaction(name)
    t = @elapsed tab = reaction_sigma_table(r, p; T_max=T_max, n=800)
    println(rpad(name, 12), " table ", round(t; digits=1), " s"); flush(stdout)
    # corners and middle of the (x_nu, r) grid
    for (xnu, rr) in ((1e3, 1e-5), (1e3, 10.0), (1.0, 1e-5), (1.0, 1.0), (1.0, 10.0), (3e-6, 1e-5), (3e-6, 1.0), (3e-6, 10.0), (100.0, 1e-3))
        T_nu = m / xnu
        t = @elapsed v = portal_point(r, p, tab, rr * T_nu, T_nu)
        println("   x_nu=", xnu, " r=", rr, "  lnref=", round.(v[1]; sigdigits=5), " ratio=", round.(v[2]; sigdigits=3),
                "  ", round(t; digits=2), " s"); flush(stdout)
    end
end
