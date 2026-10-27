"""
Lyman-alpha lengths at the network's relic roots (lyman_alpha.jl), threaded over roots.

    julia --project=. -t 8 src/run_network_lyman_alpha.jl [network_roots.csv] [out.csv] [tables.jls] [m_nu_eV]

Defaults: tmp/network_scan/network_roots.csv -> tmp/network_scan/lyman_alpha.csv, one row
per converged root (`LYMAN_ALPHA_HEADER`), with the m_phi = 100 m_N tables. With equal-mass
tables the late N1 N1 <-> N2 N2 conversions are followed with the splitting at fixed m_nu
(`mnu_splitting`; default m_nu = 0.1 eV, M_NU; 0 = none), which adds the columns x_fo, delta.
"""

include(joinpath(@__DIR__, "lyman_alpha.jl"))

function main(args)
    dir = joinpath(@__DIR__, "..", "tmp", "network_scan")
    roots = length(args) > 0 ? args[1] : joinpath(dir, "network_roots.csv")
    out = length(args) > 1 ? args[2] : joinpath(dir, "lyman_alpha.csv")
    tables = length(args) > 2 ? args[3] : rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))
    itp = RateInterpolants(load_rate_tables(tables))
    m_nu = length(args) > 3 ? parse(Float64, args[4]) * 1e-9 : M_NU
    m_nu = (m_nu > 0 && table_delta(itp) == 0) ? m_nu : nothing
    per_point = m_nu !== nothing
    tT_rel = TimeTempRelation{Float64}()
    hdr = split(readline(roots), ',')
    col(name) = findfirst(==(name), hdr)
    rows = [split(l, ',') for l in readlines(roots)[2:end]]
    rows = filter(c -> c[col("status")] == "converged", rows)
    println(length(rows), " converged roots on ", Threads.nthreads(), " threads",
            per_point ? ", splitting at m_nu = $(m_nu * 1e9) eV" : ""); flush(stdout)
    res = Vector{String}(undef, length(rows))
    done = Threads.Atomic{Int}(0)
    t0 = time()
    Threads.@threads :dynamic for i in eachindex(rows)
        m, g, s2 = (parse(Float64, rows[i][col(k)]) for k in ("m_N", "y", "s2_network"))
        r = try
            lyman_alpha_network(itp, tT_rel, m, g, s2; m_nu=m_nu)
        catch e
            @warn "failed at m=$m g=$g" exception = (e, catch_backtrace())
            failed_lengths()
        end
        res[i] = lyman_alpha_row(m, g, s2, r; delta=table_delta(itp), per_point=per_point)
        k = Threads.atomic_add!(done, 1) + 1
        k % 50 == 0 && (println("[$k/$(length(rows)), $(round((time() - t0) / 60; digits=1)) min]"); flush(stdout))
    end
    open(out, "w") do io
        println(io, lyman_alpha_header(table_delta(itp); per_point=per_point))
        foreach(l -> println(io, l), sort(res; by=l -> parse.(Float64, split(l, ',')[1:2])))
    end
    println("written ", out, " after ", round((time() - t0) / 60; digits=1), " min")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
