"""
Lyman-alpha lengths at the relic roots of the high-g solver (tmp/hg_scan/hg_roots.csv):
re-solves each converged root with pandemolator_hg.jl and computes, with the same
functions and definitions as src/run_lyman_alpha.jl for the reduced solver,
lambda_fs without kinetic decoupling (`free_streaming_length`) and kinetic
decoupling with lambda_fs after it and the sound horizon before it
(`lyman_alpha_lengths`, corrected (A6), both channels, Gamma_kd = H).

    julia --project=. src/run_hg_lyman_alpha.jl [--nproc 8] [--roots tmp/hg_scan/hg_roots.csv] [--out tmp/hg_scan/lyman_alpha_hg.csv]

f_N2 = 1/2: N1 and N2 are in chemical equilibrium in the one-fluid system.
"""

using Distributed
using Printf

function parse_opts(args)
    o = Dict("nproc" => "8", "roots" => joinpath(@__DIR__, "..", "tmp", "hg_scan", "hg_roots.csv"),
             "out" => joinpath(@__DIR__, "..", "tmp", "hg_scan", "lyman_alpha_hg.csv"))
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

rows = [split(l, ',') for l in readlines(OPTS["roots"])[2:end]]
todo = [(parse(Float64, c[1]), parse(Float64, c[2]), parse(Float64, c[7]), parse(Float64, c[8])) for c in rows if c[9] == "converged"]
println(length(todo), " converged roots"); flush(stdout)

addprocs(min(parse(Int, OPTS["nproc"]), length(todo)); exeflags=["--project=$(Base.active_project())", "--threads=1"])
@everywhere using Printf
@everywhere begin
    include(joinpath(@__DIR__, "pandemolator_hg.jl"))
    include(joinpath(@__DIR__, "relic_scan_z.jl"))
    include(joinpath(@__DIR__, "kinetic_decoupling.jl"))
    LA.BLAS.set_num_threads(1)
    const TT_REL = TimeTempRelation{Float64}()
    const ITP = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
    const PANS = Dict{Float64,Any}()
    hg_pan(m) = get!(() -> PandemolatorHG(m, TT_REL, ITP), PANS, m)

    function lya_job(m, y, s2, om_scan)
        pan = hg_pan(m)
        th = asin(sqrt(s2)) / 2
        pan.mp = ModelParams{Float64}(y, th)
        fs = (lambda_fs=NaN, lambda_fs_eq=NaN, lambda_fs_free=NaN, v_end=NaN)
        kd = (x_kd=NaN, lambda_fs=NaN, r_s=NaN, x_kd3=NaN, lambda_fs3=NaN, r_s3=NaN)
        om = NaN
        try
            for x_end in (100.0, 300.0, 1000.0)
                dw = DodelsonWidrow{Float64}(m, th, TT_REL; x_end=x_end)
                sol = pandemolate_z(TT_REL, dw, pan; reltol=1e-4)
                DE.successful_retcode(sol) || break
                if check_plateau_z(pan, sol; tol=5e-3)
                    om = final_omega_h2_z(pan, sol)
                    fs = free_streaming_length(pan, TT_REL, sol)
                    kd = lyman_alpha_lengths(pan, TT_REL, sol)
                    break
                end
            end
        catch e
            @warn "Lyman-alpha failed at m=$m y=$y" exception = (e, catch_backtrace())
        end
        return @sprintf("%.6g,%.6g,%.6g,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,%.5g,0.5", m, y, s2, om_scan, om,
                        fs.lambda_fs, fs.lambda_fs_eq, fs.lambda_fs_free, fs.v_end, kd.x_kd, kd.lambda_fs, kd.r_s)
    end
end

header = "m_N,y,sin2_2theta,omega_h2_scan,omega_h2,lambda_fs_Mpc,lambda_fs_eq_Mpc,lambda_fs_free_Mpc,v_end,x_kd,lambda_fs_kd_Mpc,r_s_Mpc,f_N2"
out_rows = String[]
next = Ref(0)
t0 = time()
@sync for w in workers()
    @async while true
        i = (next[] += 1)
        i > length(todo) && break
        r = remotecall_fetch(lya_job, w, todo[i]...)
        push!(out_rows, r)
        open(OPTS["out"], "w") do io
            println(io, header)
            foreach(l -> println(io, l), sort(out_rows; by=l -> parse.(Float64, split(l, ',')[1:2])))
        end
        println("[$(length(out_rows))/$(length(todo)), $(round((time() - t0) / 60; digits=1)) min] ", r); flush(stdout)
    end
end
println("done -> ", OPTS["out"])
