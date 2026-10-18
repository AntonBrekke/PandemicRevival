"""
Late decays at the relic roots (late_decays.jl).

    julia --project=. -t 8 src/run_late_decays.jl [--combined DIR] [--lya F1,F2,...] [--tables T]
                                                  [--out CSV] [--phi-share]

Defaults: tmp/combined_scan, the lyman_alpha.csv of tmp/{network,fluid}_scan{,_fine} and of the
combined scan (for f_N2, x_c, x_kd), the m_phi = 100 m_N tables, tmp/late_decays/late_decays.csv.
m_phi/m_N is that of the tables. `--phi-share` re-solves the network at each root (g < 1e-3) for
the largest share of the N number held in phi, the decaying fraction of a long-lived phi
(m_phi < 2 m_A'); otherwise F_phi is NaN.

Per root: f_N1; lifetimes [s] of N1 -> nu gamma, N1 -> 3 nu, A', phi; for a long-lived phi
(phi -> N1 nu) m_N/T_nu at its decay, the free streaming of the N1 it makes and Delta N_eff of its
neutrinos. Then, with the same root (sin^2 2theta at m_phi = 100 m_N as a proxy), the
m_phi < m_N case of Sec. VI of the draft at m_phi/m_N in R_LIGHT: lifetimes of N1 -> phi nu and
phi -> nu nu, the free streaming of the phi from N1 decays, and the bound on phi -> nu nu
(`ddm_bound` with the phi mass fraction of the DM).
"""

include(joinpath(@__DIR__, "network.jl"))
include(joinpath(@__DIR__, "late_decays.jl"))
using Printf

const R_LIGHT = (0.1, 0.3, 0.5, 0.7, 0.9)
const TMP = joinpath(@__DIR__, "..", "tmp")

key(m, y) = (round(m * 1e6; sigdigits=4), round(log10(y); digits=3))

function read_rows(path)
    hdr = split(readline(path), ',')
    return [Dict(zip(hdr, split(l, ','))) for l in readlines(path)[2:end]]
end

num(d, k) = haskey(d, k) ? parse(Float64, d[k]) : NaN

function parse_opts(args)
    o = Dict{String,Any}("combined" => joinpath(TMP, "combined_scan"), "lya" => nothing,
                         "tables" => rate_table_path(joinpath(TMP, "rate_tables")),
                         "out" => joinpath(TMP, "late_decays", "late_decays.csv"), "phi-share" => false)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--phi-share"
            o["phi-share"] = true
        else
            o[a[3:end]] = args[i+1]
            i += 1
        end
        i += 1
    end
    if o["lya"] === nothing
        fs = [joinpath(TMP, d, "lyman_alpha.csv") for d in ("network_scan", "network_scan_fine", "fluid_scan", "fluid_scan_fine")]
        o["lya"] = join(filter(isfile, [fs; joinpath(o["combined"], "lyman_alpha.csv")]), ",")
    end
    return o
end

"""Largest share of the N number in phi along the network solution (w_phi n_phi / n~)."""
function phi_share(itp, tT, m, g, s2)
    net = Network(itp, table_params(itp, m, g, s2), tT)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT)
    sol = solve_network(net, dw, tT; x_end=100.0, maxiters=20_000)
    sol.retcode == DE.ReturnCode.Success || return NaN
    f(u, z) = (n = state_densities(net, u, z)[1]; net.w_phi * n[4] / (n[1] + n[2] + 2n[3] + net.w_phi * n[4]))
    return maximum(f(u, z) for (u, z) in zip(sol.u, sol.t))
end

function main(args)
    o = parse_opts(args)
    itp = RateInterpolants(load_rate_tables(o["tables"]))
    rA, rphi = mass_ratios(itp)
    tT = TimeTempRelation{Float64}()
    # f_N2, x_c, x_kd per (m, y): later files fill what earlier ones lack
    info = Dict{Any,Dict{String,Float64}}()
    for f in split(o["lya"], ',')
        for r in read_rows(f)
            d = get!(info, key(num(r, "m_N"), num(r, "y")), Dict{String,Float64}())
            for k in ("f_N2", "x_c", "x_kd")
                v = num(r, k)
                (isfinite(v) || isinf(v)) && !haskey(d, k) && (d[k] = v)
            end
        end
    end
    roots = filter(r -> r["solver"] != "failed", read_rows(joinpath(o["combined"], "roots.csv")))
    println(length(roots), " roots, m_A'/m_N = $rA, m_phi/m_N = $rphi, ", Threads.nthreads(), " threads"); flush(stdout)
    hdr = ["m_N", "y", "sin2_2theta", "solver", "f_N1", "x_c", "x_kd", "tau_gamma", "tau_3nu", "tau_A", "tau_phi",
           "F_phi", "x_phi_dec", "lambda_fs_phi_dec_Mpc", "dNeff_phi_dec"]
    for r in R_LIGHT
        s = @sprintf("_r%.1f", r)
        append!(hdr, ["tau_N1phinu" * s, "tau_phinunu" * s, "lambda_fs_phi" * s * "_Mpc", "ddm" * s])
    end
    out = Vector{String}(undef, length(roots))
    t0 = time()
    Threads.@threads :dynamic for i in eachindex(roots)
        r = roots[i]
        m, g, s2 = num(r, "m_N"), num(r, "y"), num(r, "sin2_2theta")
        d = get(info, key(m, g), Dict{String,Float64}())
        fN1 = 1 - get(d, "f_N2", NaN)
        p = table_params(itp, m, g, s2)
        tauphi = 1 / gamma_phi(p)
        # long-lived phi (phi -> N1 nu only): decaying share, warm N1, neutrinos
        Fphi, xdec, lam, dneff = NaN, NaN, NaN, NaN
        if rphi < 2rA
            Fphi = o["phi-share"] && g < 1e-3 ? (try phi_share(itp, tT, m, g, s2) catch; NaN end) : NaN
            xdec = x_nu_at_time(m, tauphi)
            lam = free_stream_decay((rphi^2 - 1) / (2rphi), tauphi)
            dneff = isfinite(Fphi) ? delta_neff(Fphi, tauphi, (1 - 1 / rphi^2) / 2) : NaN
        end
        row = Any[m, g, s2, r["solver"], fN1, get(d, "x_c", NaN), get(d, "x_kd", NaN), 1 / gamma_nugamma(m, s2),
                  1 / gamma_3nu(m, s2), 1 / gamma_A(p), tauphi, Fphi, xdec, lam, dneff]
        for rl in R_LIGHT
            q = rate_params(m, g, s2; mphi_over_mN=rl, mA_over_mN=rA)
            tN, tp = 1 / gamma_tree(:N1_phinu, q), 1 / gamma_phi_nunu(q)
            fphi = fN1 * rl / (fN1 * rl + 1 - fN1)          # phi mass fraction of the DM
            push!(row, tN, tp, free_stream_decay((1 - rl^2) / (2rl), tN), isfinite(fphi) ? ddm_bound(fphi, tp, 1.0) : :unknown)
        end
        out[i] = join((v isa AbstractFloat ? @sprintf("%.5g", v) : string(v) for v in row), ",")
    end
    mkpath(dirname(o["out"]))
    open(o["out"], "w") do io
        println(io, join(hdr, ","))
        foreach(l -> println(io, l), out)
    end
    println(@sprintf("written %s after %.1f min", o["out"], (time() - t0) / 60))
end

main(ARGS)
