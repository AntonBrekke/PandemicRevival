"""
Free-streaming length lambda_fs at the network's relic roots, next to the
reduced solver's at the same (m_N, y) (tmp/relic_scan_theta_merged/lyman_alpha.csv,
column lambda_fs_Mpc: same definition, kinetic equilibrium through the
solution, frozen spectrum after x = 100, up to redshift 50).

    julia --project=. -t 8 src/run_network_lyman_alpha.jl [roots.csv] [out.csv]

The network's dark sector is Maxwell-Boltzmann wherever it runs (largest
occupation < 0.5), so the mean speed is the MB one at T_N (lyman_alpha.jl's
mean_speed with the gap capped there).
"""

include(joinpath(@__DIR__, "network.jl"))
include(joinpath(@__DIR__, "lyman_alpha.jl"))
using Printf

"""The fields lyman_alpha.jl's distance integrals need, from a Network."""
struct NetworkBackground{FS,FX}
    N1::Particle{Float64}
    ent_interp_z::FS
    dx_dt_interp_z::FX
end
NetworkBackground(net::Network) = NetworkBackground(Particle{Float64}(net.p.m1, 1, dof=2), net.ent, net.dx_dt)

"""(T_N, gap of N1) of the network state u at z."""
function network_temp_gap(net, u, z)
    n, kappa = state_densities(net, u, z)
    T_N = dark_temperature(net, n, kappa)
    return T_N, log(n_scaled_safe(net.masses[1], net.dofs[1], T_N) / n[1])
end

"""
Mean speed of an MB gas, 2(x + 1)/(x^2 e^x K_2(x)) with x = m/T: the closed form
of ∫ p^3/E e^(-E/T) dp / ∫ p^2 e^(-E/T) dp (-> 1 for x -> 0, sqrt(8T/(pi m)) for
x -> infinity).
"""
mb_mean_speed(m, T) = (x = m / T; 2 * (x + 1) / (x^2 * besselkx_safe(2, x)))

"""
lambda_fs during the solution by Simpson's rule on `n` points uniform in z (the
integrand is smooth; adaptive quadrature with lyman_alpha.jl's mean_speed costs
minutes per point), after it by lyman_alpha.jl's frozen-spectrum integral.
"""
function network_lambda_fs(itp, tT_rel, m, g, s2; n=4001)
    net = Network(itp, rate_params(m, g, s2), tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    sol = solve_network(net, dw, tT_rel; x_end=100.0, maxiters=20_000)
    sol.retcode == DE.ReturnCode.Success || return NaN
    bg = NetworkBackground(net)
    z_i, z_f = sol.t[1], sol.t[end]
    zs = range(z_i, z_f, length=n)
    h = step(zs)
    integrand(z) = mb_mean_speed(m, network_temp_gap(net, sol(z), z)[1]) / scale_factor_z(bg, z) * dt_dz(bg, z)
    vals = integrand.(zs)
    L_eq = h / 3 * (vals[1] + vals[end] + 4sum(vals[2:2:end-1]) + 2sum(vals[3:2:end-2]))
    z_end = z_at_redshift(m, tT_rel, LYA_Z_RED_END)
    T_f, gap_f = network_temp_gap(net, sol.u[end], z_f)
    a_f = scale_factor_z(bg, z_f)
    v_free(z) = mean_speed(bg.N1, T_f, gap_f; r=a_f / scale_factor_z(bg, z))
    L_free = comoving_distance(bg, v_free, z_f, z_end)
    return (L_eq + L_free) / Mpc
end

function network_lambda_fs_adaptive(itp, tT_rel, m, g, s2)
    net = Network(itp, rate_params(m, g, s2), tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    sol = solve_network(net, dw, tT_rel; x_end=100.0, maxiters=20_000)
    sol.retcode == DE.ReturnCode.Success || return NaN
    bg = NetworkBackground(net)
    z_i, z_f = sol.t[1], sol.t[end]
    z_end = z_at_redshift(m, tT_rel, LYA_Z_RED_END)
    sol_ref = Ref{Any}(sol)
    v_eq(z) = mean_speed(bg.N1, network_temp_gap(net, sol_ref[](z)::Vector{Float64}, z)...)
    L_eq = comoving_distance(bg, v_eq, z_i, z_f; breakpoints=sol.t)
    T_f, gap_f = network_temp_gap(net, sol.u[end], z_f)
    a_f = scale_factor_z(bg, z_f)
    v_free(z) = mean_speed(bg.N1, T_f, gap_f; r=a_f / scale_factor_z(bg, z))
    L_free = comoving_distance(bg, v_free, z_f, z_end)
    return (L_eq + L_free) / Mpc
end

function main(args)
    roots = length(args) > 0 ? args[1] : joinpath(@__DIR__, "..", "tmp", "network_scan", "network_roots.csv")
    out = length(args) > 1 ? args[2] : joinpath(@__DIR__, "..", "tmp", "network_scan", "lyman_alpha_network.csv")
    itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
    tT_rel = TimeTempRelation{Float64}()
    # reduced solver's lambda_fs by (m_N, y)
    la = Dict{Tuple{Float64,Float64},Float64}()
    lines = readlines(joinpath(@__DIR__, "..", "tmp", "relic_scan_theta_merged", "lyman_alpha.csv"))
    hdr = split(lines[1], ',')
    im, iy, il = findfirst(==("m_N"), hdr), findfirst(==("y"), hdr), findfirst(==("lambda_fs_Mpc"), hdr)
    for l in lines[2:end]
        c = split(l, ',')
        la[(round(parse(Float64, c[im]); sigdigits=4), round(parse(Float64, c[iy]); sigdigits=4))] = parse(Float64, c[il])
    end
    rows = [split(l, ',') for l in readlines(roots)[2:end]]
    rows = filter(c -> c[7] == "converged", rows)
    if haskey(ENV, "NETWORK_SCAN_LIMIT")
        n = parse(Int, ENV["NETWORK_SCAN_LIMIT"])
        rows = rows[round.(Int, range(1, length(rows), length=n))]
    end
    println(length(rows), " converged roots on ", Threads.nthreads(), " threads"); flush(stdout)
    res = Vector{Any}(undef, length(rows))
    t0 = time()
    Threads.@threads :dynamic for i in eachindex(rows)
        c = rows[i]
        m, y, s2n = parse(Float64, c[1]), parse(Float64, c[2]), parse(Float64, c[5])
        lam = try
            network_lambda_fs(itp, tT_rel, m, y, s2n)
        catch
            NaN
        end
        res[i] = (m, y, s2n, lam, get(la, (round(m; sigdigits=4), round(y; sigdigits=4)), NaN))
    end
    open(out, "w") do io
        println(io, "m_N,y,s2_network,lambda_fs_network_Mpc,lambda_fs_reduced_Mpc")
        for r in sort(res; by=r -> (r[1], r[2]))
            @printf(io, "%.6g,%.6g,%.6g,%.5g,%.5g\n", r...)
        end
    end
    println("written ", out, " after ", round((time() - t0) / 60; digits=1), " min")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
