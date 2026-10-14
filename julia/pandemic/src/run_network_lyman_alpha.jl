"""
Lyman-alpha lengths and final N2 fraction at the network's relic roots: the free-streaming
length without kinetic decoupling, next to the reduced solver's at the same (m_N, y)
(tmp/relic_scan_theta_merged/lyman_alpha.csv, column lambda_fs_Mpc: same definition,
kinetic equilibrium through the solution, frozen spectrum after x = 100, up to redshift 50),
and kinetic decoupling with lambda_fs after it and the sound horizon before it
(`network_kd_lengths`, the definitions of kinetic_decoupling.jl's `lyman_alpha_lengths`).

    julia --project=. -t 8 src/run_network_lyman_alpha.jl [roots.csv] [out.csv]

The network's dark sector is Maxwell-Boltzmann wherever it runs (largest
occupation < 0.5), so the mean speed is the MB one at T_N (lyman_alpha.jl's
mean_speed with the gap capped there).
"""

include(joinpath(@__DIR__, "network.jl"))
include(joinpath(@__DIR__, "kinetic_decoupling.jl"))
using Printf

"""The fields lyman_alpha.jl's distance integrals need, from a Network."""
struct NetworkBackground{FS,FX,FH}
    N1::Particle{Float64}
    ent_interp_z::FS
    dx_dt_interp_z::FX
    H_interp_z::FH
end
NetworkBackground(net::Network) = NetworkBackground(Particle{Float64}(net.p.m1, 1, dof=2), net.ent, net.dx_dt, net.H)

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
(lambda_fs, f_N2, kd lengths) at one root. lambda_fs during the solution by Simpson's rule on `n` points uniform in z (the
integrand is smooth; adaptive quadrature with lyman_alpha.jl's mean_speed costs
minutes per point), after it by lyman_alpha.jl's frozen-spectrum integral.
"""
const NO_KD = (x_c=NaN, x_kd=NaN, lambda_fs_pre=NaN, lambda_fs=NaN, r_s=NaN, coupled=false)

function network_lambda_fs(itp, tT_rel, m, g, s2; n=4001)
    net = Network(itp, rate_params(m, g, s2), tT_rel)
    dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
    sol = solve_network(net, dw, tT_rel; x_end=100.0, maxiters=20_000)
    sol.retcode == DE.ReturnCode.Success || return NaN, NaN, NO_KD
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
    # final N2 fraction of the DM (the reduced solver has 1/2 by construction)
    nf, _ = state_densities(net, sol.u[end], z_f)
    lam = (L_eq + L_free) / Mpc
    return lam, nf[2] / (nf[1] + nf[2]), network_kd_lengths(net, bg, sol, tT_rel, lam)
end

"""
Dark sector along a network solution and after it: (T_N, gaps of N1 and N2, n_N1, n_N2) at z.
After the end z_f (x = 100) the species are non-relativistic with conserved numbers in
kinetic equilibrium, T_N ∝ a^-2 at fixed gaps, n ∝ a^-3 (as kinetic_decoupling.jl's
DarkHistory).
"""
struct NetworkHistory
    net::Any
    bg::Any
    sol::Any    # untyped, see lyman_alpha.jl's `free_streaming_length`
    tab::CkdTable
    z_f::Float64
    a_f::Float64
    fin::NTuple{5,Float64}   # (T, gap1, gap2, n1, n2) at z_f
end

function network_state(net, u, z)
    n, kappa = state_densities(net, u, z)
    T = dark_temperature(net, n, kappa)
    gap(k) = log(n_scaled_safe(net.masses[k], net.dofs[k], T) / n[k])
    return (T, gap(1), gap(2), n[1], n[2])
end

function NetworkHistory(net, bg, sol)
    z_f = sol.t[end]
    p = net.p
    return NetworkHistory(net, bg, sol, CkdTable(p.m1, p.mA, p.g), z_f, scale_factor_z(bg, z_f),
                          network_state(net, sol.u[end], z_f))
end

function dark_state(h::NetworkHistory, z)
    z <= h.z_f && return network_state(h.net, h.sol(z)::Vector{Float64}, z)
    r = h.a_f / scale_factor_z(h.bg, z)
    T, g1, g2, n1, n2 = h.fin
    return (T * r^2, g1, g2, n1 * r^3, n2 * r^3)
end

"""Gamma_kd / H of N1 at z, Gamma_kd = C_kd / n_N1 with the gaps of N1 and N2 (tabulated C_kd)."""
function network_kd_ratio(h::NetworkHistory, z)
    T, g1, g2, n1, _ = dark_state(h, z)
    return C_kd(h.tab, T, g1; gap2=g2) / (h.bg.H_interp_z(z) * n1)
end

"""
Kinetic coupling and decoupling on a network solution: the first z at which Gamma_kd > H
(onset z_c) and the last z after which Gamma_kd < H for good (z_kd), on a grid of step `dz`
from the start of the solution to z_hi, both refined by bisection to 1e-4. Short dips below
H in between (e.g. at g ~ 1e-5 around x ~ 1e-3, between the DW production and the resonant
N1 N2 -> A' -> N1 N2 phase) count as coupled. Unlike `lyman_alpha_lengths`, which starts at
T_nu = 0.1 m_A' (the old code) and assumes the sector coupled before, the search starts at
the beginning: at small g the sector is produced free-streaming and couples (if at all)
only later. Returns (z_c, z_kd, coupled), with coupled = false if Gamma_kd < H throughout
and z_kd = z_hi if it is still coupled at z_hi.
"""
function network_find_kd(h::NetworkHistory, z_lo, z_hi; dz=0.05)
    r(z) = network_kd_ratio(h, z)
    # the crossing of r = 1 in [a, b], with r(a) and r(b) on either side
    function crossing(a, b)
        up = r(a) < 1
        while b - a > 1e-4
            c = (a + b) / 2
            (r(c) < 1) == up ? (a = c) : (b = c)
        end
        return b
    end
    zs = collect(range(z_lo, z_hi, length=max(2, ceil(Int, (z_hi - z_lo) / dz) + 1)))
    on = [r(z) >= 1 for z in zs]
    i_c = findfirst(on)
    i_c === nothing && return z_lo, z_lo, false
    i_kd = findlast(on)
    z_c = i_c == 1 ? z_lo : crossing(zs[i_c-1], zs[i_c])
    z_kd = i_kd == length(zs) ? z_hi : crossing(zs[i_kd], zs[i_kd+1])
    return z_c, z_kd, true
end

"""(P, rho) of the MB dark sector of the network state u at z: P = n_tot T_N, rho = sum m n + kappa."""
function network_P_rho(net, u, z)
    n, kappa = state_densities(net, u, z)
    T = dark_temperature(net, n, kappa)
    return sum(n) * T, sum(net.masses[k] * n[k] for k in 1:NSPECIES) + kappa
end

"""
Comoving sound horizon ∫ c_s/a dt from z_lo (the onset of coupling) to z_hi, in GeV^-1, with
kinetic_decoupling.jl's `sound_horizon` definitions: c_s^2 = dP/drho on a uniform z grid,
set to 0 unless dP/dt < 0, drho/dt < 0 and c_s^2 < 0.3; after the solution
c_s^2 = 5T/(3m + 7.5T).
"""
function network_sound_horizon(h::NetworkHistory, z_lo, z_hi; n_grid=2000)
    net, bg, sol = h.net, h.bg, h.sol
    z_hi > z_lo || return 0.0
    r = 0.0
    z_lo < h.z_f && (r += _sound_horizon_sol(h, z_lo, min(z_hi, h.z_f), n_grid))
    if z_hi > h.z_f
        cs(z) = (T = dark_state(h, z)[1]; sqrt(5T / (3 * net.p.m1 + 7.5T)))
        r += comoving_distance(bg, cs, max(z_lo, h.z_f), z_hi; rtol=1e-6)
    end
    return r
end

function _sound_horizon_sol(h::NetworkHistory, z_lo, z_hi, n_grid)
    net, bg, sol = h.net, h.bg, h.sol
    zs = range(z_lo, z_hi, length=n_grid)
    PR = [network_P_rho(net, sol(z)::Vector{Float64}, z) for z in zs]
    lP, lR = log.(first.(PR)), log.(last.(PR))
    r, f_prev = 0.0, 0.0
    for i in eachindex(zs)
        j0, j1 = max(i - 1, 1), min(i + 1, n_grid)
        dlP, dlR = lP[j1] - lP[j0], lR[j1] - lR[j0]
        cs2 = (PR[i][1] * dlP) / (PR[i][2] * dlR)
        f = (dlP < 0 && dlR < 0 && cs2 < 0.3) ? sqrt(cs2) / scale_factor_z(bg, zs[i]) * dt_dz(bg, zs[i]) : 0.0
        i > 1 && (r += (f + f_prev) / 2 * step(zs))
        f_prev = f
    end
    return r
end

"""∫ <v>/a dt from z_lo to z_hi in GeV^-1, MB mean speed at the dark temperature (Simpson, n points)."""
function network_free_stream(h::NetworkHistory, z_lo, z_hi; n=2001)
    z_hi > z_lo || return 0.0
    m = h.net.p.m1
    zs = range(z_lo, z_hi, length=n)
    f(z) = mb_mean_speed(m, dark_state(h, z)[1]) / scale_factor_z(h.bg, z) * dt_dz(h.bg, z)
    v = f.(zs)
    return step(zs) / 3 * (v[1] + v[end] + 4sum(v[2:2:end-1]) + 2sum(v[3:2:end-2]))
end

"""
Kinetic decoupling and the Lyman-alpha lengths at a network solution, in Mpc: r_s, the sound
horizon while coupled (from the onset x_c to x_kd), and lambda_fs, the free streaming outside
it: before the onset (`lambda_fs_pre`, at the dark temperature) plus after kd (frozen spectrum
of N1 at T_kd, to redshift 50), since what free streaming erases before the onset stays erased.
x_kd = m_N/T_nu at decoupling (Inf if still coupled at the end of the time grid). If the
sector never couples (`coupled` = false), lambda_fs is `lam_nokd` (free streaming from
production) and r_s = 0.
"""
function network_kd_lengths(net, bg, sol, tT_rel, lam_nokd)
    h = NetworkHistory(net, bg, sol)
    m = net.p.m1
    z_hi = log(m / tT_rel.T_nu_grid[end])
    z_end = z_at_redshift(m, tT_rel, LYA_Z_RED_END)
    z_c, z_kd, coupled = network_find_kd(h, sol.t[1], z_hi)
    coupled || return (x_c=NaN, x_kd=NaN, lambda_fs_pre=lam_nokd, lambda_fs=lam_nokd, r_s=0.0, coupled=false)
    pre = network_free_stream(h, sol.t[1], z_c) / Mpc
    lam = 0.0
    if z_kd < z_end
        T_kd, g1_kd = dark_state(h, z_kd)[1:2]
        a_kd = scale_factor_z(bg, z_kd)
        v(z) = mean_speed(bg.N1, T_kd, g1_kd; r=a_kd / scale_factor_z(bg, z))
        lam = comoving_distance(bg, v, z_kd, z_end; rtol=1e-6) / Mpc
    end
    return (x_c=exp(z_c), x_kd=z_kd >= z_hi ? Inf : exp(z_kd), lambda_fs_pre=pre, lambda_fs=pre + lam,
            r_s=network_sound_horizon(h, z_c, z_kd) / Mpc, coupled=true)
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
    done = Threads.Atomic{Int}(0)
    Threads.@threads :dynamic for i in eachindex(rows)
        c = rows[i]
        m, y, s2n = parse(Float64, c[1]), parse(Float64, c[2]), parse(Float64, c[5])
        lam, fN2, kd = try
            network_lambda_fs(itp, tT_rel, m, y, s2n)
        catch e
            @warn "failed at m=$m y=$y" exception = (e, catch_backtrace())
            NaN, NaN, NO_KD
        end
        res[i] = (m, y, s2n, lam, get(la, (round(m; sigdigits=4), round(y; sigdigits=4)), NaN), fN2,
                  kd.x_c, kd.x_kd, kd.lambda_fs_pre, kd.lambda_fs, kd.r_s, Int(kd.coupled))
        n_done = Threads.atomic_add!(done, 1) + 1
        n_done % 50 == 0 && (println("[$n_done/$(length(rows)), $(round((time() - t0) / 60; digits=1)) min]"); flush(stdout))
    end
    open(out, "w") do io
        println(io, "m_N,y,s2_network,lambda_fs_network_Mpc,lambda_fs_reduced_Mpc,f_N2_network,x_c,x_kd,lambda_fs_pre_Mpc,lambda_fs_kd_Mpc,r_s_Mpc,coupled")
        for r in sort(res; by=r -> (r[1], r[2]))
            @printf(io, "%.6g,%.6g,%.6g,%.5g,%.5g,%.4g,%.5g,%.5g,%.5g,%.5g,%.5g,%d\n", r...)
        end
    end
    println("written ", out, " after ", round((time() - t0) / 60; digits=1), " min")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
