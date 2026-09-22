"""
run_relic_scan_theta_z.jl

Driver for relic_scan_theta_z.jl: at one mass, root-find sin^2(2 theta) for
Omega h^2 = 0.12 for a list of couplings y, and write the result to
tmp/relic_scan_theta/m_N_<m_N in keV>keV.csv.

Usage (one process per mass, e.g. from a shell loop):
    julia --project=. src/run_relic_scan_theta_z.jl <m_N in keV> [log10y_min log10y_max n_y] [out_dir]
    julia --project=. src/run_relic_scan_theta_z.jl <m_N in keV> <y1,y2,...> [out_dir]
"""

include(joinpath(@__DIR__, "relic_scan_theta_z.jl"))

m_N_keV = parse(Float64, ARGS[1])
if length(ARGS) >= 2 && occursin(",", ARGS[2])
    ys = parse.(Float64, split(ARGS[2], ","))
    out_dir = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "../tmp/relic_scan_theta")
else
    log10y_min = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : -6.0
    log10y_max = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : -1.5
    n_y = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10
    ys = exp10.(range(log10y_min, log10y_max, length=n_y))
    out_dir = length(ARGS) >= 5 ? ARGS[5] : joinpath(@__DIR__, "../tmp/relic_scan_theta")
end

m_N = m_N_keV * 1e-6
# Optional: RELIC_ODE_RELTOL=1e-4 runs the whole search at that ODE tolerance
# (faster, but Omega h^2 has percent-level noise on the steep freeze-in branch).
cfg = if haskey(ENV, "RELIC_ODE_RELTOL")
    rt = parse(Float64, ENV["RELIC_ODE_RELTOL"])
    ScanConfigThetaZ(ode_reltol=rt, ode_reltol_coarse=max(rt, 1e-4))
else
    ScanConfigThetaZ()
end

t = @elapsed points = run_scan_theta_z(m_N, ys, cfg)
out = joinpath(out_dir, "m_N_$(m_N_keV)keV.csv")
save_results_theta_z(points, out)
println("m_N = $m_N_keV keV done in $(round(t / 60, digits=1)) min -> $out")
