"""
run_relic_scan_z.jl

As run_relic_scan.jl, but wired to the z-parameterized solver via
relic_scan_z.jl. Same benchmark particle content as test/test_pandemolate.jl.
"""

include(joinpath(@__DIR__, "relic_scan_z.jl"))
using Plots

# --- fixed "panel" parameters: masses/dof held constant across the scan ---
m_N = 1e-5   # GeV (10 keV)
m_A = 2.5 * m_N

N1 = Particle{Float64}(m_N, 1, dof=2)
N2 = Particle{Float64}(m_N, 1, dof=2)
A = Particle{Float64}(m_A, -1, dof=3)
nu = Particle{Float64}(0.0, 1, dof=2)

# --- scanned parameters: y (internal ModelParams convention) and sin2_2theta ---
# Bracket sized around the known-good benchmark point from test_pandemolate.jl:
# y=1e-4, sin2_2th=5.3e-13.
cfg = ScanConfigZ(
    sin2_2theta_min=1e-12,
    sin2_2theta_max=1e-14,
    n_theta=3,
    log10y_min=-7.0,
    log10y_max=-3.0,
    n_coarse=15,
)

@time points = run_scan_z(N1, N2, A, nu, cfg; verbose=false)
save_results_z(points, joinpath(@__DIR__, "../tmp/relic_contour_z.csv"))

ok_points = filter(p -> p.converged, points)
branches = unique(p.branch for p in ok_points)

plt = plot(
    xscale=:log10, yscale=:log10,
    xlabel="y", ylabel="sin²2θ",
    title="Ωh² = $(OMEGA_H2_TARGET_Z) contour (z-parameterized solver)",
    legend=:topright
)

for b in sort(branches)
    bp = filter(p -> p.branch == b, ok_points)
    sort!(bp, by=p -> p.sin2_2theta)
    plot!(
        plt, [p.y for p in bp], [p.sin2_2theta for p in bp],
        label="branch $b", marker=:circle, markersize=2
    )
end

savefig(plt, joinpath(@__DIR__, "../figures/relic_contour_z.pdf"))
