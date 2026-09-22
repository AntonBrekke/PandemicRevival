"""
run_relic_scan_mN_z.jl

Example driver for relic_scan_mN_z.jl: scans m_N (= m_N1 = m_N2, with
m_A = 2.5*m_N held fixed) and sin2_2theta, root-finding y for the correct
relic abundance at every (m_N, sin2_2theta) point.

m_N range: 3 keV to 100 keV, inside the "2 keV to 0.2 MeV" viable window
the draft paper quotes for this scenario.

`threaded=true` below parallelizes the m_N loop -- run this script with
`julia -t N` (or `JULIA_NUM_THREADS=N`) for it to do anything; with the
default single Julia thread it just runs the (safe, cheap) cache warm-up
and then executes serially. Measured 5.6x on 8 threads for an 8-point m_N
grid -- see the THREADING section of relic_scan_mN_z.jl's module docstring
for why this loop (unlike relic_scan_z.jl's) is safe to parallelize.
"""

include(joinpath(@__DIR__, "relic_scan_mN_z.jl"))
using Plots

cfg = ScanConfigMNZ(
    m_N_min=3e-6, m_N_max=1e-4, n_mN=6,
    sin2_2theta_min=1e-15, sin2_2theta_max=1e-11, n_theta=8,
    log10y_min=-7.0, log10y_max=-3.0, n_coarse=15,
)

@time points = run_scan_mN_z(cfg; verbose=false, threaded=true)
save_results_mN_z(points, joinpath(@__DIR__, "../tmp/relic_contour_mN_z.csv"))

ok_points = filter(p -> p.converged, points)

plt = plot(
    xscale=:log10, yscale=:log10, zscale=:log10,
    xlabel="m_N [GeV]", ylabel="sin²2θ", zlabel="y",
    title="Ωh² = $(OMEGA_H2_TARGET_Z) surface (z-parameterized solver)",
    legend=false,
    camera=(45, 30),
)
scatter!(
    plt,
    [p.m_N for p in ok_points], [p.sin2_2theta for p in ok_points], [p.y for p in ok_points],
    marker=:circle, markersize=2,
)
savefig(plt, joinpath(@__DIR__, "../figures/relic_contour_mN_z_3d.pdf"))

# 2D slices at each scanned m_N, colored, for readability alongside the 3D plot
plt2 = plot(
    xscale=:log10, yscale=:log10,
    xlabel="y", ylabel="sin²2θ",
    title="Ωh² = $(OMEGA_H2_TARGET_Z) contours vs. m_N",
    legend=:outertopright,
)
for m_N in sort(unique(p.m_N for p in ok_points))
    mp = filter(p -> p.m_N == m_N, ok_points)
    sort!(mp, by=p -> p.sin2_2theta)
    plot!(plt2, [p.y for p in mp], [p.sin2_2theta for p in mp],
        label="m_N=$(round(m_N, sigdigits=3))", marker=:circle, markersize=2)
end
savefig(plt2, joinpath(@__DIR__, "../figures/relic_contour_mN_z_slices.pdf"))
