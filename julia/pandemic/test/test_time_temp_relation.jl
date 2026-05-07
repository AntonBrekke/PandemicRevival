ENV["GKSwstype"] = "nul"
using Plots

include(joinpath(@__DIR__, "time_temp_relation.jl"))

# Use a coarse grid so the test runs quickly.
rel = TimeTempRelation(t_gp_pd=20)

p = plot(
    rel.t_grid,
    rel.T_SM_grid;
    xscale=:log10,
    yscale=:log10,
    xlabel="t [GeV^-1]",
    ylabel="Temperature [GeV]",
    label="T_SM",
    lw=2,
    legend=:topright,
    # minorgrid=true,
)
plot!(p, rel.t_grid, rel.T_nu_grid; label="T_nu", lw=2, ls=:dash)

out_path = joinpath(@__DIR__, "..", "figures", "time_temp_relation_temp_vs_time.pdf")
savefig(p, out_path)
println("Saved figure to: " * out_path)
