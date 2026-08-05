using LaTeXStrings
ENV["GKSwstype"] = "nul"
import Plots as Plt

include(joinpath(@__DIR__, "../src/time_temp_relation.jl"))
include(joinpath(@__DIR__, "../src/utils.jl"))


# Use a coarse grid so the test runs quickly.
rel = TimeTempRelation(t_gp_pd=20)

p = Plt.plot(
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
Plt.plot!(p, rel.t_grid, rel.T_nu_grid; label="T_nu", lw=2, ls=:dash)

out_path = joinpath(@__DIR__, "../figures/time_temp_relation_temp_vs_time.pdf")
Plt.savefig(p, out_path)
println("Saved figure to: " * out_path)

time_temp_arrray = [rel.t_grid;; rel.T_SM_grid;; rel.T_nu_grid]

csv_file = joinpath(@__DIR__, "../tmp/test_time_temp.csv")

export_array_to_csv(time_temp_arrray, csv_file)

# println(rel.t_grid)
# println(rel.ent_grid)

p2 = Plt.plot(
    # minorgrid=true,
    xlabel=L"$t$",
    y_label=L"$s/s_0$",
    xscale=:log10,
    yscale=:log10,
    xlim=(1e15, 1e40),
    ylim=(1e-50, 1e-20)
)
Plt.plot!(
    p2,
    rel.t_grid,
    rel.ent_grid ./ rel.ent_grid[1],
)
Plt.plot!(
    p2,
    rel.t_grid,
    rel.sf_grid[1]^3 ./ rel.sf_grid .^ 3
)
path2 = joinpath(@__DIR__, "../figures/test_ent_scalefactor.pdf")
Plt.savefig(p2, path2)

comp = rel.ent_grid .* rel.sf_grid .^ 3 ./ (rel.ent_grid[1] * rel.sf_grid[1]^3)
p3 = Plt.plot(
    xscale=:log10,
    # yscale=:log10,
    ylim=(1 - 1e-2, 1 + 1e-2),
    # ylim=(0., 2.)
)
Plt.plot!(
    rel.t_grid,
    comp
)
path3 = joinpath(@__DIR__, "../figures/test_ent_scalefactor2.pdf")
Plt.savefig(p3, path3)