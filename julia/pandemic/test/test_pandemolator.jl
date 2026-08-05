using LaTeXStrings
ENV["GKSwstype"] = "nul"
import Plots as Plt

include(joinpath(@__DIR__, "../src/time_temp_relation.jl"))
include(joinpath(@__DIR__, "../src/dodelson_widrow.jl"))
include(joinpath(@__DIR__, "../src/pandemolator.jl"))

function test_pandemolator()
    m_N = 1e-5
    k_N = 1
    dof_N = 2

    m_A = 2.5*m_N
    k_A = -1
    dof_A = 3

    m_nu = 0.
    k_nu = 1
    dof_nu = 2

    N1 = Particle{Float64}(m_N, k_N, dof=dof_N)
    N2 = Particle{Float64}(m_N, k_N, dof=dof_N)
    A = Particle{Float64}(m_A, k_A, dof=dof_A)
    nu = Particle{Float64}(m_nu, k_nu, dof=dof_nu)

    y = 1e-4
    sin2_2th = 5e-16
    th = asin(sqrt(sin2_2th)) / 2.
    mp = ModelParams{Float64}(y, th)

    tT_rel = TimeTempRelation{Float64}(t_gp_pd=20)

    dw = DodelsonWidrow{Float64}(m_N, th, tT_rel)

    pan = Pandemolator{Float64}(
        mp,
        N1, N2, A, nu,
        tT_rel,
        dw
    )

    sol = pandemolate(tT_rel, dw, pan)
    println("sol = ", sol)

    p = Plt.scatter(
        sol.t, sol.u[1]
    )
    Plt.savefig(p, "figures/first_y.pdf")
    # pn = Plt.scatter(
    #     xscale=:log10,
    #     T_nu, cn
    # )
    # Plt.savefig("figures/cn.pdf")
    # prho = Plt.scatter(
    #     xscale=:log10,
    #     T_nu, crho
    # )
    # Plt.savefig("figures/crho.pdf")

    return nothing
end

test_pandemolator()