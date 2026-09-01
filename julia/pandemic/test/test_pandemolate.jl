using LaTeXStrings
ENV["GKSwstype"] = "nul"
import Plots as Plt
import DataFrames # as DF
# import CSV

include(joinpath(@__DIR__, "../src/utils.jl"))
include(joinpath(@__DIR__, "../src/time_temp_relation.jl"))
include(joinpath(@__DIR__, "../src/dodelson_widrow.jl"))
include(joinpath(@__DIR__, "../src/pandemolator.jl"))

function test_pandemolator()
    m_N = 1e-5
    k_N = 1
    dof_N = 2

    m_A = 2.5 * m_N
    k_A = -1
    dof_A = 3

    m_nu = 0.
    k_nu = 1
    dof_nu = 2

    N1 = Particle{Float64}(m_N, k_N, dof=dof_N)
    N2 = Particle{Float64}(m_N, k_N, dof=dof_N)
    A = Particle{Float64}(m_A, k_A, dof=dof_A)
    nu = Particle{Float64}(m_nu, k_nu, dof=dof_nu)

    # TODO: Remember that y is rescaled below! Remove rescaling when we don't have to compare to Python code.
    y_pyt = 1e-5
    sin2_2th = 2.65e-11
    # y_pyt = 1e-5
    # sin2_2th = 1e-11


    # Rescale to compare with python code. 
    # D.o.f. was forgotten in Python code for collision term
    y = y_pyt / sqrt(dof_N * dof_A * dof_nu)

    th = asin(sqrt(sin2_2th)) / 2.
    mp = ModelParams{Float64}(y, th)

    tT_rel = TimeTempRelation{Float64}()

    dw = DodelsonWidrow{Float64}(m_N, th, tT_rel)

    pan = Pandemolator{Float64}(
        mp,
        N1, N2, A, nu,
        tT_rel,
        dw
    )

    sol = pandemolate(tT_rel, dw, pan)

    results = transform_sol(pan, sol)

    println(results.size)

    export_array_to_csv(DataFrames.DataFrame(results, :auto), joinpath(@__DIR__, "../tmp/sol.csv"))

    return nothing
end

test_pandemolator()
