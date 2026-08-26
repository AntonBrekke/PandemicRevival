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

    y = 1e-4
    sin2_2th = 5e-16
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

    x_pan = m_N ./ tT_rel.T_nu_grid
    println("x_0 = ", x_pan[1])
    println("x_end = ", x_pan[end])
    println("T_nu_0 = ", tT_rel.T_nu_grid[1])
    println("T_nu_end = ", tT_rel.T_nu_grid[end])
    x = logrange(x_pan[1], x_pan[end], 1000)
    T_nu = m_N ./ x
    dT_dt = pan.dT_nu_dt_interp_T_nu.(T_nu)
    ent = pan.ent_interp_T_nu.(T_nu)
    hubble = pan.H_interp_T_nu.(T_nu)
    sf = pan.sf_interp_T_nu.(T_nu)

    pandemolator_array = [x;; T_nu;; dT_dt;; ent;; hubble;; sf]
    csv_path = joinpath(@__DIR__, "../tmp/test_pandemolator.csv")
    export_array_to_csv(
        DataFrames.DataFrame(pandemolator_array, :auto),
        csv_path
    )

    return nothing
end

test_pandemolator()
