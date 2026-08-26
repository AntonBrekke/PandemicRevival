include(joinpath(@__DIR__, "../src/time_temp_relation.jl"))
include(joinpath(@__DIR__, "../src/dodelson_widrow.jl"))

function test_dw()
    m_N1 = 1e-5
    sin2_2th = 5e-16
    th = asin(sqrt(sin2_2th)) / 2.
    tt_rel = TimeTempRelation{Float64}()
    dw = DodelsonWidrow{Float64}(m_N1, th, tt_rel)

    T_ic = tt_rel.T_nu_grid[dw.i_ic]
    T_end = tt_rel.T_nu_grid[dw.i_end]

    println("T_dw = ", dw.T_dw)
    println("i_ic = ", dw.i_ic)
    println("T_ic = ", T_ic)
    println("i_end = ", dw.i_end)
    println("T_end = ", T_end)
    println("n_ic = ", dw.n_ic)
    println("rho_ic = ", dw.rho_ic)
end

test_dw()
