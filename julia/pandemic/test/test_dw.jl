include(joinpath(@__DIR__, "../src/time_temp_relation.jl"))
include(joinpath(@__DIR__, "../src/dodelson_widrow.jl"))

function test_dw()
    m_N1 = 1e-5
    th = 1e-8
    tt_rel = TimeTempRelation{Float64}(t_gp_pd=20)
    dw = DodelsonWidrow{Float64}(m_N1, th, tt_rel)
end

test_dw()
