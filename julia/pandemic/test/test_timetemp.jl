include(joinpath(@__DIR__, "../src/time_temp_relation.jl"))

function main()
    for i in 1:100
        rel = TimeTempRelation()
    end
end

@time main()
