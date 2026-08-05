mutable struct MyType{T<:Real}
    x::T
    y::T

    function MyType{T}(x::T) where T<:Real
        y = 3.

        new{T}(x, y)
    end
end

test = MyType{Float64}(1.)
println(test)