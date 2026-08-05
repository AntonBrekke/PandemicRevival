mutable struct PandemicResult{T<:Real}
    t::Vector{T}
    T_nu::Vector{T}
    H::Vector{T}
    T_dm::Vector{T}
    xi_N::Vector{T}
    xi_A::Vector{T}
    n_N::Vector{T}
    n_A::Vector{T}

    function PandemicResult{T}(n::Int64) where T <: Real
        t = Array{T}(undef, n)
        T_nu = Array{T}(undef, n)
        H = Array{T}(undef, n)
        T_N = Array{T}(undef, n)
        xi_N = Array{T}(undef, n)
        xi_A = Array{T}(undef, n)
        n_N = Array{T}(undef, n)
        n_A = Array{T}(undef, n)

        new{T}(
            t,
            T_nu,
            H,
            T_N,
            xi_N,
            xi_A,
            n_N,
            n_A,
        )
    end
end
