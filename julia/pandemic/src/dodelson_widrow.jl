include(joinpath(@__DIR__, "constants_functions.jl"))

mutable struct DodelsonWidrow{T<:Real}
    T_dw::T
    i_ic::Int64
    i_end::Int64
    n_ic::T
    rho_ic::T

    function DodelsonWidrow{T}(
            m_N::T,
            th::T,
            tt_rel::TimeTempRelation
        ) where {T<:Real}
        T_dw = T_d_dw(m_N)

        i_ic = findfirst(tt_rel.T_nu_grid .< T_dw)
        i_end = findfirst(tt_rel.T_nu_grid .< m_N/2e1)

        sf_ic_norm_0 = (s0/(s_SM_no_nu(tt_rel.T_SM_grid[i_ic]) + s_nu(tt_rel.T_nu_grid[i_ic])))^(1/3)

        n_ic = n_0_dw(m_N, th) / sf_ic_norm_0^3
        rho_ic = n_ic * avg_mom_0_dw(m_N) / sf_ic_norm_0

        new{T}(
            T_dw,
            i_ic,
            i_end,
            n_ic,
            rho_ic
        )
    end
end
