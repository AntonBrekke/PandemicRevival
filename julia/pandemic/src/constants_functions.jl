include(joinpath(@__DIR__, "interpolation_helpers.jl"))
include(joinpath(@__DIR__, "utils.jl"))

using DelimitedFiles

# Mathematical constants
const pi2 = pi * pi
const zeta3 = 1.202056903

const temp_nu_dec_sm = 1.4e-3
const hubble_T5_nu_dec = 1.62211799511e-10
const G = 6.7086096877e-39
const hbar = 6.582119514e-25
const Mpc = 1.563738357134461e38

const conv_GeV_cm_3 = 1.3014892628900395e41
const omega_d0 = 0.12
const omega_b0 = 0.02237
const rho_crit0_h2 = 1.053672e-5 / conv_GeV_cm_3
const rho_d0 = omega_d0 * rho_crit0_h2
const rho_b0 = omega_b0 * rho_crit0_h2
const rho_m0 = rho_d0 + rho_b0
const T0 = 2.72548 * 8.6173324e-14
const s0 = (2.0 * pi2 / 45.0) * (T0^3.0) * 3.9267

const _DATA_DIR = joinpath(@__DIR__, "..", "data")
const _G_STAR_DIR = joinpath(_DATA_DIR, "g_star")
const _DW_DIR = joinpath(_DATA_DIR, "dw")

function _load_table(path::AbstractString; skipstart::Int=0)
    return readdlm(path, Float64; skipstart=skipstart)
end

function _col(mat::AbstractMatrix, i::Int)
    return Vector{Float64}(mat[:, i])
end

const _gstar_cache = Ref{Any}(nothing)

function _get_gstar_cache()
    if _gstar_cache[] === nothing
        g_rho_no_nu_grid = _load_table(joinpath(_G_STAR_DIR, "g_rho_no_nu.dat"))
        g_rho_der_no_nu_grid = _load_table(joinpath(_G_STAR_DIR, "g_rho_der_no_nu.dat"))
        g_s_no_nu_grid = _load_table(joinpath(_G_STAR_DIR, "g_s_no_nu.dat"))
        g_P_no_nu_grid = _load_table(joinpath(_G_STAR_DIR, "g_P_no_nu.dat"))

        _gstar_cache[] = (
            g_rho_no_nu_grid=g_rho_no_nu_grid,
            g_rho_no_nu=LogInterp(_col(g_rho_no_nu_grid, 1), _col(g_rho_no_nu_grid, 2); extrap="b"),
            g_rho_der_no_nu_grid=g_rho_der_no_nu_grid,
            g_rho_der_no_nu=LogInterp(_col(g_rho_der_no_nu_grid, 1), _col(g_rho_der_no_nu_grid, 2); extrap="b"),
            g_s_no_nu_grid=g_s_no_nu_grid,
            g_s_no_nu=LogInterp(_col(g_s_no_nu_grid, 1), _col(g_s_no_nu_grid, 2); extrap="b"),
            g_P_no_nu_grid=g_P_no_nu_grid,
            g_P_no_nu=LogInterp(_col(g_P_no_nu_grid, 1), _col(g_P_no_nu_grid, 2); extrap="b"),
        )
    end
    return _gstar_cache[]
end

g_rho_no_nu(T) = _get_gstar_cache().g_rho_no_nu(T)
g_rho_der_no_nu(T) = _get_gstar_cache().g_rho_der_no_nu(T)
g_s_no_nu(T) = _get_gstar_cache().g_s_no_nu(T)
g_P_no_nu(T) = _get_gstar_cache().g_P_no_nu(T)

g_rho_before_nu_dec(T) = g_rho_no_nu(T) + 6.0 * 7.0 / 8.0
g_rho_der_before_nu_dec(T) = g_rho_der_no_nu(T)
g_s_before_nu_dec(T) = g_s_no_nu(T) + 6.0 * 7.0 / 8.0
g_P_before_nu_dec(T) = g_P_no_nu(T) + 6.0 * 7.0 / 8.0

rho_SM_no_nu(T) = pi2 * g_rho_no_nu(T) * (T^4.0) / 30.0
rho_der_SM_no_nu(T) = (4.0 * g_rho_no_nu(T) + g_rho_der_no_nu(T) * T) * pi2 * (T^3.0) / 30.0
P_SM_no_nu(T) = pi2 * g_P_no_nu(T) * (T^4.0) / 90.0
s_SM_no_nu(T) = pi2 * g_s_no_nu(T) * (T^3.0) * 2.0 / 45.0

rho_SM_before_nu_dec(T) = pi2 * g_rho_before_nu_dec(T) * (T^4.0) / 30.0
rho_der_SM_before_nu_dec(T) = (4.0 * g_rho_before_nu_dec(T) + g_rho_der_before_nu_dec(T) * T) * pi2 * (T^3.0) / 30.0
P_SM_before_nu_dec(T) = pi2 * g_P_before_nu_dec(T) * (T^4.0) / 90.0
s_SM_before_nu_dec(T) = pi2 * g_s_before_nu_dec(T) * (T^3.0) * 2.0 / 45.0

rho_nu(T) = pi2 * 6.0 * (7.0 / 8.0) * (T^4.0) / 30.0
s_nu(T) = pi2 * 6.0 * (7.0 / 8.0) * (T^3.0) * 2.0 / 45.0

rho_m(T_SM, T_nu) = rho_m0 * (s_SM_no_nu(T_SM) + s_nu(T_nu)) / s0

const _dw_cache = Ref{Any}(nothing)

T_d_dw(md) = 0.133 * ((1e6 * md)^(1. / 3.))
sf_nu_dec_sm() = (s0 / s_SM_before_nu_dec(temp_nu_dec_sm))^(1.0 / 3.0)

function _get_dw_cache()
    if _dw_cache[] === nothing
        data_n_dw = _load_table(joinpath(_DW_DIR, "0612182_dw_fig_4.dat"); skipstart=2)
        x_n = _col(data_n_dw, 1)
        y_n = _col(data_n_dw, 2)
        log_C_e_dw_interp = LinearInterp(
            log.(x_n),
            log.(y_n);
            extrap_low=log(y_n[1]),
            extrap_high=log(y_n[end])
        )
        data_avg_mom_dw = _load_table(joinpath(_DW_DIR, "0612182_dw_fig_8.dat"); skipstart=2)
        x_avg = _col(data_avg_mom_dw, 1)
        y_avg = _col(data_avg_mom_dw, 2)
        avg_mom_interp_dw = LinearInterp(log.(x_avg), log.(y_avg);
            extrap_low=log(y_avg[1]), extrap_high=log(y_avg[end]))

        data_Tevo_dw = _load_table(joinpath(_DW_DIR, "0612182_dw_fig_3.dat"); skipstart=2)
        data_Tevo_dw[:, 2] ./= data_Tevo_dw[1, 2]
        Tevo_dw_interp = LinearInterp(
            log10.(_col(data_Tevo_dw, 1)),
            _col(data_Tevo_dw, 2);
            extrap_low=1.0, extrap_high=0.0
        )
        _dw_cache[] = (
            data_n_dw=data_n_dw,
            log_C_e_dw_interp=log_C_e_dw_interp,
            data_avg_mom_dw=data_avg_mom_dw,
            avg_mom_interp_dw=avg_mom_interp_dw,
            data_Tevo_dw=data_Tevo_dw,
            Tevo_dw_interp=Tevo_dw_interp,
        )
    end
    return _dw_cache[]
end

C_e_dw(md) = exp(_get_dw_cache().log_C_e_dw_interp(log(md)))
O_h2_dw(md, th) = 0.11 * C_e_dw(md) * (0.5 * sin(2. * th) * md * 1e10)^2
n_0_dw(md, th) = O_h2_dw(md, th) * rho_crit0_h2 / md

avg_mom_0_dw(md) = exp(_get_dw_cache().avg_mom_interp_dw(log(md))) *
    7.0 * pi2 * pi2 * temp_nu_dec_sm * sf_nu_dec_sm() / (180.0 * zeta3)

function O_h2_dw_Tevo(T, md, th)
    O_h2 = O_h2_dw(md, th)
    T_max_prod = T_d_dw(md)
    T_ref = T_d_dw(1e-5)
    T_rescaled = T * T_ref / T_max_prod
    return _get_dw_cache().Tevo_dw_interp(log10(T_rescaled)) * O_h2
end

