"""
Tests of src/lyman_alpha.jl. Run from julia/pandemic:
    julia --project=. test/test_lyman_alpha.jl

1. `mean_speed` in the ultra-relativistic and Maxwell-Boltzmann limits, and
   its scaling for a frozen non-relativistic spectrum.
2. `comoving_distance` with the Dodelson-Widrow spectra of hep-ph/0612182
   (data/dw/0612182_dw_fig_7_*keV.dat) against the Python implementation,
   data/dw/0612182_fs_length.py (which integrates to redshift 15).
(The lengths at relic roots of both solvers are checked against the stored scans in
test/test_regression.jl.)
"""

using Test
using DelimitedFiles

include(joinpath(@__DIR__, "..", "src", "lyman_alpha.jl"))

const TT_REL = TimeTempRelation{Float64}()

"""The background interpolants lyman_alpha.jl needs, keyed on z = log(m_N/T_nu)."""
function make_bg(m_N)
    z = log.(x_grid_of(m_N, TT_REL.T_nu_grid))
    return (ent_interp_z=z_interpolation(z, TT_REL.ent_grid),
            dx_dt_interp_z=z_interpolation(z, dx_dt_grid_of(m_N, TT_REL.T_nu_grid, TT_REL.dT_nu_dt_grid)))
end

trapz(x, y) = sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1])) / 2

@testset "mean_speed limits" begin
    N = Particle{Float64}(1e-5, 1, dof=2)
    # Ultra-relativistic: <v> = 1 - O(m^2/T^2).
    @test isapprox(mean_speed(N, 1e-1, 1e-4), 1.0; atol=1e-6)
    # Maxwell-Boltzmann, non-relativistic: <v> = sqrt(8T/(pi m)) (1 + O(T/m)).
    T = 1e-5 * 1e-5
    v_mb = sqrt(8T / (pi * 1e-5))
    @test isapprox(mean_speed(N, T, 50.0), v_mb; rtol=1e-4)
    # The gap cap: the ratio is the same for any gap in the MB regime.
    @test mean_speed(N, T, 40.0) == mean_speed(N, T, 1e4)
    # Maxwell-Boltzmann particles (k = 0): the closed form equals the quadrature.
    N_mb = Particle{Float64}(1e-5, 0, dof=2)
    for Tm in (1e2, 1.0, 1e-2)
        q = thermal_moment(N_mb, Tm * 1e-5, 0.0, (E, mom) -> mom^2) / thermal_moment(N_mb, Tm * 1e-5, 0.0, (E, mom) -> E * mom)
        @test isapprox(mean_speed(N_mb, Tm * 1e-5, 0.0), q; rtol=1e-8)
    end
    # Frozen non-relativistic spectrum: <v> ∝ r.
    @test isapprox(mean_speed(N, T, 50.0; r=1e-3), 1e-3 * v_mb; rtol=1e-4)
    # Fermi-Dirac at xi = 0, massless limit of <p>: 7 pi^4 / (180 zeta3) T, checked via
    # a frozen relativistic spectrum redshifted to non-relativistic speeds: <v> -> r <p>/m.
    T_r = 1e-3
    @test isapprox(mean_speed(N, T_r, 1e-5 / T_r; r=1e-6) / 1e-6,
                   7pi^4 / (180 * zeta3) * T_r / 1e-5; rtol=1e-3)
end

@testset "DW spectra vs 0612182_fs_length.py" begin
    # Output of data/dw/0612182_fs_length.py (Python, redshift 15), in Mpc.
    ref = Dict(1 => 1.7791290567757372, 2 => 0.94711936012508, 4 => 0.49890119989184495,
               8 => 0.25548366336508693, 16 => 0.12771049319672886, 32 => 0.06231992527566675)
    for (m_keV, lam_py) in sort(collect(ref))
        m = m_keV * 1e-6
        data = readdlm(joinpath(@__DIR__, "..", "data", "dw", "0612182_dw_fig_7_$(m_keV)keV.dat"))
        x = data[:, 1]
        f = data[:, 2] ./ (exp.(x) .+ 1)          # normalisation drops out of <v>
        p = x .* 1e-3                               # momentum in GeV when T_nu = 1 MeV
        bg = make_bg(m)
        a_1MeV = scale_factor_z(bg, log(m / 1e-3))
        n = trapz(p, f .* p .^ 2)
        v_dw(z) = begin
            q = p .* (a_1MeV / scale_factor_z(bg, z))
            trapz(p, f .* p .^ 2 .* _speed.(q, m)) / n
        end
        z_i = log(m / T_d_dw(m))
        z_f = z_at_redshift(m, TT_REL, 15.0)
        lam = comoving_distance(bg, v_dw, z_i, z_f; rtol=1e-6) / Mpc
        println("  DW $m_keV keV: Julia $(round(lam, sigdigits=5)) Mpc, Python $(round(lam_py, sigdigits=5)) Mpc, " *
                "ratio $(round(lam / lam_py, sigdigits=5))")
        @test isapprox(lam, lam_py; rtol=1e-2)
    end
end

@testset "splitting at fixed m_nu" begin
    # mnu_splitting inverts m_nu = sin^2 theta_1 m_D/delta, m_D = (1 + delta) m_N
    for (m, s2, m_nu) in ((1e-5, 1e-12, 1e-10), (6e-7, 1.5e-7, 1e-10), (3e-4, 1e-20, 1e-14))
        d = mnu_splitting(m, s2; m_nu=m_nu)
        th1 = sin(asin(sqrt(s2)) / 2)
        @test th1^2 * m * (1 + d) / d ≈ m_nu rtol = 1e-10
    end
    # N2 N2 -> N1 N1 from the equal-mass tables (ln_conversion_shift) vs the delta = 1e-4 tables
    dir = joinpath(@__DIR__, "..", "tmp", "rate_tables")
    t0 = RateInterpolants(load_rate_tables(rate_table_path(dir)))
    t4 = RateInterpolants(load_rate_tables(rate_table_path(dir; delta=1e-4)))
    m = 1.057e-5
    r = reaction(:N1N1_N2N2)
    l0, l4 = (ln_norm(r, table_params(t, m, 1e-3, 3.6e-15), t.p_ref) for t in (t0, t4))
    a, b = t0.dark[:N1N1_N2N2], t4.dark[:N1N1_N2N2]
    err = maximum(range(t0.lnxN..., length=200)) do lx
        T = m / exp(lx)
        abs(b.lnRr(lx) + l4 - (a.lnRr(lx) + l0 + ln_conversion_shift(m, m * (1 + 1e-4), T)))
    end
    println("  max |ln R_r(tables, delta = 1e-4) - ln R_r(equal-mass tables + shift)| = $err")
    @test err < 0.01
end
