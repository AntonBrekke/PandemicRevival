"""
Tests of src/lyman_alpha.jl. Run from julia/pandemic:
    julia --project=. test/test_lyman_alpha.jl [--no-solve]

1. `mean_speed` in the ultra-relativistic and Maxwell-Boltzmann limits, and
   its scaling for a frozen non-relativistic spectrum.
2. `comoving_distance` with the Dodelson-Widrow spectra of hep-ph/0612182
   (data/dw/0612182_dw_fig_7_*keV.dat) against the Python implementation,
   data/dw/0612182_fs_length.py (which integrates to redshift 15).
3. (skipped with --no-solve) `free_streaming_length` on one relic root of the
   theta scan: the solve reproduces the scan's Omega h^2, and lambda_fs is
   close to the DW value at the smallest coupling, where the N spectrum is
   the DW one.
"""

using Test
using DelimitedFiles

include(joinpath(@__DIR__, "..", "src", "lyman_alpha.jl"))

const TT_REL = TimeTempRelation{Float64}()

function make_pan(m_N)
    N1 = Particle{Float64}(m_N, 1, dof=2)
    N2 = Particle{Float64}(m_N, 1, dof=2)
    A = Particle{Float64}(2.5 * m_N, -1, dof=3)
    nu = Particle{Float64}(0.0, 1, dof=2)
    return PandemolatorZ{Float64}(ModelParams{Float64}(1e-4, 1e-6), N1, N2, A, nu, TT_REL)
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
        pan = make_pan(m)
        a_1MeV = scale_factor_z(pan, log(m / 1e-3))
        n = trapz(p, f .* p .^ 2)
        v_dw(z) = begin
            q = p .* (a_1MeV / scale_factor_z(pan, z))
            trapz(p, f .* p .^ 2 .* _speed.(q, m)) / n
        end
        z_i = log(m / T_d_dw(m))
        z_f = z_at_redshift(m, TT_REL, 15.0)
        lam = comoving_distance(pan, v_dw, z_i, z_f; rtol=1e-6) / Mpc
        println("  DW $m_keV keV: Julia $(round(lam, sigdigits=5)) Mpc, Python $(round(lam_py, sigdigits=5)) Mpc, " *
                "ratio $(round(lam / lam_py, sigdigits=5))")
        @test isapprox(lam, lam_py; rtol=1e-2)
    end
end

if !("--no-solve" in ARGS)
    @testset "relic root, m_N = 10.59 keV" begin
        # Roots of tmp/relic_scan_theta/m_N_10.59keV.csv (m_N, y, sin^2 2theta, Omega h^2).
        m = 1.0589999999999999e-5
        roots = [(1.0e-6, 4.850250451471269e-10, 0.11997665067244329),
                 (1.0e-4, 5.472320061540568e-13, 0.12002813662289068)]
        pan = make_pan(m)
        res = map(roots) do (y, s, om_scan)
            theta = asin(sqrt(s)) / 2
            t = @elapsed sol, om, conv, rc = solve_relic_point(pan, TT_REL, y, theta)
            @test conv
            @test isapprox(om, om_scan; rtol=2e-2)
            t_l = @elapsed fs = free_streaming_length(pan, TT_REL, sol)
            println("  y = $y: Omega h^2 = $(round(om, sigdigits=5)) (scan $(round(om_scan, sigdigits=5))), " *
                    "lambda_fs = $(round(fs.lambda_fs, sigdigits=4)) Mpc (eq $(round(fs.lambda_fs_eq, sigdigits=4)), " *
                    "free $(round(fs.lambda_fs_free, sigdigits=4))), solve $(round(t, digits=1)) s, lambda_fs $(round(t_l, digits=2)) s")
            @test isfinite(fs.lambda_fs) && fs.lambda_fs > 0
            fs
        end
        # At y = 1e-6 the N spectrum is essentially the DW one: lambda_fs should be
        # close to the DW value, interpolated from the Python results (~1/m), up to
        # the thermal ansatz for the spectrum and the end redshift (50 vs 15).
        lam_dw = 0.25548366336508693 * (8 / 10.59)
        println("  DW (Python, z_red = 15) at 10.59 keV: ≈ $(round(lam_dw, sigdigits=3)) Mpc")
        @test 0.7 < res[1].lambda_fs / lam_dw < 1.2
    end
end
