using Test, Printf, Random
import QuadGK
import SpecialFunctions as SF

include(joinpath(@__DIR__, "../src/coll_12_34.jl"))
include(joinpath(@__DIR__, "../src/coll_12_34_legendre.jl"))

"""
Tests for the A'A' <-> N N collision integral with the amplitude of eq. (A9):

  coll_12_34           src/coll_12_34.jl: nested adaptive quadrature over
                       E1, E2, E3, s with the t-integral in closed form.
  coll_12_34_legendre  src/coll_12_34_legendre.jl: CM-frame Legendre /
                       Funk-Hecke decomposition.

Independent references:
  - explicit 4-vectors for the t-limits,
  - numerical t-integration for the closed form,
  - the cross section eq. (A10) for the amplitude,
  - the Maxwell-Boltzmann limit (k = 0 particles), where
      C = (e^{xi1+xi2} - e^{xi3+xi4}) T/(512 pi^5) ∫ds K1(√s/T)/√s ∫dt |M|^2
    and ∫dt |M|^2 = g1 g2 S 16π λ(s, mA^2, mA^2) sigma(s) is taken from eq. (A10).

Run: julia --project=. test/test_coll_12_34_AA_NN.jl
"""

const M_N = 1e-5
const M_A = 2.5 * M_N
const G_COUP = 1e-4
const MP = ModelParams{Float64}(G_COUP, 0.)

function sigma_A10(s, mA, mN, g)
    R = sqrt((s - 4mN^2) * (s - 4mA^2))
    return g^4 / (18pi * s * (s - 4mA^2)) * (
        -R * (4mN^4 + mN^2 * s + 2mA^4) / (mN^2 * (s - 4mA^2) + mA^4)
        + (-8mN^4 + 4mN^2 * (s - 2mA^2) + 4mA^4 + s^2) / (s - 2mA^2)
        * log((s - 2mA^2 + R) / (s - 2mA^2 - R))
    )
end

# ∫dt |M|^2 over the full t-range, from σ with g1 g2 = 9 and S = 2.
int_t_msq_A10(s) = 288pi * s * (s - 4M_A^2) * sigma_A10(s, M_A, M_N, G_COUP)

function coll_MB_reference(T, xi_A, xi_N)
    w_th = 2M_A
    # s = (w_th + y^2)^2 removes the threshold square root.
    f(y) = begin
        w = w_th + y^2
        4 * w * y * SF.besselk(1, w / T) / w * int_t_msq_A10(w^2)
    end
    y_max = sqrt(80T)
    I = QuadGK.quadgk(f, 0., y_max / 4, y_max / 2, y_max; rtol=1e-11)[1]
    # `dist` is the occupation per dof, so no dof product here.
    return (exp(2xi_A) - exp(2xi_N)) * T / (512pi^5) * I
end

function random_config(rng, A, N)
    while true
        e1 = M_A * (1 + 20rand(rng)); e2 = M_A * (1 + 20rand(rng)); E = e1 + e2
        e3 = M_N + (E - 2M_N) * rand(rng)
        p = Params_12_34{Float64,Float64}(A, A, N, N, (M_N, M_N, M_N, M_N), (0., 0., 0., 0.))
        p.e1 = e1; p.mom1 = momentum(A, e1); p.e2 = e2; p.mom2 = momentum(A, e2)
        p.e3 = e3; p.mom3 = momentum(N, e3); p.e4 = E - e3; p.mom4 = momentum(N, p.e4)
        smin, smax = s_min(p), s_max(p)
        smin >= smax && continue
        p.s = smin + (smax - smin) * rand(rng)
        p.a = a_theta(p)
        return p
    end
end

@testset "A'A' <-> NN collision integral" begin
    A = Particle{Float64}(M_A, -1, dof=3)
    N = Particle{Float64}(M_N, 1, dof=2)
    rng = MersenneTwister(1234)

    @testset "t-limits vs explicit 4-vectors" begin
        worst = 0.
        for _ in 1:500
            p = random_config(rng, A, N)
            E = p.e1 + p.e2
            P = sqrt(E^2 - p.s)
            c1 = (E * p.e1 - p.s / 2) / (p.mom1 * P)
            c3 = (E * p.e3 - p.s / 2) / (p.mom3 * P)
            v1 = p.mom1 .* (sqrt(1 - c1^2), 0., c1)
            Pv = (0., 0., P)
            # p2 = P - p1 must be on shell.
            v2 = Pv .- v1
            @test isapprox(sum(v2 .^ 2), p.e2^2 - M_A^2; rtol=1e-8)
            ts = map(range(0, 2pi, length=4001)) do phi
                v3 = p.mom3 .* (sqrt(1 - c3^2) * cos(phi), sqrt(1 - c3^2) * sin(phi), c3)
                v4 = Pv .- v3
                @assert isapprox(sum(v4 .^ 2), p.e4^2 - M_N^2; rtol=1e-8)
                M_A^2 + M_N^2 - 2 * (p.e1 * p.e3 - sum(v1 .* v3))
            end
            scale = 2 * p.e1 * p.e3
            # Sign convention of Bringmann et al. (arXiv:2206.10630): a < 0,
            # c_{theta,+} is the lower root, so t_lim(1) = t_min.
            @test a_theta(p) < 0
            worst = max(worst, abs(minimum(ts) - t_lim(1, p)) / scale, abs(maximum(ts) - t_lim(-1, p)) / scale)
            @test maximum(ts) <= 0
        end
        @printf("  worst relative t-limit deviation: %.2e\n", worst)
        @test worst < 1e-6
    end

    @testset "closed-form t-integral vs numerical" begin
        worst = 0.
        for _ in 1:500
            p = random_config(rng, A, N)
            s = p.s
            ana = coll_12_34_int_t_new(s, p)
            num = coll_12_34_int_t(s, p)
            worst = max(worst, abs(ana - num) / abs(num))
        end
        @printf("  worst relative deviation: %.2e\n", worst)
        @test worst < 1e-8
    end

    @testset "amplitude vs cross section (A10)" begin
        for s in 4M_A^2 .* (1.001, 1.3, 3., 30., 3e3, 3e5)
            k1 = sqrt(s / 4 - M_A^2); k3 = sqrt(s / 4 - M_N^2)
            t_lo = M_A^2 + M_N^2 - s / 2 - 2k1 * k3
            t_hi = M_A^2 + M_N^2 - s / 2 + 2k1 * k3
            u(t) = 2M_A^2 + 2M_N^2 - s - t
            p = Params_12_34{Float64,Float64}(A, A, N, N, (1., 1., 1., 1.), (0., 0., 0., 0.))
            p.s = s
            I1 = QuadGK.quadgk(t -> G_COUP^4 * coll_12_34_sq_amp(t, p), t_lo, t_hi; rtol=1e-12)[1]
            I2 = QuadGK.quadgk(t -> sq_amp_AA_NN(MP, A, A, N, N, s, t, u(t)), t_lo, t_hi; rtol=1e-12)[1]
            @test isapprox(I1, int_t_msq_A10(s); rtol=1e-7)
            @test isapprox(I2, int_t_msq_A10(s); rtol=1e-7)
        end
    end

    @testset "Maxwell-Boltzmann limit" begin
        A0 = Particle{Float64}(M_A, 0, dof=3)
        N0 = Particle{Float64}(M_N, 0, dof=2)
        xi_A, xi_N = -1., -0.3
        for x in (0.01, 0.1, 1., 5., 30.)
            T = M_N / x
            temps = (T, T, T, T); xis = (xi_A, xi_A, xi_N, xi_N)
            ref = coll_MB_reference(T, xi_A, xi_N)
            t_leg = @elapsed leg = coll_12_34_legendre(MP, A0, A0, N0, N0, temps, xis)
            @printf("  x = %5.2f: ref = % .6e   legendre rel.err = %.2e (%.3f s)\n", x, ref, leg / ref - 1, t_leg)
            @test isapprox(leg, ref; rtol=1e-5)
            # The nested integrator needs ~1 min per call at reltol = 1e-3.
            if x in (1., 5.)
                t_nest = @elapsed nest = coll_12_34(MP, A0, A0, N0, N0, temps, xis; reltol=1e-3)
                @printf("             nested   rel.err = %.2e (%.1f s)\n", nest / ref - 1, t_nest)
                @test isapprox(nest, ref; rtol=2e-3)
            end
        end
    end

    @testset "quantum statistics: legendre vs nested" begin
        cases = [
            ("x=1, xi_A = 2 xi_N < 0", M_N / 1., M_N / 1., -0.5, -1.),
            ("x=3, degenerate", M_N / 3., M_N / 3., 0.8, 1.6),
            ("x=0.3, relativistic", M_N / 0.3, M_N / 0.3, -0.1, -0.2),
            ("unequal temperatures", M_N / 1., 1.3 * M_N / 1., -0.2, -0.4),
        ]
        for (name, T_N, T_A, xi_N, xi_A) in cases
            temps = (T_A, T_A, T_N, T_N); xis = (xi_A, xi_A, xi_N, xi_N)
            t_leg = @elapsed leg = coll_12_34_legendre(MP, A, A, N, N, temps, xis)
            leg_hi = coll_12_34_legendre(MP, A, A, N, N, temps, xis; L=48, n_c=64, n_E=32, n_s=20, n_theta=32)
            t_nest = @elapsed nest = coll_12_34(MP, A, A, N, N, temps, xis; reltol=1e-3)
            @printf("  %-24s legendre = % .6e (%.3f s), conv = %.1e, nested rel.diff = %.2e (%.1f s)\n",
                name, leg, t_leg, leg / leg_hi - 1, nest / leg - 1, t_nest)
            @test isapprox(leg, leg_hi; rtol=1e-6)
            @test isapprox(nest, leg; rtol=2e-3)
        end
    end

    @testset "non-finite input returns NaN" begin
        # The ODE solver can evaluate trial states with NaN T or xi; the
        # collision term must return NaN (step rejected), not throw.
        T = M_N
        @test isnan(coll_12_34_legendre(MP, A, A, N, N, (NaN, NaN, NaN, NaN), (NaN, NaN, NaN, NaN)))
        @test isnan(coll_12_34_legendre(MP, A, A, N, N, (T, T, T, T), (-1., -1., NaN, NaN)))
        @test isnan(coll_12_34_legendre(MP, A, A, N, N, (0., 0., 0., 0.), (-1., -1., -0.5, -0.5)))
    end

    @testset "chemical equilibrium gives zero" begin
        T = M_N
        temps = (T, T, T, T); xis = (-0.4, -0.4, -0.4, -0.4)
        @test coll_12_34_legendre(MP, A, A, N, N, temps, xis) == 0
        ref = abs(coll_12_34_legendre(MP, A, A, N, N, temps, (-0.8, -0.8, -0.4, -0.4)))
        @test abs(coll_12_34(MP, A, A, N, N, temps, xis; reltol=1e-5)) < 1e-6 * ref
    end
end
