# Tests of the high-g one-fluid solver (src/pandemolator_hg.jl). No full solve (~2 min).
#   julia --project=. test/test_pandemolator_hg.jl
using Test
include(joinpath(@__DIR__, "..", "src", "pandemolator_hg.jl"))

tT_rel = TimeTempRelation{Float64}()
itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
m = 10.57e-6
g, th = 1e-2, asin(sqrt(2.6761e-17)) / 2

@testset "Delta n~ and the reaction list" begin
    @test delta_ntilde(reaction(:N1N1_AA)) == 2
    @test delta_ntilde(reaction(:A_N1N2)) == 0
    @test delta_ntilde(reaction(:phi_AA)) == 0
    @test delta_ntilde(reaction(:AA_phiphi)) == 4
    @test delta_ntilde(reaction(:N1N2_Aphi)) == 4
    @test delta_ntilde(reaction(:phi_N1N1)) == -2
    @test delta_ntilde(reaction(:N1nu_N2N2)) == 1
    @test delta_ntilde(reaction(:N2nu_Aphi)) == 5
    @test delta_ntilde(reaction(:N1nu_phiphi)) == 7
    pan = PandemolatorHG(m, tT_rel, itp)
    names = [r.name for (r, _) in pan.reactions]
    @test isempty(intersect(names, HG_EXACT))
    @test !(:N1N1_N2N2 in names) && !(:A_N1N2 in names) && !(:phi_AA in names)
    @test :N1nu_N2N2 in names && :N2nu_N1N2 in names && :AA_phiphi in names
    pan0 = PandemolatorHG(m, tT_rel, itp; with_phi=false)
    @test all(((r, _),) -> !(:phi in r.I || :phi in r.F), pan0.reactions)
end

@testset "chemical potentials of the fluid" begin
    pan = PandemolatorHG(m, tT_rel, itp)
    for (T_N, xi) in ((3m, -2.0), (0.1m, 4.0), (100m, -0.3))
        gap_A = 2.5m / T_N - 2xi
        gphi = gap_phi_of(pan, T_N, gap_A)
        @test isapprox(pan.phi.m / T_N - gphi, 4xi; rtol=1e-10, atol=1e-10)
        n = num_dens(pan, T_N, xi; gap_A=gap_A)
        parts = 2number_density(pan.N1, T_N, xi) + 2number_density(pan.A, T_N, 2xi; gap=gap_A) +
                4number_density(pan.phi, T_N, 4xi; gap=gphi)
        @test isapprox(n, parts; rtol=1e-12)
    end
end

@testset "with phi and tables off it is the reduced solver" begin
    N1 = Particle{Float64}(m, 1, dof=2); N2 = Particle{Float64}(m, 1, dof=2)
    A = Particle{Float64}(2.5m, -1, dof=3); nu = Particle{Float64}(0.0, 1, dof=2)
    panz = PandemolatorZ{Float64}(ModelParams{Float64}(g, th), N1, N2, A, nu, tT_rel)
    pan = PandemolatorHG(m, tT_rel, nothing; with_phi=false, with_tables=false)
    pan.mp = ModelParams{Float64}(g, th)
    for (z, lnxN, eta) in ((log(1e-3), log(1e-2), 0.5), (log(0.5), log(2.0), 1.2), (log(5.0), log(40.0), 3.0))
        ent = pan.ent_interp_z(z)
        T_N = m / exp(lnxN); xi = xi_from_eta(pan, eta, lnxN); gA = gap_A_from_eta(eta)
        u = [log(1.3num_dens(panz, T_N, xi; gap_A=gA) / ent), log(0.9energy_dens(panz, T_N, xi; gap_A=gA) / ent^(4 / 3)), lnxN, eta]
        du1, du2 = zeros(4), zeros(4)
        mm_func_z!(du1, u, panz, z)
        mm_func_hg!(du2, u, pan, z)
        @test du1 == du2
    end
end

@testset "detailed balance at T_N = T_nu, mu = 0" begin
    pan = PandemolatorHG(m, tT_rel, itp)
    pan.mp = ModelParams{Float64}(g, th)
    for T in (3m, 0.5m)
        gA = 2.5m / T
        # the scale: the linear response, at mu/T = +-0.02
        tp = table_collision(pan, T, T, 0.02, 2.5m / T - 0.04)
        tm = table_collision(pan, T, T, -0.02, 2.5m / T + 0.04)
        cn, cr = table_collision(pan, T, T, 0.0, gA)
        @test tp[1] * tm[1] < 0      # (not symmetric: Delta mu reaches 15 mu, e^(15 * 0.02) = 1.35)
        # the fixed point moves by less than 1e-3 in mu/T (the portal tables balance to ~1e-6-1e-4)
        @test abs(cn / ((tp[1] - tm[1]) / 0.04)) < 1e-3
        @test abs(cr / ((tp[2] - tm[2]) / 0.04)) < 1e-3
        # the exact 1 <-> 2 terms vanish at mu = 0 as well; at exactly mu = 0 the integrand is
        # rounding noise and the quadrature never meets its relative tolerance, so test that
        # the net rate is odd in mu around 0 instead
        dl = 0.02
        for f in ((T, xi) -> phi_N1nu_moments(pan, T, T, xi, 2.5m / T - 2xi)[1],
                  (T, xi) -> A_N2nu_moments(pan, T, T, xi; gap_A=2.5m / T - 2xi)[1])
            cp, cm = f(T, dl), f(T, -dl)
            @test cp * cm < 0
            @test isapprox(cp, -cm; rtol=0.1)
        end
    end
end

@testset "phi <-> N1 nu: MB limit against the table" begin
    # deep in the MB regime (dark sector dilute, T_N = T_nu): the exact FD/BE inverse decay
    # equals the tabulated MB rate up to the nu statistics, which both treat exactly
    pan = PandemolatorHG(m, tT_rel, itp)
    pan.mp = ModelParams{Float64}(g, th)
    p = hg_rates!(pan)
    for (T_N, T_nu) in ((20m, 20m), (5m, 8m))
        xi = -15.0
        gA = 2.5m / T_N - 2xi
        d = phi_N1nu_moments(pan, T_nu, T_N, xi, gA)          # net phi decay (phi -> N1 nu)
        t = itp.portal[:N1nu_phi]
        lnR, _ = pan.norms[:N1nu_phi]
        lxnu, lr = log(m / T_nu), log(T_N / T_nu)
        gN, gphi = m / T_N - xi, gap_phi_of(pan, T_N, gA)
        net_fwd = exp(t.lnRf(lxnu, lr) + lnR - gN) - exp(t.lnRr(lxnu, lr) + lnR - gphi)   # N1 nu -> phi
        @test isapprox(-d[1], net_fwd; rtol=2e-3)
    end
end
