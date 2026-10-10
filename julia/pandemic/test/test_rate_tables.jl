# src/rate_tables.jl: nu-weighted W, detailed balance, coupling factorisation,
# and A' <-> N2 nu against the solver's coll_3_12.
using Test
include(joinpath(@__DIR__, "..", "src", "rate_tables.jl"))
include(joinpath(@__DIR__, "..", "src", "coll_3_12.jl"))

m = 1e-5
p = rate_params(m, 1e-3, 1e-12)

@testset "nu tail integrals" begin
    for (dir, k) in ((:fwd, 0), (:fwd, 1), (:rev, 0), (:rev, 1)), (T_N, T_nu) in ((m, m), (0.01m, 2m), (3m, 0.5m))
        t = NuTail(dir, k, T_N, T_nu)
        for x in (1e-12 * t.T_w, 0.03t.T_w, 2t.T_w, 40t.T_w, 300t.T_w, 3e3 * t.T_w, 3e9 * t.T_w)
            # ln ∫_x^∞ w, in units of T_w (quadgk's map of [x, Inf) assumes O(1)
            # scales) and scaled by e^(x/T_w) so that it does not underflow
            u0 = x / t.T_w
            I = QuadGK.quadgk(u -> exp(log_weight(t, t.T_w * u) + u0), u0, Inf; rtol=1e-12)[1]
            @test isapprox(ln_tail(t, x), log(t.T_w * I) - u0; atol=1e-6)
        end
    end
end

@testset "W_nu: MB weight reproduces W_scaled" begin
    for (T_N, T_nu) in ((m, m), (0.1m, 0.4m), (3m, 0.5m))
        a = Species(m, 2, T_N, m / T_N)
        nu = Species(0.0, 2, T_nu, 0.0)
        t = NuTail(:mb, 0, T_N, T_nu)
        for s in ((2.5m)^2, (4.0m)^2, 400m^2)
            @test isapprox(W_nu(s, m, T_N, t), W_scaled(s, a, nu); rtol=1e-6)
        end
    end
end

@testset "W_nu: forward shift" begin
    # large threshold, cold nu: the shifted integral is O(1) and matches the unshifted one
    T_N, T_nu = 0.05m, 0.3m
    s = (4.0m)^2
    c = fwd_shift(s, m, T_N, T_nu)
    t = NuTail(:fwd, 0, T_N, T_nu)
    @test c > 1
    @test isapprox(W_nu(s, m, T_N, t; c=c) * exp(-c), W_nu(s, m, T_N, t); rtol=1e-6)
end

@testset "narrow-width s integral against thermal_rates RIS" begin
    # non-resonant: identical up to quadrature; resonant: the finite-width RIS
    # approaches the narrow-width limit linearly in Gamma/M (7e-3 at 1.9e-3, 1e-3 at 2.1e-4)
    for g in (0.3, 0.1)
        pp = rate_params(m, g, 1e-12)
        for name in (:N2nu_N1N2, :N1nu_N2N2), T in (0.5m, 3m)
            tab = reaction_sigma_table(reaction(name), pp; T_max=1e3m, n=800)
            A = Species(m, 2, T, m / T); nu = Species(0.0, 2, T, 0.0)
            ris = rate_density_scaled(tab, A, nu; propagator=:ris, rtol=1e-6)
            nw = nw_rate_density(tab, m, (T, T), s -> W_scaled(s, A, nu))
            tol = tab.M == 0 ? 1e-6 : 5 * tab.Gamma / tab.M
            @test isapprox(nw, ris; rtol=tol)
        end
    end
end

@testset "portal detailed balance at T_N = T_nu" begin
    # mu = 0 everywhere: Gamma_fwd = Gamma_rev, i.e. lnRr = lnRf + Delta/T, also for E_nu-weighted
    for name in (:N2nu_A, :N1nu_phi, :N1nu_N2N2, :N2nu_N1N2)
        r = reaction(name)
        for T in (0.3m, 2m, 20m)
            T_max = 20m
            tab = reaction_sigma_table(r, p; T_max=T_max, n=400)
            (lnRf, lnRr, lnQf, lnQr), (xRf, xRr, xQf, xQr) = portal_point(r, p, tab, T, T)
            Delta = mass_sum(r.F, p) - field_mass(only(r.I), p)
            @test isapprox(lnRr, lnRf + Delta / T; atol=2e-4)
            @test isapprox(lnQr, lnQf + Delta / T; atol=2e-4)
            @test isapprox(xRr, xRf; rtol=2e-4, atol=1e-8)
            @test isapprox(xQr, xQf; rtol=2e-4, atol=1e-8)
        end
    end
end

@testset "dark detailed balance against thermal_rates" begin
    T = 0.7m
    N1 = Species(m, 2, T, m / T); N2 = Species(m, 2, T, m / T)
    lnRf, lnRr = dark_point(reaction(:A_N1N2), p, nothing, T)
    # reverse = inverse decay N1 N2 -> A' at mu = 0, scaled by e^(2m/T)
    R_inv = rate_inverse_decay(:A_N1N2, p, N1, N2)[3]
    @test isapprox(lnRr, log(R_inv) + 2m / T; rtol=1e-6)
    # N1N1 <-> A'A' is integrated in the A'A' direction; the N1 N1 side from
    # thermal_rates (ordered pairs: twice the reactions)
    r = reaction(:N1N1_AA)
    tab = reaction_sigma_table(r, p; T_max=T, n=400)
    lnRf, lnRr = dark_point(r, p, tab, T)
    tabf = SigmaTable(_tree(:N1N1_AA), p; T_max=T, n=400)
    Rf_direct = rate_density_scaled(tabf, N1, N1) / 2
    @test isapprox(lnRf, log(Rf_direct); atol=2e-3)
end

@testset "coupling factor factorises" begin
    p2 = rate_params(3e-5, 2e-2, 1e-10)
    for r in REACTIONS
        c1 = coupling_factor(r, p2) / coupling_factor(r, p)
        c2 = coupling_factor(r, p2; s_fac=3.1, t_frac=0.81) / coupling_factor(r, p; s_fac=3.1, t_frac=0.81)
        @test isapprox(c1, c2; rtol=1e-8)
        q = _tree(r.proc)
        if length(q.initial) == 2 && q.xyz !== nothing
            for (sf, tf) in ((1.7, 0.37), (2.3, 0.6))
                s = sf^2 * max(sum(field_mass(f, p) for f in q.initial)^2, sum(field_mass(f, p) for f in q.final)^2)
                s2 = s * (p2.m1 / p.m1)^2
                t = _t_range(s, (field_mass(f, p) for f in (q.initial..., q.final...))...)
                t2 = _t_range(s2, (field_mass(f, p2) for f in (q.initial..., q.final...))...)
                x1 = q.xyz(s, t[1] + tf * (t[2] - t[1]), p)
                x2 = q.xyz(s2, t2[1] + tf * (t2[2] - t2[1]), p2)
                for k in 1:3
                    # X and Y carry mass^4 and mass^2; 1e-6: the resonant parts of
                    # N1 nu -> A'A' cancel between diagrams, and agree to ~4e-8
                    dim = (4, 2, 0)[k]
                    x1[k] == 0 || @test isapprox(x2[k] / x1[k] / (p2.m1 / p.m1)^dim, c1; rtol=1e-6)
                end
            end
        end
    end
end

@testset "A' <-> N2 nu against coll_3_12" begin
    # small fugacity, T_N != T_nu: the solver's exact FD/BE collision term
    # equals the MB-dark / FD-nu table up to O(e^xi)
    g = 1e-3
    s2 = 1e-12
    th = asin(sqrt(s2)) / 2
    pp = rate_params(m, g, s2)
    mp = ModelParams{Float64}(g, th)
    N2 = Particle{Float64}(m, 1, dof=2)
    nu = Particle{Float64}(0.0, 1, dof=2)
    A = Particle{Float64}(2.5m, -1, dof=3)
    r = reaction(:N2nu_A)
    for (x, ratio) in ((0.5, 0.3), (2.0, 1.0), (5.0, 0.6))
        T_nu = m / x
        T_N = ratio * T_nu
        xi_N = -12.0
        xi_A = 2xi_N
        c1, cEN2, cEA = coll_3_12_moments(mp, N2, nu, A, (T_N, T_nu, T_N), (xi_N, 0.0, xi_A))
        (lnRf, lnRr, lnQf, lnQr), _ = portal_point(r, pp, nothing, T_N, T_nu)
        gapN = m / T_N - xi_N
        gapA = 2.5m / T_N - xi_A
        Cn = exp(lnRf - gapN) - exp(lnRr - gapA)
        Crho = exp(lnQf - gapN) - exp(lnQr - gapA)
        @test isapprox(-c1, Cn; rtol=1e-4)
        @test isapprox(-cEA + cEN2, Crho; rtol=1e-4)
    end
end
