# src/network_reduced.jl: the closure inverts, its equations are the full network's, and the
# solutions agree at relic roots. Needs the built rate tables.
using Test
include(joinpath(@__DIR__, "..", "src", "network_reduced.jl"))

const TABLES = joinpath(@__DIR__, "..", "tmp", "rate_tables")
itp = RateInterpolants(load_rate_tables(rate_table_path(TABLES)))
tT_rel = TimeTempRelation{Float64}()
m = 1e-5
net = Network(itp, table_params(itp, m, 1e-4, 1e-12), tT_rel)
m1, m2, mA, mphi = net.masses

"""Densities in equilibrium through A' <-> N1 N2 and phi <-> A'A' at T: (n, Nt1, Nt2, kappat)."""
function equilibrium_state(n1, n2, T)
    ns(k) = n_scaled_safe(net.masses[k], net.dofs[k], T)
    nA = ns(3) / (ns(1) * ns(2)) * n1 * n2 * exp(-(mA - m1 - m2) / T)
    nphi = ns(4) / ns(3)^2 * nA^2 * exp(-(mphi - 2mA) / T)
    n = (n1, n2, nA, nphi)
    kappat = sum(n[k] * kinetic_per_particle(net.masses[k], T) for k in 1:4) +
             (mA - m1 - m2) * nA + (mphi - 2m1 - 2m2) * nphi
    return n, n1 + nA + 2nphi, n2 + nA + 2nphi, kappat
end

@testset "closure inverts" begin
    # from phi populated (T ~ m_phi) to A', phi Boltzmann-suppressed to nothing; N2 << N1 included
    for T in (1e3m, 100m, 20m, 3m, 0.4m, 0.05m, 1e-3m), r in (1.0, 1e-3, 1e-8)
        n1 = 1e-3 * n_scaled_safe(m1, 2, T)      # occupation number 1e-3
        n, Nt1, Nt2, kappat = equilibrium_state(n1, r * n1, T)
        a1, a2, lnA, lnphi, Tc = reduced_closure(net, Nt1, Nt2, kappat)
        @test isapprox(Tc, T; rtol=1e-9)
        @test isapprox(a1, n[1]; rtol=1e-8)
        @test isapprox(a2, n[2]; rtol=1e-6)
        n[3] > 1e-200 && @test isapprox(exp(lnA), n[3]; rtol=1e-6)
    end
end

@testset "partition with most of the number in A' and phi" begin
    for (KA, Kphi) in ((1e3, 0.0), (1e3, 1e3), (1e8, 1e2), (1e-3, 1e8))
        x, a = reduced_partition(1.0, 0.3, KA, Kphi)
        # the residual amplifies the rounding of 0.3 - a by K_A: reduced_densities_at avoids it there
        @test isapprox(x, KA * (1.0 - a) * (0.3 - a); rtol=1e-10 + 1e-14 * KA)
        @test 0 < a < 0.3
    end
end

@testset "dressed equations are the full network's" begin
    T_N, T_nu = 0.4m, 0.6m
    n, Nt1, Nt2, kappat = equilibrium_state(2e-15, 1e-15, T_N)
    urs = unit_rates(net, T_N, T_nu, n[1], n[2])
    lnA, lnphi = log(n[3]), log(n[4])
    C, CK, = assemble_collision(urs, n[3], n[4])
    C1, C2, Ctot, CKt, = reduced_collision(net, urs, lnA, lnphi)
    gross = maximum(abs(rate_and_kinetic(ur, lnA, lnphi)[1]) for ur in urs)
    # identities up to the rounding of the decay flux, which network.jl's C keeps and this one drops
    @test isapprox(C1, C[1] + C[3] + 2C[4]; atol=1e-10 * gross)
    @test isapprox(C2, C[2] + C[3] + 2C[4]; atol=1e-10 * gross)
    @test isapprox(Ctot, C1 + C2; atol=1e-10 * gross)
    @test isapprox(CKt, CK + (mA - m1 - m2) * C[3] + (mphi - 2m1 - 2m2) * C[4]; atol=1e-10 * m * gross)
    # the eliminated decays are there, and have zero weight (the energy one to rounding: left out)
    @test all(d -> count(ur -> ur.dN == d, urs) == 1, ELIMINATED)
    for ur in urs
        if ur.dN in ELIMINATED
            w = ur.dN[3] + 2ur.dN[4]
            @test ur.dN[1] + w == 0 && ur.dN[2] + w == 0
            @test abs((mA - m1 - m2) * ur.dN[3] + (mphi - 2m1 - 2m2) * ur.dN[4] - ur.Delta) < 1e-12 * m
        end
    end
    # the closure is the detailed balance of the tables: decay and inverse decay cancel
    for ur in urs
        if ur.dN == (1, 1, -1, 0)
            f = exp(ur.lnGf + lnA)
            @test abs(rate_and_kinetic(ur, lnA, lnphi)[1]) < 1e-3 * f
        end
    end
end

@testset "agrees with the full network at relic roots" begin
    hdr_s2(f) = parse(Float64, match(r"sin\^2 2theta = ([0-9.e+-]+)", readline(joinpath(@__DIR__, "..", "tmp", "benchmarks", f))).captures[1])
    for g in ("1e-07", "1e-05", "0.0001", "0.001")
        s2 = hdr_s2("network_m_10keV_g_$g.csv")
        n = Network(itp, table_params(itp, m, parse(Float64, g), s2), tT_rel)
        dw = DodelsonWidrow{Float64}(m, asin(sqrt(s2)) / 2, tT_rel)
        sf = solve_network(n, dw, tT_rel; maxiters=20_000)
        sr = solve_reduced_network(n, dw, tT_rel; maxiters=20_000)
        @test sr.retcode == DE.ReturnCode.Success
        @test isapprox(reduced_omega_h2(n, sr.u[end], sr.t[end]), network_omega_h2(n, sf.u[end], sf.t[end]); rtol=2e-3)
        nf, _ = state_densities(n, sf.u[end], sf.t[end])
        nr = reduced_diagnostics(n, sr.u[end], sr.t[end]).n
        @test isapprox(nr[2] / nr[1], nf[2] / nf[1]; rtol=2e-3)
    end
end

@testset "refuses a closed decay" begin
    itp15 = RateInterpolants(load_rate_tables(joinpath(TABLES, "rate_tables_mA2.5_mphi1.5.jls")))
    p15 = table_params(itp15, m, 1e-4, 1e-12)
    @test !reduced_applies(p15)
    n15 = Network(itp15, p15, tT_rel)
    @test_throws ErrorException solve_reduced_network(n15, DodelsonWidrow{Float64}(m, asin(sqrt(1e-12)) / 2, tT_rel), tT_rel)
end
