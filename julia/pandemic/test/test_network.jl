# src/network.jl: dark temperature, QSS of A' and phi, equilibrium. Needs the built
# rate tables (src/run_build_rate_tables.jl); ARGS[1] overrides the table file.
using Test
include(joinpath(@__DIR__, "..", "src", "network.jl"))

path = length(ARGS) > 0 ? ARGS[1] : rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))
itp = RateInterpolants(load_rate_tables(path))
tT_rel = TimeTempRelation{Float64}()
m = 1e-5
p = table_params(itp, m, 1e-3, 1e-13)
net = Network(itp, p, tT_rel)

@testset "kinetic energy and dark temperature" begin
    # limits: 3T - m + m^2/(2T) (relativistic); T (3/2 + 15/(8x) - 15/(8x^2) + 135/(128x^3)) (non-relativistic)
    T = 1e4m
    @test isapprox(kinetic_per_particle(m, T), 3T - m + m^2 / (2T); rtol=1e-9)
    for xN in (50.0, 5e2, 5e3, 1e6)
        T = m / xN
        @test isapprox(kinetic_per_particle(m, T), T * (1.5 + 15 / (8xN) - 15 / (8xN^2) + 135 / (128xN^3));
                       rtol=10 / xN^4 + 1e-12)
    end
    for T in (1e-6m, 1e-2m, 0.7m, 30m, 1e5m)
        n = (3e-12, 2e-12, 1e-13, 1e-20)
        kappa = sum(n[k] * kinetic_per_particle(net.masses[k], T) for k in 1:4)
        @test isapprox(dark_temperature(net, n, kappa), T; rtol=1e-12)
    end
end

@testset "QSS root of A' and phi" begin
    T_N, T_nu = 0.4m, 0.6m
    n = (2e-15, 2e-15, 1e-17, 1e-25)
    urs = unit_rates(net, T_N, T_nu, n[1], n[2])
    for k in (3, 4)
        a0, a1, a2 = qss_coefficients(urs, k, n[3], n[4])
        # the polynomial reproduces the assembled collision term
        @test isapprox(assemble_collision(urs, n[3], n[4])[1][k], a0 + a1 * n[k] + a2 * n[k]^2; rtol=1e-10)
        nq, _ = qss_root(a0, a1, a2)
        nn = collect(n); nn[k] = nq
        @test abs(assemble_collision(urs, nn[3], nn[4])[1][k]) < 1e-10 * a0
    end
end

@testset "full equilibrium with the neutrinos" begin
    # mu = 0 and T_N = T_nu: every reaction balances. Per species, the net rate
    # relative to its gross rate sum_r |dN| (Gamma_fwd + Gamma_rev) is the
    # interpolation error of the logs (~1e-3; ~1e-2 on the coarse smoke tables)
    for T in (0.3m, 2m)
        neq = ntuple(k -> n_scaled_safe(net.masses[k], net.dofs[k], T) * exp(-net.masses[k] / T), 4)
        urs = unit_rates(net, T, T, neq[1], neq[2])
        C, CK = assemble_collision(urs, neq[3], neq[4])
        lnA, lnphi = log(neq[3]), log(neq[4])
        for k in 1:4
            gross = sum(abs(ur.dN[k]) * (exp(ur.lnGf + ur.cI[1] * lnA + ur.cI[2] * lnphi) * abs(ur.xf) +
                                         exp(ur.lnGr + ur.cF[1] * lnA + ur.cF[2] * lnphi) * abs(ur.xr)) for ur in urs)
            @test abs(C[k]) < 2e-2 * gross
        end
    end
end

@testset "QSS species take their number from the N" begin
    # with A' (phi) eliminated, n_N1 + n_N2 + 2 n_A' + w_phi n_phi changes only through the
    # reactions that change it: the N pay for what the state of A' (phi) gains
    z = log(3.0)
    s = net.ent_interp_z(z)
    n = (2e-15, 1e-15, 3e-16, 1e-25)
    kappa = sum(n[k] * kinetic_per_particle(net.masses[k], 0.4m) for k in 1:4)
    u = state_vector(n, kappa, s)
    du = zeros(5)
    for (wA, wphi) in ((1.0, 0.0), (1.0, 1.0), (0.3, 0.7))
        du .= [0.1, -0.2, 2.0, -3.0, 0.5]
        du0 = copy(du)
        reservoir_correction!(du, net, n, kappa, wA, wphi)
        # change of n_k per unit z (without dilution) is n_k du_k
        dN(d) = (n[1] + n[2]) * d[1]
        @test isapprox(dN(du) - dN(du0), -(2wA * n[3] * du0[3] + wphi * net.w_phi * n[4] * du0[4]); rtol=1e-12)
        # kappa gives up the rest mass left over: m_A' - m_N1 - m_N2 per A', m_phi - w_phi m_N per phi
        m1, m2, mA, mphi = net.masses
        dE = (mA - m1 - m2) * wA * n[3] * du0[3] +
             (mphi - (net.w_phi - net.w_phi2) * m1 - net.w_phi2 * m2) * wphi * n[4] * du0[4]
        @test isapprox(kappa * (du[5] - du0[5]), -dE; rtol=1e-12)
        # the N2 share: one per A', w_phi2 per phi
        d2(d) = (n[1] + n[2]) * (n[2] / (n[1] + n[2])) * (d[1] + n[1] / (n[1] + n[2]) * d[2])
        @test isapprox(d2(du) - d2(du0), -(wA * n[3] * du0[3] + wphi * net.w_phi2 * n[4] * du0[4]); rtol=1e-10)
    end
end
