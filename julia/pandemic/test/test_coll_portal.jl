# src/coll_portal.jl: the exact (FD/BE, Pauli-blocked) portal 2 <-> 2 integrals.
# - dilute limit: the rate tables' direct calculation (MB N, FD nu; rate_tables.jl portal_point);
# - detailed balance at T_N = T_nu with chemically balanced mu, also for degenerate N: forward = backward;
# - convergence in the Legendre order and the nodes.
#   julia --project=. test/test_coll_portal.jl
using Test
include(joinpath(@__DIR__, "..", "src", "fluid.jl"))
isdefined(@__MODULE__, :coll_portal) || include(joinpath(@__DIR__, "..", "src", "coll_portal.jl"))

itp = RateInterpolants(load_rate_tables(rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"))))
tT = TimeTempRelation{Float64}()
m = 1e-5
pan = FluidSolver(m, tT, itp)
pan.mp = ModelParams{Float64}(0.3, asin(sqrt(1e-18)) / 2)
fluid_rates!(pan)
const PR = pan.p_rates
const N1, N2, NU = pan.N1, pan.N2, pan.nu
parts(name) = name === :N1nu_N2N2 ? (N1, NU, N2, N2) : (N2, NU, N1, N2)

@testset "dilute limit against the tables' direct calculation" begin
    for (T_N, T_nu, gap) in ((3m, 2m, 20.0), (0.05m, 0.3m, 25.0)), name in (:N1nu_N2N2, :N2nu_N1N2)
        r = only(filter(r -> r.name === name, REACTIONS))
        tab = reaction_sigma_table(r, PR; T_max=2max(T_N, T_nu), n=800)
        (lnref, ratio) = portal_point(r, PR, tab, T_N, T_nu)
        sym = name === :N1nu_N2N2 ? 0.5 : 1.0      # the tables carry 1/2 for identical final states
        res = coll_portal(_tree(name), PR, parts(name), (T_N, T_nu, T_N, T_N), (gap, 0.0, gap, gap))
        @test isapprox(sym * res.fwd[1], exp(lnref[1] - gap) * ratio[1]; rtol=2e-3)
        @test isapprox(sym * res.fwd[2], exp(lnref[3] - gap) * ratio[3]; rtol=2e-3)
        @test isapprox(sym * res.bwd[1], exp(lnref[2] - 2gap) * ratio[2]; rtol=2e-3)
    end
end

@testset "detailed balance at T_N = T_nu, degenerate N" begin
    # forward = backward when the chemical potentials balance: N2 nu <-> N1 N2 needs mu_N1 = 0
    # (mu_N2 free), N1 nu <-> N2 N2 needs mu_N1 = 2 mu_N2. gap = (m - mu)/T; negative = degenerate
    for T in (0.3m, 2m), gN2 in (2.0, -1.0, -4.0)
        res = coll_portal(_tree(:N2nu_N1N2), PR, parts(:N2nu_N1N2), (T, T, T, T), (gN2, 0.0, m / T, gN2))
        @test isapprox(res.fwd[1], res.bwd[1]; rtol=1e-6)
        @test isapprox(res.fwd[2], res.bwd[2]; rtol=1e-6)
        mu2 = m - gN2 * T
        gN1 = (m - 2mu2) / T
        res = coll_portal(_tree(:N1nu_N2N2), PR, parts(:N1nu_N2N2), (T, T, T, T), (gN1, 0.0, gN2, gN2))
        @test isapprox(res.fwd[1], res.bwd[1]; rtol=1e-6)
        @test isapprox(res.fwd[2], res.bwd[2]; rtol=1e-6)
    end
end

@testset "Pauli blocking lowers the forward rate of nearly degenerate N" begin
    # same N2 density factor e^-gap, but blocked final N: the exact forward rate is below its MB value
    T_N, T_nu = 0.3m, 0.4m
    dil = coll_portal(_tree(:N1nu_N2N2), PR, parts(:N1nu_N2N2), (T_N, T_nu, T_N, T_N), (20.0, 0.0, 20.0, 20.0))
    deg = coll_portal(_tree(:N1nu_N2N2), PR, parts(:N1nu_N2N2), (T_N, T_nu, T_N, T_N), (0.1, 0.0, 0.1, 0.1))
    # per initial-N occupation e^-gap: the MB forward rate scales exactly so; blocking and FD lower it
    @test deg.fwd[1] * exp(0.1) < dil.fwd[1] * exp(20.0)
end

@testset "convergence in the Legendre order and nodes" begin
    T_N, T_nu, gap = 0.1m, 0.4m, 0.5      # cold and nearly degenerate N
    for name in (:N1nu_N2N2, :N2nu_N1N2)
        a = coll_portal(_tree(name), PR, parts(name), (T_N, T_nu, T_N, T_N), (gap, 0.0, gap, gap))
        b = coll_portal(_tree(name), PR, parts(name), (T_N, T_nu, T_N, T_N), (gap, 0.0, gap, gap); L=36, ng=10, ne=10)
        @test isapprox(a.fwd[1], b.fwd[1]; rtol=2e-3)
        @test isapprox(a.bwd[1], b.bwd[1]; rtol=2e-3)
    end
end
