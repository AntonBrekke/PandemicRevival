# Built rate tables (tmp/rate_tables/, from src/run_build_rate_tables.jl):
# interpolation off the grid points, rescaling to other (m_N, g, sin^2 2theta),
# and the collision terms against a direct evaluation.
using Test
include(joinpath(@__DIR__, "..", "src", "rate_tables.jl"))

# TABLE_DELTA=<delta> checks the tables built with an N1-N2 splitting (default: equal masses)
path = rate_table_path(joinpath(@__DIR__, "..", "tmp", "rate_tables"); delta=parse(Float64, get(ENV, "TABLE_DELTA", "0")))
tabs = load_rate_tables(path)
itp = RateInterpolants(tabs)
pref = tabs.p_ref

# (T_N/m, T_nu/m) between grid points, in the physically relevant range
STATES = ((0.37, 0.53), (2.3, 4.1), (0.011, 0.27), (41.0, 57.0))

"""Direct (unnormalised) rates of reaction r at p, as stored: portal (lnref, ratio), dark (lnRf, lnRr)."""
function direct(r, p, T_N, T_nu)
    tab = reaction_sigma_table(r, p; T_max=max(T_N, T_nu) * 2, n=800)
    r.kind in (:dark2, :dark12) && return dark_point(r, p, tab, T_N)
    return portal_point(r, p, tab, T_N, T_nu)
end

@testset "interpolation off the grid points (reference parameters)" begin
    m = pref.m1
    for name in (:N1N1_N2N2, :A_N1N2, :N1N1_AA, :N2nu_A, :N1nu_phi, :N1nu_N2N2, :N2nu_N1N2, :N1nu_AA), (a, b) in STATES
        r = reaction(name)
        T_N, T_nu = a * m, b * m
        lxN, lxnu, lr = log(m / T_N), log(m / T_nu), log(T_N / T_nu)
        d = direct(r, pref, T_N, T_nu)
        if r.kind in (:dark2, :dark12)
            t = itp.dark[name]
            @test isapprox(t.lnRf(lxN), d[1]; atol=2e-3)
            @test isapprox(t.lnRr(lxN), d[2]; atol=2e-3)
        else
            t = itp.portal[name]
            (lf, lr_, lqf, lqr), (xf, xr, xqf, xqr) = d
            # the forward rates carry an exponential shift (ln R ~ -1e3 at cold
            # states): the spline error is ~1e-5 of |ln R| there, i.e. ~1% of e^-1000
            tol(v) = 2e-3 + 1e-5 * abs(v)
            @test isapprox(t.lnRf(lxnu, lr), lf; atol=tol(lf))
            @test isapprox(t.lnRr(lxnu, lr), lr_; atol=tol(lr_))
            @test isapprox(t.lnQf(lxnu, lr), lqf; atol=tol(lqf))
            # signed ratios of the resonance-subtracted rates: steep near sign changes
            @test isapprox(t.ratio_Rf(lxnu, lr), xf; rtol=1e-2, atol=5e-2)
        end
    end
end

@testset "rescaling to other parameters" begin
    for p in (table_params(itp, 3e-5, 1e-2, 1e-15), table_params(itp, 2e-6, 3e-4, 1e-11))
        norms = rate_norms(itp, p)
        m = p.m1
        for name in (:N1N1_N2N2, :N2nu_A, :N1nu_N2N2, :N2nu_N1N2)
            r = reaction(name)
            T_N, T_nu = 0.37m, 0.53m
            lxN, lxnu, lr = log(m / T_N), log(m / T_nu), log(T_N / T_nu)
            d = direct(r, p, T_N, T_nu)
            lnR = norms[name][1]
            if r.kind === :dark2
                @test isapprox(itp.dark[name].lnRf(lxN) + lnR, d[1]; atol=3e-3)
            else
                # the resonance width scales with g^2, the narrow-width limit does not
                @test isapprox(itp.portal[name].lnRf(lxnu, lr) + lnR, d[1][1]; atol=3e-3)
            end
        end
    end
end

@testset "collision terms: conservation and equilibrium" begin
    p = table_params(itp, 1e-5, 1e-3, 1e-13)
    norms = rate_norms(itp, p)
    m = p.m1
    T = 0.6m
    # full equilibrium with the neutrinos (mu = 0, T_N = T_nu): every reaction balances
    neq = ntuple(k -> (f = DARK_SPECIES[k]; n_MB(Species(field_mass(f, p), field_dof(f), T, field_mass(f, p) / T))), 4)
    dn, drho = dark_collision(itp, norms, p, T, T, neq)
    scale = maximum(abs, dark_collision(itp, norms, p, T, T, (neq[1], 1e-3neq[2], neq[3], neq[4]))[1])
    # to the interpolation accuracy of the logs (~1e-3)
    @test maximum(abs, dn) < 1e-2 * scale
    # a dilute dark sector colder than the neutrinos gains particles and energy
    dn, drho = dark_collision(itp, norms, p, 0.3m, 0.5m, (1e-6neq[1], 1e-6neq[2], 1e-9neq[3], 1e-12neq[4]))
    @test sum(dn[1:2]) + 2dn[3] > 0
    @test drho > 0
end
