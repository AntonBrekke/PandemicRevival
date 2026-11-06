# Accuracy of the fluid's A'A' <-> NN quadrature (LEGENDRE_KW_AA, tuned at m_A' = 2.5 m_N) at other
# mass ratios: against the defaults of coll_12_34_legendre and a finer rule, over states from
# relativistic to cold and degenerate to dilute.
#   julia --project=. test/check_legendre_ratio.jl <tables.jls>
include(joinpath(@__DIR__, "..", "src", "fluid.jl"))
using Printf

itp = RateInterpolants(load_rate_tables(ARGS[1]))
rA, rphi = mass_ratios(itp)
tT = TimeTempRelation{Float64}()
m = 1e-5
pan = FluidSolver(m, tT, itp)
pan.mp = ModelParams{Float64}(0.5, asin(sqrt(1e-18)) / 2)
FINE = (L=32, n_c=48, n_E=24, n_s=16, n_theta=24)
C(T_N, xi, gap_A; kw...) = coll_12_34_legendre(pan.mp, pan.A, pan.A, pan.N1, pan.N1, (T_N, T_N, T_N, T_N),
                                               (2xi, 2xi, xi, xi); gaps=(gap_A, gap_A, m / T_N - xi, m / T_N - xi), kw...)
println("m_A'/m_N = $rA: relative difference of C(A'A' <-> N1N1) to the fine rule")
println("  T_N/m      xi    gap_A   tuned (LEGENDRE_KW_AA)   default")
worst = 0.0
for T_N in (100m, 20m, 5m, 1m, 0.3m), xi in (-5.0, 0.0, 0.5)
    gap_A = rA * m / T_N - 2xi
    gap_A > 0 || continue
    f = C(T_N, xi, gap_A; FINE...)
    (isfinite(f) && f != 0) || (println(@sprintf("  %6.3g  %6.2f  %7.3g   (fine rule gives %g)", T_N / m, xi, gap_A, f)); continue)
    t = C(T_N, xi, gap_A; LEGENDRE_KW_AA...)
    d = C(T_N, xi, gap_A)
    global worst = max(worst, abs(t / f - 1))
    @printf("  %6.3g  %6.2f  %7.3g   %12.2e   %12.2e\n", T_N / m, xi, gap_A, t / f - 1, d / f - 1)
    flush(stdout)
end
@printf("largest difference of the tuned rule: %.2e\n", worst)
