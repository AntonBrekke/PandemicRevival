include(joinpath(@__DIR__, "../src/utils.jl"))
include(joinpath(@__DIR__, "../src/time_temp_relation.jl"))
include(joinpath(@__DIR__, "../src/dodelson_widrow.jl"))
include(joinpath(@__DIR__, "../src/pandemolator.jl"))
include(joinpath(@__DIR__, "../src/pandemolator_z.jl"))

"""
Compares the z-parameterized solver (`pandemolate_z`, `PandemolatorZ`)
against the original (`pandemolate`, `Pandemolator`) on the benchmark point
from test_pandemolate.jl. Both integrate in the same variable, log(x), and
solve the same equation -- they differ only in which coordinate the
background interpolants are keyed on (T_nu vs. z = log(x)) -- so persistent
disagreement at tight tolerance would indicate an actual bug in the z
version, not just solver noise.

An x-as-independent-variable version was tried first and dropped: it needed
2x+ the RHS evaluations of the z version for the same tolerance, with no
offsetting benefit. See pandemolator_z.jl's header.
"""
function test_pandemolator_z()
    m_N = 1e-5
    m_A = 2.5 * m_N

    N1 = Particle{Float64}(m_N, 1, dof=2)
    N2 = Particle{Float64}(m_N, 1, dof=2)
    A  = Particle{Float64}(m_A, -1, dof=3)
    nu = Particle{Float64}(0., 1, dof=2)

    y = 1e-4
    sin2_2th = 5.3e-13
    th = asin(sqrt(sin2_2th)) / 2.
    mp = ModelParams{Float64}(y, th)

    tT_rel = TimeTempRelation{Float64}()
    dw = DodelsonWidrow{Float64}(m_N, th, tT_rel)

    pan_T = Pandemolator{Float64}(mp, N1, N2, A, nu, tT_rel, false)
    pan_z = PandemolatorZ{Float64}(mp, N1, N2, A, nu, tT_rel, false)

    println("="^70)
    println("T-keyed (original) vs. z-keyed pandemolator")
    println("="^70)

    results = Dict{String,Any}()
    for (name, f, pan) in (
            ("T-keyed, integrates log(x)  [original]", pandemolate,   pan_T),
            ("z-keyed, integrates log(x)",              pandemolate_z, pan_z),
        )
        t = @elapsed sol = f(tT_rel, dw, pan)
        ok = DE.successful_retcode(sol)
        u_f = sol.u[end]
        results[name] = (
            y_n = exp(u_f[1]), y_rho = exp(u_f[2]),
            ln_x_N = u_f[3], eta = u_f[4],
            nsteps = length(sol.t), nf = sol.stats.nf, time = t, ok = ok,
        )
        println()
        println(name)
        println("   retcode ok : $ok")
        println("   steps      : $(length(sol.t))    RHS evals: $(sol.stats.nf)")
        println("   wall time  : $t s")
        println("   final Y_n  : $(exp(u_f[1]))")
        println("   final Y_rho: $(exp(u_f[2]))")
        println("   final ln x_N = $(u_f[3])   eta = $(u_f[4])")
    end

    println()
    println("-"^70)
    ref = results["T-keyed, integrates log(x)  [original]"].y_n
    r_z = results["z-keyed, integrates log(x)"].y_n
    rel = abs(r_z - ref) / abs(ref)
    println("Relative difference in the frozen-out yield Y_n (default tolerances):")
    println("   z vs. original: $rel")
    println("(this is solver noise at default tolerances, not a formulation")
    println("difference -- see the tolerance refinement below)")

    # abstol must stay >~ 1.3e-15 or DAE initialization rejects u0 (noted in
    # pandemolator.jl); neither version can be refined past reltol ~ 1e-4,
    # both go Unstable there with the error dominated by the ALGEBRAIC
    # components u[3], u[4] -- a DAE-conditioning ceiling shared by both
    # parameterizations, not something this comparison is expected to fix.
    println()
    println("-"^70)
    println("Tolerance refinement (abstol = 1e-14 throughout):")
    println(rpad("  reltol", 12), rpad("original Y_n", 26), rpad("z-version Y_n", 26))
    for rt in (1e-2, 1e-3, 1e-4, 1e-5)
        rT = try
            s = pandemolate(tT_rel, dw, pan_T; reltol=rt, abstol=1e-14)
            DE.successful_retcode(s) ? string(exp(s.u[end][1])) : "retcode=$(s.retcode)"
        catch e; "FAILED $(typeof(e))" end
        rz = try
            s = pandemolate_z(tT_rel, dw, pan_z; reltol=rt, abstol=1e-14)
            DE.successful_retcode(s) ? string(exp(s.u[end][1])) : "retcode=$(s.retcode)"
        catch e; "FAILED $(typeof(e))" end
        println(rpad("  $rt", 12), rpad(rT, 26), rpad(rz, 26))
    end
    println("="^70)
    return results
end

test_pandemolator_z()
