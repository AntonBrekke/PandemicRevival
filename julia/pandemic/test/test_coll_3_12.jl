using LaTeXStrings
ENV["GKSwstype"] = "nul"
import Plots as Plt

include(joinpath(@__DIR__, "../src/coll_3_12.jl"))
import .coll_3_12


function test_coll_3_12()
    energy_type = 0

    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = -20.
    xi_A = 2. * xi_N

    N = Particle(m_N, 1., xi_N)
    A = Particle(m_A, -1., xi_A)
    nu = Particle(0., 1., 0)

    y = 1e-5
    sin2_2th = 2e-11
    theta = asin(sqrt(sin2_2th)) / 2.
    model_params = ModelParams(y, theta)

    x = logrange(1e-6, 1e2, 1000)
    temp = m_N ./ x

    @time coll = coll_3_12.(temp, Ref(N), Ref(nu), Ref(A), Ref(model_params), Ref(energy_type), Ref(coll_A_Nnu_sq_amp))

    # println(coll)

    Plt.plot(
        minorgrid=true,
        xlabel=L"$x$",
        ylabel=L"Collision term",
    )
    Plt.plot!(
        xscale=:log10,
        yscale=:log10,
        ylim=(1e-100, 1e-20)
        # ylim = (0., 7e-16)
    )
    Plt.scatter!(x, coll)
    Plt.savefig("figures/test_coll.pdf")
    return 0
end


function test_coll_3_12_integral()
    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = -20.
    xi_A = 2. * xi_N

    N = Particle(m_N, 1., xi_N)
    A = Particle(m_A, -1., xi_A)
    nu = Particle(0., 1., 0)

    y = 1e-4
    sin2_2th = 1e-4
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(y, theta)

    x = 1e0
    temp = m_N / x

    println("T = ", temp)

    params = (
        temp=temp,
        p1=N,
        p2=nu,
        p3=A,
        energy_type=0,
    )

    e1_min = N.m

    n = 1000
    e1_max = logrange(1e0 * N.m, 1e8 * N.m, length=n)

    res = zeros(n)
    for i in 1:n
        problem = Integrals.IntegralProblem(coll_3_12_int_e2, (e1_min, e1_max[i]), params)
        sol = Integrals.solve(
            problem,
            Integrals.QuadGKJL(),
            abstol=1e-60,
            reltol=1e-4,
        )
        res[i] = sol[1]
    end

    Plt.plot(
        minorgrid=true,
        xlabel=L"$e_{max}$",
        ylabel=L"Integral",
    )
    Plt.plot!(
        xscale=:log10,
        yscale=:log10,
        ylim=(1e-60, 9e-10)
        # ylim = (0., 7e-16)
    )
    Plt.scatter!(e1_max / N.m, res)
    Plt.savefig("figures/test_e_max.pdf")

    # sq_amp = coll_3_12_sq_amp(model_params, N, nu, A)

    return 0
end

function test_inner_kernel()
    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = -10.
    xi_A = 2. * xi_N

    N = Particle(m_N, 1., xi_N)
    A = Particle(m_A, -1., xi_A)
    nu = Particle(0., 1., 0)

    y = 1e-4
    sin2_2th = 1e-4
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(1e-4, theta)

    x = 1e0
    temp = m_N / x

    n = 100
    e1 = 2 * N.m

    e2_m = coll_3_12_e2_min(e1, N, nu, A)
    e2_p = coll_3_12_e2_max(e1, N, nu, A)
    e2 = range(e2_m, e2_p, length=n)

    params = (
        e1=e1,
        temp=temp,
        p1=N,
        p2=nu,
        p3=A,
        energy_type=0,
    )

    println(params)

    Plt.plot(
        # xscale=:log10,
        # yscale=:log10,
        minorgrid=true,
        xlabel=L"$E_2$",
        ylabel=L"Inner Kernel",
    )
    Plt.plot!(
        # xlim=(7e-6, 1.1e-4),
        # ylim=(1e-20, 9e-10)
    )
    Plt.plot!(e2, coll_3_12_ker.(e2, Ref(params)))
    Plt.savefig("figures/test_inner_kernel.pdf")

    return 0
end

function test_outer_kernel()
    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = -10.
    xi_A = 2. * xi_N

    N = Particle(m_N, 1., xi_N)
    A = Particle(m_A, -1., xi_A)
    nu = Particle(0., 1., 0)

    y = 1e-4
    sin2_2th = 1e-4
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(y, theta)

    x = 1e-2
    temp = m_N / x

    e1_max = max(1e1*temp, 6*m_N)
    # e1_max = 3 * m_N

    n = 300
    e1 = range(N.m, e1_max, length=n)

    params = (
        temp=temp,
        p1=N,
        p2=nu,
        p3=A,
        energy_type=0,
    )

    sols = coll_3_12_int_e2.(e1, Ref(params))
    res = zeros(n)
    for i in 1:n
        res[i] = sols[i].u
    end

    Plt.plot(
        # xscale=:log10,
        # yscale=:log10,
        minorgrid=true,
        xlabel=L"$E_1$",
        ylabel=L"Outer Kernel",
    )
    Plt.plot!(
        # xlim=(7e-6, 1.1e-4),
        # ylim=(1e-20, 9e-10)
    )
    Plt.scatter!(e1, res)
    Plt.savefig("figures/test_outer_kernel.pdf")

    return 0
end

test_inner_kernel()
test_outer_kernel()
test_coll_3_12()
@time test_coll_3_12_integral()
