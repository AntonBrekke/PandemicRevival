using LaTeXStrings
ENV["GKSwstype"] = "nul"
import Plots as Plt

include(joinpath(@__DIR__, "../src/coll_3_12.jl"))
import .coll_3_12


function test_coll_3_12()
    # energy_type = Val{0}

    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = -20.
    xi_A = 2. * xi_N

    n = 1000
    x = logrange(1e-6, 1e2, n)
    temp = m_N ./ x

    N = Array{Particle{Float64}}(undef, n)
    A = Array{Particle{Float64}}(undef, n)
    nu = Array{Particle{Float64}}(undef, n)
    for i in 1:n
        N[i] = Particle{Float64}(m_N, 1., xi=xi_N, temp=temp[i])
        A[i] = Particle{Float64}(m_A, -1., xi=xi_A, temp=temp[i])
        nu[i] = Particle{Float64}(0., 1., xi=0., temp=temp[i])
    end

    y = 1e-5
    sin2_2th = 2e-11
    theta = asin(sqrt(sin2_2th)) / 2.
    model_params = ModelParams(y, theta)


    @time coll_log = coll_3_12_log.(
        N, nu, A,
        Ref(model_params),
        # sq_amp_func = Ref(coll_A_Nnu_sq_amp)
    )
    @time coll = coll_3_12.(
        N, nu, A,
        Ref(model_params),
        # Ref(energy_type),
        # Ref(coll_A_Nnu_sq_amp)
    )

    # println(coll)

    Plt.plot(
        # minorgrid=true,
        xlabel=L"$x$",
        ylabel=L"Collision term",
    )
    Plt.plot!(
        xscale=:log10,
        yscale=:log10,
        ylim=(1e-100, 1e-20)
        # ylim = (0., 7e-16)
    )
    Plt.plot!(x, coll)
    Plt.plot!(x, coll_log)
    Plt.savefig("figures/test_coll.pdf")
    return nothing
end


function test_coll_3_12_integral()
    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = -20.
    xi_A = 2. * xi_N

    x = 1e0
    temp = m_N / x

    N = Particle{Float64}(m_N, 1., xi=xi_N, temp=temp)
    A = Particle{Float64}(m_A, -1., xi=xi_A, temp=temp)
    nu = Particle{Float64}(0., 1., xi=0, temp=temp)

    y = 1e-4
    sin2_2th = 1e-4
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(y, theta)

    println("T = ", temp)

    params = Params_3_12{Float64, Val{0}}(
        model_params,
        N, nu, A
    )

    e1_min = N.m

    n = 1000
    e1_max = logrange(1e0 * N.m, 1e8 * N.m, length=n)

    res = zeros(n)
    @time for i in 1:n
        problem = Integrals.IntegralProblem(coll_3_12_int_e2, (e1_min, e1_max[i]), params)
        sol = Integrals.solve(
            problem,
            Integrals.QuadGKJL(),
            abstol=1e-60,
            reltol=1e-4,
        )
        res[i] = sol.u
    end

    y1_min = log(e1_min)
    y1_max = log.(e1_max)
    res_log = zeros(n)
    @time for i in 1:n
        problem = Integrals.IntegralProblem(
            coll_3_12_int_e2_log,
            (y1_min, y1_max[i]),
            params
        )
        sol = Integrals.solve(
            problem,
            Integrals.QuadGKJL(),
            abstol=1e-60,
            reltol=1e-4,
        )
        res_log[i] = sol[1]
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
    Plt.plot!(e1_max / N.m, res_log)
    Plt.plot!(e1_max / N.m, res)
    Plt.savefig("figures/test_e_max.pdf")

    # sq_amp = coll_3_12_sq_amp(model_params, N, nu, A)

    return nothing
end

function test_kernel()
    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = -10.
    xi_A = 2. * xi_N

    x = 1e0
    temp = m_N / x

    N = Particle{Float64}(m_N, 1., xi=xi_N, temp=temp)
    A = Particle{Float64}(m_A, -1., xi=xi_A, temp=temp)
    nu = Particle{Float64}(0., 1., xi=0, temp=temp)

    y = 1e-4
    sin2_2th = 1e-4
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(1e-4, theta)

    N.e = 2 * N.m
    N.mom = momentum(N)

    params = Params_3_12{Float64, Val{0}}(
        model_params,
        N, nu, A,
    )

    n = 1000
    e2_m = coll_3_12_e2_min(params)
    e2_p = coll_3_12_e2_max(params)
    e2 = logrange(e2_m, e2_p, length=n)

    coll = coll_3_12_ker.(e2, Ref(params))

    Plt.plot(
        # xscale=:log10,
        # yscale=:log10,
        minorgrid=true,
        xlabel=L"$E_2$",
        ylabel=L"\textrm{Inner\ Kernel}",
    )
    Plt.plot!(
        # xlim=(7e-6, 1.1e-4),
        # ylim=(1e-20, 9e-10)
    )
    Plt.scatter!(e2, coll)
    Plt.savefig("figures/test_inner_kernel.pdf")

    return nothing
end

function test_e2_int()
    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = -10.
    xi_A = 2. * xi_N

    x = 1e-2
    temp = m_N / x

    N = Particle{Float64}(m_N, 1., xi=xi_N, temp=temp)
    A = Particle{Float64}(m_A, -1., xi=xi_A, temp=temp)
    nu = Particle{Float64}(0., 1., xi=0, temp=temp)

    y = 1e-4
    sin2_2th = 1e-4
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(y, theta)


    e1_max = max(1e1*temp, 6*m_N)
    # e1_max = 3 * m_N

    n = 40000
    e1 = logrange(N.m, e1_max, length=n)

    params = Params_3_12{Float64, Val{0}}(
        model_params,
        N, nu, A,
    )

    y1 = log.(e1)
    @time sols = coll_3_12_int_e2.(e1, Ref(params))
    @time sols_log = coll_3_12_int_e2_log.(y1, Ref(params)) ./ e1

    res = zeros(n)
    res_log = zeros(n)
    for i in 1:n
        res[i] = sols[i]
        res_log[i] = sols_log[i]
    end

    Plt.plot(
        xscale=:log10,
        # yscale=:log10,
        minorgrid=true,
        xlabel=L"$E_1$",
        ylabel=L"Outer Kernel",
    )
    Plt.plot!(
        # xlim=(7e-6, 1.1e-4),
        # ylim=(1e-20, 9e-10)
    )
    Plt.plot!(e1, res)
    Plt.plot!(e1, res_log)
    Plt.savefig("figures/test_outer_kernel.pdf")

    return nothing
end

test_kernel()
# test_e2_int()
# test_coll_3_12()
# test_coll_3_12_integral()
