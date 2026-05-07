include(joinpath(@__DIR__, "../src/coll_12_34.jl"))


function test_int_e1()
    y = 1e-4
    sin2_2th = 1e-4
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(y, theta)

    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = -10.
    xi_A = 2. * xi_N

    temp = range(
        1e-3,
        1e3,
        length=100,
    )

    p1 = Particle(m_N, 1., xi_N)
    p2 = Particle(m_N, 1., xi_N)
    p3 = Particle(m_A, -1., xi_A)
    p4 = Particle(m_A, -1., xi_A)

    sol = coll_12_34_int_e1.(temp, Ref(p1), Ref(p2), Ref(p3), Ref(p4), Ref(model_params))
    println("sol = ", sol)
    p = Plt.plot(
        minorgrid=true,
        xlabel=L"$temp$",
        ylabel=L"Integral",
    )
    Plt.plot!(
        p,
        # xscale=:log10,
        # yscale=:log10,
    )
    Plt.scatter!(
        p,
        temp,
        sol,
    )
    Plt.savefig(p, "figures/coll_12_34_int_e1.pdf")
end

function fix_params()
    y = 1e-4
    sin2_2th = 1e-11
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(y, theta)

    m_N = 1e-5
    m_A = 2.5 * m_N
    # m_A = m_N
    xi_N = -10.
    xi_A = 2. * xi_N

    x = 1e0
    temp = m_N / x

    p1 = Particle(m_A, -1., xi=xi_A)
    p2 = Particle(m_A, -1., xi=xi_A)
    p3 = Particle(m_N, 1., xi=xi_N)
    p4 = Particle(m_N, 1., xi=xi_N)

    params = Params_12_34(
        model_params,
        p1,
        p2,
        p3,
        p4,
        temp,
    )

    return params
end

function test_ker()
    p = fix_params()

    p.p1.e = 1.1 * p.p1.m
    p.p1.mom = momentum(p.p1)
    p.p2.e = p.p1.e
    p.p2.mom = momentum(p.p2)

    rest_e = p.p1.e + p.p2.e - p.p3.m - p.p4.m

    # p.p3.e = p.p3.m
    p.p3.e = p.p3.m + rest_e/4.9999999
    p.p3.mom = momentum(p.p3)
    p.p4.e = p.p1.e + p.p2.e - p.p3.e
    p.p4.mom = momentum(p.p4)
    e_check = p.p1.e + p.p2.e - p.p3.e - p.p4.e
    println("e_check = ", e_check)

    smin = s_min(p)
    smax = s_max(p)
    println("smin = ", smin, ", smax = ", smax)
    if smin >= smax
        error("smin = ", smin, " >= smax = ", smax)
    end # if
    p.s = smin + (smax - smin)/1e3
    println("s = ", p.s)

    p.t_min = t_lim(-1., p)
    p.t_max = t_lim(1., p)
    p.a = a_theta(p)
    println("t_min = ", p.t_min, ", t_max = ", p.t_max)

    n = 100
    t = range(p.t_min, p.t_max, length=n)
    sq_amp = coll_12_34_sq_amp.(t, Ref(p))
    ker_vals = coll_12_34_ker.(t, Ref(p))

    # println("sq_amp = ", sq_amp)
    println("ker_vals = ", ker_vals)
    amp_plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$t$",
        ylabel=L"Squared amplitude",
        # xlims=(t_min, t_max),
        # ylims=(1e18, 1e20),
        # xscale=:log10,
        # yscale=:log10,
    )
    Plt.plot!(
        amp_plot,
        t,
        sq_amp,
    )
    Plt.savefig(amp_plot, "figures/coll_12_34_sq_amp.pdf")

    # println("ker_vals = ", ker_vals)
    ker_plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$t$",
        ylabel=L"Kernel",
        # xlims=(t_min, t_max),
        # ylims=(1e18, 1e20),
        # xscale=:log10,
        # yscale=:log10,
    )
    Plt.plot!(
        ker_plot,
        t,
        ker_vals,
    )
    Plt.savefig(ker_plot, "figures/coll_12_34_ker.pdf")
end

function test_int_t()
    p = fix_params()

    p.p1.e = 2. * p.p1.m
    p.p1.mom = momentum(p.p1)
    p.p2.e = p.p1.e
    p.p2.mom = momentum(p.p2)

    rest_e = p.p1.e + p.p2.e - p.p3.m - p.p4.m

    # p.p3.e = p.p3.m
    p.p3.e = p.p3.m + rest_e/2.
    p.p3.mom = momentum(p.p3)
    p.p4.e = p.p1.e + p.p2.e - p.p3.e
    p.p4.mom = momentum(p.p4)
    e_check = p.p1.e + p.p2.e - p.p3.e - p.p4.e
    println("e_check = ", e_check)
    println("e1 = ", p.p1.e, ", e2 = ", p.p2.e, ", e3 = ", p.p3.e, ", e4 = ", p.p4.e)

    smin = s_min(p)
    smax = s_max(p)
    println("smin = ", smin, ", smax = ", smax)
    if smin >= smax
        println("smin = ", smin, " >= smax = ", smax)
        println("Result is 0 for all t.")
    else
        # Regularisation of integral at s=s_min/s_max
        reg = (smax - smin) / 1e5
        # reg = 0.
        n = 100
        s = range(smin + reg, smax - reg, length=n)
        # s = 2e-8
        println("s = ", s)

        sol = coll_12_34_int_t.(s, Ref(p))
        println("Integral over t = ", sol)

        anal_sol = coll_12_34_int_t_anal.(s, Ref(p))
        new_sol = coll_12_34_int_t_new.(s, Ref(p))

        println("First fraction = ", sol[1] / anal_sol[1])

        plot = Plt.plot(
            minorgrid=true,
            xlabel=L"$s$",
            ylabel=L"Kernel",
            xlims=(1e-9, 1e-7),
            ylims=(1e-8, 1e-2),
            xscale=:log10,
            yscale=:log10,
        )
        Plt.plot!(
            plot,
            s,
            sol
        )
        Plt.scatter!(
            plot,
            s,
            anal_sol,
            ls=:dash
        )
        Plt.plot!(
            plot,
            s,
            new_sol,
            ls=:dashdot
        )
        Plt.savefig(plot, "figures/test_t_int.pdf")
    end # if
    return nothing
end

function test_int_s()
    p = fix_params()

    p.p1.e = 3. * p.p1.m
    p.p1.mom = momentum(p.p1)
    p.p2.e = p.p1.e
    p.p2.mom = momentum(p.p2)

    n = 1000
    e3 = range(
        p.p3.m,
        p.p1.e + p.p2.e - p.p4.m,
        length=n,
    )

    sol = coll_12_34_int_s.(e3, Ref(p))
    println("Integral over s = ", sol)

    plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$e3$",
        ylabel=L"Kernel",
        # xlims=(t_min, t_max),
        # ylims=(1e18, 1e20),
        # xscale=:log10,
        # yscale=:log10,
    )
    Plt.plot!(
        plot,
        e3,
        sol
    )
    Plt.savefig(plot, "figures/test_e3_int.pdf")
end

function test_int_e3()
    p = fix_params()

    p.p1.e = 3. * p.p1.m
    p.p1.mom = momentum(p.p1)

    max_mult = 1e3
    e2 = range(
        max(p.p2.m, p.p3.m + p.p4.m - p.p1.e),
        max(max_mult * p.temp, max_mult * p.p2.m),
        length=1000,
    )

    sol = coll_12_34_int_e3.(e2, Ref(p))
    println("Integral over e3 = ", sol)

    plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$e2$",
        ylabel=L"Kernel",
        # xlims=(t_min, t_max),
        ylims=(1e-70, 1e-35),
        # xscale=:log10,
        yscale=:log10,
    )
    Plt.plot!(
        plot,
        e2,
        -sol
    )
    Plt.savefig(plot, "figures/test_e2_int.pdf")
end # function


function test_int()
    p = fix_params()

    sol = coll_12_34(p.model_params, p.p1, p.p2, p.p3, p.p4, p.temp)

    println("Integral = ", sol)
end


# test_int_e1()
# test_ker()
# test_int_t()
# test_int_s()
# @time test_int_e3()
@time test_int()