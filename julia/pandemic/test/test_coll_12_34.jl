import BenchmarkTools as BT

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

    n = 100
    temp = logrange(
        1e-6,
        1e1,
        length=n,
    )

    p1 = Array{Particle{Float64}}(undef, n)
    p2 = Array{Particle{Float64}}(undef, n)
    p3 = Array{Particle{Float64}}(undef, n)
    p4 = Array{Particle{Float64}}(undef, n)

    for i in 1:n
        p1[i] = Particle{Float64}(m_N, 1., xi=xi_N, temp=temp[i])
        p2[i] = Particle{Float64}(m_N, 1., xi=xi_N, temp=temp[i])
        p3[i] = Particle{Float64}(m_A, -1., xi=xi_A, temp=temp[i])
        p4[i] = Particle{Float64}(m_A, -1., xi=xi_A, temp=temp[i])
    end
    params = Params_12_34{Float64}.(Ref(model_params), p1, p2, p3, p4)

    sol = coll_12_34_int_e1.(params)
    println("sol = ", sol)
    p = Plt.plot(
        minorgrid=true,
        xlabel=L"$temp$",
        ylabel=L"Integral",
    )
    Plt.plot!(
        p,
        xscale=:log10,
        yscale=:log10,
        ylim=(1e-60, 1e-10)
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
    model_params = ModelParams{Float64}(y, theta)

    m_N = 1e-5
    m_A = 2.5 * m_N
    # m_A = m_N
    xi_N = -10.
    xi_A = 2. * xi_N

    x = 1e0
    temp = m_N / x

    p1 = Particle{Float64}(m_A, -1., xi=xi_A, temp=temp)
    p2 = Particle{Float64}(m_A, -1., xi=xi_A, temp=temp)
    p3 = Particle{Float64}(m_N, 1., xi=xi_N, temp=temp)
    p4 = Particle{Float64}(m_N, 1., xi=xi_N, temp=temp)

    params = Params_12_34{Float64}(
        model_params,
        p1,
        p2,
        p3,
        p4,
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
    p.p3.e = p.p3.m + rest_e / 2.
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
    p.s = smin + (smax - smin)/1e2
    println("s = ", p.s)

    p.t_min = t_lim(-1., p)
    p.t_max = t_lim(1., p)
    p.a = a_theta(p)
    println("t_min = ", p.t_min, ", t_max = ", p.t_max)

    n = 10000
    t = range(p.t_min, p.t_max, length=n)
    sq_amp = coll_12_34_sq_amp.(t, Ref(p))
    ker_vals = coll_12_34_ker.(t, Ref(p))

    # println("sq_amp = ", sq_amp)
    # println("ker_vals = ", ker_vals)
    amp_plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$t$",
        ylabel=L"\textrm{Squared amplitude}",
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
        # xlims=(p.t_min, p.t_max),
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

function test_rescale_ker()
    p = fix_params()

    p.p1.e = 1.1 * p.p1.m
    p.p1.mom = momentum(p.p1)
    p.p2.e = p.p1.e
    p.p2.mom = momentum(p.p2)

    rest_e = p.p1.e + p.p2.e - p.p3.m - p.p4.m

    # p.p3.e = p.p3.m
    p.p3.e = p.p3.m + rest_e / 2.
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
    p.s = smin + (smax - smin)/1e2
    println("s = ", p.s)

    p.t_min = t_lim(-1., p)
    p.t_max = t_lim(1., p)
    p.a = a_theta(p)
    println("t_min = ", p.t_min, ", t_max = ", p.t_max)

    t_0 = (p.t_max + p.t_min) / 2.

    n = 4000
    t = range(t_0, p.t_max, length=n)
    sq_amp = coll_12_34_sq_amp.(t, Ref(p))
    ker_vals = coll_12_34_ker.(t, Ref(p))

    r = log.(t .- p.t_min)
    q = log.(p.t_max .- t)

    # println("sq_amp = ", sq_amp)
    # println("ker_vals = ", ker_vals)
    amp_plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$t$",
        ylabel=L"\textrm{Squared amplitude}",
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
        # xlims=(t_0, p.t_max),
        # ylims=(1e18, 1e20),
        # xscale=:log10,
        # yscale=:log10,
    )
    Plt.scatter!(
        ker_plot,
        q,
        ker_vals .* exp.(q),
    )
    Plt.savefig(ker_plot, "figures/coll_12_34_rescaled_ker.pdf")
end

function test_int_t()
    p = fix_params()

    p.p1.e = 4. * p.p1.m
    p.p1.mom = momentum(p.p1)
    p.p2.e = p.p1.e
    p.p2.mom = momentum(p.p2)

    rest_e = p.p1.e + p.p2.e - p.p3.m - p.p4.m

    p.p3.e = p.p3.m + rest_e/2.
    p.p3.mom = momentum(p.p3)
    p.p4.e = p.p1.e + p.p2.e - p.p3.e
    p.p4.mom = momentum(p.p4)
    # e_check = p.p1.e + p.p2.e - p.p3.e - p.p4.e
    # println("e_check = ", e_check)
    # println("e1 = ", p.p1.e, ", e2 = ", p.p2.e, ", e3 = ", p.p3.e, ", e4 = ", p.p4.e)

    smin = s_min(p)
    smax = s_max(p)
    println("smin = ", smin, ", smax = ", smax)
    if smin >= smax
        println("smin = ", smin, " >= smax = ", smax)
        println("Result is 0 for all t.")
    else
        # Regularisation of integral at s=(s_min and s_max)
        reg = (smax - smin) / 1e5
        # reg = 0.
        n = 1000
        s = range(smin + reg, smax - reg, length=n)

        sol = coll_12_34_int_t.(s, Ref(p))

        anal_sol = coll_12_34_int_t_anal.(s, Ref(p))
        new_sol = coll_12_34_int_t_new.(s, Ref(p))

        qr_sol = coll_12_34_int_t_qr.(s, Ref(p))

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
        Plt.plot!(
            plot,
            s,
            qr_sol,
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

    @time sol = coll_12_34_int_s.(e3, Ref(p))
    # println("Integral over s = ", sol)

    plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$e3$",
        ylabel=L"Kernel",
        xlims=(0, 1.5e-4),
        ylims=(-5e-33, 1.5e-34),
    )
    Plt.plot!(
        plot,
        e3,
        sol
    )
    Plt.savefig(plot, "figures/test_s_int.pdf")
    return nothing 
end

function test_int_e3()
    p = fix_params()

    p.p1.e = 3. * p.p1.m
    p.p1.mom = momentum(p.p1)

    max_mult = 1e2
    e2 = range(
        max(p.p2.m, p.p3.m + p.p4.m - p.p1.e),
        max(max_mult * p.p2.temp, max_mult * p.p2.m),
        length=1000,
    )

    sol = coll_12_34_int_e3.(e2, Ref(p))
    # println("Integral over e3 = ", sol)

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
    Plt.savefig(plot, "figures/test_e3_int.pdf")
end # function


function test_int()
    p = fix_params()

    sol = coll_12_34(p.model_params, p.p1, p.p2, p.p3, p.p4)

    println("Integral = ", sol)
end


# test_int_e1()
# test_ker()
# test_rescale_ker()
# test_int_t()
# test_int_s()
# @time test_int_e3()
@time test_int()
