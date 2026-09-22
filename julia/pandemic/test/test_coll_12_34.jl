using LaTeXStrings
ENV["GKSwstype"] = "nul"
import Plots as Plt
import BenchmarkTools as BT

include(joinpath(@__DIR__, "../src/coll_12_34.jl"))

"""Not up to date [15.09.26]"""
function test_int_e1()
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
        p1[i] = Particle{Float64}(m_N, 1, dof=2)
        p2[i] = Particle{Float64}(m_N, 1, dof=2)
        p3[i] = Particle{Float64}(m_A, -1, dof=3)
        p4[i] = Particle{Float64}(m_A, -1, dof=3)
    end
    params = Params_12_34{Float64}.(p1, p2, p3, p4)

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
    m_N = 1e-5
    m_A = 2.5 * m_N
    xi_N = -10.
    xi_A = 2. * xi_N

    x = 1e0
    temp = m_N / x
    temps = (temp, temp, temp, temp)
    xis = (xi_A, xi_A, xi_N, xi_N)

    p1 = Particle{Float64}(m_A, -1, dof=3)
    p2 = Particle{Float64}(m_A, -1, dof=3)
    p3 = Particle{Float64}(m_N, 1, dof=2)
    p4 = Particle{Float64}(m_N, 1, dof=2)

    check_1 = chem_pot_check(p1, temps[1], xis[1])
    check_2 = chem_pot_check(p2, temps[2], xis[2])
    check_3 = chem_pot_check(p3, temps[3], xis[3])
    check_4 = chem_pot_check(p4, temps[4], xis[4])

    if !check_1 || !check_2 || !check_3 || !check_4
        println("Error: Negative distribution function for one of the particles. Check chemical potentials, temperatures and masses.")
        println("check_1 = ", check_1, ", check_2 = ", check_2, ", check_3 = ", check_3, ", check_4 = ", check_4)
        error()
    end # if

    params = Params_12_34{Float64, Float64}(
        p1,
        p2,
        p3,
        p4,
        temps,
        xis,
    )

    return params
end

function test_ker()
    p = fix_params()

    p.e1 = 4. * p.p1.m
    p.mom1 = momentum(p.p1, p.e1)
    p.e2 = p.e1
    p.mom2 = momentum(p.p2, p.e2)

    rest_e = p.e1 + p.e2 - p.p3.m - p.p4.m

    # p.p3.e = p.p3.m
    p.e3 = p.p3.m + rest_e / 2.
    p.mom3 = momentum(p.p3, p.e3)
    p.e4 = p.e1 + p.e2 - p.e3
    p.mom4 = momentum(p.p4, p.e4)
    e_check = p.e1 + p.e2 - p.e3 - p.e4
    println("e_check = ", e_check)

    smin = s_min(p)
    smax = s_max(p)
    println("smin = ", smin, ", smax = ", smax)
    if smin >= smax
        error("smin = ", smin, " >= smax = ", smax)
    end # if
    # p.s = smin + (smax - smin)/1e2
    # p.s = 2.650522147147156e-9
    p.s = (smin + smax) / 2.
    println("s = ", p.s)

    p.t_min = t_lim(1, p)
    p.t_max = t_lim(-1, p)
    p.a = a_theta(p)
    println("t_min = ", p.t_min, ", t_max = ", p.t_max)

    # t_pole = 2*p.p1.m^2 + p.p3.m^2 - p.s
    # println("Pole in t: ", t_pole)
    # println("Amp at pole: ", coll_12_34_sq_amp_no_pole(t_pole, p))

    n = 20000
    t = range(p.t_min, p.t_max, length=n)
    sq_amp = coll_12_34_sq_amp.(t, Ref(p))
    ker_vals = coll_12_34_ker.(t, Ref(p))

    # t_less_pole = range(p.t_min, t_pole, length=n)
    # ker_less_pole = coll_12_34_ker.(t_less_pole, Ref(p))

    # sq_amp_pole = coll_12_34_sq_amp_pole.(t, Ref(p))
    # ker_pole = coll_12_34_ker_pole.(t, Ref(p))

    # sq_amp_no_pole = coll_12_34_sq_amp_no_pole.(t, Ref(p))

    # println("sq_amp = ", sq_amp)
    # println("ker_vals = ", ker_vals)
    amp_plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$t$",
        ylabel=L"\textrm{Squared amplitude}",
        xlims=(p.t_min, p.t_max),
        # ylims=(1e-20, 1e-5),
        # ylims=(-1e-13, 4e-13),
        # xscale=:log10,
        # yscale=:log10,
    )
    Plt.plot!(
        amp_plot,
        t,
        sq_amp,
        # ms=.4,
    )
    # Plt.plot!(
    #     amp_plot,
    #     t,
    #     sq_amp_pole,
    #     ls=:dash,
    # )

    Plt.savefig(amp_plot, "figures/coll_12_34_sq_amp.pdf")

    t_len = p.t_max - p.t_min

    # println("ker_vals = ", ker_vals)
    ker_plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$t$",
        ylabel=L"Kernel",
        xlims=(p.t_min, p.t_max),
        # x_lims=(t_pole - 1e-3 * t_len, t_pole + 1e-3 * t_len),
        # ylims=(-1e4, 1e4),
        # xscale=:log10,
        # yscale=:log10,
    )
    Plt.plot!(
        ker_plot,
        t,
        ker_vals,
        ms=.4,
    )
    # Plt.plot!(
    #     ker_plot,
    #     t_less_pole,
    #     ker_less_pole,
    #     ls=:dash,
    # )
    # Plt.plot!(
    #     ker_plot,
    #     t,
    #     ker_pole,
    #     ls=:dash,
    # )
    Plt.savefig(ker_plot, "figures/coll_12_34_ker.pdf")
end

"""Not up to date [15.09.26]"""
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

    p.e1 = 4. * p.p1.m
    p.mom1 = momentum(p.p1, p.e1)
    p.e2 = p.e1
    p.mom2 = momentum(p.p2, p.e2)

    rest_e = p.e1 + p.e2 - p.p3.m - p.p4.m

    p.e3 = p.p3.m + rest_e/2.
    p.mom3 = momentum(p.p3, p.e3)
    p.e4 = p.e1 + p.e2 - p.e3
    p.mom4 = momentum(p.p4, p.e4)

    smin = s_min(p)
    smax = s_max(p)
    println("smin = ", smin, ", smax = ", smax)
    println("4E1^2 = ", 4 * p.e1^2, ", 4E3^2 = ", 4 * p.e3^2)
    println("4m1^2 = ", 4 * p.p1.m^2, ", 4m3^2 = ", 4 * p.p3.m^2)
    if smin >= smax
        println("smin = ", smin, " >= smax = ", smax)
        println("Result is 0 for all t.")
        return nothing
    end # if
    # Regularisation of integral at s=(s_min and s_max)
    reg = (smax - smin) / 1e5
    # reg = 0.
    n = 200
    s = range(smin + reg, smax - reg, length=n)


    # s0 = (smax + smin) / 2.
    # int_t0 = coll_12_34_int_t_anal(s0, p)
    # println("s0 = ", s0)
    # println("Integral at s0 = ", int_t0)

    # sol = Array{Float64}(undef, n)
    # for i in 1:n
    #     try
    #         sol[i] = coll_12_34_int_t(s[i], p)
    #     catch e
    #         println("n = ", n, ", i = ", i)
    #         println("Error at s = ", s[i], ": ", e)
    #         error()
    #     end
    # end
    # sol = coll_12_34_int_t.(s, Ref(p))

    # anal_sol = coll_12_34_int_t_anal.(s, Ref(p))
    new_sol = coll_12_34_int_t_new.(s, Ref(p))

    # println(anal_sol)

    # qr_sol = coll_12_34_int_t_qr.(s, Ref(p))

    plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$s$",
        ylabel=L"Kernel",
        # xlims=(1e-9, 1e-7),
        # ylims=(-1e-5, 1e-5),
        # xscale=:log10,
        yscale=:log10,
    )
    # Plt.plot!(
    #     plot,
    #     s,
    #     sol
    # )
    # Plt.plot!(
    #     plot,
    #     s,
    #     anal_sol,
    #     # ls=:dash
    # )
    Plt.plot!(
        plot,
        s,
        new_sol,
        ls=:dashdot
    )
    Plt.savefig(plot, "figures/test_t_int.pdf")
    return nothing
end

function test_int_s()
    p = fix_params()

    p.e1 = 3. * p.p1.m
    p.mom1 = momentum(p.p1, p.e1)
    p.e2 = p.e1
    p.mom2 = momentum(p.p2, p.e2)

    n = 1000
    e3 = range(
        p.p3.m,
        p.e1 + p.e2 - p.p4.m,
        length=n,
    )

    @time sol = coll_12_34_int_s.(e3, Ref(p))
    # println("Integral over s = ", sol)

    plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$e3$",
        ylabel=L"Kernel",
        # xlims=(0, 1.5e-4),
        # ylims=(-5e-33, 2e-34),
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

    p.e1 = 3. * p.p1.m
    p.mom1 = momentum(p.p1, p.e1)

    max_mult = 1e2
    e2 = range(
        max(p.p2.m, p.p3.m + p.p4.m - p.e1),
        max(max_mult * p.temps[2], max_mult * p.p2.m),
        length=100,
    )

    sol = coll_12_34_int_e3.(e2, Ref(p))
    println("Integral over e3 = ", sol)

    plot = Plt.plot(
        minorgrid=true,
        xlabel=L"$e2$",
        ylabel=L"Kernel",
        # xlims=(t_min, t_max),
        # ylims=(1e-70, 1e-35),
        # xscale=:log10,
        # yscale=:log10,
    )
    Plt.plot!(
        plot,
        e2,
        sol
    )
    Plt.savefig(plot, "figures/test_e3_int.pdf")
end # function


function test_int()
    y = 1e-4
    sin2_2th = 1e-11
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams{Float64}(y, theta)

    p = fix_params()

    @time sol = coll_12_34(model_params, p.p1, p.p2, p.p3, p.p4, p.temps, p.xis)

    println("Integral = ", sol)
end


# test_ker()
# test_int_t()
# test_int_e1()
# test_rescale_ker()
# test_int_s()
# @time test_int_e3()
test_int()
