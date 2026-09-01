using LaTeXStrings
ENV["GKSwstype"] = "nul"
import Plots as Plt
import DataFrames

include(joinpath(@__DIR__, "../src/coll_3_12.jl"))
import .coll_3_12


function test_coll_3_12()
    # energy_type = Val{0}

    m_N = 1e-5
    m_A = 2.5 * m_N

    dof_N = 2
    dof_A = 3
    dof_nu = 2
    # dof_N = 1
    # dof_A = 1
    # dof_nu = 1

    # xi has to be non-positive for A to avoid negative distribution function
    # for small values of x.
    # (xi < m_A / T)
    xi_N = -1.
    xi_A = 2. * xi_N
    xis = (xi_N, 0., xi_A)

    n = 1000
    x = logrange(1e-6, 1e2, n)
    temp = m_N ./ x
    temps = Vector{NTuple{3, Float64}}(undef, n)
    for i in 1:n
        temps[i] = (temp[i], temp[i], temp[i])
    end

    # N = Array{Particle{Float64}}(undef, n)
    # A = Array{Particle{Float64}}(undef, n)
    # nu = Array{Particle{Float64}}(undef, n)
    # for i in 1:n
    #     N[i] = Particle{Float64}(m_N, 1, dof=2)
    #     A[i] = Particle{Float64}(m_A, -1, dof=3)
    #     nu[i] = Particle{Float64}(0., 1, dof=2)
    # end
    N = Particle{Float64}(m_N, 1, dof=dof_N)
    A = Particle{Float64}(m_A, -1, dof=dof_A)
    nu = Particle{Float64}(0., 1, dof=dof_nu)
    y = 1e-5
    sin2_2th = 1e-11
    theta = asin(sqrt(sin2_2th)) / 2.
    model_params = ModelParams(y, theta)

    @time coll = coll_3_12.(
        Ref(model_params),
        Ref(N), Ref(nu), Ref(A),
        temps,
        Ref(xis),
    )
    @time coll_log = coll_3_12_log.(
        Ref(model_params),
        Ref(N), Ref(nu), Ref(A),
        temps,
        Ref(xis),
    )

    coll_rho_A = coll_3_12.(
        Ref(model_params),
        Ref(N), Ref(nu), Ref(A),
        temps,
        Ref(xis),
        energy_type=Val(3),
    )
    coll_rho_N = coll_3_12.(
        Ref(model_params),
        Ref(N), Ref(nu), Ref(A),
        temps,
        Ref(xis),
        energy_type=Val(1),
    )
    coll_rho_nu = coll_3_12.(
        Ref(model_params),
        Ref(N), Ref(nu), Ref(A),
        temps,
        Ref(xis),
        energy_type=Val(2),
    )

    Plt.plot(
        # minorgrid=true,
        xlabel=L"$x$",
        ylabel=L"\textrm{Collision\ term}",
    )
    Plt.plot!(
        xscale=:log10,
        # yscale=:log10,
        # ylim=(1e-100, 1e-20)
        # ylim = (0., 7e-16)
    )
    Plt.plot!(x, coll)
    Plt.plot!(x, coll_log, ls=:dash)
    Plt.savefig("figures/test_coll.pdf")

    x_coll_array = [x;; coll;; coll_log;; coll_rho_A;; coll_rho_N;; coll_rho_nu]
    csv_path = joinpath(@__DIR__, "../tmp/test_coll_3_12.csv")
    export_array_to_csv(DataFrames.DataFrame(x_coll_array, :auto), csv_path)

    return nothing
end

function test_coll_3_12_fixed_temperature()
    m_N = 1e-5
    m_A = 2.5 * m_N

    dof_N = 2
    dof_A = 3
    dof_nu = 2

    N = Particle{Float64}(m_N, 1, dof=dof_N)
    A = Particle{Float64}(m_A, -1, dof=dof_A)
    nu = Particle{Float64}(0., 1, dof=dof_nu)

    y = 1e-5
    sin2_2th = 1e-11
    theta = asin(sqrt(sin2_2th)) / 2.
    model_params = ModelParams(y, theta)

    x_fixed = 1e0
    temperature = m_N / x_fixed
    temp_N = 1.2e-5
    temp_nu = 1e-5


    xi_N = range(-20., 0., length=1000)
    xi_A = 2. .* xi_N

    # temps = [temperature, temperature, temperature]
    temps = [temp_N, temp_nu, temp_N]
    coll = Vector{Float64}(undef, length(xi_N))
    coll_rho_A = similar(coll)
    coll_rho_N = similar(coll)
    coll_rho_nu = similar(coll)

    for i in eachindex(xi_N)
        xis = [xi_N[i], 0., xi_A[i]]
        coll[i] = coll_3_12(
            model_params,
            N, nu, A,
            temps,
            xis,
        )
        coll_rho_A[i] = coll_3_12(
            model_params,
            N, nu, A,
            temps,
            xis,
            energy_type=Val(3),
        )
        coll_rho_N[i] = coll_3_12(
            model_params,
            N, nu, A,
            temps,
            xis,
            energy_type=Val(1),
        )
        coll_rho_nu[i] = coll_3_12(
            model_params,
            N, nu, A,
            temps,
            xis,
            energy_type=Val(2),
        )
    end

    coll_rho = coll_rho_A .- coll_rho_N

    function test(x, scale)
        return sign(x) * log1p(abs(x)/scale)
    end
    function test2(x, scale)
        # scale = maximum(abs, x)
        return asinh.(x/scale)
    end

    function test3(x, scale)
        return log(- x/scale)
    end

    # TODO: Stupid hack to test
    scale_xi = maximum(abs, xi_N)
    # scale_xi = abs(minimum(xi_N))
    println(scale_xi)
    scale_n = abs(minimum(coll))
    scale_rho = abs(minimum(coll_rho))
    scale_rho_N = abs(minimum(coll_rho_N))
    scale_rho_A = abs(minimum(coll_rho_A))
    scale_rho_nu = abs(minimum(coll_rho_nu))

    println(xi_N[end])

    # xi_N = test2.(xi_N, scale_xi)

    println(xi_N[end])

    Plt.plot(
        xlabel=L"\xi_N",
        ylabel=L"C_n",
        label=L"C_n",
        xlim=(minimum(xi_N), maximum(xi_N)),
    )
    Plt.hline!([0.], ls=:dot, color=:black, label=L"y=0")
    # Plt.plot!(xi_N, test3.(coll, scale_n))
    Plt.plot!(xi_N, coll)
    Plt.savefig("figures/test_coll_3_12_fixed_temperature_C_n.pdf")

    Plt.plot(
        xlabel=L"\xi_N",
        ylabel=L"C_\rho",
        xlim=(minimum(xi_N), maximum(xi_N)),
    )
    Plt.hline!([0.], ls=:dot, color=:black, label=L"y=0")

    Plt.plot!(
        xi_N, test2.(coll_rho, scale_rho),
        label=L"C_\rho = C_{\rho,A} - C_{\rho,N}",
    )
    Plt.plot!(xi_N, test3.(coll_rho_N, scale_rho_N), ls=:dash, label=L"C_{\rho,N}")
    Plt.plot!(xi_N, test3.(coll_rho_A, scale_rho_A), ls=:dash, label=L"C_{\rho,A}")
    Plt.plot!(xi_N, test3.(coll_rho_nu, scale_rho_nu), ls=:dash, label=L"C_{\rho,\nu}")
    Plt.savefig("figures/test_coll_3_12_fixed_temperature_C_rho.pdf")

    x_coll_array = [xi_N;; coll;; coll_rho_A;; coll_rho_N;; coll_rho_nu]
    csv_path = joinpath(@__DIR__, "../tmp/test_coll_3_12_fixed_temperature.csv")
    export_array_to_csv(DataFrames.DataFrame(x_coll_array, :auto), csv_path)

    return nothing
end


function test_coll_3_12_integral()
    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = 1.24
    xi_A = 2. * xi_N

    x = 1e0
    temp = m_N / x
    temps = fill(temp, 3)
    xis = [xi_N, 0., xi_A]

    N = Particle{Float64}(m_N, 1, dof=2)
    A = Particle{Float64}(m_A, -1, dof=3)
    nu = Particle{Float64}(0., 1, dof=2)

    check_N = chem_pot_check(N, temps[1], xis[1])
    check_nu = chem_pot_check(nu, temps[2], xis[2])
    check_A = chem_pot_check(A,  temps[3], xis[3])

    if !check_N || !check_A || !check_nu
        println("Error: Negative distribution function for one of the particles. Check chemical potentials, temperatures and masses.")
        println("check_N = ", check_N, ", check_nu = ", check_nu, ", check_A = ", check_A)
        error()
    end

    y = 1e-5
    sin2_2th = 1e-11
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(y, theta)

    println("T = ", temp)

    params = Params_3_12{Float64, Float64, Val{0}}(
        model_params,
        N, nu, A,
        temps,
        xis,
    )

    e1_min = N.m

    n = 1000
    e1_max = logrange(1e0 * N.m, 1e8 * N.m, length=n)

    res = zeros(n)
    @time for i in 1:n
        problem = Integrals.IntegralProblem(
            coll_3_12_int_e2,
            (e1_min, e1_max[i]),
            params
        )
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
        # yscale=:log10,
        # ylim=(1e-60, 9e-10)
        # ylim = (0., 7e-16)
    )
    Plt.plot!(e1_max / N.m, res)
    Plt.plot!(e1_max / N.m, res_log, ls=:dash)
    Plt.savefig("figures/test_e_max.pdf")

    # sq_amp = coll_3_12_sq_amp(model_params, N, nu, A)

    return nothing
end

function test_kernel()
    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = 1.25
    xi_A = 2. * xi_N

    x = 1e0
    temp = m_N / x
    temps = (temp, temp, temp)
    xis = (xi_N, 0., xi_A)

    N = Particle{Float64}(m_N, 1, dof=2)
    nu = Particle{Float64}(0., 1, dof=2)
    A = Particle{Float64}(m_A, -1, dof=3)

    check_N = chem_pot_check(N, temps[1], xis[1])
    check_nu = chem_pot_check(nu, temps[2], xis[2])
    check_A = chem_pot_check(A,  temps[3], xis[3])

    if !check_N || !check_A || !check_nu
        println("Error: Negative distribution function for one of the particles. Check chemical potentials, temperatures and masses.")
        println("check_N = ", check_N, ", check_nu = ", check_nu, ", check_A = ", check_A)
        error()
    end

    y = 1e-5
    sin2_2th = 1e-11
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(1e-4, theta)

    params = Params_3_12{Float64, Float64, Float64, Val{0}}(
        model_params,
        N, nu, A,
        temps,
        xis,
    )

    # e_N = 2. * N.m
    e_N = 3.300034791125285e-5
    mom_N = momentum(N, e_N)
    params.e1 = e_N
    params.mom1 = mom_N

    n = 200
    e2_m = coll_3_12_e2_min(params)
    e2_p = coll_3_12_e2_max(params)
    e2 = logrange(e2_m, e2_p, length=n)

    coll = coll_3_12_ker.(e2, Ref(params))

    Plt.plot(
        xscale=:log10,
        # yscale=:log10,
        minorgrid=true,
        xlabel=L"$E_2$",
        ylabel=L"\textrm{Inner\ Kernel}",
    )
    Plt.plot!(
        xlim=(1e-6, 2e-4),
        # ylim=(-1e-9, 1e-9)
    )
    Plt.scatter!(e2, coll)
    Plt.savefig("figures/test_inner_kernel.pdf")

    return nothing
end

function test_e2_int()
    m_N = 1e-5
    m_A = 2.5 * m_N

    xi_N = -1.
    xi_A = 2. * xi_N

    x = 1e0
    temp = m_N / x

    temps = (temp, temp, temp)
    xis = (xi_N, 0., xi_A)

    N = Particle{Float64}(m_N, 1, dof=2)
    A = Particle{Float64}(m_A, -1, dof=3)
    nu = Particle{Float64}(0., 1, dof=2)

    check_N = chem_pot_check(N, temps[1], xis[1])
    check_nu = chem_pot_check(nu, temps[2], xis[2])
    check_A = chem_pot_check(A,  temps[3], xis[3])

    if !check_N || !check_A || !check_nu
        println("Error: Negative distribution function for one of the particles. Check chemical potentials, temperatures and masses.")
        println("check_N = ", check_N, ", check_nu = ", check_nu, ", check_A = ", check_A)
        error()
    end

    y = 1e-5
    sin2_2th = 1e-11
    theta = asin(sqrt(sin2_2th))/2
    model_params = ModelParams(y, theta)

    e1_max = max(1e1*temp, 1e1*m_N)
    # e1_max = 3 * m_N

    n = 1000
    e1 = logrange(N.m, e1_max, length=n)

    params = Params_3_12{Float64, Float64, Float64, Val{0}}(
        model_params,
        N, nu, A,
        temps,
        xis,
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
    Plt.plot!(e1, res_log, ls=:dash)
    Plt.savefig("figures/test_outer_kernel.pdf")

    return nothing
end

# test_kernel()
# test_e2_int()
test_coll_3_12()
# test_coll_3_12_fixed_temperature()
# test_coll_3_12_integral()
