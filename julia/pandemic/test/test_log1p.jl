import Plots as Plt

function test(x)
    return sign(x) * log1p(abs(x))
end

function test2(x)
    return sign(x) * log1p(x)
end

function main()
    x = range(-1., 1., length=100)
    y = test.(x)
    y2 = test2.(x)

    Plt.plot(
        x,
        y,
        label="test",
    )
    Plt.plot!(
        x,
        y2,
        label="test2",
    )
    Plt.savefig("figures/test_log1p.pdf")
end

main()