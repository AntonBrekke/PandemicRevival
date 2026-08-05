using LaTeXStrings
ENV["GKSwstype"] = "nul"
import Plots as Plt

function maxwell_boltzmann(x)
    return exp(-x)
end

function fermi_dirac(x)
    return 1. / (exp(x) + 1.)
end

function bose_einstein(x)
    return 1. / (exp(x) - 1.)
end

function test_maxwell()
    x = range(-1e1, 1e1, 1000)
    mb = maxwell_boltzmann.(x)
    fd = fermi_dirac.(x)
    be = bose_einstein.(x)

    p = Plt.plot(
        minorgrid=true,
        # xscale=:log10,
        # yscale=:log10,
        # xlim = (1e-5, 1e5),
        ylim = (-1e1, 2e1),
    )
    Plt.plot!(p, x, mb)
    Plt.plot!(p, x, fd)
    Plt.scatter!(p, x, be)
    Plt.savefig(p, "figures/test_maxwell.pdf")
    return nothing
end

test_maxwell()