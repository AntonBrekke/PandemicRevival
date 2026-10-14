"""
Runs the test suite, each file in its own Julia process (the files define overlapping
constants). From julia/pandemic:

    julia --project=. test/runtests.jl [--quick]

--quick skips the slow files (the A'A' <-> NN integral and the rate-table interpolation,
~8 min together; the regression re-solves relic roots, ~3 min). ~15 min in full.
"""

const TESTS = ["test_tree_amplitudes.jl", "test_thermal_rates.jl", "test_rate_tables.jl", "test_network.jl",
               "test_fluid.jl", "test_kinetic_decoupling.jl", "test_ckd_table.jl", "test_lyman_alpha.jl",
               "test_coll_12_34_AA_NN.jl", "test_rate_tables_eval.jl", "test_regression.jl"]
const SLOW = ["test_coll_12_34_AA_NN.jl", "test_rate_tables_eval.jl", "test_regression.jl"]

tests = "--quick" in ARGS ? setdiff(TESTS, SLOW) : TESTS
failed = String[]
for t in tests
    t0 = time()
    ok = success(pipeline(`$(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(@__DIR__, t))`;
                          stdout=devnull, stderr=devnull))
    println(rpad(t, 30), ok ? "pass" : "FAIL", "  ", round(time() - t0; digits=0), " s")
    flush(stdout)
    ok || push!(failed, t)
end
isempty(failed) || error("failed: ", join(failed, ", "), " (run them directly to see the output)")
println("all passed")
