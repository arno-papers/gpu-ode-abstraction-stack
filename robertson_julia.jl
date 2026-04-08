# Robertson stiff benchmark — DiffEqGPU GPURosenbrock23 (Level 3)
#
# Usage: julia --project=ENV robertson_julia.jl N[,N2,...] GPURosenbrock23 [rtol] [atol]
#   N: single value or comma-separated list for scaling sweep
#   rtol/atol: float values or "default" to use solver defaults

using BenchmarkTools, CUDA, DiffEqGPU, OrdinaryDiffEq, StaticArrays

Ns = parse.(Int, split(ARGS[1], ","))
RTOL = length(ARGS) >= 3 && lowercase(ARGS[3]) != "default" ? parse(Float32, ARGS[3]) : nothing
ATOL = length(ARGS) >= 4 && lowercase(ARGS[4]) != "default" ? parse(Float32, ARGS[4]) : nothing

function robertson(u, p, t)
    y1 = u[1]; y2 = u[2]; y3 = u[3]
    return @SVector [
        -0.04f0 * y1 + 1.0f4 * y2 * y3,
        0.04f0 * y1 - 1.0f4 * y2 * y3 - 3.0f7 * y2 * y2,
        3.0f7 * y2 * y2,
    ]
end

# Float64 reference (SciPy Radau at rtol=1e-13, atol=1e-14)
const ROBERTSON_REF = @SVector [1.7865921142252432e-02, 7.2747514684997379e-08, 9.8213400611023105e-01]

println("CUDA device: ", CUDA.name(CUDA.device()))
println()

u0 = @SVector [1.0f0, 0.0f0, 0.0f0]
tspan = (0.0f0, 1.0f5)
prob = ODEProblem{false}(robertson, u0, tspan)

alg = GPURosenbrock23()

# API varies across DiffEqGPU versions (2.5.1 vs 3.x)
function make_ensemble_alg()
    if isdefined(CUDA, :CUDABackend)
        be = CUDA.CUDABackend()
        applicable(EnsembleGPUKernel, be, 0.0) && return EnsembleGPUKernel(be, 0.0)
        applicable(EnsembleGPUKernel, be) && return EnsembleGPUKernel(be)
    end
    applicable(EnsembleGPUKernel, 0.0) && return EnsembleGPUKernel(0.0)
    return EnsembleGPUKernel()
end
ensemble_alg = make_ensemble_alg()

tol_kwargs = RTOL === nothing || ATOL === nothing ? NamedTuple() : (reltol = RTOL, abstol = ATOL)

mkpath("results")

for N in Ns
    println("N = $N, rtol = $(something(RTOL, "default")), atol = $(something(ATOL, "default"))")

    eprob = EnsembleProblem(prob)
    kwargs = merge((trajectories = N, dt = 1.0f-4, save_everystep = false, dense = false), tol_kwargs)

    # Warmup + verify
    println("Warming up...")
    sol = CUDA.@sync solve(eprob, alg, ensemble_alg; kwargs...)

    # Verify final state against SciPy reference
    u_final = sol[1].u[end]
    errs = abs.(Float64.(u_final) .- ROBERTSON_REF)
    max_err = maximum(errs)
    status = max_err < 1e-3 ? "OK" : "FAIL"
    println("Verification: y=[$(u_final[1]), $(u_final[2]), $(u_final[3])]")
    println("  max_err=$(max_err)  [$status]")

    println("\n=== Benchmarking GPURosenbrock23 ===")
    data = @benchmark(CUDA.@sync(solve($eprob, $alg, $ensemble_alg; $kwargs...)), samples=3, evals=1)
    t_min = minimum(data.times) / 1e6
    println("minimum = $t_min ms")
    println("median  = $(median(data.times) / 1e6) ms")

    # Append to CSV (path configurable via RESULTS_CSV env var)
    csv_path = get(ENV, "RESULTS_CSV", "results/julia_robertson.csv")
    mkpath(dirname(csv_path))
    open(csv_path, "a") do f
        println(f, "$N $t_min")
    end
    println("Appended to $csv_path")
    println("\nDone with N=$N.\n")
end
