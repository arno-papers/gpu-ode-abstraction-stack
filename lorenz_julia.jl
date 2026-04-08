# Lorenz GPU benchmark — DiffEqGPU GPUTsit5 (Level 3)
#
# Usage: julia --project=ENV lorenz_julia.jl N

using DiffEqGPU, OrdinaryDiffEq, StaticArrays, BenchmarkTools
using CUDA

N = parse(Int64, ARGS[1])

function lorenz(u, p, t)
    du1 = 10.0f0 * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - 2.666f0 * u[3]
    return SA[du1, du2, du3]
end

# Float64 reference at ρ=21 (SciPy DOP853 at rtol=1e-13, atol=1e-14)
const LORENZ_REF = SA[-6.450579147318573, -8.895211478667768, 14.64914586660417]

u0 = SA[1.0f0, 0.0f0, 0.0f0]
tspan = (0.0f0, 1.0f0)
p = SA[21.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)

parameterList = range(0.0f0, stop=21.0f0, length=N)
prob_func(prob, i, repeat) = remake(prob, p=SA[parameterList[i]])
ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)

println("Trajectories: $N")
println("CUDA device: ", CUDA.name(CUDA.device()))
println()

# Warmup + verify (last trajectory has ρ=21, matching SciPy reference)
println("Warming up...")
sol_warmup = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()),
                   trajectories=min(N, 1024), adaptive=false, dt=0.001f0,
                   save_everystep=false, dense=false)
CUDA.synchronize()

u_last = sol_warmup[end].u[end]
errs = abs.(Float64.(u_last) .- LORENZ_REF)
max_err = maximum(errs)
status = max_err < 0.01 ? "OK" : "FAIL"
println("Verification (ρ=21, fixed dt=0.001):")
println("  got=[$(u_last[1]), $(u_last[2]), $(u_last[3])]")
println("  max_err=$(max_err)  [$status]")
println()

# Fixed stepping benchmark
println("\n=== Fixed stepping (GPUTsit5, dt=0.001) ===")
b_fixed = @benchmark begin
    solve($ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()),
          trajectories=$N, adaptive=false, dt=0.001f0,
          save_everystep=false, dense=false)
    CUDA.synchronize()
end samples=10 evals=1 seconds=60
println("minimum = $(minimum(b_fixed).time / 1e6) ms")
println("median  = $(median(b_fixed).time / 1e6) ms")

# Adaptive stepping benchmark
println("\n=== Adaptive stepping (GPUTsit5, atol=rtol=1e-8) ===")
b_adaptive = @benchmark begin
    solve($ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend()),
          trajectories=$N, adaptive=true, abstol=1.0f-8, reltol=1.0f-8,
          save_everystep=false, dense=false)
    CUDA.synchronize()
end samples=10 evals=1 seconds=60
println("minimum = $(minimum(b_adaptive).time / 1e6) ms")
println("median  = $(median(b_adaptive).time / 1e6) ms")

println("\nDone.")
