#=
Lorenz GPU benchmark — DiffEqGPU.jl GPUTsit5 (Level 3).

Solves N Lorenz trajectories with ρ ∈ linspace(0, 21, N), y0 = (1,0,0),
t ∈ [0, 1]. Uses the lower-level batch GPU APIs with make_prob_compatible.

For fixed stepping on large N, explicitly request final-state-only output via
save_everystep=false so the benchmark does not allocate the full time history.

Requires DiffEqGPU ≥ 3.12.0, Julia 1.11.3, JULIA_CUDA_USE_COMPAT=false on V100.

Usage:
    julia --project=ENV lorenz_julia.jl N
=#

using DiffEqGPU, OrdinaryDiffEq, StaticArrays, BenchmarkTools, CUDA

N = parse(Int64, ARGS[1])

function lorenz(u, p, t)
    du1 = 10.0f0 * (u[2] - u[1])
    du2 = p[1] * u[1] - u[2] - u[1] * u[3]
    du3 = u[1] * u[2] - 2.666f0 * u[3]
    return @SVector [du1, du2, du3]
end

# Float64 reference at ρ=21 (SciPy DOP853 at rtol=1e-13, atol=1e-14)
const LORENZ_REF = @SVector [-6.450579147318573, -8.895211478667768, 14.64914586660417]

u0 = @SVector [1.0f0, 0.0f0, 0.0f0]
tspan = (0.0f0, 1.0f0)
p = @SArray [21.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)

println("Trajectories: $N")
println("CUDA device: ", CUDA.name(CUDA.device()))
println("DiffEqGPU:   ", pkgversion(DiffEqGPU))
println("Julia:       ", VERSION)
println()

# ── Build problems on GPU ────────────────────────────────────────────
function build_probs_gpu(n)
    params = range(0.0f0, stop=21.0f0, length=n)
    probs = map(1:n) do i
        DiffEqGPU.make_prob_compatible(
            remake(prob, p = @SArray [params[i]])
        )
    end
    return cu(probs)
end

# ── Verification (last trajectory has ρ=21) ──────────────────────────
println("Warming up + verifying...")
probs_v = build_probs_gpu(1024)
ts, us = CUDA.@sync DiffEqGPU.vectorized_asolve(
    probs_v, prob, GPUTsit5(); dt=0.001f0,
    reltol=1.0f-8, abstol=1.0f-8)
us_cpu = Array(us)
u_last = us_cpu[end, end]
errs = abs.(Float64.([u_last[1], u_last[2], u_last[3]]) .- LORENZ_REF)
max_err = maximum(errs)
status = max_err < 0.01 ? "OK" : "FAIL"
println("Verification (ρ=21, adaptive):")
println("  got=[$(u_last[1]), $(u_last[2]), $(u_last[3])]")
println("  max_err=$(max_err)  [$status]")
println()

# ── Build full problem set ───────────────────────────────────────────
probs_gpu = build_probs_gpu(N)

# Helper to avoid @benchmark macro issues with keyword args
function run_fixed(pg)
    CUDA.@sync DiffEqGPU.vectorized_solve(
        pg, prob, GPUTsit5(); dt=0.001f0, save_everystep=false)
end

function run_adaptive(pg)
    CUDA.@sync DiffEqGPU.vectorized_asolve(pg, prob, GPUTsit5();
        dt=0.001f0, reltol=1.0f-8, abstol=1.0f-8)
end

# ── Fixed stepping benchmark ─────────────────────────────────────────
println("=== Fixed stepping (GPUTsit5, dt=0.001) ===")
try
    run_fixed(probs_gpu)
    b_fixed = @benchmark(run_fixed($probs_gpu), samples=10, evals=1)
    println("minimum = $(minimum(b_fixed).time / 1e6) ms")
    println("median  = $(median(b_fixed).time / 1e6) ms")
catch e
    println("SKIPPED ($(typeof(e).name.name): fixed-step lower API failed at N=$N)")
end

# ── Adaptive stepping benchmark ──────────────────────────────────────
println("\n=== Adaptive stepping (GPUTsit5, atol=rtol=1e-8) ===")

# Warmup
run_adaptive(probs_gpu)

b_adaptive = @benchmark(run_adaptive($probs_gpu), samples=10, evals=1)
println("minimum = $(minimum(b_adaptive).time / 1e6) ms")
println("median  = $(median(b_adaptive).time / 1e6) ms")

println("\nDone.")
