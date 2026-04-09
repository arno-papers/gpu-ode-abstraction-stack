#=
Robertson stiff ODE ensemble — DiffEqGPU.jl GPURosenbrock23 (Level 3).

Solves N Robertson trajectories with k2 ∈ linspace(0, 10^4, N), y0 = (1,0,0),
t ∈ [0, 10^5].  Uses vectorized_asolve batch API with make_prob_compatible.

Requires DiffEqGPU ≥ 3.12.0, Julia 1.11.3, JULIA_CUDA_USE_COMPAT=false on V100.

Usage:
    julia --project=ENV robertson_julia.jl N1[,N2,...] [alg]
=#

using DiffEqGPU, BenchmarkTools, StaticArrays, OrdinaryDiffEq, CUDA, Printf

# ── Settings ─────────────────────────────────────────────────────────
const ALG_NAME = length(ARGS) >= 2 ? ARGS[2] : "GPURosenbrock23"
const ALG = eval(Meta.parse(ALG_NAME * "()"))
const DT0 = 0.001f0
const K2_MAX = 1.0f4

# SciPy Radau f64 reference at k2=1e4 (rtol=1e-13, atol=1e-14)
const SCIPY_REF = [1.7865921142115607e-02, 7.2747514684430595e-08, 9.8213400611036827e-01]

# ── Problem definition (parameterized k2) ────────────────────────────
function rober_f(u, p, t)
    k2 = p[1]
    du1 = -(0.04f0) * u[1] + k2 * u[2] * u[3]
    du2 = (0.04f0 * u[1] - 3.0f7 * u[2]^2) - k2 * u[2] * u[3]
    du3 = 3.0f7 * u[2]^2
    return SVector{3, eltype(u)}(du1, du2, du3)
end

function rober_jac(u, p, t)
    k2 = p[1]
    SMatrix{3, 3, eltype(u)}(
        -(0.04f0),  0.04f0,  0.0f0,
        k2*u[3],  -k2*u[3] - 6.0f7*u[2],  6.0f7*u[2],
        k2*u[2],  -k2*u[2],  0.0f0)
end

function rober_tgrad(u, p, t)
    SVector{3, eltype(u)}(0.0f0, 0.0f0, 0.0f0)
end

const u0 = @SVector Float32[1.0, 0.0, 0.0]
const tspan = (0.0f0, 1.0f5)
const prob = ODEProblem{false}(
    ODEFunction(rober_f, jac=rober_jac, tgrad=rober_tgrad),
    u0, tspan, @SArray [K2_MAX])

# ── Helpers ──────────────────────────────────────────────────────────
function build_probs_gpu(N)
    parameterList = range(0.0f0, stop=K2_MAX, length=N)
    probs = map(1:N) do i
        DiffEqGPU.make_prob_compatible(
            remake(prob, p = @SArray [parameterList[i]])
        )
    end
    return cu(probs)
end

function solve_gpu(probs_gpu)
    CUDA.@sync DiffEqGPU.vectorized_asolve(probs_gpu, prob, ALG; dt=DT0)
end

# ── Main ─────────────────────────────────────────────────────────────
function main()
    Ns = parse.(Int, split(ARGS[1], ","))

    println("=" ^ 70)
    println("DiffEqGPU.jl Robertson Benchmark (Level 3, stiff)")
    println("=" ^ 70)
    println("CUDA device: ", CUDA.name(CUDA.device()))
    println("DiffEqGPU:   ", pkgversion(DiffEqGPU))
    println("Julia:       ", VERSION)
    println("Algorithm:   ", ALG_NAME)
    println("Sweep:       k2 ∈ linspace(0, $(K2_MAX), N)")
    println()

    # ── Verification ─────────────────────────────────────────────────
    println("Verification (last trajectory, k2 ≈ $(K2_MAX)):")
    probs_v = build_probs_gpu(1024)
    _, us_v = solve_gpu(probs_v)
    us_cpu = Array(us_v)
    u_last = us_cpu[end, end]
    got = Float64.([u_last[1], u_last[2], u_last[3]])
    errs = abs.(got .- SCIPY_REF)
    err = maximum(errs)
    status = err < 1e-2 ? "OK" : "FAIL"
    @printf("  y=[%.6e, %.6e, %.6e]  max_err=%.2e  [%s]\n",
            got[1], got[2], got[3], err, status)
    println()

    # ── Scaling sweep ────────────────────────────────────────────────
    csv_path = get(ENV, "RESULTS_CSV", "results/julia_robertson.csv")
    open(csv_path, "w") do io
        for N in Ns
            println("=== N=$N ===")

            probs_gpu = build_probs_gpu(N)

            # Warmup
            print("Warmup... ")
            solve_gpu(probs_gpu)
            println("done")

            # Benchmark
            data = @benchmark(solve_gpu($probs_gpu), samples=5, evals=1)
            t_min = minimum(data.times) / 1e6
            t_med = median(data.times) / 1e6
            println("minimum = $t_min ms")
            println("median  = $t_med ms")
            println(io, "$N $t_min")
            flush(io)
            println()
        end
    end
    println("Wrote $csv_path")
end

main()
