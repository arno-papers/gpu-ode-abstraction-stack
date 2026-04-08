# Snakefile — GPU ODE Benchmark Paper
#
# Orchestrates all benchmark runs and paper compilation.
#
# Usage:
#   snakemake paper                    # compile paper locally (just pdflatex)
#   snakemake --profile slurm          # run all benchmarks on VSC genius + compile paper
#   snakemake results/jax_lorenz.out   # run a single benchmark
#   snakemake -n                       # dry-run: show what would execute
#
# Benchmarks run on VSC Tier-2 genius cluster (Tesla V100-SXM2-32GB).
# Use the slurm/ profile for SLURM submission:
#   snakemake --profile slurm --jobs 7
#
# After benchmarks complete, results/ contains raw output logs.
# The paper (gpu_ode_benchmark_paper.tex) has hardcoded numbers from these runs.

import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = 8_388_608
WORKDIR = os.environ.get("VSC_DATA", "/data/leuven/325/vsc32553") + "/gpu_bench"

JULIA_185 = f"{WORKDIR}/julia-1.8.5/bin/julia"
JULIA_1113 = f"{WORKDIR}/julia-1.11.3/bin/julia"
PAPER_REPO = f"{WORKDIR}/GPUODEBenchmarks"
VENV_JAX = f"{WORKDIR}/venv-jax"
VENV_CUDA = f"{WORKDIR}/venv-cuda"

# Shared module preamble for cluster rules (module purge clears inherited env from --export=ALL)
MODULE_PREAMBLE = "module purge 2>/dev/null || true && module load cluster/genius/gpu_v100 && module load CUDA/12.6.0"
PYTHON_MODULES = f"{MODULE_PREAMBLE} && module load Python/3.12.3-GCCcore-13.3.0"

# ---------------------------------------------------------------------------
# Default target: compile the paper
# ---------------------------------------------------------------------------
rule all:
    input:
        "gpu_ode_benchmark_paper.pdf"

# ---------------------------------------------------------------------------
# Paper compilation (runs locally, no GPU needed)
# ---------------------------------------------------------------------------
rule paper:
    input:
        tex="gpu_ode_benchmark_paper.tex",
        data_macros="data_macros.tex",
        validation_table="validation_table.tex",
        fig_bars="fig_bars.pdf",
        fig_scaling="fig_scaling.pdf",
        fig_bars_stiff="fig_bars_stiff.pdf",
        fig_scaling_stiff="fig_scaling_stiff.pdf",
        fig_validation="fig_validation.pdf"
    output:
        "gpu_ode_benchmark_paper.pdf"
    shell:
        """
        pdflatex -interaction=nonstopmode {input.tex} > /dev/null
        pdflatex -interaction=nonstopmode {input.tex} > /dev/null
        """

rule figures:
    input:
        script="generate_figures.py"
    output:
        "fig_bars.pdf",
        "fig_scaling.pdf",
        "fig_bars_stiff.pdf",
        "fig_scaling_stiff.pdf",
        "data_macros.tex",
        "validation_table.tex"
    shell:
        "python3 {input.script}"

rule validation_figure:
    input:
        script="generate_validation_figure.py"
    output:
        "fig_validation.pdf"
    shell:
        "python3 {input.script}"

# ---------------------------------------------------------------------------
# Benchmark: JAX vmap + fori_loop Lorenz (Rung 2) + Diffrax comparison (Rung 1)
# Produces: JAX vmap fixed/adaptive times, Diffrax paper-style times
# ---------------------------------------------------------------------------
rule jax_lorenz:
    input:
        "lorenz_jax.py"
    output:
        "results/jax_lorenz.out"
    resources:
        slurm_partition="gpu_v100",
        runtime=30,
        gpu=1
    shell:
        """
        {PYTHON_MODULES}

        if [ ! -d "{VENV_JAX}" ]; then
            python3 -m venv "{VENV_JAX}"
            source "{VENV_JAX}/bin/activate"
            pip install --upgrade pip
            pip install "jax[cuda12]" diffrax equinox
        else
            source "{VENV_JAX}/bin/activate"
        fi

        NVIDIA_LIB_DIRS=$(find "{VENV_JAX}/lib/python3.12/site-packages/nvidia" \
            -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)
        [ -n "${{NVIDIA_LIB_DIRS:-}}" ] && export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:${{LD_LIBRARY_PATH:-}}"

        PYTHONUNBUFFERED=1 python3 {input} 2>&1 | tee {output}
        """

# ---------------------------------------------------------------------------
# Benchmark: CUDA Lorenz kernels (Rung 4)
# Produces: CUDA fixed/adaptive Lorenz times
# ---------------------------------------------------------------------------
rule cuda_lorenz:
    input:
        "lorenz_cuda.py"
    output:
        "results/cuda_lorenz.out"
    resources:
        slurm_partition="gpu_v100",
        runtime=20,
        gpu=1
    shell:
        """
        {PYTHON_MODULES}

        if [ ! -d "{VENV_CUDA}" ]; then
            python3 -m venv "{VENV_CUDA}"
            source "{VENV_CUDA}/bin/activate"
            pip install --upgrade pip
            pip install cupy-cuda12x scipy
        else
            source "{VENV_CUDA}/bin/activate"
        fi

        PYTHONUNBUFFERED=1 python3 {input} 2>&1 | tee {output}
        """

# ---------------------------------------------------------------------------
# Benchmark: CUDA Robertson kernels (Rung 4, stiff)
# Produces: CUDA Rosenbrock23 times at default and tight tolerances
# ---------------------------------------------------------------------------
rule cuda_robertson:
    input:
        "robertson_cuda.py"
    output:
        "results/cuda_robertson.out",
        "results/cuda_robertson.csv"
    resources:
        slurm_partition="gpu_v100",
        runtime=20,
        gpu=1
    shell:
        """
        {PYTHON_MODULES}

        if [ ! -d "{VENV_CUDA}" ]; then
            python3 -m venv "{VENV_CUDA}"
            source "{VENV_CUDA}/bin/activate"
            pip install --upgrade pip
            pip install cupy-cuda12x scipy
        else
            source "{VENV_CUDA}/bin/activate"
        fi

        PYTHONUNBUFFERED=1 python3 {input} 2>&1 | tee {output}
        """

# ---------------------------------------------------------------------------
# Benchmark: JAX vmap Rosenbrock23 Robertson (Rung 2, stiff)
# Produces: JAX vmap Robertson times
# ---------------------------------------------------------------------------
rule jax_robertson:
    input:
        "robertson_jax.py"
    output:
        "results/jax_robertson.out",
        "results/jax_robertson.csv"
    resources:
        slurm_partition="gpu_v100",
        runtime=30,
        gpu=1
    shell:
        """
        {PYTHON_MODULES}

        if [ ! -d "{VENV_JAX}" ]; then
            python3 -m venv "{VENV_JAX}"
            source "{VENV_JAX}/bin/activate"
            pip install --upgrade pip
            pip install "jax[cuda12]" diffrax equinox
        else
            source "{VENV_JAX}/bin/activate"
        fi

        NVIDIA_LIB_DIRS=$(find "{VENV_JAX}/lib/python3.12/site-packages/nvidia" \
            -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)
        [ -n "${{NVIDIA_LIB_DIRS:-}}" ] && export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:${{LD_LIBRARY_PATH:-}}"

        PYTHONUNBUFFERED=1 python3 {input} 2>&1 | tee {output}
        """

# ---------------------------------------------------------------------------
# Benchmark: JAX vmap Rosenbrock23 Robertson — generic (autodiff Jacobian)
# Same solver as jax_robertson but with zero problem-specific knowledge
# ---------------------------------------------------------------------------
rule jax_robertson_generic:
    input:
        "robertson_jax_generic.py"
    output:
        "results/jax_robertson_generic.out",
        "results/jax_robertson_generic.csv",
        "results/jax_robertson_autodiff_direct.csv"
    resources:
        slurm_partition="gpu_v100",
        runtime=30,
        gpu=1
    shell:
        """
        {PYTHON_MODULES}

        if [ ! -d "{VENV_JAX}" ]; then
            python3 -m venv "{VENV_JAX}"
            source "{VENV_JAX}/bin/activate"
            pip install --upgrade pip
            pip install "jax[cuda12]" diffrax equinox
        else
            source "{VENV_JAX}/bin/activate"
        fi

        NVIDIA_LIB_DIRS=$(find "{VENV_JAX}/lib/python3.12/site-packages/nvidia" \
            -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)
        [ -n "${{NVIDIA_LIB_DIRS:-}}" ] && export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:${{LD_LIBRARY_PATH:-}}"

        PYTHONUNBUFFERED=1 python3 {input} 2>&1 | tee {output}
        """

# ---------------------------------------------------------------------------
# Benchmark: Julia paper-era Lorenz (Rung 3, DiffEqGPU 2.1.0)
# Re-runs the paper's exact bench_lorenz_gpu.jl on our V100
# ---------------------------------------------------------------------------
rule julia_paper_lorenz:
    output:
        "results/julia_paper_lorenz.out"
    resources:
        slurm_partition="gpu_v100",
        runtime=120,
        gpu=1
    params:
        n=N
    shell:
        """
        {MODULE_PREAMBLE}

        JULIA_BIN="{JULIA_185}"
        DEPOT="{WORKDIR}/julia-depot-paper-1.8.5"
        CUDA_ENV="{WORKDIR}/julia-cuda-env-1.8.5"

        # Download Julia 1.8.5 if needed
        if [ ! -x "$JULIA_BIN" ]; then
            curl -L --fail --retry 3 \
                "https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz" \
                -o "{WORKDIR}/julia-1.8.5-linux-x86_64.tar.gz"
            tar -xzf "{WORKDIR}/julia-1.8.5-linux-x86_64.tar.gz" -C "{WORKDIR}"
        fi

        # Clone paper repo if needed
        if [ ! -d "{PAPER_REPO}" ]; then
            git clone --depth 1 https://github.com/utkarsh530/GPUODEBenchmarks.git "{PAPER_REPO}"
        fi

        export JULIA_DEPOT_PATH="$DEPOT"
        export JULIA_LOAD_PATH="{PAPER_REPO}/GPU_ODE_Julia:$CUDA_ENV:@stdlib"
        export JULIA_NUM_THREADS=4
        export JULIA_CUDA_MEMORY_POOL=none

        # Install CUDA.jl 4.1 sidecar env (paper-era driver)
        "$JULIA_BIN" --project="$CUDA_ENV" -e '
            using Pkg
            Pkg.add(PackageSpec(name = "CUDA", version = "4.1"))
        '

        # Instantiate paper's pinned environment
        cd "{PAPER_REPO}"
        "$JULIA_BIN" --project=./GPU_ODE_Julia -e '
            using Pkg; Pkg.instantiate(); Pkg.precompile()
        '

        # Run the paper's exact Lorenz script
        "$JULIA_BIN" --project=./GPU_ODE_Julia \
            ./GPU_ODE_Julia/bench_lorenz_gpu.jl {params.n} 2>&1 | tee {output}
        """

# ---------------------------------------------------------------------------
# Benchmark: Julia paper-era Robertson (Rung 3, DiffEqGPU 2.5.1)
# DiffEqGPU 2.1.0 doesn't export stiff solvers; 2.5.1 is the earliest that does
# ---------------------------------------------------------------------------
rule julia_robertson_25:
    input:
        "robertson_julia.jl"
    output:
        "results/julia_robertson_25.out",
        "results/julia_robertson.csv"
    resources:
        slurm_partition="gpu_v100",
        runtime=120,
        gpu=1
    params:
        sweep="1024,10240,102400,1024000,8388608",
        n=N
    shell:
        """
        {MODULE_PREAMBLE}

        JULIA_BIN="{JULIA_185}"
        ENV_DIR="{WORKDIR}/julia-robertson-diffeqgpu25-env"
        DEPOT="{WORKDIR}/julia-depot-diffeqgpu25-1.8.5"

        if [ ! -x "$JULIA_BIN" ]; then
            curl -L --fail --retry 3 \
                "https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz" \
                -o "{WORKDIR}/julia-1.8.5-linux-x86_64.tar.gz"
            tar -xzf "{WORKDIR}/julia-1.8.5-linux-x86_64.tar.gz" -C "{WORKDIR}"
        fi

        export JULIA_DEPOT_PATH="$DEPOT"
        export JULIA_NUM_THREADS=4
        export JULIA_CUDA_MEMORY_POOL=none
        export JULIA_PKG_PRECOMPILE_AUTO=0

        mkdir -p "$ENV_DIR"

        # Install DiffEqGPU 2.5.1 (earliest with stiff GPU-kernel solvers)
        "$JULIA_BIN" --project="$ENV_DIR" -e '
            using Pkg
            Pkg.add([
                PackageSpec(name="BenchmarkTools"),
                PackageSpec(name="CUDA", version="4.1"),
                PackageSpec(name="DiffEqGPU", version="2.5.1"),
                PackageSpec(name="OrdinaryDiffEq"),
                PackageSpec(name="StaticArrays"),
            ])
        '

        # Clear stale CSV (script appends)
        rm -f results/julia_robertson.csv

        # Scaling sweep (default tolerances)
        echo "=== Robertson scaling sweep ===" | tee {output[0]}
        "$JULIA_BIN" --project="$ENV_DIR" {input} \
            {params.sweep} GPURosenbrock23 default default 2>&1 | tee -a {output[0]}

        # Tight tolerances (N=8M only, for version-sensitivity table)
        echo "" >> {output[0]}
        echo "=== Robertson tight tolerances ===" | tee -a {output[0]}
        "$JULIA_BIN" --project="$ENV_DIR" {input} \
            {params.n} GPURosenbrock23 1e-6 1e-9 2>&1 | tee -a {output[0]}
        """

# ---------------------------------------------------------------------------
# Benchmark: Latest Julia stack (DiffEqGPU 3.12.0, for version-sensitivity table)
# Runs both Lorenz and Robertson with Julia 1.11.3 + CUDA.jl 5.11
# ---------------------------------------------------------------------------
rule julia_latest:
    input:
        lorenz="lorenz_julia.jl",
        robertson="robertson_julia.jl"
    output:
        "results/julia_latest.out"
    resources:
        slurm_partition="gpu_v100",
        runtime=120,
        gpu=1
    params:
        n=N
    shell:
        """
        {MODULE_PREAMBLE}

        JULIA_BIN="{JULIA_1113}"
        DEPOT="{WORKDIR}/julia-depot-cuda12fix"
        ENV_DIR="{WORKDIR}/julia-env-cuda12fix"

        export JULIA_DEPOT_PATH="$DEPOT"
        export JULIA_NUM_THREADS=4
        export JULIA_CUDA_MEMORY_POOL=none
        export JULIA_PKG_PRECOMPILE_AUTO=0

        # CRITICAL: prevent CUDA_Driver_jll from loading compat driver v13.2
        # which drops V100 (sm_70). System driver 580.95.05 still supports V100.
        export JULIA_CUDA_USE_COMPAT=false

        echo "=== Latest Julia Lorenz ===" | tee {output}
        "$JULIA_BIN" --project="$ENV_DIR" {input.lorenz} {params.n} 2>&1 | tee -a {output}

        export RESULTS_CSV=results/julia_latest_robertson.csv
        rm -f "$RESULTS_CSV"

        echo "" >> {output}
        echo "=== Latest Julia Robertson (default tol) ===" | tee -a {output}
        "$JULIA_BIN" --project="$ENV_DIR" {input.robertson} \
            {params.n} GPURosenbrock23 default default 2>&1 | tee -a {output}

        echo "" >> {output}
        echo "=== Latest Julia Robertson (tight tol) ===" | tee -a {output}
        "$JULIA_BIN" --project="$ENV_DIR" {input.robertson} \
            {params.n} GPURosenbrock23 1e-6 1e-9 2>&1 | tee -a {output}
        """

# ---------------------------------------------------------------------------
# Benchmark: Latest Diffrax with block_until_ready() (Rung 1, proper timing)
# ---------------------------------------------------------------------------
rule diffrax_latest:
    output:
        "results/diffrax_latest.out"
    resources:
        slurm_partition="gpu_v100",
        runtime=60,
        gpu=1
    params:
        n=N
    shell:
        """
        {PYTHON_MODULES}

        if [ ! -d "{VENV_JAX}" ]; then
            python3 -m venv "{VENV_JAX}"
            source "{VENV_JAX}/bin/activate"
            pip install --upgrade pip
            pip install "jax[cuda12]" diffrax equinox
        else
            source "{VENV_JAX}/bin/activate"
        fi

        NVIDIA_LIB_DIRS=$(find "{VENV_JAX}/lib/python3.12/site-packages/nvidia" \
            -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)
        [ -n "${{NVIDIA_LIB_DIRS:-}}" ] && export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:${{LD_LIBRARY_PATH:-}}"

        # Clone paper repo for reference (paper artifact numbers come from its data/)
        if [ ! -d "{PAPER_REPO}" ]; then
            git clone --depth 1 https://github.com/utkarsh530/GPUODEBenchmarks.git "{PAPER_REPO}"
        fi

        # Inline the Diffrax benchmark with proper GPU synchronization
        python3 -u - {params.n} <<'PYEOF' 2>&1 | tee {output}
import time, sys
import diffrax, equinox as eqx, jax, jax.numpy as jnp

N = int(sys.argv[1])
print(f"JAX {{jax.__version__}}, Diffrax {{diffrax.__version__}}, device: {{jax.devices()[0]}}")
print(f"N = {{N}}")

class Lorenz(eqx.Module):
    k1: float
    def __call__(self, t, y, args):
        f0 = 10.0*(y[1] - y[0])
        f1 = self.k1 * y[0] - y[1] - y[0] * y[2]
        f2 = y[0] * y[1] - (8/3)*y[2]
        return jnp.stack([f0, f1, f2])

params = jnp.linspace(0.0, 21.0, N)

@jax.jit
@jax.vmap
def solve_fixed(k1):
    return diffrax.diffeqsolve(
        diffrax.ODETerm(Lorenz(k1)), diffrax.Tsit5(),
        0.0, 1.0, 0.001, jnp.array([1.0, 0.0, 0.0]))

@jax.jit
@jax.vmap
def solve_adaptive(k1):
    return diffrax.diffeqsolve(
        diffrax.ODETerm(Lorenz(k1)), diffrax.Tsit5(),
        0.0, 1.0, 0.001, jnp.array([1.0, 0.0, 0.0]),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8))

for fn, label in [(solve_fixed, "fixed"), (solve_adaptive, "adaptive")]:
    for _ in range(3):
        jax.block_until_ready(fn(params))
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(params))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    print(f"Diffrax {{label}} (proper timing): {{min(times):.1f}} ms (min of 20)")

# --- Robertson stiff benchmark (Kvaerno3) ---
print()
print("=== Diffrax Robertson (Kvaerno3, stiff) ===")

class Robertson(eqx.Module):
    def __call__(self, t, y, args):
        y1, y2, y3 = y[0], y[1], y[2]
        return jnp.stack([
            -0.04*y1 + 1e4*y2*y3,
             0.04*y1 - 1e4*y2*y3 - 3e7*y2*y2,
             3e7*y2*y2])

dummies = jnp.zeros(N)

@jax.jit
@jax.vmap
def solve_robertson(dummy):
    return diffrax.diffeqsolve(
        diffrax.ODETerm(Robertson()),
        diffrax.Kvaerno3(),
        0.0, 1e5, 1e-4, jnp.array([1.0, 0.0, 0.0]),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        max_steps=16384)

for _ in range(3):
    jax.block_until_ready(solve_robertson(dummies))
times = []
for _ in range(10):
    t0 = time.perf_counter()
    jax.block_until_ready(solve_robertson(dummies))
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)
print(f"Diffrax Robertson Kvaerno3 (proper timing): {{min(times):.1f}} ms (min of 10)")
PYEOF
        """

# ---------------------------------------------------------------------------
# Convenience targets
# ---------------------------------------------------------------------------
rule benchmarks:
    """Run all benchmarks (use with --profile slurm)."""
    input:
        "results/jax_lorenz.out",
        "results/jax_robertson.out",
        "results/jax_robertson_generic.out",
        "results/cuda_lorenz.out",
        "results/cuda_robertson.out",
        "results/julia_paper_lorenz.out",
        "results/julia_robertson_25.out",
        "results/julia_latest.out",
        "results/diffrax_latest.out"
