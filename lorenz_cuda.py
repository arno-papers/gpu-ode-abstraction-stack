"""
Lorenz ODE ensemble — direct CUDA Tsit5 kernels via CuPy RawKernel (Level 4).

One thread per trajectory, scalar state in registers, flat control flow.
Produces CSV sweep data for the scaling figure.
"""

import time
import shutil

import numpy as np
from scipy.integrate import solve_ivp

import cupy as cp

SIGMA = 10.0
BETA = 8.0 / 3.0


CUDA_BACKEND = "nvcc" if shutil.which("nvcc") else "nvrtc"
CUDA_OPTIONS = ("--std=c++14", "--use_fast_math", "-O3")


CUDA_SRC = r"""
#define SIGMA 10.0f
#define BETA (8.0f / 3.0f)

#define A21 0.161f
#define A31 -0.008480655492356989f
#define A32 0.335480655492357f
#define A41 2.8971530571054935f
#define A42 -6.359448489975075f
#define A43 4.3622954328695815f
#define A51 5.325864828439257f
#define A52 -11.748883564062828f
#define A53 7.4955393428898365f
#define A54 -0.09249506636175525f
#define A61 5.86145544294642f
#define A62 -12.92096931784711f
#define A63 8.159367898576159f
#define A64 -0.071584973281401f
#define A65 -0.028269050394068383f

#define B1 0.09646076681806523f
#define B2 0.01f
#define B3 0.4798896504144996f
#define B4 1.379008574103742f
#define B5 -3.290069515436081f
#define B6 2.324710524099774f

#define E1 -0.00178001105222577714f
#define E2 -0.0008164344596567469f
#define E3 0.007880878010261995f
#define E4 -0.1447110071732629f
#define E5 0.5823571654525552f
#define E6 -0.45808210592918697f
#define E7 0.015151515151515152f

__device__ __forceinline__ void lorenz_rhs(
    float x, float y, float z, float rho,
    float &dx, float &dy, float &dz
) {
    dx = SIGMA * (y - x);
    dy = x * (rho - z) - y;
    dz = x * y - BETA * z;
}

__device__ __forceinline__ void tsit5_step(
    float &x, float &y, float &z, float rho, float dt
) {
    float dx1, dy1, dz1;
    float dx2, dy2, dz2;
    float dx3, dy3, dz3;
    float dx4, dy4, dz4;
    float dx5, dy5, dz5;
    float dx6, dy6, dz6;
    float xa, ya, za;

    lorenz_rhs(x, y, z, rho, dx1, dy1, dz1);

    xa = x + dt * A21 * dx1;
    ya = y + dt * A21 * dy1;
    za = z + dt * A21 * dz1;
    lorenz_rhs(xa, ya, za, rho, dx2, dy2, dz2);

    xa = x + dt * (A31 * dx1 + A32 * dx2);
    ya = y + dt * (A31 * dy1 + A32 * dy2);
    za = z + dt * (A31 * dz1 + A32 * dz2);
    lorenz_rhs(xa, ya, za, rho, dx3, dy3, dz3);

    xa = x + dt * (A41 * dx1 + A42 * dx2 + A43 * dx3);
    ya = y + dt * (A41 * dy1 + A42 * dy2 + A43 * dy3);
    za = z + dt * (A41 * dz1 + A42 * dz2 + A43 * dz3);
    lorenz_rhs(xa, ya, za, rho, dx4, dy4, dz4);

    xa = x + dt * (A51 * dx1 + A52 * dx2 + A53 * dx3 + A54 * dx4);
    ya = y + dt * (A51 * dy1 + A52 * dy2 + A53 * dy3 + A54 * dy4);
    za = z + dt * (A51 * dz1 + A52 * dz2 + A53 * dz3 + A54 * dz4);
    lorenz_rhs(xa, ya, za, rho, dx5, dy5, dz5);

    xa = x + dt * (A61 * dx1 + A62 * dx2 + A63 * dx3 + A64 * dx4 + A65 * dx5);
    ya = y + dt * (A61 * dy1 + A62 * dy2 + A63 * dy3 + A64 * dy4 + A65 * dy5);
    za = z + dt * (A61 * dz1 + A62 * dz2 + A63 * dz3 + A64 * dz4 + A65 * dz5);
    lorenz_rhs(xa, ya, za, rho, dx6, dy6, dz6);

    x = x + dt * (B1 * dx1 + B2 * dx2 + B3 * dx3 + B4 * dx4 + B5 * dx5 + B6 * dx6);
    y = y + dt * (B1 * dy1 + B2 * dy2 + B3 * dy3 + B4 * dy4 + B5 * dy5 + B6 * dy6);
    z = z + dt * (B1 * dz1 + B2 * dz2 + B3 * dz3 + B4 * dz4 + B5 * dz5 + B6 * dz6);
}

extern "C" __global__ void tsit5_fixed_kernel(
    const float* x_ptr, const float* y_ptr, const float* z_ptr, const float* rho_ptr,
    float* xo_ptr, float* yo_ptr, float* zo_ptr,
    int N, int n_steps, float dt
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float x = x_ptr[tid];
    float y = y_ptr[tid];
    float z = z_ptr[tid];
    float rho = rho_ptr[tid];

    for (int i = 0; i < n_steps; ++i) {
        tsit5_step(x, y, z, rho, dt);
    }

    xo_ptr[tid] = x;
    yo_ptr[tid] = y;
    zo_ptr[tid] = z;
}

extern "C" __global__ void tsit5_adaptive_kernel(
    const float* x_ptr, const float* y_ptr, const float* z_ptr, const float* rho_ptr,
    float* xo_ptr, float* yo_ptr, float* zo_ptr,
    int* accepted_out, int* rejected_out,
    int N, int max_steps, float atol, float rtol, float safety
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float x = x_ptr[tid];
    float y = y_ptr[tid];
    float z = z_ptr[tid];
    float rho = rho_ptr[tid];
    float t = 0.0f;
    float dt = 0.001f;
    int n_accepted = 0;
    int n_rejected = 0;

    float dx1, dy1, dz1;
    lorenz_rhs(x, y, z, rho, dx1, dy1, dz1);

    for (int step = 0; step < max_steps && t < 1.0f; ++step) {
        float dt_eff = fminf(dt, 1.0f - t);

        float dx2, dy2, dz2;
        float dx3, dy3, dz3;
        float dx4, dy4, dz4;
        float dx5, dy5, dz5;
        float dx6, dy6, dz6;
        float dx7, dy7, dz7;
        float xa, ya, za;

        xa = x + dt_eff * A21 * dx1;
        ya = y + dt_eff * A21 * dy1;
        za = z + dt_eff * A21 * dz1;
        lorenz_rhs(xa, ya, za, rho, dx2, dy2, dz2);

        xa = x + dt_eff * (A31 * dx1 + A32 * dx2);
        ya = y + dt_eff * (A31 * dy1 + A32 * dy2);
        za = z + dt_eff * (A31 * dz1 + A32 * dz2);
        lorenz_rhs(xa, ya, za, rho, dx3, dy3, dz3);

        xa = x + dt_eff * (A41 * dx1 + A42 * dx2 + A43 * dx3);
        ya = y + dt_eff * (A41 * dy1 + A42 * dy2 + A43 * dy3);
        za = z + dt_eff * (A41 * dz1 + A42 * dz2 + A43 * dz3);
        lorenz_rhs(xa, ya, za, rho, dx4, dy4, dz4);

        xa = x + dt_eff * (A51 * dx1 + A52 * dx2 + A53 * dx3 + A54 * dx4);
        ya = y + dt_eff * (A51 * dy1 + A52 * dy2 + A53 * dy3 + A54 * dy4);
        za = z + dt_eff * (A51 * dz1 + A52 * dz2 + A53 * dz3 + A54 * dz4);
        lorenz_rhs(xa, ya, za, rho, dx5, dy5, dz5);

        xa = x + dt_eff * (A61 * dx1 + A62 * dx2 + A63 * dx3 + A64 * dx4 + A65 * dx5);
        ya = y + dt_eff * (A61 * dy1 + A62 * dy2 + A63 * dy3 + A64 * dy4 + A65 * dy5);
        za = z + dt_eff * (A61 * dz1 + A62 * dz2 + A63 * dz3 + A64 * dz4 + A65 * dz5);
        lorenz_rhs(xa, ya, za, rho, dx6, dy6, dz6);

        float xn = x + dt_eff * (B1 * dx1 + B2 * dx2 + B3 * dx3 + B4 * dx4 + B5 * dx5 + B6 * dx6);
        float yn = y + dt_eff * (B1 * dy1 + B2 * dy2 + B3 * dy3 + B4 * dy4 + B5 * dy5 + B6 * dy6);
        float zn = z + dt_eff * (B1 * dz1 + B2 * dz2 + B3 * dz3 + B4 * dz4 + B5 * dz5 + B6 * dz6);

        lorenz_rhs(xn, yn, zn, rho, dx7, dy7, dz7);

        float ex = dt_eff * (E1 * dx1 + E2 * dx2 + E3 * dx3 + E4 * dx4 + E5 * dx5 + E6 * dx6 + E7 * dx7);
        float ey = dt_eff * (E1 * dy1 + E2 * dy2 + E3 * dy3 + E4 * dy4 + E5 * dy5 + E6 * dy6 + E7 * dy7);
        float ez = dt_eff * (E1 * dz1 + E2 * dz2 + E3 * dz3 + E4 * dz4 + E5 * dz5 + E6 * dz6 + E7 * dz7);

        float sx = atol + rtol * fmaxf(fabsf(x), fabsf(xn));
        float sy = atol + rtol * fmaxf(fabsf(y), fabsf(yn));
        float sz = atol + rtol * fmaxf(fabsf(z), fabsf(zn));
        float err = sqrtf(((ex / sx) * (ex / sx) +
                           (ey / sy) * (ey / sy) +
                           (ez / sz) * (ez / sz)) / 3.0f);

        if (err <= 1.0f) {
            t += dt_eff;
            x = xn;
            y = yn;
            z = zn;
            dx1 = dx7;
            dy1 = dy7;
            dz1 = dz7;
            n_accepted++;
        } else {
            n_rejected++;
        }

        float factor = safety * powf(fmaxf(err, 1.0e-10f), -0.2f);
        factor = fminf(fmaxf(factor, 0.2f), 5.0f);
        dt *= factor;
    }

    xo_ptr[tid] = x;
    yo_ptr[tid] = y;
    zo_ptr[tid] = z;
    if (accepted_out) accepted_out[tid] = n_accepted;
    if (rejected_out) rejected_out[tid] = n_rejected;
}
"""


_RAW_MODULE_KWARGS = {
    "code": CUDA_SRC,
    "options": CUDA_OPTIONS,
    "backend": CUDA_BACKEND,
}
if CUDA_BACKEND == "nvrtc":
    _RAW_MODULE_KWARGS["name_expressions"] = ("tsit5_fixed_kernel", "tsit5_adaptive_kernel")

CUDA_MODULE = cp.RawModule(**_RAW_MODULE_KWARGS)
TSIT5_FIXED_KERNEL = CUDA_MODULE.get_function("tsit5_fixed_kernel")
TSIT5_ADAPTIVE_KERNEL = CUDA_MODULE.get_function("tsit5_adaptive_kernel")


def solve_tsit5_fixed_cuda(x0, y0, z0, rho, n_steps=1000, dt=0.001, block_size=256):
    N = int(x0.shape[0])
    xo = cp.empty_like(x0)
    yo = cp.empty_like(y0)
    zo = cp.empty_like(z0)
    grid = ((N + block_size - 1) // block_size,)
    TSIT5_FIXED_KERNEL(
        grid,
        (block_size,),
        (x0, y0, z0, rho, xo, yo, zo, np.int32(N), np.int32(n_steps), np.float32(dt)),
    )
    return xo, yo, zo


def solve_tsit5_adaptive_cuda(x0, y0, z0, rho, max_steps=500, block_size=128,
                               return_steps=False):
    N = int(x0.shape[0])
    xo = cp.empty_like(x0)
    yo = cp.empty_like(y0)
    zo = cp.empty_like(z0)
    acc_out = cp.empty(N, dtype=cp.int32)
    rej_out = cp.empty(N, dtype=cp.int32)
    grid = ((N + block_size - 1) // block_size,)
    TSIT5_ADAPTIVE_KERNEL(
        grid,
        (block_size,),
        (
            x0, y0, z0, rho, xo, yo, zo,
            acc_out, rej_out,
            np.int32(N), np.int32(max_steps),
            np.float32(1e-8), np.float32(1e-8), np.float32(0.9),
        ),
    )
    if return_steps:
        return xo, yo, zo, acc_out, rej_out
    return xo, yo, zo


def bench_cuda(fn, x0, y0, z0, rho, n_warmup=3, n_runs=20):
    for _ in range(n_warmup):
        fn(x0, y0, z0, rho)
        cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(n_runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn(x0, y0, z0, rho)
        cp.cuda.Stream.null.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times), np.median(times)


def _scalar(x):
    return float(cp.asnumpy(x)[()])


def _scipy_lorenz_reference(rho=21.0):
    """Float64 reference via SciPy DOP853 at near-machine-epsilon tolerances."""
    def rhs(t, y):
        return [SIGMA * (y[1] - y[0]),
                rho * y[0] - y[1] - y[0] * y[2],
                y[0] * y[1] - BETA * y[2]]
    sol = solve_ivp(rhs, (0.0, 1.0), [1.0, 0.0, 0.0],
                    method="DOP853", rtol=1e-13, atol=1e-14)
    return sol.y[:, -1]  # [x, y, z] at t=1

# Compute once at import time
_LORENZ_REF = _scipy_lorenz_reference(rho=21.0)


def verify(solver, label, block_size=256):
    x0 = cp.ones(block_size, dtype=cp.float32)
    y0 = cp.zeros(block_size, dtype=cp.float32)
    z0 = cp.zeros(block_size, dtype=cp.float32)
    rho = cp.full(block_size, 21.0, dtype=cp.float32)
    xf, yf, zf = solver(x0, y0, z0, rho)
    x = _scalar(xf[0])
    y = _scalar(yf[0])
    z = _scalar(zf[0])
    x_ref, y_ref, z_ref = _LORENZ_REF[0], _LORENZ_REF[1], _LORENZ_REF[2]
    err = max(abs(x - x_ref), abs(y - y_ref), abs(z - z_ref))
    status = "OK" if err < 0.01 else "FAIL"
    print(f"  {label:25s}: x={x:.6f} y={y:.6f} z={z:.6f}  err={err:.2e} [{status}]")


def _device_name():
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return name.decode() if isinstance(name, bytes) else name


if __name__ == "__main__":
    print("=" * 70)
    print("Direct CUDA Benchmark — Lorenz ODE")
    print("=" * 70)
    print(f"Device: {_device_name()}")
    print(f"Compiler backend: {CUDA_BACKEND}")
    print()

    print("Correctness:")
    verify(lambda x0, y0, z0, rho: solve_tsit5_fixed_cuda(x0, y0, z0, rho, block_size=256), "Tsit5-CUDA")
    verify(lambda x0, y0, z0, rho: solve_tsit5_adaptive_cuda(x0, y0, z0, rho, block_size=128), "Tsit5-adp-CUDA", 128)
    print()

    # Paper's exact N values: powers of 4 from 8 to 2^23.
    trajectory_counts = [8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608]

    # Collect best-block-size results for CSV output
    best_fixed = {}   # N -> min time across block sizes
    best_adaptive = {}

    for block in [64, 128, 256]:
        print(f"\n--- block_size={block} ---")
        print(f"{'N':>12s}  {'Tsit5':>10s}  {'Tsit5-adp':>10s}")

        for N in trajectory_counts:
            x0 = cp.ones(N, dtype=cp.float32)
            y0 = cp.zeros(N, dtype=cp.float32)
            z0 = cp.zeros(N, dtype=cp.float32)
            # Match paper's exact parameter distribution: linspace(0, 21, N)
            rho = cp.linspace(0, 21, N, dtype=cp.float32)

            n_runs = 20 if N <= 102_400 else 10

            try:
                t1, _ = bench_cuda(
                    lambda x, y, z, r, blk=block: solve_tsit5_fixed_cuda(x, y, z, r, block_size=blk),
                    x0, y0, z0, rho, n_runs=n_runs,
                )
            except Exception:
                t1 = float("nan")

            try:
                t2, _ = bench_cuda(
                    lambda x, y, z, r, blk=block: solve_tsit5_adaptive_cuda(x, y, z, r, block_size=blk),
                    x0, y0, z0, rho, n_runs=n_runs,
                )
            except Exception:
                t2 = float("nan")

            print(f"{N:>12,}  {t1:>8.2f}ms  {t2:>8.2f}ms")

            # Track best across block sizes
            if N not in best_fixed or t1 < best_fixed[N]:
                best_fixed[N] = t1
            if N not in best_adaptive or t2 < best_adaptive[N]:
                best_adaptive[N] = t2

    # ── Write CSV sweep data for figure generation ──
    import os
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for data, csv_name in [(best_fixed, "cuda_lorenz_fixed.csv"),
                           (best_adaptive, "cuda_lorenz_adaptive.csv")]:
        path = os.path.join(results_dir, csv_name)
        with open(path, "w") as f:
            for N_val in sorted(data.keys()):
                t = data[N_val]
                if not np.isnan(t):
                    f.write(f"{N_val} {t:.6f}\n")
        print(f"\nWrote {path}")
