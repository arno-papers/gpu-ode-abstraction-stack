"""
Robertson stiff ODE ensemble — direct CUDA Rosenbrock23 kernel via CuPy (Level 4).

Solves N Robertson trajectories with k2 ∈ linspace(0, 10^4, N).
One thread per trajectory, analytic 3×3 Jacobian and direct linear solve in registers.
"""

import shutil
import time

import numpy as np
import cupy as cp

RTOL = 1e-3
ATOL = 1e-6
DT0 = 1e-4
TF = 1e5
MAX_STEPS = 4096
TRAJECTORY_COUNTS = [1024, 10_240, 102_400, 1_024_000, 8_388_608]

# Float64 references at selected k2 values (SciPy Radau, rtol=1e-13, atol=1e-14)
ROBERTSON_REFS = {
    0:     np.array([4.4539086329753037e-53, 3.3483037787036374e-13, 9.9999999999966493e-01]),
    1:     np.array([2.1835511345657175e-10, 8.7320050186835035e-12, 9.9999999977291798e-01]),
    10:    np.array([2.1058091147169210e-08, 8.4211176575432374e-11, 9.9999997885769532e-01]),
    100:   np.array([2.0956005100413740e-06, 8.3803135613156177e-10, 9.9999790356146090e-01]),
    1000:  np.array([2.0843295464602322e-04, 8.3369708857099002e-09, 9.9979155870837855e-01]),
    5000:  np.array([4.9474452523740418e-03, 3.9766819439486466e-08, 9.9505251498081049e-01]),
    10000: np.array([1.7865921142115607e-02, 7.2747514684430595e-08, 9.8213400611036827e-01]),
}


CUDA_BACKEND = "nvcc" if shutil.which("nvcc") else "nvrtc"
CUDA_OPTIONS = ("--std=c++14", "--use_fast_math", "-O3")

CUDA_SRC = r"""
#define RB23_D 0.2928932188134524f
#define RB23_E32 7.4142135623730950f
#define BETA1 0.35f
#define BETA2 0.2f
#define QMAX_INV 0.1f
#define QMIN_INV 5.0f
#define SAFETY 0.9f

__device__ __forceinline__ void robertson_rhs(
    float y1, float y2, float y3, float k2,
    float &f1, float &f2, float &f3
) {
    f1 = -0.04f * y1 + k2 * y2 * y3;
    f2 = 0.04f * y1 - k2 * y2 * y3 - 3.0e7f * y2 * y2;
    f3 = 3.0e7f * y2 * y2;
}

__device__ __forceinline__ void solve_w_robertson(
    float y2, float y3, float k2, float gamma,
    float b1, float b2, float b3,
    float &x1, float &x2, float &x3
) {
    float w11 = 1.0f + 0.04f * gamma;
    float w12 = -k2 * gamma * y3;
    float w13 = -k2 * gamma * y2;
    float w21 = -0.04f * gamma;
    float w22 = 1.0f + gamma * (k2 * y3 + 6.0e7f * y2);
    float w23 = k2 * gamma * y2;
    float w32 = -6.0e7f * gamma * y2;

    float a12 = w12 - w13 * w32;
    float a22 = w22 - w23 * w32;
    float rhs1 = b1 - w13 * b3;
    float rhs2 = b2 - w23 * b3;
    float det = w11 * a22 - w21 * a12;

    x1 = (rhs1 * a22 - rhs2 * a12) / det;
    x2 = (w11 * rhs2 - w21 * rhs1) / det;
    x3 = b3 - w32 * x2;
}

extern "C" __global__ void robertson_ros23_adaptive_kernel(
    float* y1o, float* y2o, float* y3o, const float* k2_ptr,
    int N, int max_steps, float dt0, float atol, float rtol, float tf
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float k2 = k2_ptr[tid];
    float y1 = 1.0f;
    float y2 = 0.0f;
    float y3 = 0.0f;
    float t = 0.0f;
    float dt = dt0;
    float qold = 1.0e-4f;

    for (int attempt = 0; attempt < max_steps && t < tf; ++attempt) {
        if (dt < 1.0e-14f) {
            break;
        }

        float gamma = dt * RB23_D;

        float f01, f02, f03;
        float f11, f12, f13;
        float f21, f22, f23;
        float k11, k12, k13;
        float s21, s22, s23;
        float k21, k22, k23;
        float k31, k32, k33;

        robertson_rhs(y1, y2, y3, k2, f01, f02, f03);
        solve_w_robertson(y2, y3, k2, gamma, f01, f02, f03, k11, k12, k13);

        float ym1 = y1 + 0.5f * dt * k11;
        float ym2 = y2 + 0.5f * dt * k12;
        float ym3 = y3 + 0.5f * dt * k13;
        robertson_rhs(ym1, ym2, ym3, k2, f11, f12, f13);

        solve_w_robertson(
            y2, y3, k2, gamma,
            f11 - k11, f12 - k12, f13 - k13,
            s21, s22, s23
        );
        k21 = s21 + k11;
        k22 = s22 + k12;
        k23 = s23 + k13;

        float un1 = y1 + dt * k21;
        float un2 = y2 + dt * k22;
        float un3 = y3 + dt * k23;
        robertson_rhs(un1, un2, un3, k2, f21, f22, f23);

        solve_w_robertson(
            y2, y3, k2, gamma,
            f21 - RB23_E32 * (k21 - f11) - 2.0f * (k11 - f01),
            f22 - RB23_E32 * (k22 - f12) - 2.0f * (k12 - f02),
            f23 - RB23_E32 * (k23 - f13) - 2.0f * (k13 - f03),
            k31, k32, k33
        );

        float e1 = (dt / 6.0f) * (k11 - 2.0f * k21 + k31);
        float e2 = (dt / 6.0f) * (k12 - 2.0f * k22 + k32);
        float e3 = (dt / 6.0f) * (k13 - 2.0f * k23 + k33);

        float s1 = atol + rtol * fmaxf(fabsf(y1), fabsf(un1));
        float s2 = atol + rtol * fmaxf(fabsf(y2), fabsf(un2));
        float s3 = atol + rtol * fmaxf(fabsf(y3), fabsf(un3));

        float err1 = e1 / s1;
        float err2 = e2 / s2;
        float err3 = e3 / s3;
        float eest = sqrtf((err1 * err1 + err2 * err2 + err3 * err3) / 3.0f);

        float q = QMAX_INV;
        float q11 = 0.0f;
        if (eest != 0.0f) {
            q11 = powf(eest, BETA1);
            q = q11 / powf(qold, BETA2);
        }

        if (eest > 1.0f) {
            dt = dt / fminf(QMIN_INV, q11 / SAFETY);
            continue;
        }

        q = fmaxf(QMAX_INV, fminf(QMIN_INV, q / SAFETY));
        qold = fmaxf(eest, 1.0e-4f);
        float dtnew = dt / q;
        float remaining = fabsf(tf - t - dt);

        y1 = un1;
        y2 = un2;
        y3 = un3;

        if (remaining < 1.0e-14f) {
            t = tf;
            break;
        }

        t += dt;
        dt = fminf(fabsf(dtnew), remaining);
    }

    y1o[tid] = y1;
    y2o[tid] = y2;
    y3o[tid] = y3;
}
"""

_RAW_MODULE_KWARGS = {
    "code": CUDA_SRC,
    "options": CUDA_OPTIONS,
    "backend": CUDA_BACKEND,
}
if CUDA_BACKEND == "nvrtc":
    _RAW_MODULE_KWARGS["name_expressions"] = ("robertson_ros23_adaptive_kernel",)

CUDA_MODULE = cp.RawModule(**_RAW_MODULE_KWARGS)
ROS23_ADAPTIVE_KERNEL = CUDA_MODULE.get_function("robertson_ros23_adaptive_kernel")


def solve_robertson_ros23_cuda(N, k2_vals=None, max_steps=MAX_STEPS, dt0=DT0,
                                atol=ATOL, rtol=RTOL, tf=TF, block_size=256):
    N = int(N)
    if k2_vals is None:
        k2_vals = cp.linspace(0.0, 1e4, N, dtype=cp.float32)
    y1o = cp.empty(N, dtype=cp.float32)
    y2o = cp.empty(N, dtype=cp.float32)
    y3o = cp.empty(N, dtype=cp.float32)
    grid = ((N + block_size - 1) // block_size,)
    ROS23_ADAPTIVE_KERNEL(
        grid, (block_size,),
        (y1o, y2o, y3o, k2_vals,
         np.int32(N), np.int32(max_steps),
         np.float32(dt0), np.float32(atol), np.float32(rtol), np.float32(tf)),
    )
    return y1o, y2o, y3o


def bench_cuda(fn, n_warmup=3, n_runs=20):
    for _ in range(n_warmup):
        fn()
        cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(n_runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times)


def verify_gpu(block_size=256):
    test_k2 = [0.0, 100.0, 1000.0, 10000.0]
    k2_arr = cp.array(test_k2, dtype=cp.float32)
    y1, y2, y3 = solve_robertson_ros23_cuda(len(test_k2), k2_vals=k2_arr, block_size=block_size)

    max_err_all = 0.0
    for i, k2_val in enumerate(test_k2):
        ref = ROBERTSON_REFS[int(k2_val)]
        got = np.array(
            [float(cp.asnumpy(y1[i])), float(cp.asnumpy(y2[i])), float(cp.asnumpy(y3[i]))],
            dtype=np.float64,
        )
        errs = np.abs(got - ref)
        err = float(np.max(errs))
        max_err_all = max(max_err_all, err)
        status = "OK" if err < 1e-2 else "FAIL"
        print(f"  k2={k2_val:>8.0f}: y=[{got[0]:.6e}, {got[1]:.6e}, {got[2]:.6e}]  max_err={err:.2e}  [{status}]")
    print(f"  Overall max_err={max_err_all:.2e}")


def convergence_test():
    """Run at multiple tolerances to verify order-2 convergence (k2=1e4)."""
    ref = ROBERTSON_REFS[10000]
    print("Convergence test (Rosenbrock23, k2=1e4, single trajectory):")
    print(f"  {'rtol':>10s}  {'atol':>10s}  {'max_err':>10s}  {'y1_err':>10s}  {'y2_err':>10s}  {'y3_err':>10s}")
    prev_err = None
    for rtol_exp in [2, 3, 4, 5]:
        rtol = 10.0 ** (-rtol_exp)
        atol = rtol * 1e-3  # atol = rtol / 1000
        k2_arr = cp.array([1e4], dtype=cp.float32)
        y1, y2, y3 = solve_robertson_ros23_cuda(
            1, k2_vals=k2_arr, max_steps=65536, dt0=1e-6, atol=atol, rtol=rtol, block_size=32)
        got = np.array(
            [float(cp.asnumpy(y1[0])), float(cp.asnumpy(y2[0])), float(cp.asnumpy(y3[0]))],
            dtype=np.float64,
        )
        errs = np.abs(got - ref)
        err = float(np.max(errs))
        ratio = f"{prev_err / err:.1f}x" if prev_err is not None and err > 0 else ""
        print(f"  {rtol:10.1e}  {atol:10.1e}  {err:10.2e}  {errs[0]:10.2e}  {errs[1]:10.2e}  {errs[2]:10.2e}  {ratio}")
        prev_err = err


def _device_name():
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return name.decode() if isinstance(name, bytes) else name


if __name__ == "__main__":
    print("=" * 70)
    print("Direct CUDA Benchmark — Robertson stiff ODE (Level 4)")
    print("=" * 70)
    print(f"Device: {_device_name()}")
    print(f"Compiler backend: {CUDA_BACKEND}")
    print(f"Settings: rtol={RTOL:.1e} atol={ATOL:.1e} dt0={DT0:.1e} tf={TF:.0e}")
    print(f"Sweep: k2 ∈ linspace(0, 1e4, N)")
    print()

    print("Correctness:")
    verify_gpu(256)
    print()

    convergence_test()
    print()

    best = {}  # N -> min time across block sizes

    for block in [64, 128, 256]:
        print(f"--- block_size={block} ---")
        print(f"{'N':>12s}  {'Ros23-adp':>12s}")
        for N in TRAJECTORY_COUNTS:
            n_runs = 20 if N <= 102_400 else 10
            t1 = bench_cuda(
                lambda n=N, blk=block: solve_robertson_ros23_cuda(n, block_size=blk),
                n_runs=n_runs,
            )
            print(f"{N:>12,}  {t1:>10.2f}ms")
            if N not in best or t1 < best[N]:
                best[N] = t1
        print()

    # Write CSV for figure generation
    import os
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "cuda_robertson.csv")
    with open(path, "w") as f:
        for N_val in sorted(best.keys()):
            f.write(f"{N_val} {best[N_val]:.6f}\n")
    print(f"Wrote {path}")
