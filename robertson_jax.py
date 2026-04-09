"""
Robertson stiff ODE ensemble — JAX vmap Rosenbrock23 (Level 2).

Solves N Robertson trajectories with k2 ∈ linspace(0, 10^4, N), y0 = (1,0,0),
t ∈ [0, 10^5].  Uses a linearly implicit Rosenbrock23 method with analytic
3×3 Jacobian and direct linear solve — no Newton iteration required.
Produces CSV sweep data for the scaling figure.
"""

import os
import time

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from scipy.integrate import solve_ivp

# ── Robertson constants ──────────────────────────────────────────────
RTOL = 1e-3
ATOL = 1e-6
DT0 = 1e-4
TF = 1e5
MAX_STEPS = 4096

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

# ── Rosenbrock23 tableau constants ───────────────────────────────────
_D = 0.2928932188134524       # 1 - 1/√2
_E32 = 7.4142135623730950     # embedded error weight
_BETA1 = 0.35                 # PI controller exponent
_BETA2 = 0.2                  # PI controller exponent
_SAFETY = 0.9
_QMIN_INV = 5.0               # max step growth
_QMAX_INV = 0.1               # max step shrink


# ── Rosenbrock23 Adaptive Solver ─────────────────────────────────────
def make_rosenbrock23_solver(rtol=RTOL, atol=ATOL, dt0=DT0, tf=TF,
                              max_steps=MAX_STEPS):
    def solve(k2):
        """Solve one Robertson trajectory with rate constant k2."""

        def rhs(y1, y2, y3):
            f1 = -0.04 * y1 + k2 * y2 * y3
            f2 = 0.04 * y1 - k2 * y2 * y3 - 3e7 * y2 * y2
            f3 = 3e7 * y2 * y2
            return f1, f2, f3

        def solve_w(y2, y3, gamma, b1, b2, b3):
            """Solve W x = b where W = I − γ J (Robertson-specific).

            Exploits w31 = 0 and w33 = 1 from the Jacobian structure to
            reduce the 3×3 system to a 2×2 plus back-substitution.
            """
            w11 = 1.0 + 0.04 * gamma
            w12 = -k2 * gamma * y3
            w13 = -k2 * gamma * y2
            w21 = -0.04 * gamma
            w22 = 1.0 + gamma * (k2 * y3 + 6e7 * y2)
            w23 = k2 * gamma * y2
            w32 = -6e7 * gamma * y2

            a12 = w12 - w13 * w32
            a22 = w22 - w23 * w32
            rhs1 = b1 - w13 * b3
            rhs2 = b2 - w23 * b3

            det = w11 * a22 - w21 * a12
            x1 = (rhs1 * a22 - rhs2 * a12) / det
            x2 = (w11 * rhs2 - w21 * rhs1) / det
            x3 = b3 - w32 * x2
            return x1, x2, x3

        def cond(state):
            t, _y1, _y2, _y3, dt, _qold, step = state
            return (t < tf) & (step < max_steps) & (dt > 1e-14)

        def body(state):
            t, y1, y2, y3, dt, qold, step = state
            gamma = dt * _D

            # Stage 1
            f01, f02, f03 = rhs(y1, y2, y3)
            k11, k12, k13 = solve_w(y2, y3, gamma, f01, f02, f03)

            # Stage 2
            f11, f12, f13 = rhs(y1 + 0.5 * dt * k11,
                                 y2 + 0.5 * dt * k12,
                                 y3 + 0.5 * dt * k13)
            s21, s22, s23 = solve_w(y2, y3, gamma,
                                     f11 - k11, f12 - k12, f13 - k13)
            k21 = s21 + k11
            k22 = s22 + k12
            k23 = s23 + k13

            # Second-order solution
            un1 = y1 + dt * k21
            un2 = y2 + dt * k22
            un3 = y3 + dt * k23

            # Stage 3 (for embedded error estimate)
            f21, f22, f23 = rhs(un1, un2, un3)
            k31, k32, k33 = solve_w(
                y2, y3, gamma,
                f21 - _E32 * (k21 - f11) - 2.0 * (k11 - f01),
                f22 - _E32 * (k22 - f12) - 2.0 * (k12 - f02),
                f23 - _E32 * (k23 - f13) - 2.0 * (k13 - f03))

            # Error estimate
            e1 = (dt / 6.0) * (k11 - 2.0 * k21 + k31)
            e2 = (dt / 6.0) * (k12 - 2.0 * k22 + k32)
            e3 = (dt / 6.0) * (k13 - 2.0 * k23 + k33)

            sc1 = atol + rtol * jnp.maximum(jnp.abs(y1), jnp.abs(un1))
            sc2 = atol + rtol * jnp.maximum(jnp.abs(y2), jnp.abs(un2))
            sc3 = atol + rtol * jnp.maximum(jnp.abs(y3), jnp.abs(un3))

            eest = jnp.sqrt(
                ((e1 / sc1) ** 2 + (e2 / sc2) ** 2 + (e3 / sc3) ** 2) / 3.0)

            # PI step-size control
            q11 = jnp.where(eest > 0, jnp.power(eest, _BETA1), 0.0)
            q = jnp.where(eest > 0, q11 / jnp.power(qold, _BETA2), _QMAX_INV)

            dt_rej = dt / jnp.minimum(_QMIN_INV, q11 / _SAFETY)
            q_acc = jnp.maximum(_QMAX_INV, jnp.minimum(_QMIN_INV, q / _SAFETY))
            dtnew = dt / q_acc
            qold_new = jnp.maximum(eest, 1e-4)

            accept = eest <= 1.0
            remaining = jnp.abs(tf - t - dt)
            at_end = remaining < 1e-14

            t_new = jnp.where(accept, jnp.where(at_end, tf, t + dt), t)
            y1_new = jnp.where(accept, un1, y1)
            y2_new = jnp.where(accept, un2, y2)
            y3_new = jnp.where(accept, un3, y3)
            dt_new = jnp.where(accept,
                               jnp.minimum(jnp.abs(dtnew), remaining),
                               dt_rej)
            qold_out = jnp.where(accept, qold_new, qold)

            return (t_new, y1_new, y2_new, y3_new, dt_new, qold_out, step + 1)

        init = (jnp.float32(0.0),
                jnp.float32(1.0), jnp.float32(0.0), jnp.float32(0.0),
                jnp.float32(dt0), jnp.float32(1e-4), jnp.int32(0))
        t_f, y1_f, y2_f, y3_f, _, _, _ = lax.while_loop(cond, body, init)
        return y1_f, y2_f, y3_f

    return jax.jit(jax.vmap(solve))


# ── Benchmark harness ────────────────────────────────────────────────
def benchmark(solver_fn, dummy, n_warmup=2, n_runs=20):
    for _ in range(n_warmup):
        jax.block_until_ready(solver_fn(dummy))

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(solver_fn(dummy))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return min(times)


# ── Verification ─────────────────────────────────────────────────────
def verify():
    solver = make_rosenbrock23_solver()
    test_k2 = [0.0, 100.0, 1000.0, 10000.0]
    k2_arr = jnp.array(test_k2, dtype=jnp.float32)
    y1, y2, y3 = solver(k2_arr)

    max_err_all = 0.0
    for i, k2_val in enumerate(test_k2):
        ref = ROBERTSON_REFS[int(k2_val)]
        got = np.array([float(y1[i]), float(y2[i]), float(y3[i])], dtype=np.float64)
        errs = np.abs(got - ref)
        err = float(np.max(errs))
        max_err_all = max(max_err_all, err)
        status = "OK" if err < 1e-2 else "FAIL"
        print(f"  k2={k2_val:>8.0f}: y=[{got[0]:.6e}, {got[1]:.6e}, {got[2]:.6e}]  max_err={err:.2e}  [{status}]")

    print(f"  Overall max_err={max_err_all:.2e}")
    print()


# ── Main benchmark ───────────────────────────────────────────────────
def run_benchmarks():
    print("=" * 70)
    print("JAX vmap Robertson Benchmark (Level 2, stiff)")
    print("=" * 70)
    print(f"Device: {jax.devices()[0].device_kind} ({jax.devices()[0].platform})")
    print(f"JAX version: {jax.__version__}")
    print(f"Settings: rtol={RTOL:.1e} atol={ATOL:.1e} dt0={DT0:.1e} tf={TF:.0e}")
    print(f"Sweep: k2 ∈ linspace(0, 1e4, N)")
    print()

    trajectory_counts = [1024, 10_240, 102_400, 1_024_000, 8_388_608]
    results = []

    for N in trajectory_counts:
        k2_vals = jnp.linspace(0.0, 1e4, N, dtype=jnp.float32)
        solver = make_rosenbrock23_solver()
        n_runs = 20 if N <= 102_400 else 10
        t_min = benchmark(solver, k2_vals, n_warmup=2, n_runs=n_runs)
        results.append((N, t_min))
        print(f"  N={N:>10,}:  {t_min:8.2f} ms")

    # Write CSV
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "jax_robertson.csv")
    with open(path, "w") as f:
        for N_val, t_min in results:
            f.write(f"{N_val} {t_min:.6f}\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    verify()
    run_benchmarks()
