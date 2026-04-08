"""
Robertson stiff ODE ensemble — JAX vmap Rosenbrock23 specialization variants.

Two variants that strip away problem-specific knowledge from robertson_jax.py:

  1. "autodiff": Jacobian via jax.jacfwd, linear solve via Cramer's rule on full 3x3.
     ZERO problem-specific knowledge — same information available to a generic solver.

  2. "autodiff+direct": Jacobian via jax.jacfwd, but linear solve exploits the known
     Robertson W-matrix structure (2x2 reduction). Intermediate specialization.

Comparison with robertson_jax.py (analytical Jac + direct solve) isolates how much
of the performance comes from Jacobian knowledge vs linear-solve structure vs
DiffEqGPU.jl's kernel architecture.
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

# Float64 reference (verified against SciPy Radau at rtol=1e-13, atol=1e-14)
ROBERTSON_REF = np.array(
    [1.7865921142252432e-02, 7.2747514684997379e-08, 9.8213400611023105e-01],
    dtype=np.float64,
)

# ── Rosenbrock23 tableau constants ───────────────────────────────────
_D = 0.2928932188134524       # 1 - 1/√2
_E32 = 7.4142135623730950     # embedded error weight
_BETA1 = 0.35
_BETA2 = 0.2
_SAFETY = 0.9
_QMIN_INV = 5.0
_QMAX_INV = 0.1


# ── Generic Rosenbrock23 Solver (autodiff Jacobian, generic solve) ──
def make_rosenbrock23_solver_generic(rtol=RTOL, atol=ATOL, dt0=DT0, tf=TF,
                                      max_steps=MAX_STEPS):
    def solve(perturbation):
        """Solve one Robertson trajectory with autodiff Jacobian."""

        def rhs_vec(y):
            """RHS as a vector function for autodiff."""
            y1, y2, y3 = y[0], y[1], y[2]
            return jnp.array([
                -0.04 * y1 + 1e4 * y2 * y3,
                0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2,
                3e7 * y2 * y2,
            ])

        def solve_w_generic(y, gamma, b):
            """Solve (I - gamma * J) x = b using autodiff Jacobian and Cramer's rule."""
            J = jax.jacfwd(rhs_vec)(y)
            W = jnp.eye(3) - gamma * J

            # Cramer's rule for 3x3 (no structure exploitation)
            def det3(M):
                return (M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
                      - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
                      + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0]))

            detW = det3(W)
            x0 = det3(W.at[:, 0].set(b)) / detW
            x1 = det3(W.at[:, 1].set(b)) / detW
            x2 = det3(W.at[:, 2].set(b)) / detW
            return jnp.array([x0, x1, x2])

        def cond(state):
            t, _y, dt, _qold, step = state
            return (t < tf) & (step < max_steps) & (dt > 1e-14)

        def body(state):
            t, y, dt, qold, step = state
            gamma = dt * _D

            # Stage 1
            f0 = rhs_vec(y)
            k1 = solve_w_generic(y, gamma, f0)

            # Stage 2
            f1 = rhs_vec(y + 0.5 * dt * k1)
            s2 = solve_w_generic(y, gamma, f1 - k1)
            k2 = s2 + k1

            # Second-order solution
            un = y + dt * k2

            # Stage 3 (for embedded error estimate)
            f2 = rhs_vec(un)
            k3 = solve_w_generic(y, gamma,
                f2 - _E32 * (k2 - f1) - 2.0 * (k1 - f0))

            # Error estimate
            e = (dt / 6.0) * (k1 - 2.0 * k2 + k3)

            sc = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(un))
            eest = jnp.sqrt(jnp.sum((e / sc) ** 2) / 3.0)

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
            y_new = jnp.where(accept, un, y)
            dt_new = jnp.where(accept,
                               jnp.minimum(jnp.abs(dtnew), remaining),
                               dt_rej)
            qold_out = jnp.where(accept, qold_new, qold)

            return (t_new, y_new, dt_new, qold_out, step + 1)

        y0 = jnp.array([1.0 + perturbation, 0.0, 0.0], dtype=jnp.float32)
        init = (jnp.float32(0.0), y0,
                jnp.float32(dt0), jnp.float32(1e-4), jnp.int32(0))
        t_f, y_f, _, _, _ = lax.while_loop(cond, body, init)
        return y_f[0], y_f[1], y_f[2]

    return jax.jit(jax.vmap(solve))


# ── Intermediate: autodiff Jacobian + structure-exploiting solve ─────
def make_rosenbrock23_solver_autodiff_direct(rtol=RTOL, atol=ATOL, dt0=DT0, tf=TF,
                                              max_steps=MAX_STEPS):
    def solve(perturbation):
        """Autodiff Jacobian, but exploit Robertson W-matrix structure for the solve."""

        def rhs_vec(y):
            y1, y2, y3 = y[0], y[1], y[2]
            return jnp.array([
                -0.04 * y1 + 1e4 * y2 * y3,
                0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2,
                3e7 * y2 * y2,
            ])

        def solve_w_autodiff_direct(y, gamma, b):
            """Autodiff Jacobian, then exploit W-matrix structure (w31=0, w33=1)."""
            J = jax.jacfwd(rhs_vec)(y)
            W = jnp.eye(3) - gamma * J

            # Structure-exploiting 2x2 reduction (same as robertson_jax.py solve_w)
            w11, w12, w13 = W[0, 0], W[0, 1], W[0, 2]
            w21, w22, w23 = W[1, 0], W[1, 1], W[1, 2]
            w32 = W[2, 1]

            a12 = w12 - w13 * w32
            a22 = w22 - w23 * w32
            rhs1 = b[0] - w13 * b[2]
            rhs2 = b[1] - w23 * b[2]
            det = w11 * a22 - w21 * a12
            x0 = (rhs1 * a22 - rhs2 * a12) / det
            x1 = (w11 * rhs2 - w21 * rhs1) / det
            x2 = b[2] - w32 * x1
            return jnp.array([x0, x1, x2])

        def cond(state):
            t, _y, dt, _qold, step = state
            return (t < tf) & (step < max_steps) & (dt > 1e-14)

        def body(state):
            t, y, dt, qold, step = state
            gamma = dt * _D

            f0 = rhs_vec(y)
            k1 = solve_w_autodiff_direct(y, gamma, f0)

            f1 = rhs_vec(y + 0.5 * dt * k1)
            s2 = solve_w_autodiff_direct(y, gamma, f1 - k1)
            k2 = s2 + k1

            un = y + dt * k2

            f2 = rhs_vec(un)
            k3 = solve_w_autodiff_direct(y, gamma,
                f2 - _E32 * (k2 - f1) - 2.0 * (k1 - f0))

            e = (dt / 6.0) * (k1 - 2.0 * k2 + k3)
            sc = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(un))
            eest = jnp.sqrt(jnp.sum((e / sc) ** 2) / 3.0)

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
            y_new = jnp.where(accept, un, y)
            dt_new = jnp.where(accept,
                               jnp.minimum(jnp.abs(dtnew), remaining),
                               dt_rej)
            qold_out = jnp.where(accept, qold_new, qold)

            return (t_new, y_new, dt_new, qold_out, step + 1)

        y0 = jnp.array([1.0 + perturbation, 0.0, 0.0], dtype=jnp.float32)
        init = (jnp.float32(0.0), y0,
                jnp.float32(dt0), jnp.float32(1e-4), jnp.int32(0))
        t_f, y_f, _, _, _ = lax.while_loop(cond, body, init)
        return y_f[0], y_f[1], y_f[2]

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
    print("Reference (SciPy Radau, f64):", ROBERTSON_REF)

    for name, make_solver in [("autodiff+direct", make_rosenbrock23_solver_autodiff_direct),
                               ("autodiff+cramer", make_rosenbrock23_solver_generic)]:
        solver = make_solver()
        dummy = jnp.zeros(1, dtype=jnp.float32).at[0].set(1e-5)
        y1, y2, y3 = solver(dummy)
        got = np.array([float(y1[0]), float(y2[0]), float(y3[0])], dtype=np.float64)
        errs = np.abs(got - ROBERTSON_REF)
        err = float(np.max(errs))
        status = "OK" if err < 1e-4 else "FAIL"
        print(f"Rosenbrock23-JAX-{name} (f32): y=[{got[0]:.8f}, {got[1]:.8e}, {got[2]:.8f}]")
        print(f"  max_err={err:.2e}  components=[{errs[0]:.2e}, {errs[1]:.2e}, {errs[2]:.2e}]  [{status}]")
    print()


# ── Main benchmark ───────────────────────────────────────────────────
def run_benchmarks():
    print("=" * 70)
    print("JAX vmap Robertson — Specialization Variants")
    print("=" * 70)
    print(f"Device: {jax.devices()[0].device_kind} ({jax.devices()[0].platform})")
    print(f"JAX version: {jax.__version__}")
    print(f"Settings: rtol={RTOL:.1e} atol={ATOL:.1e} dt0={DT0:.1e} tf={TF:.0e}")
    print()

    trajectory_counts = [1024, 10_240, 102_400, 1_024_000, 8_388_608]
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    variants = [
        ("autodiff+direct", make_rosenbrock23_solver_autodiff_direct,
         "jax_robertson_autodiff_direct.csv"),
        ("autodiff+cramer (generic)", make_rosenbrock23_solver_generic,
         "jax_robertson_generic.csv"),
    ]

    for var_name, make_solver, csv_name in variants:
        print(f"\n--- {var_name} ---")
        results = []
        for N in trajectory_counts:
            dummy = jnp.linspace(0, 1e-5, N, dtype=jnp.float32)
            solver = make_solver()
            n_runs = 20 if N <= 102_400 else 10
            t_min = benchmark(solver, dummy, n_warmup=2, n_runs=n_runs)
            results.append((N, t_min))
            print(f"  N={N:>10,}:  {t_min:8.2f} ms")

        path = os.path.join(results_dir, csv_name)
        with open(path, "w") as f:
            for N_val, t_min in results:
                f.write(f"{N_val} {t_min:.6f}\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    verify()
    run_benchmarks()
