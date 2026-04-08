"""
Lorenz ODE ensemble benchmark — JAX vmap + fori_loop Tsit5 (Level 2).

Solves N Lorenz trajectories with ρ ∈ linspace(0, 21, N), y0 = (1,0,0), t ∈ [0,1].
Produces CSV sweep data for the scaling figure.
"""

import os
import time

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from scipy.integrate import solve_ivp

SIGMA = 10.0
BETA = 8.0 / 3.0

# ── Tsit5 Butcher tableau (Tsitouras 2011) ──────────────────────────
_a21 = 0.161
_a31 = -0.008480655492356989;  _a32 = 0.335480655492357
_a41 = 2.8971530571054935;     _a42 = -6.359448489975075;     _a43 = 4.3622954328695815
_a51 = 5.325864828439257;      _a52 = -11.748883564062828;    _a53 = 7.4955393428898365;    _a54 = -0.09249506636175525
_a61 = 5.86145544294642;       _a62 = -12.92096931784711;     _a63 = 8.159367898576159;     _a64 = -0.071584973281401;     _a65 = -0.028269050394068383
_a71 = 0.09646076681806523;    _a72 = 0.01;                   _a73 = 0.4798896504144996;    _a74 = 1.379008574103742;      _a75 = -3.290069515436081;     _a76 = 2.324710524099774

_b1 = _a71; _b2 = _a72; _b3 = _a73; _b4 = _a74; _b5 = _a75; _b6 = _a76
_c2 = 0.161; _c3 = 0.327; _c4 = 0.9; _c5 = 0.9800255409045097; _c6 = 1.0

_e1 = -0.00178001105222577714
_e2 = -0.0008164344596567469
_e3 =  0.007880878010261995
_e4 = -0.1447110071732629
_e5 =  0.5823571654525552
_e6 = -0.45808210592918697
_e7 =  1.0 / 66.0


# ── Tsit5 Fixed-Step Solver ──────────────────────────────────────────
def make_tsit5_fixed_solver(n_steps=1000, dt=0.001, unroll=5):
    n_iters = n_steps // unroll

    def solve(x0, y0, z0, rho):
        def lorenz(x, y, z):
            return SIGMA * (y - x), x * (rho - z) - y, x * y - BETA * z

        def tsit5_step(x, y, z):
            dx1, dy1, dz1 = lorenz(x, y, z)
            dx2, dy2, dz2 = lorenz(
                x + dt * _a21 * dx1, y + dt * _a21 * dy1, z + dt * _a21 * dz1)
            dx3, dy3, dz3 = lorenz(
                x + dt * (_a31 * dx1 + _a32 * dx2),
                y + dt * (_a31 * dy1 + _a32 * dy2),
                z + dt * (_a31 * dz1 + _a32 * dz2))
            dx4, dy4, dz4 = lorenz(
                x + dt * (_a41 * dx1 + _a42 * dx2 + _a43 * dx3),
                y + dt * (_a41 * dy1 + _a42 * dy2 + _a43 * dy3),
                z + dt * (_a41 * dz1 + _a42 * dz2 + _a43 * dz3))
            dx5, dy5, dz5 = lorenz(
                x + dt * (_a51 * dx1 + _a52 * dx2 + _a53 * dx3 + _a54 * dx4),
                y + dt * (_a51 * dy1 + _a52 * dy2 + _a53 * dy3 + _a54 * dy4),
                z + dt * (_a51 * dz1 + _a52 * dz2 + _a53 * dz3 + _a54 * dz4))
            dx6, dy6, dz6 = lorenz(
                x + dt * (_a61 * dx1 + _a62 * dx2 + _a63 * dx3 + _a64 * dx4 + _a65 * dx5),
                y + dt * (_a61 * dy1 + _a62 * dy2 + _a63 * dy3 + _a64 * dy4 + _a65 * dy5),
                z + dt * (_a61 * dz1 + _a62 * dz2 + _a63 * dz3 + _a64 * dz4 + _a65 * dz5))
            x_new = x + dt * (_b1 * dx1 + _b2 * dx2 + _b3 * dx3 + _b4 * dx4 + _b5 * dx5 + _b6 * dx6)
            y_new = y + dt * (_b1 * dy1 + _b2 * dy2 + _b3 * dy3 + _b4 * dy4 + _b5 * dy5 + _b6 * dy6)
            z_new = z + dt * (_b1 * dz1 + _b2 * dz2 + _b3 * dz3 + _b4 * dz4 + _b5 * dz5 + _b6 * dz6)
            return x_new, y_new, z_new

        def body(i, state):
            x, y, z = state
            for _ in range(unroll):
                x, y, z = tsit5_step(x, y, z)
            return (x, y, z)

        return lax.fori_loop(0, n_iters, body, (x0, y0, z0))

    return jax.jit(jax.vmap(solve))


# ── Tsit5 Adaptive-Step Solver ───────────────────────────────────────
def make_tsit5_adaptive_solver(atol=1e-8, rtol=1e-8, safety=0.9,
                                min_factor=0.2, max_factor=5.0,
                                max_steps=100000):
    def solve(x0, y0, z0, rho):
        def lorenz(x, y, z):
            return SIGMA * (y - x), x * (rho - z) - y, x * y - BETA * z

        def cond(state):
            t, x, y, z, dt, step_count = state
            return (t < 1.0) & (step_count < max_steps)

        def body(state):
            t, x, y, z, dt, step_count = state
            dt_eff = jnp.minimum(dt, 1.0 - t)

            dx1, dy1, dz1 = lorenz(x, y, z)
            dx2, dy2, dz2 = lorenz(
                x + dt_eff * _a21 * dx1, y + dt_eff * _a21 * dy1, z + dt_eff * _a21 * dz1)
            dx3, dy3, dz3 = lorenz(
                x + dt_eff * (_a31 * dx1 + _a32 * dx2),
                y + dt_eff * (_a31 * dy1 + _a32 * dy2),
                z + dt_eff * (_a31 * dz1 + _a32 * dz2))
            dx4, dy4, dz4 = lorenz(
                x + dt_eff * (_a41 * dx1 + _a42 * dx2 + _a43 * dx3),
                y + dt_eff * (_a41 * dy1 + _a42 * dy2 + _a43 * dy3),
                z + dt_eff * (_a41 * dz1 + _a42 * dz2 + _a43 * dz3))
            dx5, dy5, dz5 = lorenz(
                x + dt_eff * (_a51 * dx1 + _a52 * dx2 + _a53 * dx3 + _a54 * dx4),
                y + dt_eff * (_a51 * dy1 + _a52 * dy2 + _a53 * dy3 + _a54 * dy4),
                z + dt_eff * (_a51 * dz1 + _a52 * dz2 + _a53 * dz3 + _a54 * dz4))
            dx6, dy6, dz6 = lorenz(
                x + dt_eff * (_a61 * dx1 + _a62 * dx2 + _a63 * dx3 + _a64 * dx4 + _a65 * dx5),
                y + dt_eff * (_a61 * dy1 + _a62 * dy2 + _a63 * dy3 + _a64 * dy4 + _a65 * dy5),
                z + dt_eff * (_a61 * dz1 + _a62 * dz2 + _a63 * dz3 + _a64 * dz4 + _a65 * dz5))

            x_new = x + dt_eff * (_b1 * dx1 + _b2 * dx2 + _b3 * dx3 + _b4 * dx4 + _b5 * dx5 + _b6 * dx6)
            y_new = y + dt_eff * (_b1 * dy1 + _b2 * dy2 + _b3 * dy3 + _b4 * dy4 + _b5 * dy5 + _b6 * dy6)
            z_new = z + dt_eff * (_b1 * dz1 + _b2 * dz2 + _b3 * dz3 + _b4 * dz4 + _b5 * dz5 + _b6 * dz6)

            dx7, dy7, dz7 = lorenz(x_new, y_new, z_new)

            err_x = dt_eff * (_e1 * dx1 + _e2 * dx2 + _e3 * dx3 + _e4 * dx4 + _e5 * dx5 + _e6 * dx6 + _e7 * dx7)
            err_y = dt_eff * (_e1 * dy1 + _e2 * dy2 + _e3 * dy3 + _e4 * dy4 + _e5 * dy5 + _e6 * dy6 + _e7 * dy7)
            err_z = dt_eff * (_e1 * dz1 + _e2 * dz2 + _e3 * dz3 + _e4 * dz4 + _e5 * dz5 + _e6 * dz6 + _e7 * dz7)

            sc_x = atol + rtol * jnp.maximum(jnp.abs(x), jnp.abs(x_new))
            sc_y = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
            sc_z = atol + rtol * jnp.maximum(jnp.abs(z), jnp.abs(z_new))
            err_norm = jnp.sqrt(((err_x / sc_x) ** 2 + (err_y / sc_y) ** 2 + (err_z / sc_z) ** 2) / 3.0)

            accept = err_norm <= 1.0
            t_new = jnp.where(accept, t + dt_eff, t)
            x_out = jnp.where(accept, x_new, x)
            y_out = jnp.where(accept, y_new, y)
            z_out = jnp.where(accept, z_new, z)

            err_safe = jnp.maximum(err_norm, 1e-10)
            factor = safety * err_safe ** (-0.2)
            factor = jnp.clip(factor, min_factor, max_factor)
            dt_new = dt * factor

            return (t_new, x_out, y_out, z_out, dt_new, step_count + 1)

        init = (0.0, x0, y0, z0, 0.001, 0)
        t_f, x_f, y_f, z_f, dt_f, steps_f = lax.while_loop(cond, body, init)
        return x_f, y_f, z_f

    return jax.jit(jax.vmap(solve))


# ── Benchmark harness ────────────────────────────────────────────────
def benchmark(solver_fn, x0, y0, z0, rho, n_warmup=2, n_runs=20):
    for _ in range(n_warmup):
        jax.block_until_ready(solver_fn(x0, y0, z0, rho))

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(solver_fn(x0, y0, z0, rho))
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return min(times)


# ── Verification ─────────────────────────────────────────────────────
def _scipy_lorenz_reference(rho=21.0):
    """Float64 reference via SciPy DOP853."""
    def rhs(t, y):
        return [SIGMA * (y[1] - y[0]), rho * y[0] - y[1] - y[0] * y[2], y[0] * y[1] - BETA * y[2]]
    sol = solve_ivp(rhs, (0.0, 1.0), [1.0, 0.0, 0.0], method="DOP853", rtol=1e-13, atol=1e-14)
    return sol.y[:, -1]


def verify():
    ref = _scipy_lorenz_reference(rho=21.0)
    print(f"Reference (SciPy DOP853, f64): x={ref[0]:.10f}, y={ref[1]:.10f}, z={ref[2]:.10f}")

    x0 = jnp.array([1.0], dtype=jnp.float32)
    y0 = jnp.array([0.0], dtype=jnp.float32)
    z0 = jnp.array([0.0], dtype=jnp.float32)
    rho = jnp.array([21.0], dtype=jnp.float32)

    xf, yf, zf = make_tsit5_fixed_solver()(x0, y0, z0, rho)
    err_fix = max(abs(float(xf[0]) - ref[0]), abs(float(yf[0]) - ref[1]), abs(float(zf[0]) - ref[2]))
    print(f"Tsit5 fixed  (f32, dt=0.001): max_err={err_fix:.2e}")

    xa, ya, za = make_tsit5_adaptive_solver()(x0, y0, z0, rho)
    err_adp = max(abs(float(xa[0]) - ref[0]), abs(float(ya[0]) - ref[1]), abs(float(za[0]) - ref[2]))
    print(f"Tsit5 adaptive (f32, 1e-8):   max_err={err_adp:.2e}")
    print()


# ── Main benchmark ───────────────────────────────────────────────────
def run_benchmarks():
    print("=" * 70)
    print("JAX vmap Lorenz Benchmark (Level 2)")
    print("=" * 70)
    print(f"Device: {jax.devices()[0].device_kind} ({jax.devices()[0].platform})")
    print(f"JAX version: {jax.__version__}")
    print()

    trajectory_counts = [8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608]

    solvers = {
        "Tsit5-fixed":    make_tsit5_fixed_solver(n_steps=1000, dt=0.001, unroll=5),
        "Tsit5-adaptive": make_tsit5_adaptive_solver(atol=1e-8, rtol=1e-8),
    }

    results = {name: [] for name in solvers}

    for N in trajectory_counts:
        rho = jnp.linspace(0.0, 21.0, N, dtype=jnp.float32)
        x0 = jnp.ones(N, dtype=jnp.float32)
        y0 = jnp.zeros(N, dtype=jnp.float32)
        z0 = jnp.zeros(N, dtype=jnp.float32)
        n_runs = 50 if N <= 10_000 else (20 if N <= 100_000 else 10)

        for name, solver in solvers.items():
            t_min = benchmark(solver, x0, y0, z0, rho, n_warmup=2, n_runs=n_runs)
            results[name].append((N, t_min))
            print(f"  {name:18s}  N={N:>10,}:  {t_min:8.2f} ms")
        print()

    # Write CSV sweep data for figure generation
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    for name, csv_name in [("Tsit5-fixed", "jax_vmap_fixed.csv"),
                           ("Tsit5-adaptive", "jax_vmap_adaptive.csv")]:
        path = os.path.join(results_dir, csv_name)
        with open(path, "w") as f:
            for N_val, t_min in results[name]:
                f.write(f"{N_val} {t_min:.6f}\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    verify()
    run_benchmarks()
