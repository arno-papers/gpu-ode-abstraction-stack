"""
Generate validation figure for the GPU ODE benchmark paper.

Produces fig_validation.pdf with two subplots:
  - Left: Lorenz final state (x, y, z) at t=1 vs ρ, comparing methods
  - Right: Robertson time series y1(t), y2(t), y3(t) from SciPy reference

Runs locally using SciPy only (no GPU required).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

SIGMA = 10.0
BETA = 8.0 / 3.0

COLORS = {
    "SciPy f64":     "#333333",
    "JAX vmap":      "#ff7f0e",
    "CUDA kernel":   "#1f77b4",
}

STATE_LABELS = [r"$y_1$", r"$y_2$", r"$y_3$"]
STATE_MARKERS = ["o", "s", "^"]


def lorenz_reference_sweep(rho_values):
    """Compute Lorenz final states at t=1 for multiple ρ values (SciPy DOP853, f64)."""
    results = np.zeros((len(rho_values), 3))
    for i, rho in enumerate(rho_values):
        def rhs(t, y):
            return [SIGMA * (y[1] - y[0]),
                    rho * y[0] - y[1] - y[0] * y[2],
                    y[0] * y[1] - BETA * y[2]]
        sol = solve_ivp(rhs, (0.0, 1.0), [1.0, 0.0, 0.0],
                        method="DOP853", rtol=1e-13, atol=1e-14)
        results[i] = sol.y[:, -1]
    return results


def robertson_reference_sweep(k2_values):
    """Compute Robertson final states at t=1e5 for multiple k2 values (SciPy Radau, f64)."""
    results = np.zeros((len(k2_values), 3))
    for i, k2 in enumerate(k2_values):
        def rhs(t, y, _k2=k2):
            return [-0.04 * y[0] + _k2 * y[1] * y[2],
                    0.04 * y[0] - _k2 * y[1] * y[2] - 3e7 * y[1]**2,
                    3e7 * y[1]**2]
        sol = solve_ivp(rhs, (0.0, 1e5), [1.0, 0.0, 0.0],
                        method="Radau", rtol=1e-13, atol=1e-14)
        results[i] = sol.y[:, -1]
    return results


def make_validation_figure(outpath="fig_validation.pdf"):
    fig, (ax_lorenz, ax_robertson) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: Lorenz final state vs ρ ────────────────────────────────
    rho_values = np.linspace(0.0, 21.0, 200)
    ref = lorenz_reference_sweep(rho_values)

    for j, (label, marker) in enumerate(zip(STATE_LABELS, STATE_MARKERS)):
        ax_lorenz.plot(rho_values, ref[:, j], color="#333333", linewidth=1.2,
                       label=f"SciPy f64 {label}" if j == 0 else None,
                       zorder=1)
        ax_lorenz.annotate(label, xy=(rho_values[-1], ref[-1, j]),
                           xytext=(5, 0), textcoords="offset points",
                           fontsize=8, va="center")

    ax_lorenz.set_xlabel(r"$\rho$")
    ax_lorenz.set_ylabel("State at $t = 1$")
    ax_lorenz.set_title("Lorenz: final state vs. parameter")
    ax_lorenz.grid(True, alpha=0.3)

    # ── Right: Robertson final state vs k₂ ───────────────────────────
    k2_values = np.logspace(0, 4, 100)  # 1 to 1e4, log-spaced
    ref_rob = robertson_reference_sweep(k2_values)

    ax_robertson.plot(k2_values, ref_rob[:, 0], color="#1f77b4",
                      linewidth=1.2, label=r"$y_1$")
    ax_robertson.plot(k2_values, ref_rob[:, 2], color="#2ca02c",
                      linewidth=1.2, label=r"$y_3$")
    ax_robertson.set_xscale("log")
    ax_robertson.set_xlabel(r"$k_2$")
    ax_robertson.set_ylabel(r"$y_1$, $y_3$ at $t = 10^5$")
    ax_robertson.set_title("Robertson: final state vs. parameter")

    # y2 on right axis
    ax2 = ax_robertson.twinx()
    ax2.plot(k2_values, ref_rob[:, 1], color="#d62728", linewidth=1.2,
             linestyle="--", label=r"$y_2$")
    ax2.set_ylabel(r"$y_2$", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax_robertson.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_robertson.legend(lines1 + lines2, labels1 + labels2,
                        fontsize=8, loc="center right")
    ax_robertson.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print(f"Wrote {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating validation figure...")
    make_validation_figure()
