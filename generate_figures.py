"""
Generate figures for the GPU ODE benchmark paper.

Produces:
  - fig_scaling.pdf: log-log scaling plot (time vs N) for all 4 rungs
  - fig_bars.pdf: horizontal bar chart at N=8M showing rung gaps
  - fig_bars_stiff.pdf: horizontal bar chart for stiff Robertson at N=8M
  - fig_scaling_stiff.pdf: log-log scaling for stiff Robertson

Data sources (bar charts read from result files, never hardcoded):
  Lorenz:
    - Diffrax:      results/diffrax_latest.out
    - JAX vmap:     results/jax_vmap_{fixed,adaptive}.csv
    - DiffEqGPU.jl: results/julia_lorenz.out
    - CUDA kernel:  results/cuda_lorenz_{fixed,adaptive}.csv
  Robertson:
    - JAX vmap:     results/jax_robertson.csv
    - DiffEqGPU.jl: results/julia_robertson.csv
    - CUDA kernel:  results/cuda_robertson.csv
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────
PAPER_ARTIFACTS = os.path.join(
    "GPUODEBenchmarks", "paper_artifacts", "data", "Tesla_V100"
)
RESULTS_DIR = "results"
TARGET_N = 8_388_608

COLORS = {
    "Diffrax":               "#d62728",  # red
    "JAX vmap":              "#ff7f0e",  # orange
    "DiffEqGPU.jl":         "#2ca02c",  # green
    "CUDA kernel":           "#1f77b4",  # blue
}
MARKERS = {
    "Diffrax":               "s",
    "JAX vmap":              "^",
    "DiffEqGPU.jl":         "o",
    "CUDA kernel":           "D",
}


def load_two_column(path):
    """Load a 2-column (N, time_ms) text/csv file."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0].astype(int), data[:, 1]


def get_n8m_from_csv(path):
    """Extract the N=8,388,608 timing from a two-column CSV."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    N, t = load_two_column(path)
    mask = N == TARGET_N
    if not mask.any():
        print(f"  WARNING: N={TARGET_N} not found in {path}")
        return None
    return float(t[mask][0])


def load_diffrax_latest():
    """Parse results/diffrax_latest.out for fixed and adaptive times."""
    path = os.path.join(RESULTS_DIR, "diffrax_latest.out")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None, None
    text = open(path).read()
    fixed = re.search(r"Diffrax fixed \(proper timing\):\s+([\d.]+)\s+ms", text)
    adaptive = re.search(r"Diffrax adaptive \(proper timing\):\s+([\d.]+)\s+ms", text)
    return (float(fixed.group(1)) if fixed else None,
            float(adaptive.group(1)) if adaptive else None)


def load_julia_lorenz():
    """Load the paper's pre-regression Lorenz timings.

    The paper intentionally uses results/julia_paper_lorenz.out for the
    DiffEqGPU Lorenz row because the latest tested stack still regresses
    adaptive performance.
    """
    # Keep the pre-regression Lorenz results in the paper.
    # On the latest tested stack, fixed stepping can still match the
    # old number, but adaptive remains substantially slower; see the
    # paper's version-sensitivity discussion.
    path = os.path.join(RESULTS_DIR, "julia_paper_lorenz.out")
    if os.path.exists(path):
        text = open(path).read()
        times = re.findall(r"Minimum time:\s+([\d.]+)\s+ms", text)
        if len(times) >= 2:
            return float(times[0]), float(times[1])

    print(f"  WARNING: no Julia Lorenz results found")
    return None, None


def load_scaling_data():
    """Load all available scaling data, returning dict of {label: (N_arr, time_arr)}."""
    data = {"fixed": {}, "adaptive": {}}

    # Rung 1: Diffrax (paper artifacts — these are Diffrax timings labeled as "JAX")
    for mode, fname in [("fixed", "Jax_times_unadaptive.txt"),
                        ("adaptive", "Jax_times_adaptive.txt")]:
        path = os.path.join(PAPER_ARTIFACTS, "JAX", fname)
        if os.path.exists(path):
            N, t = load_two_column(path)
            data[mode]["Diffrax"] = (N, t)

    # Rung 3: DiffEqGPU.jl (paper artifacts)
    for mode, fname in [("fixed", "Julia_times_unadaptive.txt"),
                        ("adaptive", "Julia_times_adaptive.txt")]:
        path = os.path.join(PAPER_ARTIFACTS, "Julia", fname)
        if os.path.exists(path):
            N, t = load_two_column(path)
            data[mode]["DiffEqGPU.jl"] = (N, t)

    # Rung 2: JAX vmap (our benchmark output)
    for mode, fname in [("fixed", "jax_vmap_fixed.csv"),
                        ("adaptive", "jax_vmap_adaptive.csv")]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            N, t = load_two_column(path)
            data[mode]["JAX vmap"] = (N, t)

    # Rung 4: CUDA kernel (our benchmark output)
    for mode, fname in [("fixed", "cuda_lorenz_fixed.csv"),
                        ("adaptive", "cuda_lorenz_adaptive.csv")]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            N, t = load_two_column(path)
            data[mode]["CUDA kernel"] = (N, t)

    return data


def make_scaling_figure(data, outpath="fig_scaling.pdf"):
    """Log-log scaling plot: two subplots (fixed, adaptive), 4 lines each."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for ax, mode, title in zip(axes, ["fixed", "adaptive"],
                                ["Fixed-step Tsit5", "Adaptive Tsit5"]):
        mode_data = data[mode]
        # Plot in order: Diffrax, JAX vmap, DiffEqGPU.jl, CUDA
        for label in ["Diffrax", "JAX vmap", "DiffEqGPU.jl", "CUDA kernel"]:
            if label not in mode_data:
                continue
            N, t = mode_data[label]
            ax.plot(N, t, marker=MARKERS[label], color=COLORS[label],
                    label=label, linewidth=1.5, markersize=5)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Trajectories ($N$)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print(f"Wrote {outpath}")
    plt.close(fig)


def make_bar_figure(outpath="fig_bars.pdf"):
    """Horizontal bar chart at N=8M showing gaps between rungs."""
    # Load all N=8M Lorenz results from data files
    cuda_fixed = get_n8m_from_csv(os.path.join(RESULTS_DIR, "cuda_lorenz_fixed.csv"))
    cuda_adaptive = get_n8m_from_csv(os.path.join(RESULTS_DIR, "cuda_lorenz_adaptive.csv"))
    jax_fixed = get_n8m_from_csv(os.path.join(RESULTS_DIR, "jax_vmap_fixed.csv"))
    jax_adaptive = get_n8m_from_csv(os.path.join(RESULTS_DIR, "jax_vmap_adaptive.csv"))
    diffrax_fixed, diffrax_adaptive = load_diffrax_latest()
    julia_fixed, julia_adaptive = load_julia_lorenz()

    missing = []
    for name, val in [("CUDA fixed", cuda_fixed), ("CUDA adaptive", cuda_adaptive),
                      ("JAX fixed", jax_fixed), ("JAX adaptive", jax_adaptive),
                      ("Diffrax fixed", diffrax_fixed), ("Diffrax adaptive", diffrax_adaptive),
                      ("Julia fixed", julia_fixed), ("Julia adaptive", julia_adaptive)]:
        if val is None:
            missing.append(name)
    if missing:
        print(f"  Skipping Lorenz bar chart — missing: {', '.join(missing)}")
        return

    methods_fixed = [
        ("CUDA kernel",   cuda_fixed),
        ("DiffEqGPU.jl",  julia_fixed),
        ("JAX vmap",      jax_fixed),
        ("Diffrax",       diffrax_fixed),
    ]
    methods_adaptive = [
        ("CUDA kernel",   cuda_adaptive),
        ("DiffEqGPU.jl",  julia_adaptive),
        ("JAX vmap",      jax_adaptive),
        ("Diffrax",       diffrax_adaptive),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))

    for ax, methods, title in zip(axes,
                                   [methods_fixed, methods_adaptive],
                                   ["Fixed-step Tsit5", "Adaptive Tsit5"]):
        labels = [m[0] for m in methods]
        times = [m[1] for m in methods]
        colors = [COLORS[m[0]] for m in methods]

        y_pos = range(len(methods))
        bars = ax.barh(y_pos, times, color=colors, height=0.6, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xscale("log")
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"{title}, $N = 8{{,}}388{{,}}608$")
        ax.invert_yaxis()

        # Annotate gaps between adjacent bars
        for i in range(len(methods) - 1):
            ratio = times[i + 1] / times[i]
            mid_y = (y_pos[i] + y_pos[i + 1]) / 2
            mid_x = (times[i] * times[i + 1]) ** 0.5  # geometric mean
            ax.annotate(
                f"{ratio:.1f}$\\times$",
                xy=(mid_x, mid_y),
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray",
                          alpha=0.8),
            )

        # Add time labels — outside bar for small bars, inside for large
        for bar, t in zip(bars, times):
            w = bar.get_width()
            y_center = bar.get_y() + bar.get_height() / 2
            # Place label to the right of the bar for short bars
            if t < 300:
                ax.text(w * 1.3, y_center, f"{t:,.0f} ms",
                        va="center", ha="left", fontsize=7, fontweight="bold")
            else:
                ax.text(w * 0.6, y_center, f"{t:,.0f} ms",
                        va="center", ha="right", fontsize=7,
                        color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print(f"Wrote {outpath}")
    plt.close(fig)


def load_stiff_scaling_data():
    """Load Robertson scaling data from CSV files."""
    data = {}
    for label, fname in [("JAX vmap", "jax_robertson.csv"),
                         ("CUDA kernel", "cuda_robertson.csv"),
                         ("DiffEqGPU.jl", "julia_robertson.csv")]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            N, t = load_two_column(path)
            data[label] = (N, t)
    return data


def make_stiff_scaling_figure(data, outpath="fig_scaling_stiff.pdf"):
    """Log-log scaling plot for stiff Robertson Rosenbrock23."""
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

    for label in ["JAX vmap", "DiffEqGPU.jl", "CUDA kernel"]:
        if label not in data:
            continue
        N, t = data[label]
        ax.plot(N, t, marker=MARKERS[label], color=COLORS[label],
                label=label, linewidth=1.5, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Trajectories ($N$)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Robertson Rosenbrock23 (adaptive)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print(f"Wrote {outpath}")
    plt.close(fig)


def make_stiff_bar_figure(outpath="fig_bars_stiff.pdf"):
    """Horizontal bar chart for stiff Robertson at N=8M."""
    cuda_rob = get_n8m_from_csv(os.path.join(RESULTS_DIR, "cuda_robertson.csv"))
    jax_rob = get_n8m_from_csv(os.path.join(RESULTS_DIR, "jax_robertson.csv"))
    julia_rob = get_n8m_from_csv(os.path.join(RESULTS_DIR, "julia_robertson.csv"))

    missing = []
    for name, val in [("CUDA Robertson", cuda_rob), ("JAX Robertson", jax_rob),
                      ("Julia Robertson", julia_rob)]:
        if val is None:
            missing.append(name)
    if missing:
        print(f"  Skipping Robertson bar chart — missing: {', '.join(missing)}")
        return

    methods = [
        ("CUDA kernel",   cuda_rob),
        ("DiffEqGPU.jl",  julia_rob),
        ("JAX vmap",      jax_rob),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))

    labels = [m[0] for m in methods]
    times = [m[1] for m in methods]
    colors = [COLORS[m[0]] for m in methods]

    y_pos = range(len(methods))
    bars = ax.barh(y_pos, times, color=colors, height=0.5, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Robertson Rosenbrock23, $N = 8{{,}}388{{,}}608$")
    ax.invert_yaxis()

    # Annotate gaps between adjacent bars
    for i in range(len(methods) - 1):
        ratio = times[i + 1] / times[i]
        mid_y = (y_pos[i] + y_pos[i + 1]) / 2
        mid_x = (times[i] * times[i + 1]) ** 0.5
        ax.annotate(
            f"{ratio:.0f}$\\times$",
            xy=(mid_x, mid_y),
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray",
                      alpha=0.8),
        )

    # Time labels
    for bar, t in zip(bars, times):
        w = bar.get_width()
        y_center = bar.get_y() + bar.get_height() / 2
        if t < 100:
            ax.text(w * 1.5, y_center, f"{t:,.2f} ms",
                    va="center", ha="left", fontsize=7, fontweight="bold")
        else:
            ax.text(w * 0.6, y_center, f"{t:,.0f} ms",
                    va="center", ha="right", fontsize=7,
                    color="white", fontweight="bold")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    print(f"Wrote {outpath}")
    plt.close(fig)




def fmt_ms(val):
    """Format milliseconds for LaTeX: 12345.6 -> '12{,}345'."""
    rounded = round(val)
    if rounded >= 1000:
        s = f"{rounded:,}".replace(",", "{,}")
    else:
        # Keep one decimal for values under 1000
        if val < 10:
            s = f"{val:.2f}"
        elif val < 100:
            s = f"{val:.1f}"
        else:
            s = str(rounded)
    return s


def load_verification_data():
    """Parse verification errors from all result files."""
    rows = []

    # JAX Lorenz (Level 2)
    path = os.path.join(RESULTS_DIR, "jax_lorenz.out")
    if os.path.exists(path):
        text = open(path).read()
        m = re.search(r"Tsit5 fixed.*?max_err=([\d.e+-]+)", text)
        if m:
            rows.append(("Lorenz", 2, "JAX vmap", "Tsit5 fixed", float(m.group(1))))
        m = re.search(r"Tsit5 adaptive.*?max_err=([\d.e+-]+)", text)
        if m:
            rows.append(("Lorenz", 2, "JAX vmap", "Tsit5 adaptive", float(m.group(1))))

    # CUDA Lorenz (Level 4)
    path = os.path.join(RESULTS_DIR, "cuda_lorenz.out")
    if os.path.exists(path):
        text = open(path).read()
        for label, pattern in [("Tsit5 fixed", r"Tsit5-CUDA\s+:.*?err=([\d.e+-]+)"),
                                ("Tsit5 adaptive", r"Tsit5-adp-CUDA\s+:.*?err=([\d.e+-]+)")]:
            m = re.search(pattern, text)
            if m:
                rows.append(("Lorenz", 4, "CUDA kernel", label, float(m.group(1))))

    # Julia Lorenz (Level 3)
    path = os.path.join(RESULTS_DIR, "julia_lorenz.out")
    if not os.path.exists(path):
        path = os.path.join(RESULTS_DIR, "julia_paper_lorenz.out")
    if os.path.exists(path):
        text = open(path).read()
        m = re.search(r"max_err=([\d.e+-]+)", text)
        if m:
            rows.append(("Lorenz", 3, "DiffEqGPU.jl", "Tsit5 adaptive", float(m.group(1))))

    # JAX Robertson (Level 2)
    path = os.path.join(RESULTS_DIR, "jax_robertson.out")
    if os.path.exists(path):
        text = open(path).read()
        m = re.search(r"Overall max_err=([\d.e+-]+)", text)
        if not m:
            m = re.search(r"max_err=([\d.e+-]+)", text)
        if m:
            rows.append(("Robertson", 2, "JAX vmap", "Rosenbrock23", float(m.group(1))))

    # CUDA Robertson (Level 4)
    path = os.path.join(RESULTS_DIR, "cuda_robertson.out")
    if os.path.exists(path):
        text = open(path).read()
        m = re.search(r"Overall max_err=([\d.e+-]+)", text)
        if not m:
            m = re.search(r"max_err=([\d.e+-]+)", text)
        if m:
            rows.append(("Robertson", 4, "CUDA kernel", "Rosenbrock23", float(m.group(1))))

    # Julia Robertson (Level 3)
    path = os.path.join(RESULTS_DIR, "julia_robertson.out")
    if os.path.exists(path):
        text = open(path).read()
        m = re.search(r"max_err=([\d.e+-]+)", text)
        if m:
            rows.append(("Robertson", 3, "DiffEqGPU.jl", "Rosenbrock23", float(m.group(1))))

    return rows


def generate_validation_table(rows, outpath="validation_table.tex"):
    """Generate LaTeX validation table from parsed verification data."""
    if not rows:
        print("  No verification data found — skipping validation table")
        return

    with open(outpath, "w") as f:
        f.write("% Auto-generated by generate_figures.py — do not edit by hand.\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\begin{tabular}{@{}cllll@{}}\n\\toprule\n")
        f.write("Level & Method & Problem & Solver & Max error vs.\\ SciPy f64 \\\\\n")
        f.write("\\midrule\n")

        # Sort by problem then level
        for problem in ["Lorenz", "Robertson"]:
            problem_rows = sorted([r for r in rows if r[0] == problem], key=lambda r: r[1])
            for prob, level, method, solver, err in problem_rows:
                f.write(f"{level} & {method} & {prob} & {solver} & ${err:.2e}$ \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Verification of all GPU implementations against SciPy\n")
        f.write("float64 reference solutions (DOP853 for Lorenz at $\\rho = 21$,\n")
        f.write("Radau for Robertson at $t_f = 10^5$).\n")
        f.write("All solvers use float32; errors reflect single-precision limits.}\n")
        f.write("\\label{tab:validation}\n\\end{table}\n")

    print(f"Wrote {outpath} ({len(rows)} rows)")


def generate_data_macros(outpath="data_macros.tex"):
    """Generate LaTeX \\newcommand definitions for all benchmark numbers."""
    macros = {}

    # Lorenz N=8M
    cuda_f = get_n8m_from_csv(os.path.join(RESULTS_DIR, "cuda_lorenz_fixed.csv"))
    cuda_a = get_n8m_from_csv(os.path.join(RESULTS_DIR, "cuda_lorenz_adaptive.csv"))
    jax_f = get_n8m_from_csv(os.path.join(RESULTS_DIR, "jax_vmap_fixed.csv"))
    jax_a = get_n8m_from_csv(os.path.join(RESULTS_DIR, "jax_vmap_adaptive.csv"))
    dfx_f, dfx_a = load_diffrax_latest()
    jul_f, jul_a = load_julia_lorenz()

    if cuda_f is not None: macros["CUDALorenzFixed"] = cuda_f
    if cuda_a is not None: macros["CUDALorenzAdaptive"] = cuda_a
    if jax_f is not None: macros["JAXLorenzFixed"] = jax_f
    if jax_a is not None: macros["JAXLorenzAdaptive"] = jax_a
    if dfx_f is not None: macros["DiffraxLorenzFixed"] = dfx_f
    if dfx_a is not None: macros["DiffraxLorenzAdaptive"] = dfx_a
    if jul_f is not None: macros["JuliaLorenzFixed"] = jul_f
    if jul_a is not None: macros["JuliaLorenzAdaptive"] = jul_a

    # Robertson N=8M
    cuda_rob = get_n8m_from_csv(os.path.join(RESULTS_DIR, "cuda_robertson.csv"))
    jax_rob = get_n8m_from_csv(os.path.join(RESULTS_DIR, "jax_robertson.csv"))
    julia_rob = get_n8m_from_csv(os.path.join(RESULTS_DIR, "julia_robertson.csv"))

    if cuda_rob is not None: macros["CUDARobertson"] = cuda_rob
    if jax_rob is not None: macros["JAXRobertson"] = jax_rob
    if julia_rob is not None: macros["JuliaRobertson"] = julia_rob

    # Derived ratios — Lorenz
    if dfx_f and jax_f:
        macros["RatioDiffraxJAXFixed"] = dfx_f / jax_f
    if jax_f and jul_f:
        macros["RatioJAXJuliaFixed"] = jax_f / jul_f
    if jul_f and cuda_f:
        macros["RatioJuliaCUDAFixed"] = jul_f / cuda_f
    if dfx_a and jul_a:
        macros["RatioDiffraxJuliaAdaptive"] = dfx_a / jul_a
    if cuda_a and jul_a:
        macros["RatioJuliaCUDAAdaptive"] = jul_a / cuda_a

    # Derived ratios — Robertson (CUDA < DiffEqGPU < JAX ordering)
    if cuda_rob and julia_rob:
        macros["RatioCUDAJuliaRobertson"] = julia_rob / cuda_rob
    if julia_rob and jax_rob:
        macros["RatioJuliaJAXRobertson"] = jax_rob / julia_rob
    if cuda_rob and jax_rob:
        macros["RatioCUDAJAXRobertson"] = jax_rob / cuda_rob


    # Write tex file
    with open(outpath, "w") as f:
        f.write("% Auto-generated by generate_figures.py — do not edit by hand.\n")
        f.write("% All values derived from result files in results/.\n\n")
        for name, val in sorted(macros.items()):
            if "Ratio" in name or "Regression" in name:
                # Format ratios as "94" or "1.8" or "5.6"
                if val >= 10:
                    f.write(f"\\newcommand{{\\{name}}}{{{val:.0f}}}\n")
                else:
                    f.write(f"\\newcommand{{\\{name}}}{{{val:.1f}}}\n")
            elif "Pct" in name:
                f.write(f"\\newcommand{{\\{name}}}{{{val:.1f}}}\n")
            else:
                f.write(f"\\newcommand{{\\{name}}}{{{fmt_ms(val)}}}\n")
    print(f"Wrote {outpath} ({len(macros)} macros)")


if __name__ == "__main__":
    data = load_scaling_data()

    # Report what data was found
    for mode in ["fixed", "adaptive"]:
        found = list(data[mode].keys())
        print(f"{mode}: {found}")

    # Generate data macros for paper
    print("\nData macros:")
    generate_data_macros()

    # Generate validation table
    print("\nValidation table:")
    vrows = load_verification_data()
    generate_validation_table(vrows)

    # Generate bar charts from result files
    print("\nLorenz bar chart:")
    make_bar_figure()
    print("\nRobertson bar chart:")
    make_stiff_bar_figure()

    # Stiff scaling figure
    stiff_data = load_stiff_scaling_data()
    stiff_labels = list(stiff_data.keys())
    print(f"\nStiff scaling data: {stiff_labels}")
    if len(stiff_data) >= 2:
        make_stiff_scaling_figure(stiff_data)
    else:
        print("Insufficient stiff scaling data — skipping fig_scaling_stiff.pdf")

    # Generate scaling plot if we have at least 2 lines
    has_data = any(len(data[mode]) >= 2 for mode in ["fixed", "adaptive"])
    if has_data:
        make_scaling_figure(data)
    else:
        print("\nInsufficient scaling data — skipping fig_scaling.pdf")
        print("Run benchmarks first: snakemake --profile slurm benchmarks")
