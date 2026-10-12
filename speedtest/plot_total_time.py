#!/usr/bin/env python3
"""Compare Plummer and Hernquist-cusp PeTar integration timings."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics


def load_results(path: Path, expected: str) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("status") != "ok":
                continue
            if row.get("stellar_evolution") != "2":
                raise ValueError(f"{path} is not a stellar-evolution mode-2 result")
            if row.get("profile") != expected:
                raise ValueError(f"{path} has profile={row.get('profile')!r}; expected {expected!r}")
            if not row.get("profiled_step_seconds"):
                raise ValueError(f"profiled_step_seconds is missing in {path}")
            grouped.setdefault(int(row["n"]), []).append(float(row["profiled_step_seconds"]))
    if not grouped:
        raise ValueError(f"no successful benchmark rows in {path}")
    return grouped


def summarize(grouped: dict[int, list[float]]) -> dict[int, tuple[float, int]]:
    return {n: (statistics.median(values), len(values))
            for n, values in sorted(grouped.items())}


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plummer", type=Path, default=root / "results_plummer.csv")
    parser.add_argument("--hernquist", type=Path, default=root / "results_hernquist.csv")
    parser.add_argument("--output", type=Path, default=root / "model_comparison.png")
    parser.add_argument("--pdf", type=Path, default=root / "model_comparison.pdf")
    parser.add_argument("--plot-data", type=Path, default=root / "model_comparison.csv")
    parser.add_argument("--title", default="PeTar single-core scaling: core vs cusp")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {
        "Plummer (finite-density core)": summarize(load_results(args.plummer, "plummer")),
        r"Hernquist ($\rho\propto r^{-1}$ cusp)": summarize(
            load_results(args.hernquist, "hernquist")),
    }
    all_points = [(n, measured) for values in series.values()
                  for n, (measured, _) in values.items()]
    normalization_n, maximum_time = max(all_points, key=lambda item: item[1])
    scale = maximum_time / (normalization_n * math.log(normalization_n))
    theory_n = sorted({n for n, _ in all_points})
    theory = {n: scale * n * math.log(n) for n in theory_n}

    args.plot_data.parent.mkdir(parents=True, exist_ok=True)
    with args.plot_data.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "n", "measured_profile_seconds_median",
                         "successful_repeats", "nlogn_reference_seconds"))
        for model, values in series.items():
            for n, (measured, repeats) in values.items():
                writer.writerow((model, n, measured, repeats, theory[n]))

    fig, (ax, info_ax) = plt.subplots(
        1, 2, figsize=(12.5, 6.5), constrained_layout=True,
        gridspec_kw={"width_ratios": (3.2, 1.35)},
    )
    styles = (("#1f77b4", "o"), ("#ff7f0e", "s"))
    for (label, values), (color, marker) in zip(series.items(), styles):
        n_values = list(values)
        ax.plot(n_values, [values[n][0] for n in n_values], marker=marker,
                color=color, linewidth=2, markersize=6, label=label)
    ax.plot(theory_n, [theory[n] for n in theory_n], "k--", linewidth=1.8,
            label=rf"$C\,N\ln N$ (normalized to maximum: {maximum_time:.3g} s)")
    ax.set(xscale="log", yscale="log", xlabel="Particle number N",
           ylabel="Profiled integration time [s]", title=args.title)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left")

    feature_text = (
        "Common benchmark configuration\n"
        "• SSE/BSE stellar evolution mode 2\n"
        "• Dynamical tides + hyperbolic GW loss\n"
        "• Tree + Hermite/AR/H4 integration\n"
        "• 1 MPI rank, 1 OpenMP thread, 1 CPU core\n"
        "• Equal-mass stars; no primordial binaries\n\n"
        "Matched between models\n"
        "• N, total mass, stellar mass\n"
        "• 3D half-mass radius\n\n"
        "N log N normalization\n"
        f"• Maximum measured time: {maximum_time:.4g} s\n"
        f"• Occurs at N={normalization_n:,}"
    )
    info_ax.axis("off")
    info_ax.text(0.03, 0.98, feature_text, transform=info_ax.transAxes,
                 ha="left", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                           edgecolor="0.55", alpha=0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    if args.pdf:
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf)
    print(f"Wrote {args.output}")
    if args.pdf:
        print(f"Wrote {args.pdf}")
    print(f"Wrote {args.plot_data}")


if __name__ == "__main__":
    main()
