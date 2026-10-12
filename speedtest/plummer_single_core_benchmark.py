#!/usr/bin/env python3
"""Generate equal-mass Plummer models and benchmark PeTar on one CPU core."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np


G = 0.00449830997959438  # pc^3 / (Msun Myr^2)


def geometric_grid(start: int = 1_000, factor: int = 3,
                   target: int = 1_000_000) -> tuple[int, ...]:
    """Return start, factor*start, ... and always include target exactly."""
    if start <= 0 or factor <= 1 or target < start:
        raise ValueError("require start > 0, factor > 1, and target >= start")
    values = []
    value = start
    while value < target:
        values.append(value)
        value *= factor
    if not values or values[-1] != target:
        values.append(target)
    return tuple(values)


DEFAULT_N = geometric_grid()


def unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    vectors = rng.normal(size=(n, 3))
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    return vectors


def make_model(path: Path, n: int, stellar_mass: float, rho0: float, seed: int) -> dict:
    """Write a 34-column PeTar/BSE input with stellar type kw=1."""
    rng = np.random.default_rng(seed)
    total_mass = n * stellar_mass
    # rho(0) = 3 M / (4 pi a^3) for a Plummer sphere.
    scale = (3.0 * total_mass / (4.0 * math.pi * rho0)) ** (1.0 / 3.0)

    u = rng.random(n)
    radius = scale / np.sqrt(np.power(u, -2.0 / 3.0) - 1.0)
    position = radius[:, None] * unit_vectors(rng, n)

    # Exact isotropic Plummer DF: (v/v_escape)^2 ~ Beta(3/2, 9/2).
    q = np.sqrt(rng.beta(1.5, 4.5, size=n))
    escape_speed = np.sqrt(2.0 * G * total_mass / np.sqrt(radius**2 + scale**2))
    velocity = (q * escape_speed)[:, None] * unit_vectors(rng, n)

    # Put the finite-N realization in its centre-of-mass frame.
    position -= position.mean(axis=0)
    velocity -= velocity.mean(axis=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write(f"0 {n} 0\n")
        chunk = 100_000
        for first in range(0, n, chunk):
            last = min(first + chunk, n)
            data = np.zeros((last - first, 34))
            data[:, 0] = stellar_mass
            data[:, 1:4] = position[first:last]
            data[:, 4:7] = velocity[first:last]
            data[:, 12] = 1                 # BSE stellar type (kw)
            data[:, 13:15] = stellar_mass  # initial/current stellar mass
            data[:, 23] = np.arange(first + 1, last + 1)
            np.savetxt(stream, data, fmt="%.16g")

    return {
        "profile": "plummer",
        "n": n,
        "stellar_mass_msun": stellar_mass,
        "total_mass_msun": total_mass,
        "central_density_msun_pc3": rho0,
        "plummer_scale_pc": scale,
        "half_mass_radius_pc": scale / math.sqrt(2.0 ** (2.0 / 3.0) - 1.0),
        "seed": seed,
        "stellar_type_kw": 1,
    }


def hernquist_df(q: np.ndarray) -> np.ndarray:
    """Dimensionless isotropic Hernquist distribution function."""
    q = np.clip(q, 1.0e-7, 1.0 - 1.0e-10)
    omq2 = 1.0 - q*q
    polynomial = (1.0 - 2.0*q*q) * (8.0*q**4 - 8.0*q*q - 3.0)
    numerator = 3.0*np.arcsin(q) + q*np.sqrt(omq2)*polynomial
    return np.maximum(numerator / np.power(omq2, 2.5), 0.0)


def sample_hernquist_speeds(rng: np.random.Generator, x: np.ndarray,
                            total_mass: float, scale: float) -> np.ndarray:
    """Sample the isotropic Hernquist DF through tabulated conditional CDFs."""
    x_grid = np.geomspace(max(x.min()*0.8, 1.0e-7), max(x.max()*1.2, 1.0e-6), 384)
    y_grid = np.linspace(0.0, 1.0, 1025)  # y = v/v_escape
    bins = np.clip(np.searchsorted(x_grid, x), 0, len(x_grid) - 1)
    uniform = rng.random(len(x))
    y = np.empty(len(x))
    for index in np.unique(bins):
        q_max = 1.0 / math.sqrt(1.0 + x_grid[index])
        q = q_max * np.sqrt(np.maximum(1.0 - y_grid*y_grid, 0.0))
        pdf = y_grid*y_grid * hernquist_df(q)
        cdf = np.empty_like(y_grid)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum((pdf[:-1] + pdf[1:]) * np.diff(y_grid) * 0.5)
        if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
            raise RuntimeError(f"invalid Hernquist speed CDF at r/a={x_grid[index]}")
        cdf /= cdf[-1]
        selected = bins == index
        y[selected] = np.interp(uniform[selected], cdf, y_grid)
    escape = np.sqrt(2.0 * G * total_mass / (scale * (1.0 + x)))
    return y * escape


def make_cusp_model(path: Path, n: int, stellar_mass: float, rho0: float,
                    seed: int) -> dict:
    """Write an isotropic Hernquist rho~r^-1 cusp with Plummer-matched r_h."""
    rng = np.random.default_rng(seed)
    total_mass = n * stellar_mass
    plummer_scale = (3.0 * total_mass / (4.0 * math.pi * rho0)) ** (1.0 / 3.0)
    half_mass_radius = plummer_scale / math.sqrt(2.0**(2.0/3.0) - 1.0)
    scale = half_mass_radius / (1.0 + math.sqrt(2.0))
    sqrt_u = np.sqrt(rng.random(n))
    x = sqrt_u / (1.0 - sqrt_u)
    radius = scale * x
    position = radius[:, None] * unit_vectors(rng, n)
    velocity = sample_hernquist_speeds(rng, x, total_mass, scale)[:, None] * unit_vectors(rng, n)
    # Keep the analytic cusp at the origin. A distant finite-sample halo star
    # can make COM recentering spuriously displace and smear the cusp.
    velocity -= velocity.mean(axis=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write(f"0 {n} 0\n")
        for first in range(0, n, 100_000):
            last = min(first + 100_000, n)
            data = np.zeros((last-first, 34))
            data[:, 0] = stellar_mass
            data[:, 1:4] = position[first:last]
            data[:, 4:7] = velocity[first:last]
            data[:, 12] = 1
            data[:, 13:15] = stellar_mass
            data[:, 23] = np.arange(first+1, last+1)
            np.savetxt(stream, data, fmt="%.16g")
    return {"profile": "hernquist", "central_cusp": "rho proportional to r^-1",
            "n": n, "stellar_mass_msun": stellar_mass,
            "total_mass_msun": total_mass, "hernquist_scale_pc": scale,
            "half_mass_radius_pc": half_mass_radius,
            "matched_plummer_central_density_msun_pc3": rho0,
            "seed": seed, "stellar_type_kw": 1}


def choose_cpu(requested: int | None) -> int:
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("no CPU is available in the current affinity mask")
    if requested is None:
        return allowed[0]
    if requested not in allowed:
        raise ValueError(f"CPU {requested} is unavailable; allowed CPUs: {allowed}")
    return requested


def run_one(exe: Path, input_path: Path, run_dir: Path, cpu: int,
            dt: float, steps: int, timeout: float) -> tuple[float, list[str]]:
    run_dir.mkdir(parents=True, exist_ok=False)
    end_time = dt * steps
    command = [
        "taskset", "-c", str(cpu), str(exe),
        "-u", "1", "-b", "0", "--stellar-evolution", "2",
        "-s", f"{dt:.17g}", "-t", f"{end_time:.17g}",
        # One profile record per tree step, needed for per-step timings.
        "-o", f"{dt:.17g}", "-w", "3", "-a", "0",
        "--write-group-info", "0", str(input_path),
    ]
    (run_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1", OMP_DYNAMIC="false", OMP_PROC_BIND="true",
        OMP_PLACES="cores", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )
    start = time.perf_counter()
    with (run_dir / "output.log").open("w") as log:
        subprocess.run(command, cwd=run_dir, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=timeout)
    return time.perf_counter() - start, command


def read_step_profile(path: Path, n: int, repeat: int) -> list[dict]:
    """Read the single rank's per-output-interval PeTar timing profile."""
    with path.open() as stream:
        names = stream.readline().split()
        records = [line.split() for line in stream if line.strip()]
    hard_fields = ("PP_single", "PP_cluster", "PP_cross", "PP_intrpt*")
    required = ("Time", "N_steps", "Total", "AR_step_sum", "H4_step_sum", *hard_fields)
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{path} lacks profile columns: {missing}")

    index = {name: names.index(name) for name in required}
    rows = []
    previous_time = 0.0
    for sequence, values in enumerate(records, start=1):
        physical_time = float(values[index["Time"]])
        internal_steps = int(float(values[index["N_steps"]]))
        total_seconds = float(values[index["Total"]])
        ar_steps = int(float(values[index["AR_step_sum"]]))
        h4_steps = int(float(values[index["H4_step_sum"]]))
        adaptive_steps = ar_steps + h4_steps
        hard_seconds = sum(float(values[index[name]]) for name in hard_fields)
        rows.append({
            "n": n,
            "repeat": repeat,
            "profile_record": sequence,
            "physical_time_myr": f"{physical_time:.17g}",
            "selected_tree_dt_myr": f"{physical_time - previous_time:.17g}",
            "tree_steps_in_record": internal_steps,
            "total_step_seconds": f"{total_seconds:.12g}",
            "seconds_per_tree_step": f"{total_seconds / internal_steps:.12g}",
            "ar_adaptive_steps": ar_steps,
            "h4_adaptive_steps": h4_steps,
            "adaptive_steps_total": adaptive_steps,
            "hard_integration_seconds": f"{hard_seconds:.12g}",
            "seconds_per_adaptive_hard_step": (
                f"{hard_seconds / adaptive_steps:.12g}" if adaptive_steps else ""
            ),
        })
        previous_time = physical_time
    if not rows:
        raise ValueError(f"no timing records in {path}")
    return rows


def main(profile: str = "plummer") -> None:
    root = Path(__file__).resolve().parent
    default_exe = root.parent / "PeTar_new_version/build/petar.mpi.omp.avx2.bse"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=default_exe)
    parser.add_argument("--n", nargs="+", type=int, default=list(DEFAULT_N))
    parser.add_argument("--mass", type=float, default=1.0, help="mass of every star [Msun]")
    parser.add_argument("--rho0", type=float, default=1.0e5,
                        help="analytic central mass density [Msun/pc^3]")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--dt", type=float, default=2.0**-16, help="tree step [Myr]")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--cpu", type=int, help="logical CPU ID (default: first allowed CPU)")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace models and run directories")
    args = parser.parse_args()
    if profile not in ("plummer", "hernquist"):
        raise ValueError(f"unknown profile: {profile}")

    if args.generate_only and args.run_only:
        parser.error("--generate-only and --run-only are mutually exclusive")
    if any(n <= 0 for n in args.n) or args.mass <= 0 or args.rho0 <= 0:
        parser.error("N, mass, and rho0 must be positive")
    if args.steps <= 0 or args.repeats <= 0 or args.dt <= 0:
        parser.error("steps, repeats, and dt must be positive")

    model_dir = root / "models"
    run_root = root / "runs" / profile
    result_path = root / f"results_{profile}.csv"
    step_result_path = root / f"step_results_{profile}.csv"
    summary_path = root / f"summary_{profile}.json"
    metadata = []
    for n in args.n:
        path = model_dir / f"{profile}_N{n}.input"
        if not args.run_only and (args.force or not path.exists()):
            print(f"Generating N={n:,}: {path}", flush=True)
            generator = make_model if profile == "plummer" else make_cusp_model
            metadata.append(generator(path, n, args.mass, args.rho0, args.seed + n))
        elif path.exists():
            metadata.append({"n": n, "input": str(path), "reused": True})
        else:
            raise FileNotFoundError(f"missing model for --run-only: {path}")
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / f"{profile}_models.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.generate_only:
        return

    exe = args.exe.resolve(strict=True)
    if not os.access(exe, os.X_OK):
        raise PermissionError(f"not executable: {exe}")
    if shutil.which("taskset") is None:
        raise RuntimeError("taskset is required to enforce single-core execution")
    cpu = choose_cpu(args.cpu)
    if args.force and run_root.exists():
        # Only this script's narrowly scoped generated run tree is removed.
        shutil.rmtree(run_root)

    fields = ["n", "repeat", "wall_seconds", "cpu", "mpi_ranks", "omp_threads",
              "dt_myr", "steps", "end_time_myr", "profile", "stellar_evolution",
              "status", "error",
              "profiled_tree_steps", "profiled_step_seconds"]
    rows: list[dict] = []
    step_rows: list[dict] = []
    run_root.mkdir(parents=True, exist_ok=True)
    for n in args.n:
        input_path = (model_dir / f"{profile}_N{n}.input").resolve()
        for repeat in range(1, args.repeats + 1):
            run_dir = run_root / f"N{n}_repeat{repeat}"
            if run_dir.exists():
                raise FileExistsError(f"run directory exists (use --force): {run_dir}")
            row = dict(n=n, repeat=repeat, cpu=cpu, mpi_ranks=1, omp_threads=1,
                       dt_myr=args.dt, steps=args.steps, end_time_myr=args.dt * args.steps,
                       profile=profile, stellar_evolution=2,
                       status="failed", wall_seconds="", error="",
                       profiled_tree_steps="", profiled_step_seconds="")
            print(f"Running N={n:,}, repeat={repeat}/{args.repeats} on CPU {cpu}", flush=True)
            try:
                wall, _ = run_one(exe, input_path, run_dir, cpu, args.dt, args.steps, args.timeout)
                case_steps = read_step_profile(run_dir / "data.prof.rank.0", n, repeat)
                step_rows.extend(case_steps)
                row.update(status="ok", wall_seconds=f"{wall:.9f}",
                           profiled_tree_steps=sum(x["tree_steps_in_record"] for x in case_steps),
                           profiled_step_seconds=f"{sum(float(x['total_step_seconds']) for x in case_steps):.12g}")
            except Exception as error:
                row["error"] = repr(error)
                print(f"FAILED: {error}", flush=True)
            rows.append(row)
            with result_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            if step_rows:
                with step_result_path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(step_rows[0]))
                    writer.writeheader()
                    writer.writerows(step_rows)

    summary = []
    for n in args.n:
        values = [float(row["wall_seconds"]) for row in rows
                  if row["n"] == n and row["status"] == "ok"]
        n_steps = [int(row["profiled_tree_steps"]) for row in rows
                   if row["n"] == n and row["status"] == "ok"]
        step_times = [float(step["seconds_per_tree_step"]) for step in step_rows
                      if step["n"] == n]
        summary.append({"n": n, "successful_repeats": len(values),
                        "median_wall_seconds": statistics.median(values) if values else None,
                        "profiled_tree_steps_per_repeat": n_steps,
                        "median_profiled_seconds_per_tree_step": (
                            statistics.median(step_times) if step_times else None),
                        "wall_seconds": values})
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if any(row["status"] != "ok" for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
