# Single-core Plummer speed test

`plummer_single_core_benchmark.py` generates an equal-mass Plummer-model grid
starting at 1,000 particles, increasing by a factor of three, and ending at
the exact target of 1,000,000 particles:

```text
1000, 3000, 9000, 27000, 81000, 243000, 729000, 1000000
```

Every star has mass `1 Msun` and BSE stellar type `kw=1`. The Plummer scale
radius is chosen separately for each N so that the analytic central density is
exactly `rho(0) = 1e5 Msun/pc^3`. Supplying `--n` explicitly overrides the
default grid.

The run uses `--stellar-evolution 2`: SSE/BSE evolution is active, together
with PeTar's mode-2 dynamical tides and hyperbolic gravitational-wave losses.
There are no primordial binaries (`-b 0`), although binaries may form
dynamically.

## Central-cusp benchmark

Use the separate entry point below for a centrally cusped model:

```bash
./cusp_single_core_benchmark.py --force
```

It generates isotropic Hernquist models with `rho ~ r^-1` at the centre.
For each N, total mass, equal stellar mass, and 3D half-mass radius match the
corresponding Plummer benchmark. Velocities are sampled from the isotropic
Hernquist distribution function. This exposes the extra neighbour and hard
integration work caused by central concentration without changing the global
cluster size.

The models use independent output names and run directories:

| Model | Results | Per-step results | Summary | Runs |
|---|---|---|---|---|
| Plummer | `results_plummer.csv` | `step_results_plummer.csv` | `summary_plummer.json` | `runs/plummer/` |
| Hernquist | `results_hernquist.csv` | `step_results_hernquist.csv` | `summary_hernquist.json` | `runs/hernquist/` |

Thus `--force` replaces only the selected model's outputs and does not remove
the other model's runs.

Run the complete benchmark:

```bash
cd /home/joojie/PeTar_source/speedtest
./plummer_single_core_benchmark.py
```

The PeTar process is pinned with `taskset` and uses one MPI rank (the MPI
executable is launched directly) and one OpenMP thread. Each case integrates
the same short physical interval: five steps of `2^-16 Myr`. Three repetitions
are run by default. Run-level wall times and profiled step counts are in the
model-named results CSV; medians are in the model-named summary JSON.

Each `step_results_<model>.csv` contains one row per tree-step output interval. It
records physical time, the selected tree-step interval, the measured `Total`
time, time per tree step, AR/H4 adaptive hard-step counts, hard-integration
time, and hard-integration seconds per adaptive hard step. The last quantity is
blank when no adaptive hard step occurred. PeTar's profile reports aggregate
adaptive hard-step counts rather than every individual AR/H4 physical step
size; the script does not claim those unreported step sizes.

Plot total time against particle number after the benchmark finishes:

```bash
./plot_total_time.py
```

This reads both model result files and creates `model_comparison.png`,
`model_comparison.pdf`, and `model_comparison.csv`. Both median profiled
integration-time curves appear in the same plot. One theoretical `C N ln(N)`
curve is normalized to whichever measured point has the largest time across
both datasets. The comparison CSV records measured and theoretical values.

Useful alternatives:

```bash
# Generate inputs without running PeTar
./plummer_single_core_benchmark.py --generate-only

# Use another executable, logical CPU, or equal stellar mass
./plummer_single_core_benchmark.py --exe /path/to/petar --cpu 3 --mass 0.7 --force

# Quick smoke test
./plummer_single_core_benchmark.py --n 1000 --steps 1 --repeats 1 --force
```

`--force` replaces generated input files and removes only the selected model's
run directory. Generation time is not included in the reported wall-clock time.
