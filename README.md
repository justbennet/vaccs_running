# VACC's Running?

A colorful terminal UI for checking your jobs on the Vermont Advanced Computing
Cluster. It is intentionally small: it uses Python's standard `curses` module
and the Slurm commands already available on VACC.

## Quick Start

From this directory:

```bash
./vaccs-running
```

For a plain one-shot table without opening the TUI:

```bash
./vaccs-running --once
```

The TUI auto-refreshes the active view every 0.25 seconds by default. To change
that:

```bash
./vaccs-running --refresh 2
```

## Keys

- `j`: show your jobs
- `n`: show all node availability and load
- `g`: in node view, toggle GPU nodes only
- `f`: in node view, toggle nodes with free GPUs only
- `down`: move down
- `k` / `up`: move up
- `left` / `right`: jump one visible page up or down
- `d`: show `scontrol show job` or `scontrol show node` details
- `p`: in node view, peek at all jobs running on the selected node
- `e`: show `my_job_statistics` output, falling back to `seff`
- `q`: quit

## Install As A Command

If you want `vaccs-running` on your path:

```bash
python3 -m pip install --user .
vaccs-running
```

## Why Slurm?

VACC uses Slurm for job scheduling. This app reads from `squeue`, shows selected
job details with `scontrol`, reads node load and resource allocation from
`scontrol show node`, and shows historical job efficiency with VACC's
`my_job_statistics` helper when available.

Run the app from a VACC login node. It executes Slurm commands directly.

In the node view:

- `CPU` is allocated CPUs over total CPUs.
- `MEM` is Slurm allocated memory over configured node memory.
- `GPU` is allocated GPUs over total GPUs, when Slurm reports GPU resources.
