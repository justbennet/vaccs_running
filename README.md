# VACC's Running?

A colorful terminal UI for checking your jobs on the Vermont Advanced Computing Cluster.

> This project is not affiliated in any way with UVM, VACC, or the Complex Systems Center.

## Quick Start

From this directory:

```bash
./vaccs-running
```

The TUI auto-refreshes the active view every 0.25 seconds by default. To change
that:

```bash
./vaccs-running --refresh 2
```

## Install As A Command

If you want `vaccs-running` on your path:

```bash
python3 -m pip install --user .
vaccs-running
```
