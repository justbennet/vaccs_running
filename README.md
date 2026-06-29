# VACC's Running?

A colorful terminal UI for checking your jobs on the Vermont Advanced Computing Cluster and viewing the node availability.

> This project is not affiliated in any way with UVM, VACC, or the Vermont Complex Systems Institute.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/deringezgin/vaccs_running.git
cd vaccs_running
```

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
cd vaccs_running
python3 -m pip install --user .
vaccs-running
```
