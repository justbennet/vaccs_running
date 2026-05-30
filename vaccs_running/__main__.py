from __future__ import annotations

import argparse
import sys

from .slurm import SlurmClient, SlurmError
from .ui import VaccsRunningApp, print_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaccs-running",
        description="Terminal UI for viewing and managing your VACC Slurm jobs.",
    )
    parser.add_argument(
        "--user",
        help="Slurm username to inspect. Defaults to the current VACC shell user.",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=0.25,
        help="Auto-refresh interval in seconds. Use 0 to disable. Default: 0.25.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print a one-shot job table and exit instead of opening the TUI.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = SlurmClient(user=args.user)

    try:
        if args.once:
            print_once(client)
            return 0

        if not sys.stdin.isatty() or not sys.stdout.isatty():
            print_once(client)
            return 0

        VaccsRunningApp(
            client=client,
            refresh_seconds=max(0.0, args.refresh),
        ).run()
        return 0
    except KeyboardInterrupt:
        return 130
    except SlurmError as exc:
        print(f"vaccs-running: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
