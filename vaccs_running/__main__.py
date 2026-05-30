from __future__ import annotations

import argparse

from .slurm import SlurmClient, SlurmError
from .ui import VaccsRunningApp


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = SlurmClient(user=args.user)

    try:
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
