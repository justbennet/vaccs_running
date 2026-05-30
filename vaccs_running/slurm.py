from __future__ import annotations

from dataclasses import dataclass
import getpass
import os
import re
import shlex
import subprocess
from typing import Iterable


SQUEUE_FIELDS = [
    "job_id",
    "name",
    "state",
    "partition",
    "nodes",
    "reason",
    "elapsed",
    "limit",
    "node_count",
    "cpus",
    "gres",
    "submit_time",
    "start_time",
]

SQUEUE_FORMAT = "%i|%j|%T|%P|%N|%R|%M|%l|%D|%C|%b|%V|%S"
NODE_JOBS_FIELDS = [
    "job_id",
    "user",
    "state",
    "elapsed",
    "cpus",
    "gres",
    "name",
]
NODE_JOBS_FORMAT = "%i|%u|%T|%M|%C|%b|%j"


class SlurmError(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    job_id: str
    name: str
    state: str
    partition: str
    nodes: str
    reason: str
    elapsed: str
    limit: str
    node_count: str
    cpus: str
    gres: str
    submit_time: str
    start_time: str

    @property
    def array_parent(self) -> str:
        return self.job_id.split("_", 1)[0]

    @property
    def is_running(self) -> bool:
        return self.state.upper() == "RUNNING"

    @property
    def location(self) -> str:
        if self.nodes and self.nodes not in {"(null)", "N/A"}:
            return self.nodes
        if self.reason and self.reason not in {"None", "N/A"}:
            return f"pending: {self.reason}"
        return "-"


@dataclass(frozen=True)
class Node:
    name: str
    state: str
    partitions: str
    cpu_alloc: int
    cpu_total: int
    cpu_load: float
    real_memory_mb: int
    alloc_memory_mb: int
    free_memory_mb: int
    gres: str
    alloc_tres: str
    features: str

    @property
    def base_state(self) -> str:
        return re.split(r"[+~-]", self.state, maxsplit=1)[0].upper() or "UNKNOWN"

    @property
    def free_cpus(self) -> int:
        return max(0, self.cpu_total - self.cpu_alloc)

    @property
    def cpu_percent(self) -> float:
        if not self.cpu_total:
            return 0.0
        return 100.0 * self.cpu_alloc / self.cpu_total

    @property
    def memory_percent(self) -> float:
        if not self.real_memory_mb:
            return 0.0
        return 100.0 * self.alloc_memory_mb / self.real_memory_mb

    @property
    def gpu_total(self) -> int:
        return sum(int(match) for match in re.findall(r"gpu(?::[^:,]+)*:(\d+)", self.gres))

    @property
    def has_gpus(self) -> bool:
        return self.gpu_total > 0

    @property
    def gpu_alloc(self) -> int:
        match = re.search(r"(?:^|,)gres/gpu=(\d+)", self.alloc_tres)
        if not match:
            return 0
        return int(match.group(1))

    @property
    def gpu_free(self) -> int:
        return max(0, self.gpu_total - self.gpu_alloc)

    @property
    def gpu_text(self) -> str:
        if self.gpu_total == 0:
            return "-"
        return f"{self.gpu_alloc}/{self.gpu_total}"

    @property
    def memory_text(self) -> str:
        return f"{human_mb(self.alloc_memory_mb)}/{human_mb(self.real_memory_mb)}"


class CommandRunner:
    def run(self, args: Iterable[str], timeout: float = 12.0) -> str:
        argv = list(args)

        try:
            proc = subprocess.run(
                argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except OSError as exc:
            command = " ".join(shlex.quote(part) for part in argv)
            raise SlurmError(f"could not run {command}: {exc}") from exc
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            command = " ".join(shlex.quote(part) for part in argv)
            raise SlurmError(stderr or f"command failed: {command}")
        return proc.stdout


class SlurmClient:
    def __init__(self, user: str | None = None):
        self.user = user or os.environ.get("USER") or getpass.getuser()
        self.runner = CommandRunner()

    def fetch_jobs(self) -> list[Job]:
        output = self.runner.run(
            ["squeue", "--array", "-h", "-u", self.user, "-o", SQUEUE_FORMAT],
            timeout=15.0,
        )
        jobs: list[Job] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            jobs.append(parse_squeue_line(line))
        return jobs

    def fetch_nodes(self) -> list[Node]:
        output = self.runner.run(["scontrol", "show", "node"], timeout=20.0)
        return parse_scontrol_nodes(output)

    def show_job(self, job_id: str) -> str:
        return self.runner.run(["scontrol", "show", "job", job_id], timeout=12.0)

    def show_node(self, node_name: str) -> str:
        return self.runner.run(["scontrol", "show", "node", node_name], timeout=12.0)

    def node_jobs(self, node_name: str) -> str:
        output = self.runner.run(
            ["squeue", "-a", "-h", "-w", node_name, "-o", NODE_JOBS_FORMAT],
            timeout=12.0,
        )
        body = output.strip()
        if not body:
            return f"No jobs found on {node_name}."
        jobs = [parse_node_job_line(line) for line in body.splitlines() if line.strip()]
        return format_node_jobs(jobs)

    def job_statistics(self, job_id: str) -> str:
        try:
            return self.runner.run(["my_job_statistics", job_id], timeout=20.0)
        except SlurmError as first_error:
            try:
                return self.runner.run(["seff", job_id], timeout=20.0)
            except SlurmError:
                raise first_error


def parse_squeue_line(line: str) -> Job:
    parts = line.rstrip("\n").split("|")
    if len(parts) < len(SQUEUE_FIELDS):
        parts.extend([""] * (len(SQUEUE_FIELDS) - len(parts)))
    elif len(parts) > len(SQUEUE_FIELDS):
        head = parts[: len(SQUEUE_FIELDS) - 1]
        tail = "|".join(parts[len(SQUEUE_FIELDS) - 1 :])
        parts = [*head, tail]

    values = {field: value.strip() for field, value in zip(SQUEUE_FIELDS, parts)}
    return Job(**values)


def parse_node_job_line(line: str) -> dict[str, str]:
    parts = line.rstrip("\n").split("|")
    if len(parts) < len(NODE_JOBS_FIELDS):
        parts.extend([""] * (len(NODE_JOBS_FIELDS) - len(parts)))
    elif len(parts) > len(NODE_JOBS_FIELDS):
        head = parts[: len(NODE_JOBS_FIELDS) - 1]
        tail = "|".join(parts[len(NODE_JOBS_FIELDS) - 1 :])
        parts = [*head, tail]
    return {
        field: value.strip()
        for field, value in zip(NODE_JOBS_FIELDS, parts)
    }


def format_node_jobs(jobs: list[dict[str, str]]) -> str:
    columns = [
        ("job_id", "JOBID", 12),
        ("user", "USER", 10),
        ("state", "STATE", 8),
        ("elapsed", "ELAPSED", 8),
        ("cpus", "CPUS", 4),
        ("gres", "GRES", 12),
        ("name", "JOB", 18),
    ]
    widths = []
    for key, label, minimum in columns:
        widths.append(max(minimum, len(label), *(len(job.get(key, "")) for job in jobs)))

    header = "  ".join(label.ljust(width) for (_, label, _), width in zip(columns, widths))
    divider = "-" * len(header)
    rows = [
        "  ".join(job.get(key, "").ljust(width) for (key, _, _), width in zip(columns, widths))
        for job in jobs
    ]
    return "\n".join([header, divider, *rows])


def summarize_jobs(jobs: Iterable[Job]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for job in jobs:
        key = job.state.upper() or "UNKNOWN"
        summary[key] = summary.get(key, 0) + 1
    return summary


def parse_scontrol_nodes(output: str) -> list[Node]:
    nodes: list[Node] = []
    current: dict[str, str] = {}

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("NodeName=") and current:
            nodes.append(node_from_fields(current))
            current = {}
        current.update(parse_key_values(stripped))

    if current:
        nodes.append(node_from_fields(current))
    return nodes


def parse_key_values(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def node_from_fields(fields: dict[str, str]) -> Node:
    return Node(
        name=fields.get("NodeName", ""),
        state=fields.get("State", ""),
        partitions=fields.get("Partitions", ""),
        cpu_alloc=parse_int(fields.get("CPUAlloc", "")),
        cpu_total=parse_int(fields.get("CPUTot", "")),
        cpu_load=parse_float(fields.get("CPULoad", "")),
        real_memory_mb=parse_int(fields.get("RealMemory", "")),
        alloc_memory_mb=parse_int(fields.get("AllocMem", "")),
        free_memory_mb=parse_int(fields.get("FreeMem", "")),
        gres=fields.get("Gres", ""),
        alloc_tres=fields.get("AllocTRES", ""),
        features=fields.get("ActiveFeatures") or fields.get("AvailableFeatures", ""),
    )


def parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def human_mb(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f}T"
    if value >= 1024:
        return f"{value / 1024:.0f}G"
    return f"{value}M"


def summarize_nodes(nodes: Iterable[Node]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for node in nodes:
        key = node.base_state
        summary[key] = summary.get(key, 0) + 1
    return summary
