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
class JobGroup:
    array_parent: str
    name: str
    total: int
    completed: int
    running: int
    pending: int
    failed: int
    other: int
    longest_running_elapsed: str
    limit: str

    @property
    def done_text(self) -> str:
        return f"{self.completed}/{self.total}"

    @property
    def dominant_state(self) -> str:
        if self.running:
            return "RUNNING"
        if self.pending:
            return "PENDING"
        if self.failed:
            return "FAILED"
        if self.completed == self.total and self.total:
            return "COMPLETED"
        return "UNKNOWN"


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


@dataclass(frozen=True)
class UserUsage:
    user: str
    tasks: int
    cpus: int
    gpus: int
    memory_mb: int | None = None


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
            [
                "squeue",
                "--array",
                "-h",
                "-u",
                self.user,
                "-t",
                "all",
                "-o",
                SQUEUE_FORMAT,
            ],
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

    def cluster_usage(self) -> str:
        output = self.runner.run(["scontrol", "show", "job"], timeout=20.0)
        usage = parse_scontrol_job_usage(output)
        return format_user_usage(aggregate_user_usage(usage))


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


def group_jobs(jobs: Iterable[Job]) -> list[JobGroup]:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for job in jobs:
        key = (job.array_parent, job.name)
        group = groups.setdefault(
            key,
            {
                "array_parent": job.array_parent,
                "name": job.name,
                "total": 0,
                "completed": 0,
                "running": 0,
                "pending": 0,
                "failed": 0,
                "other": 0,
                "longest_running_elapsed": "-",
                "longest_running_seconds": -1,
                "limit": job.limit or "-",
            },
        )
        group["total"] = int(group["total"]) + 1
        state = job.state.upper()
        if state == "COMPLETED":
            group["completed"] = int(group["completed"]) + 1
        elif state == "RUNNING":
            group["running"] = int(group["running"]) + 1
            seconds = parse_elapsed_seconds(job.elapsed)
            if seconds > int(group["longest_running_seconds"]):
                group["longest_running_seconds"] = seconds
                group["longest_running_elapsed"] = job.elapsed or "-"
        elif state == "PENDING":
            group["pending"] = int(group["pending"]) + 1
        elif state in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}:
            group["failed"] = int(group["failed"]) + 1
        else:
            group["other"] = int(group["other"]) + 1

    return [
        JobGroup(
            array_parent=str(group["array_parent"]),
            name=str(group["name"]),
            total=int(group["total"]),
            completed=int(group["completed"]),
            running=int(group["running"]),
            pending=int(group["pending"]),
            failed=int(group["failed"]),
            other=int(group["other"]),
            longest_running_elapsed=str(group["longest_running_elapsed"]),
            limit=str(group["limit"]),
        )
        for group in groups.values()
    ]


def aggregate_user_usage(tasks: Iterable[dict[str, str]]) -> list[UserUsage]:
    usage: dict[str, dict[str, int | bool]] = {}
    for task in tasks:
        user = task.get("user") or "unknown"
        row = usage.setdefault(
            user,
            {"tasks": 0, "cpus": 0, "gpus": 0, "memory_mb": 0, "has_memory": False},
        )
        row["tasks"] = int(row["tasks"]) + 1
        row["cpus"] = int(row["cpus"]) + parse_int(task.get("cpus", ""))
        row["gpus"] = int(row["gpus"]) + parse_gpu_count(task.get("tres", ""))
        memory_mb = parse_memory_mb(task.get("memory", ""))
        if memory_mb is not None:
            row["memory_mb"] = int(row["memory_mb"]) + memory_mb
            row["has_memory"] = True

    summaries = [
        UserUsage(
            user=user,
            tasks=int(row["tasks"]),
            cpus=int(row["cpus"]),
            gpus=int(row["gpus"]),
            memory_mb=int(row["memory_mb"]) if row["has_memory"] else None,
        )
        for user, row in usage.items()
    ]
    return sorted(
        summaries,
        key=lambda row: (-row.gpus, -row.cpus, -row.tasks, row.user),
    )


def format_user_usage(usage: list[UserUsage]) -> str:
    if not usage:
        return "No running tasks found."

    total_tasks = sum(row.tasks for row in usage)
    total_cpus = sum(row.cpus for row in usage)
    total_gpus = sum(row.gpus for row in usage)
    show_memory = any(row.memory_mb is not None for row in usage)
    total_memory = sum(row.memory_mb or 0 for row in usage)
    columns = [
        ("user", "USER"),
        ("tasks", "TASKS"),
        ("cpus", "CPUS"),
        ("gpus", "GPUS"),
    ]
    if show_memory:
        columns.append(("memory", "RAM_ALLOC"))

    rows: list[dict[str, str]] = []
    for row in usage:
        values = {
            "user": row.user,
            "tasks": str(row.tasks),
            "cpus": str(row.cpus),
            "gpus": str(row.gpus),
        }
        if show_memory:
            values["memory"] = (
                human_mb(row.memory_mb) if row.memory_mb is not None else "-"
            )
        rows.append(values)

    total = {
        "user": "TOTAL",
        "tasks": str(total_tasks),
        "cpus": str(total_cpus),
        "gpus": str(total_gpus),
    }
    if show_memory:
        total["memory"] = human_mb(total_memory)
    rows.append(total)

    widths = [
        max(len(label), *(len(row[key]) for row in rows))
        for key, label in columns
    ]
    header = "  ".join(label.ljust(width) for (_, label), width in zip(columns, widths))
    divider = "-" * len(header)
    body = [
        "  ".join(row[key].ljust(width) for (key, _), width in zip(columns, widths))
        for row in rows
    ]
    people = "person" if len(usage) == 1 else "people"
    tasks_word = "task" if total_tasks == 1 else "tasks"
    title = f"{len(usage)} {people} running {total_tasks} {tasks_word}"
    return "\n".join([title, "", header, divider, *body])


def parse_scontrol_job_usage(output: str) -> list[dict[str, str]]:
    usage: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("JobId="):
            append_job_usage(usage, current)
            current = {}
        current.update(parse_key_values(stripped))

    append_job_usage(usage, current)
    return usage


def append_job_usage(
    usage: list[dict[str, str]],
    fields: dict[str, str],
) -> None:
    if not fields or fields.get("JobState", "").upper() != "RUNNING":
        return
    tres = fields.get("AllocTRES") or fields.get("ReqTRES", "")
    usage.append(
        {
            "job_id": fields.get("JobId", ""),
            "user": parse_user_id(fields.get("UserId", "")),
            "cpus": parse_tres_value(tres, "cpu") or fields.get("NumCPUs", ""),
            "tres": tres,
            "memory": parse_tres_value(tres, "mem") or fields.get("MinMemoryNode", ""),
        }
    )


def parse_user_id(value: str) -> str:
    if not value:
        return ""
    return value.split("(", 1)[0]


def parse_tres_value(tres: str, key: str) -> str:
    for part in tres.split(","):
        name, separator, value = part.partition("=")
        if separator and name == key:
            return value
    return ""


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


def parse_gpu_count(value: str) -> int:
    return sum(
        int(match)
        for match in re.findall(r"(?:gres/)?gpu(?::[^,;:=()]+)*[:=](\d+)", value)
    )


def parse_memory_mb(value: str) -> int | None:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "NONE", "(NULL)"}:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTkmgt]?)([cnCN]?)", stripped)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper()
    multiplier = {
        "": 1,
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024 * 1024,
    }[unit]
    memory_mb = int(amount * multiplier)
    return memory_mb if memory_mb > 0 else None


def parse_elapsed_seconds(value: str) -> int:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "UNLIMITED"}:
        return -1
    days = 0
    time_part = stripped
    if "-" in stripped:
        day_part, time_part = stripped.split("-", 1)
        days = parse_int(day_part)
    pieces = time_part.split(":")
    if len(pieces) == 2:
        hours = 0
        minutes, seconds = pieces
    elif len(pieces) == 3:
        hours, minutes, seconds = pieces
    else:
        return -1
    return (
        days * 24 * 60 * 60
        + parse_int(hours) * 60 * 60
        + parse_int(minutes) * 60
        + parse_int(seconds)
    )


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
