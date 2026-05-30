from __future__ import annotations

import curses
from collections.abc import Callable
import textwrap
import time
from dataclasses import dataclass

from .slurm import Job, Node, SlurmClient, SlurmError, summarize_jobs, summarize_nodes


STATE_COLORS = {
    "RUNNING": 1,
    "PENDING": 2,
    "COMPLETED": 3,
    "FAILED": 4,
    "CANCELLED": 4,
    "TIMEOUT": 4,
}

NODE_COLORS = {
    "IDLE": 1,
    "MIXED": 2,
    "ALLOCATED": 3,
    "DOWN": 4,
    "DRAIN": 4,
    "DRAINED": 4,
}

BORDER_PAIR = 9
TEXT_PAIR = 10
ACTIVE_TAB_PAIR = 11
MUTED_PAIR = 13
SURFACE_PAIR = 14


@dataclass
class AppState:
    jobs: list[Job]
    nodes: list[Node]
    view: str = "jobs"
    selected: int = 0
    scroll: int = 0
    message: str = ""
    last_refresh: float = 0.0
    gpu_nodes_only: bool = False
    free_gpu_only: bool = False


class VaccsRunningApp:
    def __init__(
        self,
        client: SlurmClient,
        refresh_seconds: float,
        initial_view: str = "jobs",
    ):
        self.client = client
        self.refresh_seconds = refresh_seconds
        self.state = AppState(
            jobs=[],
            nodes=[],
            view="nodes" if initial_view == "nodes" else "jobs",
        )
        self.colors_enabled = False

    def run(self) -> None:
        curses.wrapper(self._main)

    def _main(self, stdscr: curses.window) -> None:
        safe_curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        self._init_colors()
        self._refresh_current()

        while True:
            self._draw(stdscr)
            key = stdscr.getch()
            if key != -1 and not self._handle_key(stdscr, key):
                return

            now = time.monotonic()
            if (
                self.refresh_seconds
                and now - self.state.last_refresh >= self.refresh_seconds
            ):
                self._refresh_current()

            if key == -1:
                time.sleep(0.05)

    def _init_colors(self) -> None:
        try:
            curses.start_color()
            curses.use_default_colors()
            orange = self._orange_color()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_CYAN, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)
            curses.init_pair(5, orange, -1)
            curses.init_pair(6, orange, -1)
            curses.init_pair(7, curses.COLOR_BLACK, orange)
            curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_RED)
            curses.init_pair(BORDER_PAIR, orange, -1)
            curses.init_pair(TEXT_PAIR, curses.COLOR_WHITE, -1)
            curses.init_pair(ACTIVE_TAB_PAIR, curses.COLOR_BLACK, orange)
            curses.init_pair(MUTED_PAIR, curses.COLOR_WHITE, -1)
            curses.init_pair(SURFACE_PAIR, curses.COLOR_WHITE, -1)
            self.colors_enabled = True
        except curses.error:
            self.colors_enabled = False

    def _orange_color(self) -> int:
        if curses.COLORS > 16 and curses.can_change_color():
            try:
                curses.init_color(16, 851, 467, 341)  # #D97757
                return 16
            except curses.error:
                pass
        return 173 if curses.COLORS > 173 else curses.COLOR_YELLOW

    def _refresh_current(self) -> None:
        if self.state.view == "nodes":
            message = self._refresh_nodes()
        else:
            message = self._refresh_jobs()
        self.state.last_refresh = time.monotonic()
        self.state.message = f"refreshed {message}"
        self._clamp_selection()

    def _refresh_jobs(self) -> str:
        try:
            self.state.jobs = self.client.fetch_jobs()
            return f"{len(self.state.jobs)} jobs"
        except SlurmError as exc:
            return f"jobs: {exc}"

    def _refresh_nodes(self) -> str:
        try:
            self.state.nodes = self.client.fetch_nodes()
            return f"{len(self.state.nodes)} nodes"
        except SlurmError as exc:
            return f"nodes: {exc}"

    def _visible_jobs(self) -> list[Job]:
        return self.state.jobs

    def _visible_nodes(self) -> list[Node]:
        visible = self.state.nodes
        if self.state.gpu_nodes_only:
            visible = [node for node in visible if node.has_gpus]
        if self.state.free_gpu_only:
            visible = [node for node in visible if node.gpu_free > 0]
        return visible

    def _visible_count(self) -> int:
        if self.state.view == "nodes":
            return len(self._visible_nodes())
        return len(self._visible_jobs())

    def _handle_key(self, stdscr: curses.window, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == curses.KEY_DOWN:
            self.state.selected += 1
        elif key in (curses.KEY_UP, ord("k")):
            self.state.selected -= 1
        elif key == curses.KEY_NPAGE:
            self.state.selected += 10
        elif key == curses.KEY_PPAGE:
            self.state.selected -= 10
        elif key == curses.KEY_RIGHT:
            self._jump_page(stdscr, 1)
        elif key == curses.KEY_LEFT:
            self._jump_page(stdscr, -1)
        elif key == curses.KEY_HOME:
            self.state.selected = 0
        elif key == curses.KEY_END:
            self.state.selected = self._visible_count() - 1
        elif key == ord("n"):
            self._switch_view("nodes")
        elif key == ord("j"):
            self._switch_view("jobs")
        elif key == ord("g"):
            if self.state.view == "nodes":
                enabled = not self.state.gpu_nodes_only
                self.state.gpu_nodes_only = enabled
                if enabled:
                    self.state.free_gpu_only = False
                self.state.selected = 0
                self.state.scroll = 0
                state = "on" if self.state.gpu_nodes_only else "off"
                self.state.message = f"GPU node filter {state}"
        elif key == ord("f"):
            if self.state.view == "nodes":
                enabled = not self.state.free_gpu_only
                self.state.free_gpu_only = enabled
                if enabled:
                    self.state.gpu_nodes_only = False
                self.state.selected = 0
                self.state.scroll = 0
                state = "on" if self.state.free_gpu_only else "off"
                self.state.message = f"free GPU filter {state}"
        elif key == ord("d"):
            self._show_detail(stdscr)
        elif key == ord("p"):
            if self.state.view == "nodes":
                self._show_node_jobs(stdscr)
        elif key == ord("e"):
            if self.state.view == "jobs":
                self._show_efficiency(stdscr)

        self._clamp_selection()
        return True

    def _page_size(self, stdscr: curses.window) -> int:
        height, _ = stdscr.getmaxyx()
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        return max(1, table_height - 3)

    def _jump_page(self, stdscr: curses.window, direction: int) -> None:
        count = self._visible_count()
        if count == 0:
            self.state.selected = 0
            self.state.scroll = 0
            return

        page_size = self._page_size(stdscr)
        current_page_top = (self.state.selected // page_size) * page_size
        next_page_top = current_page_top + direction * page_size
        last_page_top = ((count - 1) // page_size) * page_size
        next_page_top = max(0, min(next_page_top, last_page_top))
        self.state.selected = next_page_top
        self.state.scroll = next_page_top

    def _switch_view(self, view: str) -> None:
        if self.state.view == view:
            return
        self.state.view = view
        self.state.selected = 0
        self.state.scroll = 0
        self._refresh_current()
        self._clamp_selection()

    def _clamp_selection(self) -> None:
        count = self._visible_count()
        if count == 0:
            self.state.selected = 0
            self.state.scroll = 0
            return
        self.state.selected = max(0, min(self.state.selected, count - 1))

    def _selected_job(self) -> Job | None:
        visible = self._visible_jobs()
        if not visible:
            return None
        self._clamp_selection()
        return visible[self.state.selected]

    def _selected_node(self) -> Node | None:
        visible = self._visible_nodes()
        if not visible:
            return None
        self._clamp_selection()
        return visible[self.state.selected]

    def _draw(self, stdscr: curses.window) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        self._draw_header(stdscr, width)
        if self.state.view == "nodes":
            self._draw_nodes_table(stdscr, self._visible_nodes(), height, width)
            self._draw_node_detail(stdscr, height, width)
        else:
            self._draw_jobs_table(stdscr, self._visible_jobs(), height, width)
            self._draw_job_detail(stdscr, height, width)
        stdscr.refresh()

    def _draw_header(self, stdscr: curses.window, width: int) -> None:
        title = " VACC's Running? "
        clock = time.strftime("%H:%M:%S")
        right = f"refresh {self.refresh_seconds:.2f}s  {clock}"
        self._draw_box(stdscr, 0, 0, 3, width)

        job_summary = summarize_jobs(self.state.jobs)
        node_summary = summarize_nodes(self.state.nodes)
        jobs = compact_summary(job_summary, ["RUNNING", "PENDING", "FAILED", "CANCELLED"])
        nodes = compact_summary(node_summary, ["IDLE", "MIXED", "ALLOCATED", "DOWN"])
        meta = f" jobs {jobs or 'none'}"
        if nodes:
            meta += f"   nodes {nodes}"
        self._addstr(stdscr, 1, 2, meta[: max(0, width - 4)], self._pair(MUTED_PAIR))
        title_x = max(1, (width - len(title)) // 2)
        self._addstr(stdscr, 1, title_x, title[: max(0, width - 2)], self._pair(5) | curses.A_BOLD)
        if width > len(right) + 2:
            self._addstr(stdscr, 1, width - len(right) - 2, right, self._pair(MUTED_PAIR))
        jobs_attr = self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD if self.state.view == "jobs" else self._pair(MUTED_PAIR)
        nodes_attr = self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD if self.state.view == "nodes" else self._pair(MUTED_PAIR)
        self._addstr(stdscr, 3, 1, " j Jobs ", jobs_attr)
        self._addstr(stdscr, 3, 10, " n Nodes ", nodes_attr)
        if self.state.view == "nodes":
            x = 21
            gpu_filter_text = " g gpu-nodes "
            self._addstr(
                stdscr,
                3,
                x,
                gpu_filter_text,
                self._pair(ACTIVE_TAB_PAIR if self.state.gpu_nodes_only else MUTED_PAIR),
            )
            x += len(gpu_filter_text) + 1
            free_filter_text = " f free-gpu "
            self._addstr(
                stdscr,
                3,
                x,
                free_filter_text,
                self._pair(ACTIVE_TAB_PAIR if self.state.free_gpu_only else MUTED_PAIR),
            )
            x += len(free_filter_text) + 1
            self._addstr(stdscr, 3, x, " d detail  p peek  q quit", self._pair(MUTED_PAIR))
        else:
            self._addstr(stdscr, 3, 21, " d detail  q quit", self._pair(MUTED_PAIR))

    def _draw_jobs_table(
        self,
        stdscr: curses.window,
        visible: list[Job],
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        self._draw_box(stdscr, table_top, 0, table_height, width, " jobs ")
        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        row_values = [
            [
                job.job_id,
                job.state,
                job.partition,
                job.elapsed,
                job.limit,
                job.cpus,
                job.location,
            ]
            for job in visible
        ]
        columns = fit_columns(
            [
                ("JOBID", 10, 22),
                ("STATE", 8, 14),
                ("PARTITION", 10, 22),
                ("ELAPSED", 8, 14),
                ("LIMIT", 8, 14),
                ("CPUS", 4, 8),
                ("WHERE / WHY", 18, 56),
            ],
            row_values,
            max(1, width - 4),
        )
        headers = [
            ("JOBID", columns[0]),
            ("STATE", columns[1]),
            ("PARTITION", columns[2]),
            ("ELAPSED", columns[3]),
            ("LIMIT", columns[4]),
            ("CPUS", columns[5]),
            ("WHERE / WHY", columns[6]),
        ]
        x = 2
        for label, size in headers:
            self._addstr(stdscr, header_y, x, label[:size].ljust(size), self._pair(MUTED_PAIR) | curses.A_BOLD)
            x += size + 1

        if self.state.selected < self.state.scroll:
            self.state.scroll = self.state.selected
        if self.state.selected >= self.state.scroll + rows:
            self.state.scroll = self.state.selected - rows + 1
        page_label = page_status(self.state.selected, len(visible), rows)
        if page_label:
            footer = f" {page_label} "
            footer_x = max(2, width - len(footer) - 2)
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                footer_x,
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, job in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._state_attr(job.state)
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            cells = [
                (job.job_id, columns[0]),
                (job.state, columns[1]),
                (job.partition, columns[2]),
                (job.elapsed, columns[3]),
                (job.limit, columns[4]),
                (job.cpus, columns[5]),
                (job.location, columns[6]),
            ]
            x = 2
            for value, size in cells:
                text = value[:size].ljust(size)
                if x < width:
                    self._addstr(stdscr, screen_row, x, text[: max(0, width - x - 1)], attr)
                x += size + 1

    def _draw_nodes_table(
        self,
        stdscr: curses.window,
        visible: list[Node],
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        self._draw_box(stdscr, table_top, 0, table_height, width, " nodes ")
        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        row_values = []
        cpu_count_width = resource_count_width(
            [(node.cpu_alloc, node.cpu_total) for node in visible]
        )
        gpu_count_width = resource_text_width([node.gpu_text for node in visible])
        memory_count_width = resource_text_width(
            [node.memory_text for node in visible]
        )
        for node in visible:
            gpu_percent = pct(node.gpu_alloc, node.gpu_total)
            row_values.append(
                [
                    node.name,
                    node.state,
                    node.partitions,
                    resource_meter(
                        node.cpu_alloc,
                        node.cpu_total,
                        node.cpu_percent,
                        meter_width=16,
                        count_width=cpu_count_width,
                    ),
                    resource_text_meter(
                        node.memory_text,
                        node.memory_percent,
                        meter_width=14,
                        count_width=memory_count_width,
                    ),
                    resource_text_meter(
                        node.gpu_text,
                        gpu_percent,
                        meter_width=12,
                        count_width=gpu_count_width,
                    ),
                    node.gres,
                ]
            )
        columns = fit_columns(
            [
                ("NODE", 10, 22),
                ("STATE", 8, 18),
                ("PARTITION", 10, 22),
                ("CPU", 24, 38),
                ("MEM", 24, 38),
                ("GPU", 18, 30),
                ("GRES", 12, 48),
            ],
            row_values,
            max(1, width - 4),
        )
        headers = [
            ("NODE", columns[0]),
            ("STATE", columns[1]),
            ("PARTITION", columns[2]),
            ("CPU", columns[3]),
            ("MEM", columns[4]),
            ("GPU", columns[5]),
            ("GRES", columns[6]),
        ]
        x = 2
        for label, size in headers:
            self._addstr(stdscr, header_y, x, label[:size].ljust(size), self._pair(MUTED_PAIR) | curses.A_BOLD)
            x += size + 1

        if self.state.selected < self.state.scroll:
            self.state.scroll = self.state.selected
        if self.state.selected >= self.state.scroll + rows:
            self.state.scroll = self.state.selected - rows + 1
        page_label = page_status(self.state.selected, len(visible), rows)
        if page_label:
            footer = f" {page_label} "
            footer_x = max(2, width - len(footer) - 2)
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                footer_x,
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, node in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._node_attr(node)
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            gpu_percent = pct(node.gpu_alloc, node.gpu_total)
            cells = [
                (node.name, columns[0]),
                (node.state, columns[1]),
                (node.partitions, columns[2]),
                (
                    resource_meter(
                        node.cpu_alloc,
                        node.cpu_total,
                        node.cpu_percent,
                        meter_width=16,
                        count_width=cpu_count_width,
                    ),
                    columns[3],
                ),
                (
                    resource_text_meter(
                        node.memory_text,
                        node.memory_percent,
                        meter_width=14,
                        count_width=memory_count_width,
                    ),
                    columns[4],
                ),
                (
                    resource_text_meter(
                        node.gpu_text,
                        gpu_percent,
                        meter_width=12,
                        count_width=gpu_count_width,
                    ),
                    columns[5],
                ),
                (node.gres, columns[6]),
            ]
            x = 2
            for value, size in cells:
                text = value[:size].ljust(size)
                if x < width:
                    self._addstr(stdscr, screen_row, x, text[: max(0, width - x - 1)], attr)
                x += size + 1

    def _draw_job_detail(self, stdscr: curses.window, height: int, width: int) -> None:
        panel_height = min(8, max(4, height // 4))
        top = max(4, height - panel_height)
        job = self._selected_job()
        self._draw_box(stdscr, top, 0, panel_height, width, " selected job ")
        if not job:
            self._addstr(stdscr, top + 1, 2, "No jobs found.", self._pair(2))
            return

        lines = [
            f"{job.name}  job={job.job_id}  array-parent={job.array_parent}",
            f"state={job.state}  partition={job.partition}  nodes={job.nodes or '-'}",
            f"submitted={job.submit_time}  started={job.start_time}  reason={job.reason}",
            f"resources: nodes={job.node_count}  cpus={job.cpus}  gres={job.gres}",
        ]
        for offset, line in enumerate(lines):
            if top + 1 + offset < height - 1:
                self._addstr(stdscr, top + 1 + offset, 2, line[: max(0, width - 4)], self._state_attr(job.state))

    def _draw_node_detail(self, stdscr: curses.window, height: int, width: int) -> None:
        panel_height = min(8, max(4, height // 4))
        top = max(4, height - panel_height)
        node = self._selected_node()
        self._draw_box(stdscr, top, 0, panel_height, width, " selected node ")
        if not node:
            self._addstr(stdscr, top + 1, 2, "No nodes found.", self._pair(2))
            return

        gpu_percent = pct(node.gpu_alloc, node.gpu_total)
        lines = [
            f"{node.name}  state={node.state}  partition={node.partitions}",
            (
                f"cpu alloc={node.cpu_alloc}/{node.cpu_total} "
                f"free={node.free_cpus}  {meter(node.cpu_percent, 18)}  live-load={node.cpu_load:.2f}"
            ),
            (
                f"mem alloc={node.memory_text} ({node.memory_percent:.1f}%) "
                f"{meter(node.memory_percent, 18)}  free-os={node.free_memory_mb // 1024}G"
            ),
            f"gpu alloc={node.gpu_text} free={node.gpu_free}  {meter(gpu_percent, 18)}  tres={node.alloc_tres or '-'}",
            f"features={node.features}",
        ]
        for offset, line in enumerate(lines):
            if top + 1 + offset < height - 1:
                self._addstr(stdscr, top + 1 + offset, 2, line[: max(0, width - 4)], self._node_attr(node))

    def _state_attr(self, state: str) -> int:
        return self._pair(STATE_COLORS.get(state.upper(), 3))

    def _node_attr(self, node: Node) -> int:
        return self._pair(NODE_COLORS.get(node.base_state, 3))

    def _pair(self, pair_id: int) -> int:
        if not self.colors_enabled:
            return 0
        return curses.color_pair(pair_id)

    def _addstr(
        self,
        win: curses.window,
        y: int,
        x: int,
        text: str,
        attr: int = 0,
    ) -> None:
        max_y, max_x = win.getmaxyx()
        if y < 0 or y >= max_y or x < 0 or x >= max_x:
            return
        width = max_x - x
        if y == max_y - 1:
            width -= 1
        if width <= 0:
            return
        try:
            win.addstr(y, x, text[:width], attr)
        except curses.error:
            pass

    def _addch(
        self,
        win: curses.window,
        y: int,
        x: int,
        ch: int,
        attr: int = 0,
    ) -> None:
        max_y, max_x = win.getmaxyx()
        if y < 0 or y >= max_y or x < 0 or x >= max_x:
            return
        try:
            win.addch(y, x, ch, attr)
        except curses.error:
            pass

    def _draw_box(
        self,
        win: curses.window,
        top: int,
        left: int,
        height: int,
        width: int,
        title: str = "",
    ) -> None:
        if height < 2 or width < 2:
            return
        attr = self._pair(BORDER_PAIR) | curses.A_DIM
        right = left + width - 1
        bottom = top + height - 1
        self._addstr(win, top, left, "╭", attr)
        self._addstr(win, top, right, "╮", attr)
        self._addstr(win, bottom, left, "╰", attr)
        self._addstr(win, bottom, right, "╯", attr)
        for x in range(left + 1, right):
            self._addstr(win, top, x, "─", attr)
            self._addstr(win, bottom, x, "─", attr)
        for y in range(top + 1, bottom):
            self._addstr(win, y, left, "│", attr)
            self._addstr(win, y, right, "│", attr)
        if title:
            self._addstr(win, top, left + 2, title[: max(0, width - 4)], self._pair(5) | curses.A_BOLD)

    def _show_detail(self, stdscr: curses.window) -> None:
        if self.state.view == "nodes":
            node = self._selected_node()
            if not node:
                return
            self._popup_command(
                stdscr,
                f"scontrol show node {node.name}",
                self.client.show_node,
                node.name,
                close_keys=(ord("d"),),
            )
            return
        job = self._selected_job()
        if not job:
            return
        self._popup_command(
            stdscr,
            f"scontrol show job {job.job_id}",
            self.client.show_job,
            job.job_id,
            close_keys=(ord("d"),),
        )

    def _show_node_jobs(self, stdscr: curses.window) -> None:
        node = self._selected_node()
        if not node:
            return
        self._popup_command(
            stdscr,
            f"squeue -a -w {node.name}",
            self.client.node_jobs,
            node.name,
            close_keys=(ord("p"),),
        )

    def _show_efficiency(self, stdscr: curses.window) -> None:
        job = self._selected_job()
        if not job:
            return
        self._popup_command(
            stdscr,
            f"my_job_statistics {job.job_id}",
            self.client.job_statistics,
            job.job_id,
            close_keys=(ord("e"),),
        )

    def _popup_command(
        self,
        stdscr: curses.window,
        title: str,
        fn,
        job_id: str,
        close_keys: tuple[int, ...] = (),
    ) -> None:
        self._popup(
            stdscr,
            title,
            lambda: command_text(fn, job_id),
            close_keys=close_keys,
        )

    def _popup(
        self,
        stdscr: curses.window,
        title: str,
        text: str | Callable[[], str],
        close_keys: tuple[int, ...] = (),
    ) -> None:
        get_text = text if callable(text) else lambda: text
        current_text = get_text()
        last_refresh = time.monotonic()
        height, width = stdscr.getmaxyx()
        top, left, box_height, box_width = popup_geometry(height, width, title, current_text)
        win = curses.newwin(box_height, box_width, top, left)
        win.nodelay(True)
        win.keypad(True)
        scroll = 0
        wrapped = wrap_lines(current_text, box_width - 4)
        while True:
            now = time.monotonic()
            if self.refresh_seconds and now - last_refresh >= self.refresh_seconds:
                self._refresh_current()
                self._draw(stdscr)
                height, width = stdscr.getmaxyx()
                current_text = get_text()
                top, left, box_height, box_width = popup_geometry(height, width, title, current_text)
                win.resize(box_height, box_width)
                win.mvwin(top, left)
                wrapped = wrap_lines(current_text, box_width - 4)
                body_height = box_height - 4
                scroll = min(scroll, max(0, len(wrapped) - body_height))
                last_refresh = now

            win.erase()
            win.border()
            self._addstr(win, 0, 2, f" {title} ", self._pair(6) | curses.A_BOLD)
            body_height = box_height - 4
            for idx, line in enumerate(wrapped[scroll : scroll + body_height], start=2):
                self._addstr(win, idx, 2, line[: box_width - 4])
            footer = " up/down scroll  q/esc close "
            self._addstr(win, box_height - 1, 2, footer[: box_width - 4], self._pair(5))
            win.refresh()
            key = win.getch()
            if key == -1:
                time.sleep(0.05)
                continue
            if key in (ord("q"), 27, ord("\n"), *close_keys):
                return
            if key in (curses.KEY_DOWN, ord("j")):
                scroll = min(max(0, len(wrapped) - body_height), scroll + 1)
            elif key in (curses.KEY_UP, ord("k")):
                scroll = max(0, scroll - 1)
            elif key == curses.KEY_NPAGE:
                scroll = min(max(0, len(wrapped) - body_height), scroll + body_height)
            elif key == curses.KEY_PPAGE:
                scroll = max(0, scroll - body_height)


def command_text(fn, job_id: str) -> str:
    try:
        return fn(job_id).strip() or "No output."
    except SlurmError as exc:
        return str(exc)


def wrap_lines(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        if not line:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(line, width=width, replace_whitespace=False) or [""])
    return lines


def popup_geometry(
    screen_height: int,
    screen_width: int,
    title: str,
    text: str,
) -> tuple[int, int, int, int]:
    footer = " up/down scroll  q/esc close "
    max_box_width = max(20, screen_width - 8)
    max_box_height = max(6, screen_height - 4)
    longest_line = max((len(line) for line in text.splitlines()), default=0)
    content_width = max(len(title) + 4, len(footer), longest_line)
    box_width = min(max_box_width, max(40, content_width + 4))
    body_width = max(1, box_width - 4)
    wrapped = wrap_lines(text, body_width)
    box_height = min(max_box_height, max(8, len(wrapped) + 4))
    top = max(1, (screen_height - box_height) // 2)
    left = max(1, (screen_width - box_width) // 2)
    return top, left, box_height, box_width


def compact_summary(summary: dict[str, int], preferred: list[str]) -> str:
    bits: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if summary.get(key):
            bits.append(f"{key[:4]}:{summary[key]}")
            seen.add(key)
    for key, value in sorted(summary.items()):
        if key not in seen and value:
            bits.append(f"{key[:4]}:{value}")
    return " ".join(bits)


def meter(percent: float, width: int) -> str:
    bounded = max(0.0, min(100.0, percent))
    inner = max(1, width)
    filled = round(inner * bounded / 100.0)
    return "[" + "|" * filled + "." * (inner - filled) + "]"


def resource_count_width(pairs: list[tuple[int, int]]) -> int:
    return max((len(f"{used}/{total}") for used, total in pairs), default=0)


def resource_text_width(values: list[str]) -> int:
    return max((len(value) for value in values), default=0)


def resource_meter(
    used: int,
    total: int,
    percent: float,
    *,
    meter_width: int,
    count_width: int,
) -> str:
    count = f"{used}/{total}".rjust(count_width)
    return f"{count} {meter(percent, meter_width)}"


def resource_text_meter(
    text: str,
    percent: float,
    *,
    meter_width: int,
    count_width: int,
) -> str:
    return f"{text.rjust(count_width)} {meter(percent, meter_width)}"


def pct(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * value / total


def page_status(selected: int, total_items: int, page_size: int) -> str:
    if total_items <= 0 or page_size <= 0:
        return "0/0"
    page_count = (total_items + page_size - 1) // page_size
    current_page = min(page_count, max(0, selected) // page_size + 1)
    return f"{current_page}/{page_count}"


def fit_columns(
    specs: list[tuple[str, int, int]],
    rows: list[list[str]],
    available_width: int,
) -> list[int]:
    widths: list[int] = []
    for index, (label, min_width, max_width) in enumerate(specs):
        content_width = len(label)
        for row in rows:
            if index < len(row):
                content_width = max(content_width, len(row[index]))
        widths.append(min(max(content_width, min_width), max_width))

    gaps = max(0, len(widths) - 1)
    target = max(1, available_width - gaps)
    while sum(widths) > target:
        candidates = [
            index
            for index, width in enumerate(widths)
            if width > specs[index][1]
        ]
        if not candidates:
            break
        widest = max(candidates, key=lambda index: widths[index])
        widths[widest] -= 1
    return widths


def safe_curs_set(visibility: int) -> None:
    try:
        curses.curs_set(visibility)
    except curses.error:
        pass


def print_once(client: SlurmClient) -> None:
    jobs = client.fetch_jobs()
    summary = summarize_jobs(jobs)
    print("VACC's Running?")
    print(" ".join(f"{state}:{count}" for state, count in sorted(summary.items())) or "no jobs")
    print()
    print(f"{'JOBID':<16} {'STATE':<10} {'PARTITION':<14} {'ELAPSED':<10} {'LIMIT':<10} WHERE / WHY")
    print("-" * 90)
    for job in jobs:
        print(
            f"{job.job_id:<16} {job.state:<10} {job.partition:<14} "
            f"{job.elapsed:<10} {job.limit:<10} {job.location}"
        )
