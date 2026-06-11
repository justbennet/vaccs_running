import curses
import unittest

from vaccs_running.slurm import Job, Node
from vaccs_running.ui import (
    VaccsRunningApp,
    command_text,
    page_status,
    popup_geometry,
    resource_count_width,
    resource_meter,
    resource_text_meter,
    resource_text_width,
    status_title,
    terminal_too_small,
    wrap_detail_lines,
)


class FakeClient:
    def fetch_jobs(self):
        return []

    def fetch_nodes(self):
        return []

    def node_jobs(self, node_name):
        return f"jobs for {node_name}"

    def cluster_usage(self):
        return "usage by user"


class FakeScreen:
    def __init__(self, height=64, width=120):
        self.height = height
        self.width = width
        self.writes = []
        self.erase_count = 0
        self.refresh_count = 0

    def getmaxyx(self):
        return self.height, self.width

    def addstr(self, y, x, text, attr=0):
        self.writes.append((y, x, text, attr))

    def erase(self):
        self.erase_count += 1

    def refresh(self):
        self.refresh_count += 1


class FakePopupWindow(FakeScreen):
    def __init__(self, keys):
        super().__init__(height=1, width=1)
        self.keys = list(keys)
        self.refresh_count = 0
        self.sizes = []
        self.positions = []

    def keypad(self, value):
        self.keypad_value = value

    def nodelay(self, value):
        self.nodelay_value = value

    def erase(self):
        pass

    def border(self):
        pass

    def refresh(self):
        self.refresh_count += 1

    def getch(self):
        return self.keys.pop(0) if self.keys else ord("q")

    def resize(self, height, width):
        self.height = height
        self.width = width
        self.sizes.append((height, width))

    def mvwin(self, top, left):
        self.positions.append((top, left))


def make_node(name, gres, alloc_tres="", state="IDLE"):
    return Node(
        name=name,
        state=state,
        partitions="nvgpu",
        cpu_alloc=0,
        cpu_total=1,
        cpu_load=0.0,
        real_memory_mb=1,
        alloc_memory_mb=0,
        free_memory_mb=1,
        gres=gres,
        alloc_tres=alloc_tres,
        features="",
    )


def make_job(job_id, state="RUNNING"):
    return Job(
        job_id=job_id,
        name="job",
        state=state,
        partition="nvgpu",
        nodes="h2node01",
        reason="",
        elapsed="0:01",
        limit="1:00:00",
        node_count="1",
        cpus="1",
        gres="",
        submit_time="",
        start_time="",
    )


class NodeFilterTests(unittest.TestCase):
    def test_header_draws_jobs_and_nodes_tabs_on_top_bar_left(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        screen = FakeScreen(height=12, width=100)

        app._draw_header(screen, 100)

        self.assertIn((1, 2, " j Jobs ", curses.A_BOLD), screen.writes)
        self.assertIn((1, 11, " n Nodes ", 0), screen.writes)

    def test_header_does_not_show_refresh_interval(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0.25)
        screen = FakeScreen(height=12, width=100)

        app._draw_header(screen, 100)

        written = " ".join(write[2] for write in screen.writes)
        self.assertNotIn("refresh", written)
        self.assertNotIn("0.25s", written)

    def test_terminal_too_small_uses_minimum_size(self):
        self.assertTrue(terminal_too_small(69, 32))
        self.assertTrue(terminal_too_small(70, 15))
        self.assertFalse(terminal_too_small(70, 16))

    def test_draw_shows_terminal_too_small_message(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        screen = FakeScreen(height=32, width=69)

        app._draw(screen)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("Terminal size too small:", written)
        self.assertIn("Width = 69 Height = 32", written)
        self.assertIn("Needed for current config:", written)
        self.assertIn("Width = 70 Height = 16", written)
        self.assertNotIn("VACC's Running?", written)
        self.assertEqual(screen.refresh_count, 1)

    def test_status_title_uses_full_state_names(self):
        self.assertEqual(
            status_title(
                "Jobs",
                {"RUNNING": 2, "PENDING": 1, "FAILED": 1},
                ["RUNNING", "PENDING", "FAILED"],
            ),
            " Jobs: RUNNING:2 PENDING:1 FAILED:1 ",
        )
        self.assertEqual(status_title("Nodes", {}, ["IDLE"]), " Nodes: none ")

    def test_detail_lines_wrap_with_indent(self):
        wrapped = wrap_detail_lines(
            ["submitted=2026-05-31T10:04:06  started=2026-05-31T11:00:00"],
            width=36,
        )

        self.assertEqual(
            wrapped,
            ["submitted=2026-05-31T10:04:06", "  started=2026-05-31T11:00:00"],
        )

    def test_box_draws_bottom_right_corner(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        screen = FakeScreen(height=6, width=20)

        app._draw_box(screen, 2, 0, 4, screen.width, " selected job ")

        self.assertIn((5, 19, "╯", curses.A_DIM), screen.writes)

    def test_jobs_table_title_includes_overall_job_status(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [
            make_job("1", "RUNNING"),
            make_job("2", "RUNNING"),
            make_job("3", "PENDING"),
        ]
        screen = FakeScreen(height=40, width=140)

        app._draw_jobs_table(screen, app._visible_jobs(), screen.height, screen.width)

        self.assertIn(
            (5, 2, " Jobs: RUNNING:2 PENDING:1 ", curses.A_BOLD),
            screen.writes,
        )

    def test_jobs_table_hides_limit_before_cpus_when_narrow(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [make_job("1")]
        screen = FakeScreen(height=40, width=70)

        app._draw_jobs_table(screen, app._visible_jobs(), screen.height, screen.width)

        written = " ".join(write[2].strip() for write in screen.writes)
        self.assertNotIn("LIMIT", written)
        self.assertIn("CPUS", written)

    def test_jobs_table_then_hides_cpus_when_narrower(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [make_job("1")]
        screen = FakeScreen(height=40, width=62)

        app._draw_jobs_table(screen, app._visible_jobs(), screen.height, screen.width)

        written = " ".join(write[2].strip() for write in screen.writes)
        self.assertNotIn("LIMIT", written)
        self.assertNotIn("CPUS", written)

    def test_nodes_table_title_includes_overall_node_status(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.nodes = [
            make_node("node01", "(null)", state="IDLE"),
            make_node("node02", "(null)", state="MIXED"),
            make_node("node03", "(null)", state="ALLOCATED"),
        ]
        screen = FakeScreen(height=40, width=140)

        app._draw_nodes_table(screen, app._visible_nodes(), screen.height, screen.width)

        self.assertIn(
            (5, 2, " Nodes: IDLE:1 MIXED:1 ALLOCATED:1 ", curses.A_BOLD),
            screen.writes,
        )

    def test_nodes_table_removes_resource_bars_when_narrow(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.nodes = [make_node("node01", "gpu:h200:4", "gres/gpu=1")]
        screen = FakeScreen(height=40, width=100)

        app._draw_nodes_table(screen, app._visible_nodes(), screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertNotIn("[", written)
        self.assertNotIn("]", written)
        self.assertIn("0/1", written)
        self.assertIn("0M/1M", written)
        self.assertIn("1/4", written)

    def test_nodes_table_keeps_resource_bars_when_wide(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.nodes = [make_node("node01", "gpu:h200:4", "gres/gpu=1")]
        screen = FakeScreen(height=40, width=140)

        app._draw_nodes_table(screen, app._visible_nodes(), screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("[", written)
        self.assertIn("]", written)

    def test_j_and_n_switch_main_views(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)

        self.assertEqual(app.state.view, "jobs")

        self.assertTrue(app._handle_key(None, ord("n")))
        self.assertEqual(app.state.view, "nodes")

        self.assertTrue(app._handle_key(None, ord("j")))
        self.assertEqual(app.state.view, "jobs")

    def test_g_toggles_gpu_node_filter(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.view = "nodes"
        app.state.nodes = [
            make_node("cpu01", "(null)"),
            make_node("gpu-full", "gpu:h200:4", "gres/gpu=4"),
            make_node("gpu-free", "gpu:h200:4", "gres/gpu=1"),
        ]

        self.assertEqual(
            [node.name for node in app._visible_nodes()],
            ["cpu01", "gpu-full", "gpu-free"],
        )

        self.assertTrue(app._handle_key(None, ord("g")))

        self.assertTrue(app.state.gpu_nodes_only)
        self.assertEqual(
            [node.name for node in app._visible_nodes()],
            ["gpu-full", "gpu-free"],
        )
        self.assertEqual(app.state.message, "GPU node filter on")

        self.assertTrue(app._handle_key(None, ord("g")))

        self.assertFalse(app.state.gpu_nodes_only)
        self.assertEqual(
            [node.name for node in app._visible_nodes()],
            ["cpu01", "gpu-full", "gpu-free"],
        )

    def test_node_filters_are_mutually_exclusive(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.view = "nodes"
        app.state.nodes = [
            make_node("cpu01", "(null)"),
            make_node("gpu-full", "gpu:h200:4", "gres/gpu=4"),
            make_node("gpu-free", "gpu:h200:4", "gres/gpu=1"),
        ]

        self.assertTrue(app._handle_key(None, ord("g")))
        self.assertTrue(app.state.gpu_nodes_only)
        self.assertFalse(app.state.free_gpu_only)
        self.assertEqual(
            [node.name for node in app._visible_nodes()],
            ["gpu-full", "gpu-free"],
        )

        self.assertTrue(app._handle_key(None, ord("f")))
        self.assertFalse(app.state.gpu_nodes_only)
        self.assertTrue(app.state.free_gpu_only)
        self.assertEqual([node.name for node in app._visible_nodes()], ["gpu-free"])

        self.assertTrue(app._handle_key(None, ord("g")))
        self.assertTrue(app.state.gpu_nodes_only)
        self.assertFalse(app.state.free_gpu_only)
        self.assertEqual(
            [node.name for node in app._visible_nodes()],
            ["gpu-full", "gpu-free"],
        )

    def test_p_peeks_at_selected_node_jobs(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        calls = []
        app.state.view = "nodes"
        app.state.nodes = [make_node("h2node01", "gpu:h200:4")]
        app._popup_command = lambda stdscr, title, fn, arg, close_keys=(): calls.append(
            (title, fn(arg), close_keys)
        )

        self.assertTrue(app._handle_key(None, ord("p")))

        self.assertEqual(
            calls,
            [("squeue -a -w h2node01", "jobs for h2node01", (ord("p"),))],
        )

    def test_nodes_header_shows_usage_shortcut(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="nodes")
        screen = FakeScreen(height=12, width=120)

        app._draw_header(screen, 120)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" i usage ", written)

    def test_i_opens_cluster_usage_from_nodes_view(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        calls = []
        app.state.view = "nodes"
        app.state.nodes = [make_node("h2node01", "gpu:h200:4")]
        app._popup = (
            lambda stdscr, title, text, close_keys=(), refresh_while_open=True: calls.append(
                (title, text, close_keys, refresh_while_open)
            )
        )

        self.assertTrue(app._handle_key(None, ord("i")))

        self.assertEqual(
            calls,
            [("running usage by user", "usage by user", (ord("i"),), True)],
        )

    def test_i_is_nodes_only(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        calls = []
        app.state.view = "jobs"
        app._popup = (
            lambda stdscr, title, text, close_keys=(), refresh_while_open=True: calls.append(
                (title, text, close_keys, refresh_while_open)
            )
        )

        self.assertTrue(app._handle_key(None, ord("i")))

        self.assertEqual(calls, [])

    def test_popup_command_passes_close_keys_to_popup(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        calls = []
        app._popup = lambda stdscr, title, get_text, close_keys=(): calls.append(
            (title, get_text(), close_keys)
        )

        app._popup_command(
            None,
            "title",
            lambda value: f"text for {value}",
            "node01",
            close_keys=(ord("p"),),
        )

        self.assertEqual(calls, [("title", "text for node01", (ord("p"),))])

    def test_command_text_formats_empty_and_errors(self):
        self.assertEqual(command_text(lambda value: "", "job"), "No output.")
        self.assertEqual(command_text(lambda value: " text\n", "job"), "text")

    def test_popup_refreshes_live_text(self):
        import vaccs_running.ui as ui

        app = VaccsRunningApp(FakeClient(), refresh_seconds=0.25)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[-1, ord("q")])
        calls = []
        background_calls = []
        times = [0.0, 0.0, 0.30]
        original_newwin = curses.newwin
        original_monotonic = ui.time.monotonic
        original_sleep = ui.time.sleep
        try:
            curses.newwin = lambda height, width, top, left: popup
            ui.time.monotonic = lambda: times.pop(0) if times else 0.30
            ui.time.sleep = lambda seconds: None
            app._refresh_current = lambda: background_calls.append("refresh")
            app._draw = lambda stdscr: background_calls.append("draw")
            app._popup(screen, "title", lambda: calls.append("call") or f"text {len(calls)}")
        finally:
            curses.newwin = original_newwin
            ui.time.monotonic = original_monotonic
            ui.time.sleep = original_sleep

        self.assertEqual(calls, ["call", "call"])
        self.assertEqual(background_calls, ["refresh", "draw"])
        self.assertGreaterEqual(popup.refresh_count, 2)

    def test_popup_refreshes_background_for_static_text(self):
        import vaccs_running.ui as ui

        app = VaccsRunningApp(FakeClient(), refresh_seconds=0.25)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[-1, ord("q")])
        background_calls = []
        times = [0.0, 0.0, 0.30]
        original_newwin = curses.newwin
        original_monotonic = ui.time.monotonic
        original_sleep = ui.time.sleep
        try:
            curses.newwin = lambda height, width, top, left: popup
            ui.time.monotonic = lambda: times.pop(0) if times else 0.30
            ui.time.sleep = lambda seconds: None
            app._refresh_current = lambda: background_calls.append("refresh")
            app._draw = lambda stdscr: background_calls.append("draw")
            app._popup(screen, "title", "snapshot")
        finally:
            curses.newwin = original_newwin
            ui.time.monotonic = original_monotonic
            ui.time.sleep = original_sleep

        self.assertEqual(background_calls, ["refresh", "draw"])
        self.assertGreaterEqual(popup.refresh_count, 2)

    def test_popup_can_disable_live_refresh(self):
        import vaccs_running.ui as ui

        app = VaccsRunningApp(FakeClient(), refresh_seconds=0.25)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[-1, ord("q")])
        calls = []
        background_calls = []
        times = [0.0, 0.0, 0.30]
        original_newwin = curses.newwin
        original_monotonic = ui.time.monotonic
        original_sleep = ui.time.sleep
        try:
            curses.newwin = lambda height, width, top, left: popup
            ui.time.monotonic = lambda: times.pop(0) if times else 0.30
            ui.time.sleep = lambda seconds: None
            app._refresh_current = lambda: background_calls.append("refresh")
            app._draw = lambda stdscr: background_calls.append("draw")
            app._popup(
                screen,
                "title",
                lambda: calls.append("call") or f"text {len(calls)}",
                refresh_while_open=False,
            )
        finally:
            curses.newwin = original_newwin
            ui.time.monotonic = original_monotonic
            ui.time.sleep = original_sleep

        self.assertEqual(calls, ["call"])
        self.assertEqual(background_calls, [])
        self.assertGreaterEqual(popup.refresh_count, 2)

    def test_popup_footer_draws_on_bottom_border(self):
        import vaccs_running.ui as ui

        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[ord("q")])
        original_newwin = curses.newwin
        try:
            def fake_newwin(height, width, top, left):
                popup.height = height
                popup.width = width
                return popup

            curses.newwin = fake_newwin
            app._popup(screen, "title", "body")
        finally:
            curses.newwin = original_newwin

        footer_writes = [
            write for write in popup.writes if "up/down scroll" in write[2]
        ]
        self.assertEqual(len(footer_writes), 1)
        self.assertEqual(footer_writes[0][0], popup.height - 1)

    def test_popup_geometry_shrinks_to_short_content(self):
        top, left, height, width = popup_geometry(
            screen_height=60,
            screen_width=160,
            title="peek",
            text="one short line",
        )

        self.assertEqual((height, width), (8, 40))
        self.assertEqual(top, 26)
        self.assertEqual(left, 60)

    def test_popup_geometry_caps_to_screen_for_long_content(self):
        long_line = "x" * 300

        top, left, height, width = popup_geometry(
            screen_height=30,
            screen_width=100,
            title="detail",
            text="\n".join([long_line] * 40),
        )

        self.assertEqual((height, width), (26, 92))
        self.assertEqual(top, 2)
        self.assertEqual(left, 4)

    def test_left_and_right_arrows_jump_visible_page(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [make_job(str(index)) for index in range(60)]
        screen = FakeScreen(height=64)

        self.assertEqual(app._page_size(screen), 48)

        app.state.selected = 0
        self.assertTrue(app._handle_key(screen, curses.KEY_RIGHT))
        self.assertEqual(app.state.selected, 48)
        self.assertEqual(app.state.scroll, 48)

        self.assertTrue(app._handle_key(screen, curses.KEY_LEFT))
        self.assertEqual(app.state.selected, 0)
        self.assertEqual(app.state.scroll, 0)

        app.state.selected = 20
        app.state.scroll = 0
        self.assertTrue(app._handle_key(screen, curses.KEY_LEFT))
        self.assertEqual(app.state.selected, 0)
        self.assertEqual(app.state.scroll, 0)

    def test_right_arrow_stops_at_partial_last_page_start(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [make_job(str(index)) for index in range(101)]
        screen = FakeScreen(height=64)

        self.assertEqual(app._page_size(screen), 48)

        app.state.selected = 48
        app.state.scroll = 48
        self.assertTrue(app._handle_key(screen, curses.KEY_RIGHT))
        self.assertEqual(app.state.selected, 96)
        self.assertEqual(app.state.scroll, 96)

        self.assertTrue(app._handle_key(screen, curses.KEY_RIGHT))
        self.assertEqual(app.state.selected, 96)
        self.assertEqual(app.state.scroll, 96)

    def test_page_status_uses_selected_item_page(self):
        self.assertEqual(page_status(0, total_items=150, page_size=50), "1/3")
        self.assertEqual(page_status(49, total_items=150, page_size=50), "1/3")
        self.assertEqual(page_status(50, total_items=150, page_size=50), "2/3")
        self.assertEqual(page_status(149, total_items=150, page_size=50), "3/3")
        self.assertEqual(page_status(150, total_items=150, page_size=50), "3/3")
        self.assertEqual(page_status(0, total_items=0, page_size=50), "0/0")

    def test_resource_meter_aligns_count_prefix(self):
        count_width = resource_count_width([(1, 8), (12, 192), (192, 192)])

        rows = [
            resource_meter(1, 8, 12.5, meter_width=4, count_width=count_width),
            resource_meter(12, 192, 6.25, meter_width=4, count_width=count_width),
            resource_meter(192, 192, 100.0, meter_width=4, count_width=count_width),
        ]

        self.assertEqual(count_width, len("192/192"))
        self.assertEqual([row.index("[") for row in rows], [8, 8, 8])

    def test_resource_text_meter_aligns_text_prefix(self):
        count_width = resource_text_width(["-", "0/4", "12/16"])

        rows = [
            resource_text_meter("-", 0.0, meter_width=4, count_width=count_width),
            resource_text_meter("0/4", 0.0, meter_width=4, count_width=count_width),
            resource_text_meter("12/16", 75.0, meter_width=4, count_width=count_width),
        ]

        self.assertEqual(count_width, len("12/16"))
        self.assertEqual([row.index("[") for row in rows], [6, 6, 6])

    def test_resource_text_meter_aligns_memory_prefix(self):
        count_width = resource_text_width(["0M/8G", "120G/1000G", "1.0T/1.0T"])

        rows = [
            resource_text_meter("0M/8G", 0.0, meter_width=4, count_width=count_width),
            resource_text_meter(
                "120G/1000G",
                12.0,
                meter_width=4,
                count_width=count_width,
            ),
            resource_text_meter(
                "1.0T/1.0T",
                100.0,
                meter_width=4,
                count_width=count_width,
            ),
        ]

        self.assertEqual(count_width, len("120G/1000G"))
        self.assertEqual([row.index("[") for row in rows], [11, 11, 11])


if __name__ == "__main__":
    unittest.main()
