import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from vaccs_running import __version__
from vaccs_running.__main__ import build_parser, make_client


class CliTests(unittest.TestCase):
    def test_state_option_normalizes_slurm_state_list(self):
        args = build_parser().parse_args(["--state", "pd, running"])
        self.assertEqual(args.states, "PD,RUNNING")

        alias_args = build_parser().parse_args(["--states", "all"])
        self.assertEqual(alias_args.states, "all")

    def test_state_option_rejects_invalid_tokens(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--state", "PD;RUNNING"])

    def test_version_option_prints_package_version(self):
        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            build_parser().parse_args(["--version"])

        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f"vaccs-running {__version__}")

    def test_user_all_starts_with_all_users_filter(self):
        client = make_client("all", "PD")

        self.assertTrue(client.job_all_principals)
        self.assertEqual(client.job_users, set())
        self.assertEqual(client.job_groups, set())
        self.assertEqual(client.job_user_label, "all users")
        self.assertEqual(client.squeue_states, "PD")

    def test_user_name_starts_with_single_user_filter(self):
        client = make_client("testuser", "all")

        self.assertFalse(client.job_all_principals)
        self.assertEqual(client.user, "testuser")
        self.assertEqual(client.job_users, {"testuser"})


if __name__ == "__main__":
    unittest.main()
