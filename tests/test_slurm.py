import unittest

from vaccs_running.slurm import (
    NODE_JOBS_FORMAT,
    SlurmClient,
    format_node_jobs,
    parse_node_job_line,
    parse_scontrol_nodes,
    parse_squeue_line,
    summarize_jobs,
)


class FakeRunner:
    def __init__(self, output=""):
        self.calls = []
        self.output = output

    def run(self, args, timeout=12.0):
        self.calls.append((args, timeout))
        return self.output


class SlurmParsingTests(unittest.TestCase):
    def test_fetch_jobs_expands_array_tasks(self):
        client = SlurmClient(user="dgezgin")
        fake_runner = FakeRunner()
        client.runner = fake_runner

        self.assertEqual(client.fetch_jobs(), [])
        self.assertEqual(fake_runner.calls[0][0][:2], ["squeue", "--array"])

    def test_parse_squeue_line_running_job(self):
        job = parse_squeue_line(
            "4340534_1|lcb-w2d-lr|RUNNING|nvgpu|h2xnode05|h2xnode05|19:21|"
            "6:00:00|1|4|N/A|2026-05-30T12:00:26|2026-05-30T12:00:27"
        )

        self.assertEqual(job.job_id, "4340534_1")
        self.assertEqual(job.array_parent, "4340534")
        self.assertEqual(job.location, "h2xnode05")
        self.assertTrue(job.is_running)

    def test_parse_squeue_line_pending_job(self):
        job = parse_squeue_line(
            "4340534|lcb-w2d-lr|PENDING|nvgpu,gpu-preempt||Resources|0:00|"
            "6:00:00|1|4|N/A|2026-05-30T12:00:26|2026-05-30T14:31:45"
        )

        self.assertEqual(job.location, "pending: Resources")
        self.assertEqual(summarize_jobs([job]), {"PENDING": 1})

    def test_parse_scontrol_node_load(self):
        nodes = parse_scontrol_nodes(
            """NodeName=h2node01 Arch=x86_64 CoresPerSocket=96
   CPUAlloc=13 CPUEfctv=192 CPUTot=192 CPULoad=4.74
   AvailableFeatures=GPU_SKU:H200,GPU_FP:FP64,GPU_ANY,h200
   ActiveFeatures=GPU_SKU:H200,GPU_FP:FP64,GPU_ANY,h200
   Gres=gpu:h200:4
   RealMemory=1000000 AllocMem=198656 FreeMem=942714
   State=MIXED+PLANNED ThreadsPerCore=1 TmpDisk=0 Weight=1
   Partitions=nvgpu
   AllocTRES=cpu=13,mem=194G,gres/gpu=4
"""
        )

        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.name, "h2node01")
        self.assertEqual(node.base_state, "MIXED")
        self.assertEqual(node.free_cpus, 179)
        self.assertEqual(node.cpu_load, 4.74)
        self.assertTrue(node.has_gpus)
        self.assertEqual(node.gpu_text, "4/4")
        self.assertEqual(node.gpu_free, 0)

    def test_node_jobs_queries_selected_node(self):
        client = SlurmClient(user="dgezgin")
        fake_runner = FakeRunner(
            "4341591_1|dgezgin|RUNNING|12:34|4|gpu:h200:1|train\n"
        )
        client.runner = fake_runner

        output = client.node_jobs("h2node01")

        self.assertEqual(
            fake_runner.calls[0][0],
            ["squeue", "-a", "-h", "-w", "h2node01", "-o", NODE_JOBS_FORMAT],
        )
        self.assertIn("JOBID", output)
        self.assertIn("USER", output)
        self.assertIn("dgezgin", output)
        self.assertIn("train", output)

    def test_node_jobs_reports_empty_node(self):
        client = SlurmClient(user="dgezgin")
        client.runner = FakeRunner("")

        self.assertEqual(client.node_jobs("h2node01"), "No jobs found on h2node01.")

    def test_parse_node_job_line_strips_fields(self):
        job = parse_node_job_line(
            " 4341679_19 | dgezgin | RUNNING | 44:18 | 4 | N/A | lcb-ant-omni-lr "
        )

        self.assertEqual(job["job_id"], "4341679_19")
        self.assertEqual(job["user"], "dgezgin")
        self.assertEqual(job["state"], "RUNNING")
        self.assertEqual(job["name"], "lcb-ant-omni-lr")

    def test_format_node_jobs_aligns_rows(self):
        text = format_node_jobs(
            [
                parse_node_job_line("4341591_66|dgezgin|RUNNING|58:18|4|N/A|lcb-ant-lr"),
                parse_node_job_line("4341679_19|dgezgin|RUNNING|44:18|4|N/A|lcb-ant-omni-lr"),
            ]
        )
        lines = text.splitlines()

        self.assertEqual(lines[0].index("USER"), lines[2].index("dgezgin"))
        self.assertEqual(lines[0].index("STATE"), lines[2].index("RUNNING"))
        self.assertEqual(lines[0].rindex("JOB"), lines[2].index("lcb-ant-lr"))


if __name__ == "__main__":
    unittest.main()
