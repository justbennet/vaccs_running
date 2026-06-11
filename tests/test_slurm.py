import unittest

from vaccs_running.slurm import (
    NODE_JOBS_FORMAT,
    SlurmClient,
    aggregate_user_usage,
    format_node_jobs,
    format_user_usage,
    parse_node_job_line,
    parse_scontrol_nodes,
    parse_scontrol_job_usage,
    parse_squeue_line,
    parse_gpu_count,
    parse_memory_mb,
    parse_tres_value,
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

    def test_cluster_usage_queries_running_tasks_across_all_nodes(self):
        client = SlurmClient(user="dgezgin")
        fake_runner = FakeRunner(
            """JobId=4341591 ArrayJobId=4341591 ArrayTaskId=1 JobName=train
   UserId=dgezgin(512550) GroupId=pi-ncheney(170095)
   JobState=RUNNING Reason=None Dependency=(null)
   NumCPUs=4 ReqTRES=cpu=4,mem=16G,node=1,billing=4,gres/gpu=1
   AllocTRES=cpu=4,mem=16G,node=1,billing=4,gres/gpu=1
JobId=4341592 ArrayJobId=4341592 ArrayTaskId=7 JobName=train
   UserId=other(1234) GroupId=pi-example(5678)
   JobState=RUNNING Reason=None Dependency=(null)
   NumCPUs=8 ReqTRES=cpu=8,mem=32G,node=1,billing=8
   AllocTRES=cpu=8,mem=32G,node=1,billing=8
"""
        )
        client.runner = fake_runner

        output = client.cluster_usage()

        self.assertEqual(
            fake_runner.calls[0][0],
            ["scontrol", "show", "job"],
        )
        self.assertIn("2 people running 2 tasks", output)
        self.assertIn("dgezgin", output)
        self.assertIn("other", output)
        self.assertIn("TOTAL", output)

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

    def test_parse_gpu_count_handles_slurm_gres_shapes(self):
        self.assertEqual(parse_gpu_count("gpu:h200:1"), 1)
        self.assertEqual(parse_gpu_count("gpu:2"), 2)
        self.assertEqual(parse_gpu_count("gres/gpu=4"), 4)
        self.assertEqual(parse_gpu_count("cpu=4,mem=64G,gres/gpu=1"), 1)
        self.assertEqual(parse_gpu_count("gpu:h200:1,gpu:a100:2"), 3)
        self.assertEqual(parse_gpu_count("N/A"), 0)

    def test_parse_memory_mb_handles_slurm_units_and_unavailable_values(self):
        self.assertEqual(parse_memory_mb("4096M"), 4096)
        self.assertEqual(parse_memory_mb("16G"), 16384)
        self.assertEqual(parse_memory_mb("1.5T"), 1572864)
        self.assertEqual(parse_memory_mb("2Gc"), 2048)
        self.assertIsNone(parse_memory_mb("N/A"))
        self.assertIsNone(parse_memory_mb("0"))

    def test_aggregate_user_usage_sums_tasks_and_requested_resources(self):
        usage = aggregate_user_usage(
            [
                {
                    "job_id": "1",
                    "user": "alice",
                    "cpus": "4",
                    "tres": "cpu=4,mem=16G,node=1,gres/gpu=1",
                    "memory": "16G",
                },
                {
                    "job_id": "2",
                    "user": "alice",
                    "cpus": "2",
                    "tres": "cpu=2,mem=8G,node=1",
                    "memory": "8G",
                },
                {
                    "job_id": "3",
                    "user": "bob",
                    "cpus": "16",
                    "tres": "cpu=16,node=1,gres/gpu=2",
                    "memory": "N/A",
                },
            ]
        )

        self.assertEqual([row.user for row in usage], ["bob", "alice"])
        self.assertEqual(usage[0].tasks, 1)
        self.assertEqual(usage[0].cpus, 16)
        self.assertEqual(usage[0].gpus, 2)
        self.assertIsNone(usage[0].memory_mb)
        self.assertEqual(usage[1].tasks, 2)
        self.assertEqual(usage[1].cpus, 6)
        self.assertEqual(usage[1].gpus, 1)
        self.assertEqual(usage[1].memory_mb, 24576)

    def test_format_user_usage_omits_ram_when_unavailable(self):
        usage = aggregate_user_usage(
            [
                {
                    "job_id": "1",
                    "user": "alice",
                    "cpus": "4",
                    "tres": "cpu=4,gres/gpu=1",
                    "memory": "N/A",
                },
                {
                    "job_id": "2",
                    "user": "bob",
                    "cpus": "8",
                    "tres": "cpu=8",
                    "memory": "N/A",
                },
            ]
        )

        text = format_user_usage(usage)

        self.assertIn("2 people running 2 tasks", text)
        self.assertIn("USER", text)
        self.assertIn("TASKS", text)
        self.assertIn("CPUS", text)
        self.assertIn("GPUS", text)
        self.assertNotIn("RAM", text)

    def test_parse_scontrol_job_usage_counts_running_alloc_tres(self):
        usage = parse_scontrol_job_usage(
            """JobId=4414236 ArrayJobId=4413548 ArrayTaskId=235 JobName=ae-pert-cand
   UserId=dgezgin(512550) GroupId=pi-ncheney(170095)
   JobState=RUNNING Reason=None Dependency=(null)
   NumCPUs=4 NumTasks=1 CPUs/Task=4
   ReqTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1
   AllocTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1
JobId=4414192 ArrayJobId=4413548 ArrayTaskId=214 JobName=ae-pert-cand
   UserId=dgezgin(512550) GroupId=pi-ncheney(170095)
   JobState=COMPLETED Reason=None Dependency=(null)
   NumCPUs=4 NumTasks=1 CPUs/Task=4
   ReqTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1
   AllocTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1
"""
        )

        self.assertEqual(
            usage,
            [
                {
                    "job_id": "4414236",
                    "user": "dgezgin",
                    "cpus": "4",
                    "tres": "cpu=4,mem=64G,node=1,billing=4,gres/gpu=1",
                    "memory": "64G",
                }
            ],
        )

    def test_parse_tres_value_extracts_named_value(self):
        self.assertEqual(
            parse_tres_value("cpu=4,mem=64G,node=1,billing=4,gres/gpu=1", "mem"),
            "64G",
        )
        self.assertEqual(parse_tres_value("cpu=4,node=1", "mem"), "")


if __name__ == "__main__":
    unittest.main()
