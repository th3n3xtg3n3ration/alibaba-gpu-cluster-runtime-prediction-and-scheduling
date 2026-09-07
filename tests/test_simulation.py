"""
tests/test_simulation.py

Unit tests for src.simulation.

Verifies the correctness of the discrete-event simulator and
scheduling policies (FIFO, SJF) using deterministic workloads.
"""
import unittest
import pandas as pd
from src.simulation import (
    ClusterSimulator,
    FIFOScheduler,
    Machine,
    MultiNodeClusterSimulator,
    SJFScheduler,
    provision_heterogeneous_gpu_cluster,
)

class TestSimulation(unittest.TestCase):
    def test_fifo_logic(self):
        """Test First-In-First-Out (FIFO) scheduling logic."""
        jobs = pd.DataFrame({
            "job_id": [1, 2],
            "submit_time": [0.0, 5.0],
            "runtime": [10.0, 10.0]
        })
        sim = ClusterSimulator(FIFOScheduler())
        res = sim.run(jobs)

        # FIFO: Job 1 starts at 0, ends at 10. Job 2 starts at 10 (since 10 > 5), ends at 20.
        self.assertEqual(res.iloc[0]["start_time"], 0.0)
        self.assertEqual(res.iloc[1]["start_time"], 10.0)
        self.assertEqual(res.iloc[1]["completion_time"], 20.0)

    def test_sjf_logic(self):
        """Test Shortest-Job-First (SJF) scheduling logic."""
        jobs = pd.DataFrame({
            "job_id": [1, 2],
            "submit_time": [0.0, 0.0],
            "runtime": [20.0, 5.0]
        })
        sim = ClusterSimulator(SJFScheduler())
        res = sim.run(jobs)

        # SJF: Job 2 (runtime 5) should go first
        self.assertEqual(res.iloc[0]["job_id"], 2)
        self.assertEqual(res.iloc[0]["completion_time"], 5.0)
        self.assertEqual(res.iloc[1]["job_id"], 1)
        self.assertEqual(res.iloc[1]["start_time"], 5.0)

    def test_multinode_simulator_uses_additional_machine(self):
        """Multi-node simulator should place concurrent jobs on separate machines when possible."""
        jobs = pd.DataFrame({
            "job_id": [1, 2],
            "submit_time": [0.0, 0.0],
            "runtime": [10.0, 10.0],
            "num_cpu": [2.0, 2.0],
            "num_gpu": [1.0, 1.0],
        })
        machines = [
            Machine(machine_id=0, cpu_capacity=8.0, gpu_capacity=1.0),
            Machine(machine_id=1, cpu_capacity=8.0, gpu_capacity=1.0),
        ]

        sim = MultiNodeClusterSimulator(FIFOScheduler(), machines)
        res = sim.run(jobs).sort_values("job_id").reset_index(drop=True)

        self.assertEqual(res["start_time"].tolist(), [0.0, 0.0])
        self.assertEqual(sorted(res["machine_id"].tolist()), [0, 1])

    def test_machine_release_matches_by_job_id_not_footprint(self):
        """release() must drop the finishing job's own reservation record,
        not any other running job's record that happens to share the same
        (cpu, gpu) footprint.

        Regression test for code_bugs-4: the old implementation matched by
        (cpu, gpu) alone and removed the *first* matching entry in
        running_detail, so releasing job B here (allocated second, with a
        later expected_finish) would incorrectly delete job A's still-live
        reservation instead, corrupting earliest_fit()'s backfill window
        for every job still running on this machine.
        """
        machine = Machine(machine_id=0, cpu_capacity=8.0, gpu_capacity=4.0)
        job_a = pd.Series({"job_id": 1})
        job_b = pd.Series({"job_id": 2})

        # Same (cpu, gpu) footprint, different expected_finish, A allocated first.
        machine.allocate(job_a, job_cpu=2.0, job_gpu=1.0, expected_finish=100.0)
        machine.allocate(job_b, job_cpu=2.0, job_gpu=1.0, expected_finish=200.0)
        self.assertEqual(len(machine.running_detail), 2)

        # Release B (it finished first in this scenario), A must remain.
        machine.release(job_b, job_cpu=2.0, job_gpu=1.0)

        self.assertEqual(len(machine.running_detail), 1)
        remaining_finish, remaining_cpu, remaining_gpu, remaining_job_id = machine.running_detail[0]
        self.assertEqual(remaining_job_id, 1)
        self.assertEqual(remaining_finish, 100.0)
        self.assertNotIn(2, machine.running_jobs)
        self.assertIn(1, machine.running_jobs)

    def test_multinode_simulator_fifo_tiebreak_is_deterministic(self):
        """Simultaneous arrivals must be run in a fixed, deterministic order
        (the DataFrame row order they were seeded in), not whatever order
        heapq's internal layout happens to produce for equal timestamps.

        Regression test for simulator-4: without a tie-breaking sequence
        number, FIFO's "earliest arrival first" guarantee only held between
        *distinct* timestamps, simultaneous arrivals could be scheduled in
        an order unrelated to how they were submitted, and re-running the
        identical input could reorder them.
        """
        n = 8
        jobs = pd.DataFrame({
            "job_id": list(range(1, n + 1)),
            "submit_time": [0.0] * n,  # all simultaneous
            "runtime": [10.0] * n,
            "num_cpu": [1.0] * n,
            "num_gpu": [1.0] * n,
        })
        # One machine at a time: only one job can start per round, so start
        # order fully reveals the tie-break order.
        machines = [Machine(machine_id=0, cpu_capacity=1.0, gpu_capacity=1.0)]

        first_run = MultiNodeClusterSimulator(FIFOScheduler(), machines).run(jobs)
        first_order = first_run.sort_values("start_time", kind="mergesort")["job_id"].tolist()

        # Re-running the identical input must reproduce the identical order.
        second_run = MultiNodeClusterSimulator(FIFOScheduler(), machines).run(jobs)
        second_order = second_run.sort_values("start_time", kind="mergesort")["job_id"].tolist()

        self.assertEqual(first_order, list(range(1, n + 1)))
        self.assertEqual(first_order, second_order)

    def test_multinode_simulator_rejects_unplaceable_job(self):
        """A job requesting more resources than any machine can ever provide
        must fail loudly instead of silently vanishing from the results.

        Regression test for simulator-6 / code_bugs-7: previously such a job
        (and, under strict HoL, every job queued behind it) was dropped with
        no error, the simulation just returned fewer rows than jobs given.
        """
        jobs = pd.DataFrame({
            "job_id": [1],
            "submit_time": [0.0],
            "runtime": [10.0],
            "num_cpu": [1.0],
            "num_gpu": [16.0],  # exceeds every machine's gpu_capacity below
        })
        machines = [Machine(machine_id=0, cpu_capacity=8.0, gpu_capacity=8.0)]

        with self.assertRaises(ValueError):
            MultiNodeClusterSimulator(FIFOScheduler(), machines).run(jobs)

    def test_bounded_slowdown_caps_short_job_blowup(self):
        """A very short job with a long wait should post a much smaller
        bounded_slowdown than unbounded slowdown, that gap is exactly the
        point of bounded slowdown (statistics-8 / simulator-9): unbounded
        slowdown lets a 1s job dominate a mean simply by being short, not by
        reflecting the cluster's actual queueing behaviour.
        """
        jobs = pd.DataFrame({
            "job_id": [1, 2],
            "submit_time": [0.0, 0.0],
            "runtime": [100.0, 1.0],  # job 2 is short and queues behind job 1
            "num_cpu": [1.0, 1.0],
            "num_gpu": [1.0, 1.0],
        })
        machines = [Machine(machine_id=0, cpu_capacity=1.0, gpu_capacity=1.0)]
        result = MultiNodeClusterSimulator(FIFOScheduler(), machines).run(jobs)

        short_job = result[result["job_id"] == 2].iloc[0]
        # turnaround = 100 (wait) + 1 (run) = 101; unbounded = 101/1 = 101,
        # bounded (tau=10) = 101/10 = 10.1.
        self.assertAlmostEqual(short_job["slowdown"], 101.0)
        self.assertAlmostEqual(short_job["bounded_slowdown"], 10.1)
        self.assertLess(short_job["bounded_slowdown"], short_job["slowdown"])

    def test_provision_heterogeneous_gpu_cluster_counts(self):
        """Provisioned heterogeneous cluster should return the requested machine mix."""
        machines = provision_heterogeneous_gpu_cluster(n_high=2, n_mid=3, n_cpu=1)

        self.assertEqual(len(machines), 6)
        self.assertEqual([m.gpu_capacity for m in machines[:2]], [8.0, 8.0])
        self.assertEqual([m.gpu_capacity for m in machines[2:5]], [2.0, 2.0, 2.0])
        self.assertEqual(machines[5].gpu_capacity, 0.0)

if __name__ == "__main__":
    unittest.main()
