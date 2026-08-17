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

    def test_provision_heterogeneous_gpu_cluster_counts(self):
        """Provisioned heterogeneous cluster should return the requested machine mix."""
        machines = provision_heterogeneous_gpu_cluster(n_high=2, n_mid=3, n_cpu=1)

        self.assertEqual(len(machines), 6)
        self.assertEqual([m.gpu_capacity for m in machines[:2]], [8.0, 8.0])
        self.assertEqual([m.gpu_capacity for m in machines[2:5]], [2.0, 2.0, 2.0])
        self.assertEqual(machines[5].gpu_capacity, 0.0)

if __name__ == "__main__":
    unittest.main()
