"""
tests/test_simulation.py

Unit tests for src.simulation.

Verifies the correctness of the discrete-event simulator and 
scheduling policies (FIFO, SJF) using deterministic workloads.
"""
import unittest
import pandas as pd
from src.simulation import ClusterSimulator, FIFOScheduler, SJFScheduler

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

if __name__ == "__main__":
    unittest.main()
