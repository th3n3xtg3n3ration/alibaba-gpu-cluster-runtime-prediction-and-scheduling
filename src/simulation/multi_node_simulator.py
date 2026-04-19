"""
multi_node_simulator.py

Multi-Node Event-Driven Cluster Simulator

This module implements a heap-based discrete-event simulator for evaluating
scheduling policies across a heterogeneous multi-machine GPU cluster. Each
machine tracks CPU, GPU, and (optionally) memory resources independently.

Key Components
--------------
JobEvent
    Dataclass representing an ARRIVAL or FINISH event in the priority queue.
Machine
    Single cluster node with resource accounting.
MultiNodeClusterSimulator
    Full event-driven simulator supporting multi-machine placement.
provision_heterogeneous_gpu_cluster
    Factory that builds a realistic heterogeneous cluster configuration.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .scheduler_simulator import SchedulerBase


__all__ = [
    "JobEvent",
    "Machine",
    "MultiNodeClusterSimulator",
    "provision_heterogeneous_gpu_cluster",
]


# ---------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------


@dataclass
class JobEvent:
    """
    A single simulation event (arrival or completion).

    Parameters
    ----------
    time : float
        Simulation clock time when this event occurs.
    event_type : {"ARRIVAL", "FINISH"}
        Event category.
    job : pd.Series
        The job associated with this event.
    machine_id : int, optional
        Machine assigned to this job (only set for FINISH events).
    """

    time: float
    event_type: str
    job: pd.Series
    machine_id: Optional[int] = None

    def __lt__(self, other: "JobEvent") -> bool:
        return self.time < other.time


# ---------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------


class Machine:
    """
    Represents a single cluster node with CPU, GPU, and memory resources.

    Parameters
    ----------
    machine_id : int
        Unique identifier for this node.
    cpu_capacity : float
        Total CPU cores available.
    gpu_capacity : float
        Total GPUs available.
    mem_capacity : float, default 0.0
        Total memory available (in arbitrary units). Set to 0.0 to
        disable memory checks.
    """

    def __init__(
        self,
        machine_id: int,
        cpu_capacity: float,
        gpu_capacity: float,
        mem_capacity: float = 0.0,
    ) -> None:
        self.machine_id = machine_id
        self.cpu_capacity = cpu_capacity
        self.gpu_capacity = gpu_capacity
        self.mem_capacity = mem_capacity

        self.cpu_used: float = 0.0
        self.gpu_used: float = 0.0
        self.mem_used: float = 0.0
        self.running_jobs: List[Any] = []

    def can_fit(
        self, job_cpu: float, job_gpu: float, job_mem: float = 0.0
    ) -> bool:
        """
        Check whether this machine has sufficient free resources.

        Parameters
        ----------
        job_cpu : float
            CPU cores required by the job.
        job_gpu : float
            GPUs required by the job.
        job_mem : float, default 0.0
            Memory required by the job. Ignored when ``mem_capacity == 0``.

        Returns
        -------
        bool
        """
        if (self.cpu_used + job_cpu) > self.cpu_capacity + 1e-5:
            return False
        if (self.gpu_used + job_gpu) > self.gpu_capacity + 1e-5:
            return False
        # Only enforce memory constraint when the machine has a memory budget
        if self.mem_capacity > 0.0 and (self.mem_used + job_mem) > self.mem_capacity + 1e-5:
            return False
        return True

    def allocate(
        self, job: pd.Series, job_cpu: float, job_gpu: float, job_mem: float = 0.0
    ) -> None:
        """Reserve resources for a scheduled job."""
        self.cpu_used += job_cpu
        self.gpu_used += job_gpu
        self.mem_used += job_mem
        self.running_jobs.append(job["job_id"])

    def release(
        self, job: pd.Series, job_cpu: float, job_gpu: float, job_mem: float = 0.0
    ) -> None:
        """Free resources after a job finishes."""
        self.cpu_used = max(0.0, self.cpu_used - job_cpu)
        self.gpu_used = max(0.0, self.gpu_used - job_gpu)
        self.mem_used = max(0.0, self.mem_used - job_mem)
        job_id = job["job_id"]
        if job_id in self.running_jobs:
            self.running_jobs.remove(job_id)

    def __repr__(self) -> str:
        return (
            f"Machine({self.machine_id}, "
            f"CPU: {self.cpu_used}/{self.cpu_capacity}, "
            f"GPU: {self.gpu_used}/{self.gpu_capacity})"
        )


# ---------------------------------------------------------------------
# Multi-node simulator
# ---------------------------------------------------------------------


class MultiNodeClusterSimulator:
    """
    Heap-based discrete-event simulator for multi-node GPU clusters.

    Supports any :class:`~src.simulation.SchedulerBase` policy. Jobs are
    placed on the first machine that can accommodate their resource request
    (First-Fit). Head-of-line blocking is used: if the highest-priority job
    cannot be placed, the scheduler waits for the next FINISH event.

    Parameters
    ----------
    scheduler : SchedulerBase
        Scheduling policy used to order the pending queue.
    machines : list of Machine
        Cluster nodes available for job placement.
    """

    def __init__(
        self, scheduler: SchedulerBase, machines: List[Machine]
    ) -> None:
        self.scheduler = scheduler
        self.machines = machines
        self.time: float = 0.0
        self.results: List[dict] = []
        self.utilization_history: List[dict] = []

    def _get_avg_utilization(self) -> Tuple[float, float]:
        """Compute mean CPU and GPU utilization ratios across all machines."""
        total_cpu_cap = sum(m.cpu_capacity for m in self.machines) or 1.0
        total_gpu_cap = sum(m.gpu_capacity for m in self.machines) or 1.0
        total_cpu_used = sum(m.cpu_used for m in self.machines)
        total_gpu_used = sum(m.gpu_used for m in self.machines)
        return total_cpu_used / total_cpu_cap, total_gpu_used / total_gpu_cap

    def run(self, jobs: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the simulation on the provided workload.

        Parameters
        ----------
        jobs : pd.DataFrame
            Job workload. Required columns:

            - ``job_id``
            - ``submit_time``  (float, seconds)
            - ``runtime``      (float, seconds)
            - ``num_cpu``      (float, optional)
            - ``num_gpu``      (float, optional)
            - ``predicted_runtime`` (float, optional, for SJF-Pred)

        Returns
        -------
        pd.DataFrame
            Per-job result table with:

            - ``job_id``, ``submit_time``, ``start_time``
            - ``completion_time``, ``waiting_time``, ``turnaround_time``
            - ``slowdown``, ``machine_id``
        """
        # Reset state
        self.time = 0.0
        self.results = []
        self.utilization_history = []

        for m in self.machines:
            m.cpu_used = 0.0
            m.gpu_used = 0.0
            m.mem_used = 0.0
            m.running_jobs = []

        # Seed event queue with all arrivals
        events: List[JobEvent] = []
        for _, job in jobs.iterrows():
            heapq.heappush(events, JobEvent(float(job["submit_time"]), "ARRIVAL", job))

        # Use a persistent DataFrame for the queue to avoid expensive O(N) conversions
        pending_df = jobs.iloc[0:0].copy()

        while events or not pending_df.empty:
            # --- Try to schedule pending jobs ---
            if not pending_df.empty:
                scheduled_any = True
                while scheduled_any and not pending_df.empty:
                    scheduled_any = False
                    best_job_row = self.scheduler.select_job(pending_df)

                    req_cpu = float(best_job_row.get("num_cpu", 0))
                    req_gpu = float(best_job_row.get("num_gpu", 0))

                    allocated_machine: Optional[Machine] = None
                    for m in self.machines:
                        if m.can_fit(req_cpu, req_gpu):
                            m.allocate(best_job_row, req_cpu, req_gpu)
                            allocated_machine = m
                            break

                    if allocated_machine is not None:
                        # Remove from pending_df using its index
                        pending_df = pending_df.drop(best_job_row.name)
                        scheduled_any = True

                        start_time = self.time
                        actual_runtime = float(best_job_row["runtime"])
                        finish_time = start_time + actual_runtime
                        waiting_time = start_time - float(best_job_row["submit_time"])
                        turnaround_time = finish_time - float(best_job_row["submit_time"])
                        slowdown = (
                            turnaround_time / actual_runtime
                            if actual_runtime > 0
                            else float("inf")
                        )

                        heapq.heappush(
                            events,
                            JobEvent(finish_time, "FINISH", best_job_row, allocated_machine.machine_id),
                        )
                        self.results.append(
                            {
                                "job_id": best_job_row["job_id"],
                                "submit_time": best_job_row["submit_time"],
                                "start_time": start_time,
                                "completion_time": finish_time,
                                "waiting_time": waiting_time,
                                "turnaround_time": turnaround_time,
                                "slowdown": slowdown,
                                "machine_id": allocated_machine.machine_id,
                            }
                        )
                    # else: head-of-line blocking — wait for a FINISH event

            # Record utilization snapshot
            cpu_util, gpu_util = self._get_avg_utilization()
            self.utilization_history.append(
                {
                    "time": self.time,
                    "cpu_util": cpu_util,
                    "gpu_util": gpu_util,
                    "pending_jobs": len(pending_df),
                }
            )

            if not events:
                break

            # Advance clock
            event = heapq.heappop(events)
            self.time = event.time

            if event.event_type == "ARRIVAL":
                new_jobs = [event.job]
                # Optimization: Admit any other jobs that arrive at the exact same time
                while events and events[0].time <= self.time and events[0].event_type == "ARRIVAL":
                    new_jobs.append(heapq.heappop(events).job)
                
                pending_df = pd.concat([pending_df, pd.DataFrame(new_jobs)])
            elif event.event_type == "FINISH":
                machine = next(
                    (m for m in self.machines if m.machine_id == event.machine_id), None
                )
                if machine is not None:
                    req_cpu = float(event.job.get("num_cpu", 0))
                    req_gpu = float(event.job.get("num_gpu", 0))
                    machine.release(event.job, req_cpu, req_gpu)

        return pd.DataFrame(self.results)


# ---------------------------------------------------------------------
# Cluster factory
# ---------------------------------------------------------------------


def provision_heterogeneous_gpu_cluster(
    n_high: int = 25,
    n_mid: int = 100,
    n_cpu: int = 50,
) -> List[Machine]:
    """
    Create a heterogeneous cluster of :class:`Machine` objects.

    Parameters
    ----------
    n_high : int, default 25
        Number of High-Performance nodes (8 GPU, 96 CPU cores).
    n_mid : int, default 100
        Number of Mid-Range nodes (2 GPU, 64 CPU cores).
    n_cpu : int, default 50
        Number of CPU-only nodes (0 GPU, 64 CPU cores).

    Returns
    -------
    list of Machine
        Ordered list: high-perf nodes first, then mid-range, then CPU-only.
    """
    machines: List[Machine] = []
    mid = 0

    for _ in range(n_high):
        machines.append(Machine(mid, cpu_capacity=96.0, gpu_capacity=8.0))
        mid += 1

    for _ in range(n_mid):
        machines.append(Machine(mid, cpu_capacity=64.0, gpu_capacity=2.0))
        mid += 1

    for _ in range(n_cpu):
        machines.append(Machine(mid, cpu_capacity=64.0, gpu_capacity=0.0))
        mid += 1

    return machines
