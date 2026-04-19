"""
scheduler_simulator.py

Single-Queue Discrete-Event Scheduler Simulator

This module implements a simplified discrete-event simulator for evaluating
scheduling algorithms in GPU clusters. It models a single-server queue and
supports three scheduling policies.

Schedulers Implemented
----------------------
FIFOScheduler
    Baseline First-In-First-Out (arrival order).
SJFScheduler
    Oracle Shortest-Job-First using true job runtime.
SJFPredScheduler
    ML-based SJF using the ``predicted_runtime`` column.

Metrics Computed
----------------
- ``waiting_time``    : time from submission to job start.
- ``turnaround_time`` : time from submission to job completion.
- ``completion_time`` : absolute job finish time.
- ``slowdown``        : turnaround_time / job_runtime (≥ 1).

Expected Input Schema
---------------------
A :class:`pandas.DataFrame` with columns:

- ``job_id``
- ``submit_time``
- ``runtime``             (true runtime in seconds)
- ``predicted_runtime``   (required for :class:`SJFPredScheduler`)

This simulator is used in:
    ``notebooks/05_scheduler_evaluation.ipynb``
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


__all__ = [
    "SchedulerBase",
    "FIFOScheduler",
    "SJFScheduler",
    "SJFPredScheduler",
    "ClusterSimulator",
]


# ============================================================
# Scheduler base class
# ============================================================


class SchedulerBase:
    """Abstract base class for scheduling algorithms."""

    def select_job(self, queue: pd.DataFrame) -> pd.Series:
        """Select the next job from the ready queue.

        Parameters
        ----------
        queue : pd.DataFrame
            Ready queue containing jobs that have arrived and are waiting.

        Returns
        -------
        pd.Series
            The selected job row.
        """
        raise NotImplementedError


# ============================================================
# FIFO scheduler
# ============================================================


class FIFOScheduler(SchedulerBase):
    """First-In-First-Out scheduler — selects the earliest-arrived job."""

    def select_job(self, queue: pd.DataFrame) -> pd.Series:
        return queue.iloc[0]


# ============================================================
# Oracle SJF scheduler
# ============================================================


class SJFScheduler(SchedulerBase):
    """Oracle SJF — selects the job with the smallest *true* runtime."""

    def select_job(self, queue: pd.DataFrame) -> pd.Series:
        idx = queue["runtime"].idxmin()
        return queue.loc[idx]


# ============================================================
# ML-based SJF scheduler (SJF-Pred)
# ============================================================


class SJFPredScheduler(SchedulerBase):
    """
    SJF-Pred — selects the job with the smallest *ML-predicted* runtime.

    Requires column ``predicted_runtime`` in the ready queue.
    """

    def select_job(self, queue: pd.DataFrame) -> pd.Series:
        idx = queue["predicted_runtime"].idxmin()
        return queue.loc[idx]


# ============================================================
# Single-queue cluster simulator
# ============================================================


class ClusterSimulator:
    """
    Discrete-event simulator for evaluating single-queue scheduling policies.

    Models one virtual "GPU server" that processes jobs one at a time.
    The simulation clock advances to each job's completion time after starting
    a job, so there is no idle wait between jobs when the queue is non-empty.

    Parameters
    ----------
    scheduler : SchedulerBase
        An instance of :class:`FIFOScheduler`, :class:`SJFScheduler`, or
        :class:`SJFPredScheduler`.
    gpu_capacity : int, default 1
        Reserved for future multi-GPU extensions. Currently unused.
    """

    def __init__(self, scheduler: SchedulerBase, gpu_capacity: int = 1) -> None:
        self.scheduler = scheduler
        self.gpu_capacity = gpu_capacity

    # ----------------------------------------------------------
    # Main simulation loop
    # ----------------------------------------------------------

    def run(self, jobs: pd.DataFrame) -> pd.DataFrame:
        """
        Run the scheduler on a workload.

        Parameters
        ----------
        jobs : pd.DataFrame
            Must contain:

            - ``job_id``      (any hashable): unique job identifier.
            - ``submit_time`` (float):        arrival time in seconds.
            - ``runtime``     (float):        true job duration in seconds.
            - ``predicted_runtime`` (float, optional): required for SJF-Pred.

        Returns
        -------
        pd.DataFrame
            Per-job result table with columns:

            - ``job_id``
            - ``start_time``
            - ``completion_time``
            - ``waiting_time``
            - ``turnaround_time``
            - ``slowdown``
        """
        jobs = jobs.sort_values("submit_time").reset_index(drop=True)

        current_time: float = 0.0
        results = []
        # Use a list of row dicts to avoid pandas concat dtype issues
        queue_rows: list = []
        remaining = jobs.copy()

        while len(remaining) > 0 or len(queue_rows) > 0:
            # Admit jobs that have arrived by current_time
            mask = remaining["submit_time"] <= current_time
            newly_arrived = remaining[mask]
            queue_rows.extend(newly_arrived.to_dict("records"))
            remaining = remaining[~mask].reset_index(drop=True)

            if not queue_rows:
                # Advance clock directly to the next arriving job (no +1 drift)
                current_time = float(remaining["submit_time"].iloc[0])
                continue

            # Build queue DataFrame fresh each round (avoids concat dtype mismatches)
            queue = pd.DataFrame(queue_rows)

            # Select next job via the chosen scheduler
            job = self.scheduler.select_job(queue)

            # Remove selected job from queue by job_id
            queue_rows = [r for r in queue_rows if r["job_id"] != job["job_id"]]

            start_time = max(current_time, float(job["submit_time"]))
            completion_time = start_time + float(job["runtime"])
            waiting_time = start_time - float(job["submit_time"])
            turnaround_time = completion_time - float(job["submit_time"])

            runtime = float(job["runtime"])
            slowdown = turnaround_time / runtime if runtime > 0 else float("inf")

            results.append(
                {
                    "job_id": job["job_id"],
                    "start_time": start_time,
                    "completion_time": completion_time,
                    "waiting_time": waiting_time,
                    "turnaround_time": turnaround_time,
                    "slowdown": slowdown,
                }
            )

            current_time = completion_time  # advance clock

        return pd.DataFrame(results)