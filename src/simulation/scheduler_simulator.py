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

Refusals
--------
DegeneratePredictionError
    Raised by :meth:`SJFPredScheduler.validate_workload` before either
    simulator starts, when the prediction column holds one value for every
    job and therefore ranks nothing. Callers that replay a list of policies
    catch it and record that policy as excluded.

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

import numpy as np
import pandas as pd

from src.models.evaluation import prediction_ranks_nothing


__all__ = [
    "DegeneratePredictionError",
    "SchedulerBase",
    "FIFOScheduler",
    "SJFScheduler",
    "SJFPredScheduler",
    "SRFScheduler",
    "ClusterSimulator",
]


# ============================================================
# Refusal raised before a run starts
# ============================================================


class DegeneratePredictionError(ValueError):
    """
    A policy's ranking column carries no ordering, so the run is refused.

    Raised only by :meth:`SJFPredScheduler.validate_workload`, only for the
    constant-column case, and only before the simulation loop starts. It is a
    distinct type rather than a bare ``ValueError`` because a caller that
    replays a whole list of policies has to be able to record *this* policy as
    refused and carry on, while every other ``ValueError`` a run can produce
    (a missing column, a job larger than any machine) is a wiring or
    provisioning bug that must still abort the run. ``except ValueError``
    around the call would not tell the two apart.

    A refusal is a result and is meant to be reported as one: the attributes
    carry what a results table needs to name it -- which policy was refused,
    on which column, and what the single value was -- so the row can read
    "excluded: predictions collapsed to a constant" instead of being reported
    as an evaluated predictor that happened to score 0.00%.

    Attributes
    ----------
    policy : str
        Caller's name for the refused policy (see
        :class:`SJFPredScheduler`'s ``policy_name``).
    column : str
        Ranking column that turned out to be constant.
    value : float
        The single value every job carried.
    n_jobs : int
        Number of jobs that shared it.
    """

    def __init__(
        self,
        message: str,
        *,
        policy: str,
        column: str,
        value: float,
        n_jobs: int,
    ) -> None:
        super().__init__(message)
        self.policy = policy
        self.column = column
        self.value = value
        self.n_jobs = n_jobs


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

    def order_jobs(self, queue: pd.DataFrame) -> pd.DataFrame:
        """
        Return the queue sorted by this policy's priority, best first.

        Backfilling has to walk the queue in priority order. Doing that through
        repeated ``select_job`` + ``drop`` calls is quadratic in the queue
        length, which dominates the whole simulation once thousands of jobs are
        pending. Subclasses override this with a single vectorised sort.
        """
        return queue

    def validate_workload(self, jobs: pd.DataFrame) -> None:
        """
        Reject a workload this policy cannot rank, before the run starts.

        Both simulators call this once on the full job table, as the first
        statement of ``run``. The base policy ranks on arrival order, which is
        always well defined, so there is nothing to check here; subclasses that
        rank on a data column check that the column actually carries an
        ordering and raise :class:`DegeneratePredictionError` when it does not.
        """
        return None


# ============================================================
# FIFO scheduler
# ============================================================


class FIFOScheduler(SchedulerBase):
    """First-In-First-Out scheduler — selects the earliest-arrived job."""

    def select_job(self, queue: pd.DataFrame) -> pd.Series:
        return queue.iloc[0]

    def order_jobs(self, queue: pd.DataFrame) -> pd.DataFrame:
        return queue


# ============================================================
# Oracle SJF scheduler
# ============================================================


class SJFScheduler(SchedulerBase):
    """Oracle SJF — selects the job with the smallest *true* runtime."""

    def select_job(self, queue: pd.DataFrame) -> pd.Series:
        idx = queue["runtime"].idxmin()
        return queue.loc[idx]

    def order_jobs(self, queue: pd.DataFrame) -> pd.DataFrame:
        return queue.sort_values("runtime", kind="mergesort")


# ============================================================
# ML-based SJF scheduler (SJF-Pred)
# ============================================================


class SJFPredScheduler(SchedulerBase):
    """
    SJF-Pred — selects the job with the smallest *ML-predicted* runtime.

    Requires column ``predicted_runtime`` in the ready queue.

    Parameters
    ----------
    policy_name : str, optional
        The caller's name for the policy this instance implements, e.g.
        ``"SJF-CNN-LSTM (Numeric Sequence)"``. Every prediction-driven policy
        shares this one class, so a refusal that reported only the class name
        would not say *which* of the comparison's policies was refused. Passed
        through to :class:`DegeneratePredictionError` so a caller replaying a
        list of policies can label the refused row without tracking the name
        itself. Defaults to the class name.
    """

    def __init__(self, policy_name: str | None = None) -> None:
        self.policy_name = policy_name or type(self).__name__

    def select_job(self, queue: pd.DataFrame) -> pd.Series:
        idx = queue["predicted_runtime"].idxmin()
        return queue.loc[idx]

    def order_jobs(self, queue: pd.DataFrame) -> pd.DataFrame:
        return queue.sort_values("predicted_runtime", kind="mergesort")

    def validate_workload(self, jobs: pd.DataFrame) -> None:
        """
        Refuse a prediction column that expresses no priority at all.

        ``idxmin`` on a constant column returns the first row of the ready
        queue, and the queue is held in arrival order, so a predictor that
        emits one value for every job turns this policy into FIFO while it is
        still reported — with its own MAE/R2 — as a distinct ML scheduler. A
        shipped checkpoint did exactly that (one value, 4128.124023, for all
        16,437 test jobs), and its run matched the FIFO baseline in every
        digit of every metric; nothing in the pipeline noticed. A degenerate
        predictor must stop the run here rather than be silently renamed.

        The constant case raises :class:`DegeneratePredictionError`, which a
        caller comparing many policies is expected to catch and record as a
        refused row. The two cases below stay bare ``ValueError``s on purpose:
        an absent or all-non-finite prediction column is a broken pipeline,
        not a policy whose predictor collapsed, and must not be swallowed by
        the same handler and filed as a reportable refusal.
        """
        if "predicted_runtime" not in jobs.columns:
            raise ValueError(
                "SJFPredScheduler ranks on 'predicted_runtime', which this "
                "workload does not have."
            )

        values = pd.to_numeric(jobs["predicted_runtime"], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(
                "'predicted_runtime' holds no finite value, so no job can be "
                "ranked ahead of another."
            )
        # A single-job workload is trivially constant and still perfectly
        # schedulable, so only a real queue is judged. The rule itself lives in
        # prediction_ranks_nothing, shared with the metrics-side verdict in
        # src/models/evaluation.py: when the two owned separate thresholds they
        # disagreed about the same model -- notebook 04 printed Exp A LightGBM
        # (Numeric) as EXCLUDED while this class simulated it and notebook 05
        # reported its 20.8% JCT improvement.
        if (
            len(jobs) > 1
            and finite.size == values.size
            and prediction_ranks_nothing(int(np.unique(finite).size))
        ):
            constant = float(finite[0])
            raise DegeneratePredictionError(
                f"{self.policy_name}: 'predicted_runtime' is the constant {constant!r} "
                f"across all {len(jobs)} jobs: this policy would order the queue "
                "exactly as FIFO does while being reported as a distinct predictor. "
                "Repair the predictor, or record the policy as excluded -- it is the "
                "no-prediction baseline, not a predictor that scored no improvement.",
                policy=self.policy_name,
                column="predicted_runtime",
                value=constant,
                n_jobs=int(len(jobs)),
            )


# ============================================================
# Greedy Resource Heuristic (SRF)
# ============================================================


class SRFScheduler(SchedulerBase):
    """
    Smallest Resource First (SRF) — selects the job with the smallest resource demand.
    
    Acts as a greedy heuristic baseline. It mimics a simple rule-based approach 
    that ignores runtime and just clears small jobs (based on GPU demand) first.
    """

    def select_job(self, queue: pd.DataFrame) -> pd.Series:
        # Check which column name is used for GPU demand
        gpu_col = "num_gpu" if "num_gpu" in queue.columns else "gpu_demand"
        
        if gpu_col in queue.columns:
            # idxmin() returns the first occurrence of the minimum value,
            # which inherently breaks ties in FIFO order (arrival time)
            idx = queue[gpu_col].idxmin()
            return queue.loc[idx]
            
        # Fallback to FIFO if no resource columns exist
        return queue.iloc[0]

    def order_jobs(self, queue: pd.DataFrame) -> pd.DataFrame:
        gpu_col = "num_gpu" if "num_gpu" in queue.columns else "gpu_demand"
        if gpu_col in queue.columns:
            return queue.sort_values(gpu_col, kind="mergesort")
        return queue


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
        # Once, before any state is built: a policy that ranks on a data column
        # checks here that the column can rank at all. A constant prediction
        # column raises DegeneratePredictionError -- the one exception a caller
        # replaying many policies is meant to catch and record, which is why
        # nothing further down this method raises that type.
        self.scheduler.validate_workload(jobs)

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