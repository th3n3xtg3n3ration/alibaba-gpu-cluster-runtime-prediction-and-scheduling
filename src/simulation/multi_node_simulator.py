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
import itertools
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .scheduler_simulator import SchedulerBase


# Floor for a predicted runtime used in reservation arithmetic. Predictions of
# zero or less mean "essentially instant"; they must stay positive so a window
# can be computed, but must not be replaced by the true runtime.
_MIN_ESTIMATE = 1.0

# Denominator floor for bounded slowdown (Feitelson & Rudolph): turnaround /
# max(runtime, BOUNDED_SLOWDOWN_TAU). Unbounded slowdown (turnaround / runtime)
# lets sub-tau jobs dominate the mean: a 1s job waiting a few hours already
# posts a slowdown in the thousands, so mean unbounded slowdown mostly
# reflects how this handful of very short jobs fared, not overall queueing
# behaviour (statistics-8 / simulator-9). 10s follows the same literature's
# common choice; both metrics are reported side by side rather than one
# replacing the other, so this floor changes nothing about turnaround_time or
# the unbounded `slowdown` column itself.
BOUNDED_SLOWDOWN_TAU = 10.0

__all__ = [
    "BOUNDED_SLOWDOWN_TAU",
    "JobEvent",
    "Machine",
    "MultiNodeClusterSimulator",
    "provision_heterogeneous_gpu_cluster",
]


def _gpu_request(job: pd.Series) -> float:
    """
    Read a job's GPU request, accepting either column name.

    Job tables built by :mod:`src.feature_engineering` name this column
    ``gpu_demand``, while raw trace frames name it ``num_gpu``. Looking up only
    one of the two silently yields 0.0 for every job on the other layout, which
    removes the GPU constraint from the simulation entirely. Mirrors the
    same-named fallback in :class:`~src.simulation.scheduler_simulator.SRFScheduler`.
    """
    for col in ("num_gpu", "gpu_demand"):
        if col in job.index:
            return _finite(job[col])
    return 0.0


def _finite(value: Any) -> float:
    """
    Coerce a resource request to a usable non-negative float.

    A NaN request would defeat resource accounting silently rather than
    loudly: every ``can_fit`` comparison against NaN evaluates False, so the
    machine reports that the job fits, and once a NaN is added to ``gpu_used``
    the node accepts unlimited work from then on. Mapping NaN to 0.0 keeps the
    accounting arithmetic sound.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v) or v < 0.0:
        return 0.0
    return v


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
    seq : int, default 0
        Monotonic tie-breaker for events at an identical ``time``. heapq is a
        binary heap, not a stable structure: with only ``time`` to compare,
        two events sharing a timestamp pop in whatever order the heap's
        internal layout happens to produce, not necessarily the order they
        were pushed in. For simultaneous ARRIVALs this made FIFO's "earliest
        arrival first" tie-break effectively random (simulator-4); ``seq``
        (assigned in push order by the simulator) makes it deterministic.
    """

    time: float
    event_type: str
    job: pd.Series
    machine_id: Optional[int] = None
    seq: int = 0

    def __lt__(self, other: "JobEvent") -> bool:
        return (self.time, self.seq) < (other.time, other.seq)


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
        # (expected_finish, cpu, gpu, job_id) per running job. "Expected" means
        # the scheduler's estimate, not ground truth: EASY backfilling reserves
        # a future window from what it believes the running jobs will take, so
        # a poor runtime estimate degrades backfill quality exactly as it does
        # in a production scheduler. Actual completion always uses the true
        # runtime. job_id is carried so release() can drop the exact record for
        # the job that finished, rather than any other running job that
        # happens to request the same (cpu, gpu) footprint.
        self.running_detail: List[Tuple[float, float, float, Any]] = []

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
        self, job: pd.Series, job_cpu: float, job_gpu: float, job_mem: float = 0.0,
        expected_finish: Optional[float] = None,
    ) -> None:
        """Reserve resources for a scheduled job."""
        self.cpu_used += job_cpu
        self.gpu_used += job_gpu
        self.mem_used += job_mem
        self.running_jobs.append(job["job_id"])
        if expected_finish is not None:
            self.running_detail.append((expected_finish, job_cpu, job_gpu, job["job_id"]))

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
        # Drop this job's own reservation record, by job_id -- not by (cpu,
        # gpu) footprint, which two concurrently running jobs can share. A
        # footprint match could previously delete a *different*, still-running
        # job's record instead of the one that actually finished, leaving a
        # stale entry that corrupts later earliest_fit()/backfill decisions.
        for i, (_, _c, _g, jid) in enumerate(self.running_detail):
            if jid == job_id:
                self.running_detail.pop(i)
                break

    def earliest_fit(self, job_cpu: float, job_gpu: float, now: float) -> Optional[float]:
        """
        Earliest time this machine could accommodate the request.

        Returns ``now`` when the job fits immediately, the expected finish time
        of the running job whose departure first frees enough capacity, or
        ``None`` when the request exceeds this machine's total capacity and can
        therefore never be satisfied here.
        """
        if job_cpu > self.cpu_capacity + 1e-5 or job_gpu > self.gpu_capacity + 1e-5:
            return None
        if self.can_fit(job_cpu, job_gpu):
            return now
        freed_cpu = freed_gpu = 0.0
        # Sort by finish time only: two records can legitimately share the
        # same finish/cpu/gpu, and job_id (the 4th field) has no meaningful
        # ordering here.
        for finish, c, g, _jid in sorted(self.running_detail, key=lambda rec: rec[0]):
            freed_cpu += c
            freed_gpu += g
            if (self.cpu_used - freed_cpu + job_cpu) <= self.cpu_capacity + 1e-5 and \
               (self.gpu_used - freed_gpu + job_gpu) <= self.gpu_capacity + 1e-5:
                return max(finish, now)
        return None

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
        self,
        scheduler: SchedulerBase,
        machines: List[Machine],
        backfill: bool = False,
        estimate_col: str = "runtime",
        backfill_depth: int = 100,
    ) -> None:
        self.scheduler = scheduler
        self.machines = machines
        self.backfill = backfill
        self.estimate_col = estimate_col
        self.backfill_depth = backfill_depth
        self.time: float = 0.0
        self.results: List[dict] = []
        self.utilization_history: List[dict] = []
        self.backfilled_jobs: int = 0
        # How many backfills actually had to respect the reservation window.
        # If this stays near zero the estimate quality cannot influence results,
        # and any 'estimates do not matter' conclusion would be an artefact.
        self.backfilled_on_reserved: int = 0

    def _estimate(self, job: pd.Series) -> float:
        """
        Runtime the scheduler believes this job will take.

        Reservation windows are built from these values, never from ground
        truth, so an inaccurate predictor shrinks or inflates the backfill
        window just as it would in production. ``estimate_col`` selects the
        column: ``"runtime"`` reproduces a perfect-estimate (oracle) scheduler,
        while a prediction column measures what the model is actually worth.
        """
        col = self.estimate_col
        if col and col in job.index:
            v = _finite(job[col])
            # A prediction column is present, so this policy must live with what
            # its model produced. Regressors trained on an unbounded target do
            # emit negatives; falling back to the true runtime here would hand
            # exactly those jobs an oracle-quality reservation window, which
            # flatters whichever model produces the most negative predictions.
            # A non-positive prediction means "essentially instant", so that is
            # what the window is built from.
            return v if v > 0 else _MIN_ESTIMATE
        # No prediction column at all (FIFO, SRF, SJF-Oracle): these policies
        # have no model of their own and are given the true runtime.
        return _finite(job.get("runtime", 0.0))

    def _reservation(self, req_cpu: float, req_gpu: float):
        """Shadow time and reserved machine for the blocked head-of-line job."""
        best_t = None
        best_m = None
        for m in self.machines:
            t = m.earliest_fit(req_cpu, req_gpu, self.time)
            if t is None:
                continue
            if best_t is None or t < best_t:
                best_t, best_m = t, m
        return best_t, best_m

    def _try_backfill(self, pending_df: pd.DataFrame, shadow_time: float, reserved):
        """
        First pending job that may start now without delaying the reservation.

        A candidate placed on a machine other than the reserved one cannot
        affect the reservation at all. On the reserved machine it may run only
        if it is expected to finish by the shadow time. Candidates are examined
        in the active policy's priority order, so backfilling never overrides
        the scheduler's ranking, it only fills gaps the ranking would waste.
        """
        self._last_backfill_on_reserved = False
        ordered = self.scheduler.order_jobs(pending_df)
        if self.backfill_depth:
            # SLURM caps how many queued jobs its backfill scheduler examines per
            # cycle (bf_max_job_test, default 100) because an unbounded scan is
            # too costly on a busy queue. Mirroring that keeps the simulation
            # both tractable and faithful to production behaviour.
            ordered = ordered.head(self.backfill_depth)

        cpus = ordered["num_cpu"].to_numpy(dtype=float) if "num_cpu" in ordered.columns \
            else np.zeros(len(ordered))
        gcol = "num_gpu" if "num_gpu" in ordered.columns else "gpu_demand"
        gpus = ordered[gcol].to_numpy(dtype=float) if gcol in ordered.columns \
            else np.zeros(len(ordered))
        ests = ordered[self.estimate_col].to_numpy(dtype=float) \
            if self.estimate_col in ordered.columns else ordered["runtime"].to_numpy(dtype=float)
        if self.estimate_col in ordered.columns:
            # Same rule as _estimate(): a non-positive prediction is clamped, not
            # replaced by ground truth, so the shadow-time check and the booking
            # in allocate() agree and no model is rewarded for emitting negatives.
            ests = np.where(np.isfinite(ests) & (ests > 0), ests, _MIN_ESTIMATE)
        else:
            _rt = ordered["runtime"].to_numpy(dtype=float)
            ests = np.where(np.isfinite(ests) & (ests > 0), ests, _rt)

        for pos in range(len(ordered)):
            c_cpu = float(cpus[pos]) if np.isfinite(cpus[pos]) and cpus[pos] > 0 else 0.0
            c_gpu = float(gpus[pos]) if np.isfinite(gpus[pos]) and gpus[pos] > 0 else 0.0
            fits_by_shadow = (self.time + float(ests[pos])) <= shadow_time + 1e-9
            for m in self.machines:
                if not m.can_fit(c_cpu, c_gpu):
                    continue
                if m is reserved and not fits_by_shadow:
                    continue
                if m is reserved:
                    self._last_backfill_on_reserved = True
                return ordered.iloc[pos], m
        return None, None

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
            - ``num_gpu`` or ``gpu_demand`` (float, optional) -- GPU request per
              job; may be fractional, since Alibaba PAI supports GPU sharing.
            - ``predicted_runtime`` (float, optional, for SJF-Pred)

        Returns
        -------
        pd.DataFrame
            Per-job result table with:

            - ``job_id``, ``submit_time``, ``start_time``
            - ``completion_time``, ``waiting_time``, ``turnaround_time``
            - ``slowdown`` (unbounded, turnaround / runtime), ``bounded_slowdown``
              (turnaround / max(runtime, BOUNDED_SLOWDOWN_TAU) -- see that
              constant's docstring for why both are reported), ``machine_id``
        """
        # Once, before any state is reset: a policy that ranks on a data column
        # checks here that the column can rank at all -- a constant prediction
        # reproduces FIFO under the ML policy's name (see
        # SJFPredScheduler.validate_workload). That case raises
        # DegeneratePredictionError, the one exception a caller replaying many
        # policies is meant to catch and record as a refused policy; the
        # capacity ValueError and the completeness RuntimeError below are
        # deliberately different types, since neither is a reportable result.
        self.scheduler.validate_workload(jobs)

        # Reset state
        self.time = 0.0
        self.results = []
        self.utilization_history = []

        self.backfilled_jobs = 0
        self.backfilled_on_reserved = 0
        for m in self.machines:
            m.cpu_used = 0.0
            m.gpu_used = 0.0
            m.mem_used = 0.0
            m.running_jobs = []
            m.running_detail = []

        # A job requesting more than any single machine's total capacity can
        # never be placed: the event queue eventually empties with it (and
        # everything queued behind it, under a strict-HoL policy) still in
        # pending_df, and the loop below exits with those jobs silently
        # missing from the returned results -- a caller comparing len(jobs)
        # against len(results) would have no other signal that this
        # happened (simulator-6 / code_bugs-7). Not triggered by the trace
        # this thesis uses (max num_cpu=90 <= 96, max gpu_demand=8 <= 8), but
        # silent under a different cluster profile or workload, so it is
        # checked explicitly rather than left to be discovered downstream.
        max_cpu_capacity = max((m.cpu_capacity for m in self.machines), default=0.0)
        max_gpu_capacity = max((m.gpu_capacity for m in self.machines), default=0.0)
        for _, job in jobs.iterrows():
            req_cpu = _finite(job.get("num_cpu", 0))
            req_gpu = _gpu_request(job)
            if req_cpu > max_cpu_capacity + 1e-5 or req_gpu > max_gpu_capacity + 1e-5:
                raise ValueError(
                    f"Job {job.get('job_id')!r} requests (cpu={req_cpu}, gpu={req_gpu}), "
                    f"which exceeds every provisioned machine's capacity "
                    f"(max cpu={max_cpu_capacity}, max gpu={max_gpu_capacity}); it could "
                    "never be scheduled and would silently vanish from the results."
                )

        # Seed event queue with all arrivals. seq is a monotonic tie-breaker
        # for heapq comparisons at equal timestamps (see JobEvent.seq); using
        # a single counter across both ARRIVAL and FINISH pushes below keeps
        # tie-breaking consistent for the whole run.
        _seq = itertools.count()
        events: List[JobEvent] = []
        for _, job in jobs.iterrows():
            heapq.heappush(events, JobEvent(float(job["submit_time"]), "ARRIVAL", job, seq=next(_seq)))

        # Use a persistent DataFrame for the queue to avoid expensive O(N) conversions
        pending_df = jobs.iloc[0:0].copy()

        while events or not pending_df.empty:
            # --- Try to schedule pending jobs ---
            if not pending_df.empty:
                scheduled_any = True
                while scheduled_any and not pending_df.empty:
                    scheduled_any = False
                    best_job_row = self.scheduler.select_job(pending_df)

                    req_cpu = _finite(best_job_row.get("num_cpu", 0))
                    req_gpu = _gpu_request(best_job_row)

                    allocated_machine: Optional[Machine] = None
                    for m in self.machines:
                        if m.can_fit(req_cpu, req_gpu):
                            m.allocate(best_job_row, req_cpu, req_gpu,
                                       expected_finish=self.time + self._estimate(best_job_row))
                            allocated_machine = m
                            break

                    if allocated_machine is None and self.backfill:
                        # Head-of-line job is blocked. Reserve the earliest slot
                        # for it, then let a lower-priority job use capacity that
                        # would otherwise idle until that slot opens.
                        shadow_time, reserved = self._reservation(req_cpu, req_gpu)
                        if shadow_time is not None:
                            cand, m = self._try_backfill(
                                pending_df.drop(best_job_row.name), shadow_time, reserved
                            )
                            if cand is not None:
                                best_job_row = cand
                                req_cpu = _finite(cand.get("num_cpu", 0))
                                req_gpu = _gpu_request(cand)
                                m.allocate(cand, req_cpu, req_gpu,
                                           expected_finish=self.time + self._estimate(cand))
                                allocated_machine = m
                                self.backfilled_jobs += 1
                                if getattr(self, '_last_backfill_on_reserved', False):
                                    self.backfilled_on_reserved += 1

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
                        bounded_slowdown = turnaround_time / max(actual_runtime, BOUNDED_SLOWDOWN_TAU)

                        heapq.heappush(
                            events,
                            JobEvent(finish_time, "FINISH", best_job_row, allocated_machine.machine_id, seq=next(_seq)),
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
                                "bounded_slowdown": bounded_slowdown,
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
                    req_cpu = _finite(event.job.get("num_cpu", 0))
                    req_gpu = _gpu_request(event.job)
                    machine.release(event.job, req_cpu, req_gpu)

        # Close the history at the makespan. Snapshots are taken at the top of
        # the loop, before the clock advances, so the loop exits one event
        # short: the last recorded time is the second-to-last event, never the
        # final completion. Consumers integrate this history as a left Riemann
        # sum (each snapshot's value holds until the next one), which then both
        # drops the drain interval and never weights the last snapshot's value
        # at all -- the tail of a run is precisely when the cluster empties, so
        # time-weighted utilization came out too high: 0.51% of the makespan
        # went unmeasured on a backfilled 800-job run (gpu_util 0.8804 instead
        # of 0.8765), and 2-3% on shorter replays. With this terminating
        # snapshot the left sum reproduces the analytic integral
        # sum(runtime*gpu) / (makespan * capacity) to machine precision. Its
        # own value carries zero duration by construction.
        cpu_util, gpu_util = self._get_avg_utilization()
        self.utilization_history.append(
            {
                "time": self.time,
                "cpu_util": cpu_util,
                "gpu_util": gpu_util,
                "pending_jobs": len(pending_df),
            }
        )

        # Defense in depth alongside the pre-flight capacity check above: if
        # any job is still missing from the results for a reason that check
        # didn't anticipate, fail loudly rather than silently return a
        # shorter-than-expected table (simulator-6 / code_bugs-7).
        if len(self.results) != len(jobs):
            missing = set(jobs["job_id"]) - {r["job_id"] for r in self.results}
            raise RuntimeError(
                f"Simulation returned {len(self.results)} results for {len(jobs)} input jobs; "
                f"{len(missing)} job(s) never completed (job_id sample: {sorted(missing)[:5]})."
            )

        return pd.DataFrame(self.results)


# ---------------------------------------------------------------------
# Cluster factory
# ---------------------------------------------------------------------


def provision_heterogeneous_gpu_cluster(
    n_high: int = 25,
    n_mid: int = 100,
    n_cpu: int = 0,
) -> List[Machine]:
    """
    Create a heterogeneous cluster of :class:`Machine` objects.

    Parameters
    ----------
    n_high : int, default 25
        Number of High-Performance nodes (8 GPU, 96 CPU cores).
    n_mid : int, default 100
        Number of Mid-Range nodes (2 GPU, 64 CPU cores).
    n_cpu : int, default 0
        Number of CPU-only nodes (0 GPU, 64 CPU cores). Defaults to none: a
        GPU-less node can only ever admit a job requesting zero GPUs, and the
        job table drops ``num_gpu <= 0`` rows, so on this trace such nodes stay
        idle for the entire run while still counting toward cluster capacity.
        Raise it only for a workload that actually contains GPU-free jobs.

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
