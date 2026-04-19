"""
src.simulation

Scheduling Simulation Package

This sub-package implements a discrete-event simulator for evaluating scheduling
algorithms in large-scale GPU clusters. Two simulator levels are provided:

- :class:`ClusterSimulator`
    Single-queue, single-resource simulator suitable for policy comparison.
- :class:`MultiNodeClusterSimulator`
    Multi-machine, event-driven simulator with per-node resource tracking.

Scheduling Policies
-------------------
FIFOScheduler
    Baseline First-In-First-Out (arrive order).
SJFScheduler
    Oracle Shortest-Job-First (true runtime known).
SJFPredScheduler
    ML-Based Shortest-Job-First (predicted runtime).

Cluster Helpers
---------------
Machine
    Represents a single cluster node with CPU, GPU, and memory capacity.
provision_heterogeneous_gpu_cluster
    Factory function for a typical heterogeneous cluster configuration.

Used by:
    ``notebooks/05_scheduler_evaluation.ipynb``
"""

from .scheduler_simulator import (
    SchedulerBase,
    FIFOScheduler,
    SJFScheduler,
    SJFPredScheduler,
    ClusterSimulator,
)
from .multi_node_simulator import (
    Machine,
    MultiNodeClusterSimulator,
    provision_heterogeneous_gpu_cluster,
)

__all__ = [
    "SchedulerBase",
    "FIFOScheduler",
    "SJFScheduler",
    "SJFPredScheduler",
    "ClusterSimulator",
    "Machine",
    "MultiNodeClusterSimulator",
    "provision_heterogeneous_gpu_cluster",
]