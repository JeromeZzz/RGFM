"""
In-context label table and samplers.

This module materializes the "Online In-Context Label Generator" stage described
in the RGFM workflow. It maintains a high-throughput table of historical labels
for each entity type and exposes deterministic sampling strategies that can be
used by the context sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict
from bisect import bisect_right
from datetime import datetime
import random
import numpy as np
import torch


@dataclass(frozen=True)
class ContextLabelRecord:
    """Record stored inside the in-context label table."""

    entity: Tuple[str, int]
    timestamp: float  # POSIX seconds
    label: Any
    metadata: Optional[Dict[str, Any]] = None


class InContextLabelTable:
    """
    High-throughput container for historical labels.

    The table keeps per-node-type timelines sorted by timestamp so that
    downstream samplers can grab historical contexts efficiently with
    strategies like uniform / most_recent / fixed_interval.
    """

    def __init__(self, max_records_per_type: int = 2_000_000):
        self.max_records_per_type = max_records_per_type
        self._timelines: Dict[str, List[ContextLabelRecord]] = defaultdict(list)
        self._finalized: bool = False

    def add_record(
        self,
        entity: Tuple[str, int],
        timestamp: datetime,
        label: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Buffer a record; sorting happens during finalize()."""
        node_type, _ = entity
        record = ContextLabelRecord(
            entity=entity,
            timestamp=float(timestamp.timestamp()),
            label=self._normalize_label(label),
            metadata=metadata,
        )
        self._timelines[node_type].append(record)
        self._finalized = False

    def extend_from_split(
        self,
        entities: Sequence[Tuple[str, int]],
        timestamps: Sequence[datetime],
        labels: Sequence[Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Bulk-ingest a dataset split (train/val/test)."""
        for entity, ts, label in zip(entities, timestamps, labels):
            self.add_record(entity, ts, label, metadata)

    def finalize(self) -> None:
        """Sort and truncate timelines for efficient sampling."""
        for node_type, records in self._timelines.items():
            if not records:
                continue
            records.sort(key=lambda r: r.timestamp)
            if len(records) > self.max_records_per_type:
                self._timelines[node_type] = records[-self.max_records_per_type :]
        self._finalized = True

    def _get_slice(
        self, node_type: str, end_time: float
    ) -> List[ContextLabelRecord]:
        """Return all records for node_type with timestamp <= end_time."""
        if not self._finalized:
            self.finalize()
        records = self._timelines.get(node_type, [])
        if not records:
            return []
        # binary search upper bound
        timestamps = [rec.timestamp for rec in records]
        hi = bisect_right(timestamps, end_time)
        if hi <= 0:
            return []
        return records[:hi]

    @staticmethod
    def _normalize_label(label: Any) -> Any:
        """Convert tensors/np scalars to Python scalars for serialization."""
        if isinstance(label, torch.Tensor):
            if label.numel() == 1:
                return label.detach().cpu().item()
            return label.detach().cpu().tolist()
        if isinstance(label, np.generic):
            return label.item()
        return label

    def sample(
        self,
        node_type: str,
        before_time: float,
        k: int,
        strategy: str = "uniform",
        fixed_interval_seconds: float = 86400.0,
    ) -> List[ContextLabelRecord]:
        """Sample records according to the requested strategy."""
        history = self._get_slice(node_type, before_time)
        if not history or k <= 0:
            return []

        if strategy == "uniform":
            if len(history) <= k:
                return history.copy()
            indices = random.sample(range(len(history)), k)
            indices.sort()
            return [history[i] for i in indices]

        if strategy == "most_recent":
            return history[-k:]

        if strategy == "fixed_interval":
            selected: List[ContextLabelRecord] = []
            next_time = before_time
            idx = len(history) - 1
            while idx >= 0 and len(selected) < k:
                record = history[idx]
                if record.timestamp <= next_time:
                    selected.append(record)
                    next_time = record.timestamp - fixed_interval_seconds
                idx -= 1
            return selected

        # fallback: uniform
        return history[-min(len(history), k) :]


class ForwardLabelSampler:
    """
    Responsible for retrieving future (ground-truth) labels and populating the
    in-context table. For RelBench we simply ingest the provided splits, which
    already contain timestamped labels.
    """

    def __init__(self, label_table: InContextLabelTable):
        self.label_table = label_table

    def ingest_relbench_split(
        self,
        split: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entities = split["entities"]
        timestamps = split["timestamps"]
        labels = split["labels"]
        self.label_table.extend_from_split(entities, timestamps, labels, metadata)
