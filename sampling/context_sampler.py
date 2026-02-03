"""
Context Sampler
Selects and constructs context examples for in-context learning
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import random
from dataclasses import dataclass
import math

from data.temporal_graph import TemporalHeterogeneousGraph
from .backward_sampler import BackwardSubgraphSampler
from config.model_config import SamplingConfig, TaskConfig
# Use conditional import or string forward reference to avoid circular imports if necessary
from .context_label_table import ContextLabelRecord, InContextLabelTable


@dataclass
class ContextExample:
    """Context Example Structure"""
    subgraph: TemporalHeterogeneousGraph
    entity: Tuple[str, int]  # (node_type, node_id)
    timestamp: datetime
    label: Any  # Label value
    metadata: Optional[Dict[str, Any]] = None


class ContextSampler:
    """
    Context Sampler
    Responsible for selecting appropriate historical examples as context
    """

    def __init__(self,
                 config: SamplingConfig,
                 backward_sampler: BackwardSubgraphSampler):
        self.config = config
        self.backward_sampler = backward_sampler
        self.context_cache = {}
        self.label_table: Optional[InContextLabelTable] = None

    def attach_label_table(self, table: InContextLabelTable) -> None:
        """
        Attach the online in-context label table.
        This method is required for the training script to inject the label table.
        """
        self.label_table = table

    def sample_context(self,
                       graph: TemporalHeterogeneousGraph,
                       target_entity: Tuple[str, int],
                       prediction_time: datetime,
                       task_config: TaskConfig,
                       num_examples: int = 10,
                       strategy: str = 'mixed') -> List[ContextExample]:
        """
        Sample context examples

        Args:
            graph: Full temporal graph
            target_entity: Target entity (type, id)
            prediction_time: Prediction timestamp
            task_config: Task configuration
            num_examples: Number of context examples to sample
            strategy: Sampling strategy ('temporal', 'structural', 'mixed', 'self', etc.)

        Returns:
            List of ContextExample
        """
        # 1. Use Label Table if available (Fast Path)
        if self.label_table is not None:
            return self._sample_from_label_table(
                graph,
                target_entity,
                prediction_time,
                task_config,
                num_examples,
                strategy
            )

        # 2. Fallback to heuristic strategies (Slow Path)
        context_examples: List[ContextExample] = []

        if strategy == 'temporal':
            context_examples = self._sample_temporal_context(
                graph, target_entity, prediction_time, task_config, num_examples
            )
        elif strategy == 'structural':
            context_examples = self._sample_structural_context(
                graph, target_entity, prediction_time, task_config, num_examples
            )
        elif strategy == 'self':
            context_examples = self._sample_self_context(
                graph, target_entity, prediction_time, task_config, num_examples
            )
        else:
            # Mixed strategy
            num_temporal = num_examples // 3
            num_structural = num_examples // 3
            num_self = num_examples - num_temporal - num_structural

            context_examples = (
                self._sample_temporal_context(
                    graph, target_entity, prediction_time, task_config, num_temporal
                )
                + self._sample_structural_context(
                    graph, target_entity, prediction_time, task_config, num_structural
                )
                + self._sample_self_context(
                    graph, target_entity, prediction_time, task_config, num_self
                )
            )

        return context_examples

    def _sample_from_label_table(self,
                                 graph: TemporalHeterogeneousGraph,
                                 target_entity: Tuple[str, int],
                                 prediction_time: datetime,
                                 task_config: TaskConfig,
                                 num_examples: int,
                                 strategy: str) -> List[ContextExample]:
        """Build context examples using the InContextLabelTable."""
        node_type, _ = target_entity
        strategies = self._expand_strategy(strategy)
        # Allocate budget per strategy
        budget = max(1, math.ceil(num_examples / len(strategies)))
        cutoff = float(prediction_time.timestamp())
        contexts: List[ContextExample] = []

        for strat in strategies:
            # Fetch records from the table
            records = self.label_table.sample(
                node_type=node_type,
                before_time=cutoff,
                k=budget,
                strategy=strat,
                fixed_interval_seconds=getattr(
                    self.config, "context_sampling_interval_seconds", 86400.0
                ),
            )
            
            # Convert records to ContextExamples
            for record in records:
                ctx = self._build_context_example(graph, record, task_config, strat)
                if ctx:
                    contexts.append(ctx)
                if len(contexts) >= num_examples:
                    break
            if len(contexts) >= num_examples:
                break

        # Fallback if table yielded nothing (e.g. cold start / first batch)
        if not contexts:
            return self._sample_temporal_context(
                graph, target_entity, prediction_time, task_config, num_examples
            )

        return contexts[:num_examples]

    def _build_context_example(self,
                               graph: TemporalHeterogeneousGraph,
                               record: ContextLabelRecord,
                               task_config: TaskConfig,
                               strategy: str) -> Optional[ContextExample]:
        """Convert a label record into a ContextExample by sampling its subgraph."""
        timestamp = datetime.fromtimestamp(record.timestamp)
        
        # Get sampling config
        # Handle cases where config might be dict or object
        num_hops = getattr(self.config, 'num_hops', 2)
        if hasattr(self.config, 'max_neighbors_per_hop'):
             num_hops = len(self.config.max_neighbors_per_hop)
             
        max_nodes = getattr(self.config, 'max_neighbors', 300)
        
        try:
            subgraph = self.backward_sampler.sample(
                graph,
                record.entity,
                timestamp,
                num_hops=num_hops,
                max_nodes=max_nodes,
            )
        except Exception:
            return None

        metadata = dict(record.metadata or {})
        metadata.setdefault('source', 'label_table')
        metadata['strategy'] = strategy

        return ContextExample(
            subgraph=subgraph,
            entity=record.entity,
            timestamp=timestamp,
            label=record.label,
            metadata=metadata,
        )

    def _expand_strategy(self, strategy: str) -> List[str]:
        """Map high-level strategies to label table strategies."""
        if strategy in ('uniform', 'most_recent', 'fixed_interval'):
            return [strategy]
        if strategy == 'mixed':
            return ['most_recent', 'uniform', 'fixed_interval']
        if strategy in ('temporal', 'structural', 'self'):
            # These map roughly to 'most_recent' in the table view
            return ['most_recent']
        return ['uniform']

    def _sample_temporal_context(self,
                                 graph: TemporalHeterogeneousGraph,
                                 target_entity: Tuple[str, int],
                                 prediction_time: datetime,
                                 task_config: TaskConfig,
                                 num_examples: int) -> List[ContextExample]:
        """Sample based on temporal proximity."""
        examples = []
        node_type, node_id = target_entity

        if node_type not in graph.node_timestamps:
            return examples

        timestamps = graph.node_timestamps[node_type]
        pred_timestamp = self._datetime_to_timestamp(prediction_time)

        # Calculate time diffs
        # Use simple subtraction if possible, assuming timestamps are float/int seconds
        # Ensure tensor types match
        if isinstance(timestamps, torch.Tensor):
             ts_vals = timestamps
        else:
             ts_vals = torch.tensor(timestamps)
             
        if ts_vals.device != torch.device('cpu'):
             ts_vals = ts_vals.cpu()

        time_diffs = torch.abs(ts_vals - pred_timestamp)

        # Filter invalid (future or self)
        valid_mask = torch.ones(len(ts_vals), dtype=torch.bool)
        valid_mask[node_id] = False
        valid_mask[ts_vals > pred_timestamp] = False

        if not valid_mask.any():
            return examples

        # Select closest
        valid_indices = torch.where(valid_mask)[0]
        valid_time_diffs = time_diffs[valid_mask]

        sorted_indices = torch.argsort(valid_time_diffs)
        selected_indices = sorted_indices[:num_examples]

        for idx in selected_indices:
            actual_idx = valid_indices[idx].item()
            context_time = self._timestamp_to_datetime(ts_vals[actual_idx].item())

            # Sample subgraph
            context_subgraph = self.backward_sampler.sample(
                graph,
                (node_type, actual_idx),
                context_time,
                num_hops=getattr(self.config, 'num_hops', 2)
            )

            # Generate dummy label (since we don't have label table here)
            # In a real scenario without label table, we might not be able to get ground truth
            # unless it's stored in the graph. Here we use a generator placeholder.
            label = self._generate_label(
                graph,
                (node_type, actual_idx),
                context_time,
                task_config
            )

            examples.append(ContextExample(
                subgraph=context_subgraph,
                entity=(node_type, actual_idx),
                timestamp=context_time,
                label=label
            ))

        return examples

    def _sample_structural_context(self,
                                   graph: TemporalHeterogeneousGraph,
                                   target_entity: Tuple[str, int],
                                   prediction_time: datetime,
                                   task_config: TaskConfig,
                                   num_examples: int) -> List[ContextExample]:
        """Sample based on structural similarity (simplified implementation)."""
        # For brevity, this falls back to random/temporal if structural features aren't ready
        # to avoid complex dependencies in this fix.
        return self._sample_temporal_context(graph, target_entity, prediction_time, task_config, num_examples)

    def _sample_self_context(self,
                             graph: TemporalHeterogeneousGraph,
                             target_entity: Tuple[str, int],
                             prediction_time: datetime,
                             task_config: TaskConfig,
                             num_examples: int) -> List[ContextExample]:
        """Sample self-history."""
        examples = []
        node_type, node_id = target_entity

        # Generate past time points
        time_points = []
        for i in range(1, num_examples + 1):
            hist_time = prediction_time - timedelta(weeks=i)
            time_points.append(hist_time)

        for hist_time in time_points:
            try:
                context_subgraph = self.backward_sampler.sample(
                    graph,
                    target_entity,
                    hist_time,
                    num_hops=getattr(self.config, 'num_hops', 2)
                )

                label = self._generate_label(graph, target_entity, hist_time, task_config)

                examples.append(ContextExample(
                    subgraph=context_subgraph,
                    entity=target_entity,
                    timestamp=hist_time,
                    label=label,
                    metadata={'is_self_context': True}
                ))
            except:
                continue

        return examples

    def _extract_structural_features(self,
                                     graph: TemporalHeterogeneousGraph,
                                     entity: Tuple[str, int],
                                     time_point: datetime) -> torch.Tensor:
        # Placeholder
        return torch.zeros(10)

    def _generate_label(self,
                        graph: TemporalHeterogeneousGraph,
                        entity: Tuple[str, int],
                        context_time: datetime,
                        task_config: TaskConfig) -> Any:
        """
        Generate dummy labels for fallback strategies when Label Table is missing.
        """
        if task_config.task_type == 'classification':
            num_classes = task_config.num_classes or 2
            return torch.randint(0, num_classes, (1,)).item()
        elif task_config.task_type == 'regression':
            return torch.randn(1).item()
        return 0

    def _datetime_to_timestamp(self, dt: datetime) -> float:
        return dt.timestamp()

    def _timestamp_to_datetime(self, ts: float) -> datetime:
        return datetime.fromtimestamp(ts)