"""
RelBench Dataset Adapter
Converts RelBench data to KumoRFM format
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

try:
    import relbench
    from relbench.base import Database, Table, Dataset, TaskType
    from relbench.datasets import (
        AmazonDataset,
        StackDataset,
        F1Dataset,
        TrialDataset,
        AvitoDataset,
        EventDataset,
        HMDataset
    )
except ImportError:
    raise ImportError("Please install RelBench: pip install relbench")

from data.temporal_graph import TemporalHeterogeneousGraph
from data.database import Database as KumoDatabase, Table as KumoTable
from config.model_config import TaskConfig

logger = logging.getLogger(__name__)


class RelBenchAdapter:
    """
    Adapter from RelBench to KumoRFM
    """

    # RelBench Dataset Mapping
    DATASET_MAPPING = {
        'amazon': AmazonDataset,
        'stack': StackDataset,
        'f1': F1Dataset,
        'trial': TrialDataset,
        'avito': AvitoDataset,
        'event': EventDataset,
        'hm': HMDataset
    }

    def __init__(self, dataset_name: str):
        """
        Args:
            dataset_name: RelBench dataset name
        """
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(f"Unsupported dataset: {dataset_name}. "
                             f"Supported datasets: {list(self.DATASET_MAPPING.keys())}")

        self.dataset_name = dataset_name
        self.dataset = None
        self.database = None
        self.graph = None
        self.tasks = {}

    def load_dataset(self) -> 'RelBenchDataset':
        """Load RelBench dataset"""
        logger.info(f"Loading RelBench dataset: {self.dataset_name}")

        # Create dataset instance
        dataset_class = self.DATASET_MAPPING[self.dataset_name]
        self.dataset = dataset_class()

        # Download data (if necessary)
        self.dataset.download()

        logger.info(f"Dataset loaded: {self.dataset.name}")
        logger.info(f"  Number of tables: {len(self.dataset.db.table_dict)}")
        logger.info(f"  Number of tasks: {len(self.dataset.tasks)}")

        return self.dataset

    def convert_database(self) -> KumoDatabase:
        """Convert RelBench database to KumoRFM database"""
        if self.dataset is None:
            self.load_dataset()

        logger.info("Converting database format...")

        # Create KumoRFM database
        self.database = RelBenchDatabase(self.dataset.db)

        return self.database

    def build_temporal_graph(self) -> TemporalHeterogeneousGraph:
        """Build temporal heterogeneous graph"""
        if self.database is None:
            self.convert_database()

        logger.info("Building temporal heterogeneous graph...")

        self.graph = TemporalHeterogeneousGraph()

        # Add node types
        for table_name, table in self.dataset.db.table_dict.items():
            df = table.df
            num_nodes = len(df)

            # Get node features
            features = self._extract_node_features(table_name, df)

            # Get timestamps (if available)
            timestamps = self._extract_timestamps(table_name, df)

            self.graph.add_node_type(
                table_name,
                num_nodes,
                features,
                timestamps
            )

            logger.info(f"  Added node type {table_name}: {num_nodes} nodes")

        # Add edges (based on foreign key relationships)
        edge_count = 0
        for fkey in self.dataset.db.foreign_keys:
            edge_name = f"{fkey.src_table_name}_to_{fkey.dst_table_name}"
            src_table = self.dataset.db.table_dict[fkey.src_table_name]

            # Build edge index
            src_df = src_table.df
            src_col = fkey.src_col_name
            dst_col = fkey.dst_col_name

            # Get valid edges
            edges = []
            for idx, row in src_df.iterrows():
                if pd.notna(row[src_col]):
                    src_idx = idx
                    dst_idx = int(row[src_col])  # Assume foreign key is an integer index
                    edges.append([src_idx, dst_idx])

            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t()

                # Add edge timestamps (if available)
                edge_timestamps = self._extract_edge_timestamps(
                    fkey.src_table_name, src_df, edges
                )

                self.graph.add_edge_type(
                    edge_name,
                    (fkey.src_table_name, fkey.dst_table_name),
                    edge_index,
                    edge_timestamps
                )

                edge_count += len(edges)
                logger.info(f"  Added edge type {edge_name}: {len(edges)} edges")

        logger.info(f"Graph building complete: {self.graph.total_nodes} nodes, {edge_count} edges")

        return self.graph

    def get_task_config(self, task_name: str) -> TaskConfig:
        """Get task configuration"""
        if self.dataset is None:
            self.load_dataset()

        if task_name not in self.dataset.tasks:
            raise ValueError(f"Task {task_name} does not exist. "
                             f"Available tasks: {list(self.dataset.tasks.keys())}")

        task = self.dataset.tasks[task_name]

        # Convert task type
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            task_type = 'classification'
            num_classes = 2
        elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            task_type = 'classification'
            num_classes = task.num_classes
        elif task.task_type == TaskType.REGRESSION:
            task_type = 'regression'
            num_classes = None
        else:
            task_type = 'regression'  # Default
            num_classes = None

        config = TaskConfig(
            task_type=task_type,
            num_classes=num_classes,
            target_column=task.target_col,
            time_window_start=-task.eval_timestamp_delta_days if hasattr(task, 'eval_timestamp_delta_days') else None,
            time_window_end=0
        )

        return config

    def get_train_test_split(self, task_name: str) -> Dict[str, Any]:
        """Get train-test split"""
        if self.dataset is None:
            self.load_dataset()

        task = self.dataset.tasks[task_name]

        # Get train, validation, test tables
        train_table = task.train_table
        val_table = task.val_table
        test_table = task.test_table

        # Extract entities and labels
        result = {
            'train': self._extract_entities_and_labels(train_table, task),
            'val': self._extract_entities_and_labels(val_table, task),
            'test': self._extract_entities_and_labels(test_table, task)
        }

        logger.info(f"Dataset split - {task_name}:")
        logger.info(f"  Train set: {len(result['train']['entities'])} samples")
        logger.info(f"  Validation set: {len(result['val']['entities'])} samples")
        logger.info(f"  Test set: {len(result['test']['entities'])} samples")

        return result

    def _extract_node_features(self, table_name: str, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """Extract node features"""
        # Select numerical columns as features
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return None

        # Fill missing values
        features = df[numeric_cols].fillna(0).values

        return torch.tensor(features, dtype=torch.float)

    def _extract_timestamps(self, table_name: str, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """Extract timestamps"""
        # Find time columns
        time_cols = []
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    # Try converting to timestamp
                    timestamps = pd.to_datetime(df[col]).astype(np.int64) / 1e9
                    return torch.tensor(timestamps.values, dtype=torch.float)
                except:
                    continue

        return None

    def _extract_edge_timestamps(self, table_name: str, df: pd.DataFrame,
                                 edges: List[List[int]]) -> Optional[torch.Tensor]:
        """Extract edge timestamps"""
        timestamps = self._extract_timestamps(table_name, df)

        if timestamps is not None:
            edge_times = []
            for src_idx, _ in edges:
                edge_times.append(timestamps[src_idx].item())
            return torch.tensor(edge_times, dtype=torch.float)

        return None

    def _extract_entities_and_labels(self, table: Table, task: Any) -> Dict[str, Any]:
        """Extract entities and labels"""
        df = table.df

        # Get entity IDs (assuming the first column is the entity ID)
        entity_col = df.columns[0]
        entities = [(task.entity_table, row[entity_col]) for _, row in df.iterrows()]

        # Get labels
        if task.target_col in df.columns:
            labels = df[task.target_col].values
        else:
            labels = np.zeros(len(df))  # Placeholder

        # Get timestamps
        timestamps = []
        time_col = task.timestamp_col if hasattr(task, 'timestamp_col') else None

        if time_col and time_col in df.columns:
            timestamps = pd.to_datetime(df[time_col]).tolist()
        else:
            # Use default timestamp
            base_time = datetime.now()
            timestamps = [base_time] * len(df)

        return {
            'entities': entities,
            'labels': labels,
            'timestamps': timestamps
        }


class RelBenchDatabase(KumoDatabase):
    """
    RelBench Database Wrapper
    """

    def __init__(self, relbench_db: Database):
        super().__init__()
        self.relbench_db = relbench_db

        # Convert tables
        for table_name, table in relbench_db.table_dict.items():
            self.tables[table_name] = KumoTable(table_name, table.df)

        # Convert relationships
        for fkey in relbench_db.foreign_keys:
            self.relationships.append({
                'source_table': fkey.src_table_name,
                'source_column': fkey.src_col_name,
                'target_table': fkey.dst_table_name,
                'target_column': fkey.dst_col_name,
                'relationship_type': 'many-to-one'  # RelBench default
            })

    def load_from_source(self, source: Any) -> None:
        """Already loaded during initialization"""
        pass

    def get_table(self, table_name: str) -> KumoTable:
        """Get table"""
        return self.tables.get(table_name)

    def get_schema(self) -> Dict[str, List[str]]:
        """Get schema"""
        return {name: list(table.data.columns)
                for name, table in self.tables.items()}

    def get_relationships(self) -> List[Dict[str, Any]]:
        """Get relationships"""
        return self.relationships


class RelBenchGraphConverter:
    """
    RelBench Graph Converter
    Converts RelBench database to temporal heterogeneous graph
    """

    def __init__(self):
        self.node_mapping = {}
        self.edge_mapping = {}

    def convert(self, database: RelBenchDatabase,
                dataset: Dataset) -> TemporalHeterogeneousGraph:
        """
        Convert database to graph

        This method is already implemented in RelBenchAdapter.build_temporal_graph
        A simplified interface is provided here
        """
        adapter = RelBenchAdapter(dataset.name)
        adapter.dataset = dataset
        adapter.database = database

        return adapter.build_temporal_graph()


def get_database_schema_from_relbench(dataset: Dataset) -> Dict[str, Dict[str, str]]:
    """
    Infer database schema from RelBench dataset
    """
    schema = {}

    for table_name, table in dataset.db.table_dict.items():
        df = table.df
        column_types = {}

        for col in df.columns:
            dtype = df[col].dtype

            # Infer column type
            if pd.api.types.is_numeric_dtype(dtype):
                if df[col].nunique() < 100:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'numerical'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                column_types[col] = 'time'
            elif pd.api.types.is_string_dtype(dtype) or dtype == object:
                if df[col].nunique() < 1000:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'text'
            else:
                column_types[col] = 'categorical'

        schema[table_name] = column_types

    return schema