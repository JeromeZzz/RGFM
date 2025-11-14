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
    import relbench  # type: ignore
    # These imports may vary by version; guard them.
    try:
        from relbench.base import Database, Table, Dataset, TaskType  # type: ignore
    except Exception:
        Database = Table = Dataset = TaskType = object  # soft fallback types

    try:
        from relbench.datasets import (
            AmazonDataset,
            StackDataset,
            F1Dataset,
            TrialDataset,
            AvitoDataset,
            EventDataset,
            HMDataset,
            get_dataset,
        )  # type: ignore
    except Exception:
        # Some versions export only get_dataset
        from relbench.datasets import get_dataset  # type: ignore
        AmazonDataset = StackDataset = F1Dataset = TrialDataset = AvitoDataset = EventDataset = HMDataset = None  # type: ignore
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

    # RelBench Dataset Mapping (best-effort; may be None depending on version)
    DATASET_MAPPING = {
        'amazon': AmazonDataset,
        'stack': StackDataset,
        'f1': F1Dataset,
        'trial': TrialDataset,
        'avito': AvitoDataset,
        'event': EventDataset,
        'hm': HMDataset,
    }

    # Name aliases for get_dataset
    NAME_ALIASES = {
        'amazon': ['amazon', 'rel-amazon'],
        'stack': ['stack', 'rel-stack', 'stackexchange', 'rel-stackexchange'],
        'f1': ['f1', 'rel-f1'],
        'trial': ['trial', 'rel-trial'],
        'avito': ['avito', 'rel-avito'],
        'event': ['event', 'rel-event'],
        'hm': ['hm', 'rel-hm', 'hm-merchandise'],
    }

    def __init__(self, dataset_name: str):
        """
        Args:
            dataset_name: RelBench dataset name
        """
        if dataset_name not in self.NAME_ALIASES:
            raise ValueError(f"Unsupported dataset: {dataset_name}. "
                             f"Supported datasets: {list(self.NAME_ALIASES.keys())}")

        self.dataset_name = dataset_name
        self.dataset = None
        self.database = None
        self.graph = None
        self.tasks = {}

    def load_dataset(self) -> 'RelBenchDataset':
        """Load RelBench dataset"""
        logger.info(f"Loading RelBench dataset: {self.dataset_name}")

        # Resolve dataset via get_dataset first, then class fallback
        self.dataset = self._resolve_dataset_instance(self.dataset_name)

        # Download data if supported
        if self.dataset is not None and hasattr(self.dataset, 'download'):
            try:
                self.dataset.download()  # type: ignore
            except Exception as e:
                logger.warning(f"Dataset download failed or skipped: {e}")

        # Best-effort dataset info across RelBench versions
        try:
            ds_name = getattr(self.dataset, 'name', self.dataset.__class__.__name__)
            logger.info(f"Dataset loaded: {ds_name}")
        except Exception:
            pass
        try:
            db = self._get_db()
            logger.info(f"  Number of tables: {len(db.table_dict)}")
        except Exception:
            pass
        try:
            tasks = getattr(self.dataset, 'tasks')
            logger.info(f"  Number of tasks: {len(tasks)}")
        except Exception:
            pass

        return self.dataset

    # Internal helpers
    def _resolve_dataset_instance(self, dataset_key: str):
        """Create a RelBench dataset instance robustly across versions."""
        # Try get_dataset with name aliases
        try:
            for name in self.NAME_ALIASES.get(dataset_key, [dataset_key]):
                try:
                    ds = get_dataset(name, download=False)  # type: ignore
                    if ds is not None:
                        return ds
                except Exception:
                    continue
        except Exception:
            pass

        # Fallback to class mapping if available
        dataset_class = self.DATASET_MAPPING.get(dataset_key)
        if dataset_class is not None:
            try:
                return dataset_class()
            except Exception:
                pass

        raise RuntimeError(f"Failed to resolve RelBench dataset for key '{dataset_key}'. "
                           f"Tried aliases {self.NAME_ALIASES.get(dataset_key)} and class mapping.")

    def _get_db(self):
        """Return the underlying RelBench Database object from the dataset."""
        if hasattr(self.dataset, 'get_db'):
            return self.dataset.get_db()
        if hasattr(self.dataset, 'db'):
            return getattr(self.dataset, 'db')
        raise RuntimeError("Dataset does not expose get_db() or .db")

    def convert_database(self) -> KumoDatabase:
        """Convert RelBench database to KumoRFM database"""
        if self.dataset is None:
            self.load_dataset()

        logger.info("Converting database format...")

        # Create KumoRFM database
        db = self._get_db()
        self.database = RelBenchDatabase(db)

        return self.database

    def build_temporal_graph(self) -> TemporalHeterogeneousGraph:
        """Build temporal heterogeneous graph"""
        if self.database is None:
            self.convert_database()

        logger.info("Building temporal heterogeneous graph...")

        self.graph = TemporalHeterogeneousGraph()

        # Add node types
        db = self._get_db()
        for table_name, table in db.table_dict.items():
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

            logger.debug(f"  Added node type {table_name}: {num_nodes} nodes")

        # Add edges (based on foreign key relationships)
        edge_count = 0
        # Foreign keys may be .foreign_keys or .relationships
        foreign_keys = []
        db = self._get_db()
        if hasattr(db, 'foreign_keys'):
            foreign_keys = getattr(db, 'foreign_keys')
        elif hasattr(db, 'relationships'):
            foreign_keys = getattr(db, 'relationships')

        for fkey in foreign_keys:
            # Support both object attributes and dicts
            def _f(obj, name):
                return getattr(obj, name) if hasattr(obj, name) else obj.get(name)

            src_table_name = _f(fkey, 'src_table_name')
            dst_table_name = _f(fkey, 'dst_table_name')
            src_col_name = _f(fkey, 'src_col_name')
            dst_col_name = _f(fkey, 'dst_col_name')

            if src_table_name is None or dst_table_name is None:
                continue

            edge_name = f"{src_table_name}_to_{dst_table_name}"
            src_table = db.table_dict[src_table_name]

            # Build edge index
            src_df = src_table.df
            src_col = src_col_name
            dst_col = dst_col_name

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
                logger.debug(f"  Added edge type {edge_name}: {len(edges)} edges")

        logger.info(f"Graph building complete: {self.graph.total_nodes} nodes, {edge_count} edges")

        return self.graph

    def get_task_config(self, task_name: str) -> TaskConfig:
        """Get task configuration"""
        if self.dataset is None:
            self.load_dataset()

        # Newer RelBench may not expose .tasks; build a default TaskConfig
        task = None
        if hasattr(self.dataset, 'tasks'):
            tasks = getattr(self.dataset, 'tasks')
            if task_name not in tasks:
                logger.warning(f"Task {task_name} not found in dataset.tasks; using default task config")
            else:
                task = tasks[task_name]

        if task is None:
            # Fallback: attempt to infer from dataset attributes
            target_col = getattr(self.dataset, 'target_col', 'label')
            task_type = 'classification'
            num_classes = 2
            return TaskConfig(
                task_type=task_type,
                num_classes=num_classes,
                target_column=target_col,
                time_window_start=None,
                time_window_end=0,
            )

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

        task = None
        if hasattr(self.dataset, 'tasks'):
            task = getattr(self.dataset, 'tasks').get(task_name)
        if task is None:
            # Fallback: random split on first table as entity table
            logger.warning("Dataset has no tasks; creating a random split on first table")
            db = self._get_db()
            table_name, table = next(iter(db.table_dict.items()))
            dummy_task = type('T', (), {})()
            dummy_task.entity_table = getattr(self.dataset, 'entity_table', table_name)
            dummy_task.target_col = getattr(self.dataset, 'target_col', table.df.columns[-1] if len(table.df.columns) > 0 else None)

            full = self._extract_entities_and_labels(table, dummy_task)
            n = len(full['entities'])
            idx = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(idx)
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)
            train_idx = idx[:n_train]
            val_idx = idx[n_train:n_train + n_val]
            test_idx = idx[n_train + n_val:]

            def take(split_idx):
                return {
                    'entities': [full['entities'][i] for i in split_idx],
                    'labels': [full['labels'][i] for i in split_idx],
                    'timestamps': [full['timestamps'][i] for i in split_idx],
                }

            return {
                'train': take(train_idx),
                'val': take(val_idx),
                'test': take(test_idx),
            }

        # Get train, validation, test tables
        train_table = getattr(task, 'train_table', None)
        val_table = getattr(task, 'val_table', None)
        test_table = getattr(task, 'test_table', None)

        if train_table is None or val_table is None or test_table is None:
            # Try attributes that hold names
            db = self._get_db()
            def table_from_name(name):
                return db.table_dict[name] if name in db.table_dict else None
            train_table = train_table or table_from_name(getattr(task, 'train_table_name', ''))
            val_table = val_table or table_from_name(getattr(task, 'val_table_name', ''))
            test_table = test_table or table_from_name(getattr(task, 'test_table_name', ''))

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
        # Select numerical columns as features (coerce potential mixed types)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            # Try to coerce all columns to numeric and pick those with success
            coerced = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))
            numeric_cols = coerced.columns[(~coerced.isna()).any()].tolist()
            if not numeric_cols:
                return None
            values = coerced[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32)
        else:
            values = df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors='coerce')).fillna(0.0).to_numpy(dtype=np.float32)

        return torch.from_numpy(values.astype(np.float32))

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

        # Use row positions as node ids to match graph construction
        node_type = getattr(task, 'entity_table', None) or getattr(table, 'name', None) or 'entity'
        entities = [(node_type, i) for i in range(len(df))]

        # Get labels
        if hasattr(task, 'target_col') and task.target_col in df.columns:
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
        if hasattr(relbench_db, 'foreign_keys'):
            fk_iter = relbench_db.foreign_keys
        elif hasattr(relbench_db, 'relationships'):
            fk_iter = relbench_db.relationships
        else:
            fk_iter = []

        for fkey in fk_iter:
            def _f(obj, name):
                return getattr(obj, name) if hasattr(obj, name) else obj.get(name)

            self.relationships.append({
                'source_table': _f(fkey, 'src_table_name'),
                'source_column': _f(fkey, 'src_col_name'),
                'target_table': _f(fkey, 'dst_table_name'),
                'target_column': _f(fkey, 'dst_col_name'),
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

    # Support dataset.get_db() or .db
    db = dataset.get_db() if hasattr(dataset, 'get_db') else getattr(dataset, 'db', None)
    if db is None:
        return {}

    for table_name, table in db.table_dict.items():
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

    # --- Helpers ---

    
    
