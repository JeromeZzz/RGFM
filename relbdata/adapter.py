"""
RelBench Dataset Adapter
Converts RelBench data to KumoRFM format
(Final Version: Offline-First, Local Task Loading, Unique Edge Naming)
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import warnings
import os
from pathlib import Path

# Suppress specific pandas warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

try:
    import relbench  # type: ignore
    try:
        from relbench.base import Database, Table, Dataset, TaskType  # type: ignore
    except Exception:
        Database = Table = Dataset = TaskType = object

    try:
        from relbench.datasets import get_dataset
    except Exception:
        get_dataset = None
except ImportError:
    raise ImportError("Please install RelBench: pip install relbench")

# Ensure TaskType exists for comparison
if not hasattr(TaskType, 'BINARY_CLASSIFICATION'):
    class MockTaskType:
        BINARY_CLASSIFICATION = 'BINARY_CLASSIFICATION'
        MULTICLASS_CLASSIFICATION = 'MULTICLASS_CLASSIFICATION'
        REGRESSION = 'REGRESSION'
    TaskType = MockTaskType

from data.temporal_graph import TemporalHeterogeneousGraph
from data.database import Database as KumoDatabase, Table as KumoTable
from config.model_config import TaskConfig

logger = logging.getLogger(__name__)


class RelBenchAdapter:
    """
    Adapter from RelBench to KumoRFM
    """

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
        self.dataset_name = dataset_name
        self.dataset = None
        self.database = None
        self.graph = None
        
        # ID Mapping: table_name -> {raw_id -> tensor_index}
        self.node_id_map = {}

    def load_dataset(self) -> Any:
        """
        Load dataset and force load local tasks.
        """
        logger.info(f"Loading RelBench dataset: {self.dataset_name} (Local/Docker Mode)")
        
        # 1. Load the Database object (Offline)
        self.dataset = self._resolve_dataset(self.dataset_name)
        
        # 2. FORCE load local tasks from the 'tasks' folder
        # We do this unconditionally to ensure we use the mounted files
        self._attempt_load_local_tasks()
            
        if hasattr(self.dataset, 'tasks') and self.dataset.tasks:
            logger.info(f"Loaded tasks from local storage: {list(self.dataset.tasks.keys())}")
        else:
            logger.warning("No local tasks found in 'tasks' folder. Please check your data mount.")
            
        return self.dataset

    def _resolve_dataset(self, key: str):
        """
        Resolve dataset object WITHOUT downloading.
        """
        candidates = self.NAME_ALIASES.get(key, [key])
        
        for name in candidates:
            try:
                # STRICTLY download=False. We assume data is mounted.
                ds = get_dataset(name, download=False)
                if ds: return ds
            except Exception:
                continue
        
        # If we fail, raise a clear error for Docker/Offline usage
        raise RuntimeError(
            f"Could not load dataset '{key}' locally. \n"
            f"Ensure the dataset 'db' folder is mounted at ~/.cache/relbench/{key} or ./data/{key}.\n"
            f"Do not attempt to re-download in this environment."
        )

    def _attempt_load_local_tasks(self):
        """
        Manually scan for 'tasks' folder and load parquet files.
        Overrides any internal RelBench task logic.
        """
        candidates = self.NAME_ALIASES.get(self.dataset_name, [self.dataset_name])
        
        # Define search paths (Docker mount points or local cache)
        roots = [Path.home() / ".cache" / "relbench" / name for name in candidates]
        roots.append(Path("data") / self.dataset_name) # Local fallback
        
        found_root = None
        for r in roots:
            if (r / "tasks").exists():
                found_root = r
                break
        
        if not found_root:
            logger.warning(f"Could not find a 'tasks' directory in {roots}")
            return

        logger.info(f"Found local tasks directory at: {found_root}/tasks")
        
        # Initialize tasks dict if missing
        if not hasattr(self.dataset, 'tasks') or self.dataset.tasks is None:
            self.dataset.tasks = {}
            
        tasks_dir = found_root / "tasks"
        
        # Scan subdirectories in 'tasks' folder
        for task_path in tasks_dir.iterdir():
            if not task_path.is_dir(): continue
            
            task_name = task_path.name
            
            # Check for standard parquet files
            train_f = task_path / "train.parquet"
            val_f = task_path / "val.parquet"
            test_f = task_path / "test.parquet"
            
            if train_f.exists() and val_f.exists() and test_f.exists():
                # Only load if not already manually loaded (or overwrite to be safe)
                try:
                    # Inner wrapper classes to mimic RelBench objects
                    class MockTableWrapper:
                        def __init__(self, df, name=""): 
                            self.df = df
                            self.name = name
                        
                    class LocalTask:
                        def __init__(self, t_path, t_name):
                            # Load Parquet directly
                            self.train_table = MockTableWrapper(pd.read_parquet(t_path / "train.parquet"), "train")
                            self.val_table = MockTableWrapper(pd.read_parquet(t_path / "val.parquet"), "val")
                            self.test_table = MockTableWrapper(pd.read_parquet(t_path / "test.parquet"), "test")
                            
                            # Infer Metadata from Train Table
                            cols = self.train_table.df.columns
                            self.timestamp_col = 'timestamp' if 'timestamp' in cols else None
                            
                            # Heuristic for target column
                            if 'y' in cols: self.target_col = 'y'
                            elif 'label' in cols: self.target_col = 'label'
                            elif 'churn' in cols: self.target_col = 'churn'
                            else: self.target_col = cols[-1] # Fallback
                            
                            # Heuristic for Task Type
                            target_series = self.train_table.df[self.target_col]
                            # Simple heuristic: float & many unique -> Regression, else Classification
                            is_float = pd.api.types.is_float_dtype(target_series)
                            nunique = target_series.nunique()
                            
                            if is_float and nunique > 50:
                                self.task_type = TaskType.REGRESSION
                                self.num_classes = None
                            elif nunique == 2:
                                self.task_type = TaskType.BINARY_CLASSIFICATION
                                self.num_classes = 2
                            else:
                                self.task_type = TaskType.MULTICLASS_CLASSIFICATION
                                self.num_classes = nunique
                                
                            self.entity_table = None # Will be inferred later

                    # Register the task
                    self.dataset.tasks[task_name] = LocalTask(task_path, task_name)
                    logger.debug(f"  -> Registered local task: {task_name}")
                    
                except Exception as e:
                    logger.warning(f"  Failed to load local task {task_name}: {e}")

    def _get_db(self):
        if hasattr(self.dataset, 'get_db'):
            return self.dataset.get_db()
        return getattr(self.dataset, 'db', None)

    def convert_database(self) -> KumoDatabase:
        if not self.dataset: self.load_dataset()
        db = self._get_db()
        self.database = RelBenchDatabase(db)
        return self.database

    def build_temporal_graph(self) -> TemporalHeterogeneousGraph:
        """
        Build graph using table-level fkey_col_to_pkey_table metadata.
        FIXED: Uses unique edge names including column name to handle multiple FKs to same table.
        """
        if not self.database: self.convert_database()
        
        logger.info("Building temporal heterogeneous graph...")
        self.graph = TemporalHeterogeneousGraph()
        db = self._get_db()
        
        # -------------------------------------------------------
        # Phase 1: Build Nodes & ID Maps
        # -------------------------------------------------------
        self.node_id_map = {}
        
        for table_name, table in db.table_dict.items():
            df = table.df
            num_nodes = len(df)
            
            # Map Raw ID (DataFrame Index) -> Tensor Index (0..N-1)
            self.node_id_map[table_name] = {uid: i for i, uid in enumerate(df.index)}
            
            # Features
            features = self._extract_node_features(table_name, df)
            
            # Timestamps
            timestamps = self._extract_timestamps(table_name, df)
            if timestamps is None:
                timestamps = torch.zeros(num_nodes)

            self.graph.add_node_type(table_name, num_nodes, features, timestamps)
            logger.debug(f"  Added node type {table_name}: {num_nodes} nodes")

        # -------------------------------------------------------
        # Phase 2: Build Edges using fkey_col_to_pkey_table
        # -------------------------------------------------------
        edge_count = 0
        
        for src_table_name, src_table in db.table_dict.items():
            # Check for foreign keys defined on this table
            fkeys = getattr(src_table, 'fkey_col_to_pkey_table', {})
            
            for src_col, dst_table_name in fkeys.items():
                if dst_table_name not in self.node_id_map:
                    continue
                
                # [FIX] Unique edge name including source column to prevent collisions
                # e.g., 'posts_parent_id_to_posts' vs 'posts_accepted_answer_id_to_posts'
                edge_name = f"{src_table_name}_{src_col}_to_{dst_table_name}"
                
                src_df = src_table.df
                
                # Filter valid rows (FK not null)
                valid_mask = src_df[src_col].notna()
                if not valid_mask.any():
                    continue
                
                valid_df = src_df[valid_mask]
                
                # 1. Get Source Tensor Indices
                src_indices_np = np.where(valid_mask)[0]
                
                # 2. Get Dest Tensor Indices
                dst_raw_ids = valid_df[src_col].values
                dst_map = self.node_id_map[dst_table_name]
                
                # Map Dst Raw IDs -> Tensor Indices
                dst_indices_series = pd.Series(dst_raw_ids).map(dst_map)
                
                # Filter out edges where FK points to non-existent ID
                valid_edge_mask = dst_indices_series.notna()
                
                if valid_edge_mask.sum() == 0:
                    continue
                    
                final_src = src_indices_np[valid_edge_mask.values]
                final_dst = dst_indices_series[valid_edge_mask].astype(int).values
                
                edge_index = torch.stack([
                    torch.from_numpy(final_src),
                    torch.from_numpy(final_dst)
                ], dim=0).long()
                
                # Edge Timestamps (inherit from source)
                edge_ts = None
                if src_table_name in self.graph.node_timestamps:
                    edge_ts = self.graph.node_timestamps[src_table_name][final_src]

                # Add Forward Edge
                self.graph.add_edge_type(edge_name, (src_table_name, dst_table_name), edge_index, edge_ts)
                
                # Add Reverse Edge
                # [FIX] Also make reverse edge name unique
                rev_name = f"{dst_table_name}_rev_{src_table_name}_{src_col}"
                rev_index = torch.stack([
                    torch.from_numpy(final_dst),
                    torch.from_numpy(final_src)
                ], dim=0).long()
                self.graph.add_edge_type(rev_name, (dst_table_name, src_table_name), rev_index, edge_ts)
                
                cnt = len(final_src)
                edge_count += cnt * 2
                logger.debug(f"  Added edge {edge_name}: {cnt}")

        logger.info(f"Graph built: {self.graph.total_nodes} nodes, {edge_count} edges")
        return self.graph

    def get_task_config(self, task_name: str) -> TaskConfig:
        """Infer task config from loaded tasks"""
        if not self.dataset: self.load_dataset()
        
        task = None
        if hasattr(self.dataset, 'tasks'):
            tasks = getattr(self.dataset, 'tasks')
            
            # Handle 'auto' selection
            if task_name == 'auto' and tasks:
                task_name = list(tasks.keys())[0]
                logger.info(f"Auto-selected task: {task_name}")
            
            task = tasks.get(task_name)
            
        # Determine Config
        task_type = 'regression'
        num_classes = None
        target_col = 'label'
        
        if task:
            target_col = getattr(task, 'target_col', 'label')
            tt = getattr(task, 'task_type', 'regression')
            
            # Map RelBench types to string
            if str(tt) == 'TaskType.BINARY_CLASSIFICATION' or tt == TaskType.BINARY_CLASSIFICATION:
                task_type = 'classification'
                num_classes = 2
            elif str(tt) == 'TaskType.MULTICLASS_CLASSIFICATION' or tt == TaskType.MULTICLASS_CLASSIFICATION:
                task_type = 'classification'
                num_classes = getattr(task, 'num_classes', 2)
            else:
                task_type = 'regression'
                num_classes = None
            
        return TaskConfig(
            task_type=task_type,
            num_classes=num_classes,
            target_column=target_col,
            time_window_end=0
        )

    def get_train_test_split(self, task_name: str) -> Dict[str, Any]:
        """Get dataset splits from loaded tasks"""
        if not self.dataset: self.load_dataset()
        
        task = None
        if hasattr(self.dataset, 'tasks'):
            tasks = getattr(self.dataset, 'tasks')
            if task_name == 'auto' and tasks:
                task_name = list(tasks.keys())[0]
            task = tasks.get(task_name)
            
        if not task:
            logger.warning(f"Task '{task_name}' not found. Using random split.")
            return self._create_random_split()

        # Resolve tables
        def get_table(prefix):
            t = getattr(task, f"{prefix}_table", None)
            if t: return t
            name = getattr(task, f"{prefix}_table_name", None)
            return self._get_db().table_dict.get(name) if name else None

        train = get_table('train')
        val = get_table('val')
        test = get_table('test')
        
        if not train: 
            logger.warning("Train table not found in task object. Using random split.")
            return self._create_random_split()
        
        return {
            'train': self._extract_data(train, task),
            'val': self._extract_data(val, task),
            'test': self._extract_data(test, task)
        }

    def _create_random_split(self):
        logger.warning("Creating random split as fallback.")
        db = self._get_db()
        if not db.table_dict:
            raise RuntimeError("Database has no tables!")
            
        tname, table = next(iter(db.table_dict.items()))
        
        # Mock task
        class MockTask:
            entity_table = tname
            target_col = table.df.columns[-1]
            timestamp_col = None
        
        full = self._extract_data(table, MockTask())
        n = len(full['entities'])
        idx = np.random.permutation(n)
        
        def sub(i):
            return {k: [full[k][x] for x in i] for k in full}
            
        return {
            'train': sub(idx[:int(0.7*n)]),
            'val': sub(idx[int(0.7*n):int(0.85*n)]),
            'test': sub(idx[int(0.85*n):])
        }

    def _extract_node_features(self, table_name, df) -> torch.Tensor:
        # Numeric only
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            return torch.zeros((len(df), 16)) # Fallback
        return torch.from_numpy(num_df.fillna(0).values.astype(np.float32))

    def _extract_timestamps(self, table_name, df) -> Optional[torch.Tensor]:
        """Vectorized timestamp extraction"""
        # Heuristic: look for 'date' or 'time' in column names
        cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        
        for col in cols:
            try:
                # Fast vectorized parsing
                if pd.__version__ >= '2.0.0':
                    ts = pd.to_datetime(df[col], utc=True, format='mixed', errors='coerce')
                else:
                    ts = pd.to_datetime(df[col], utc=True, errors='coerce')
                
                if ts.notna().sum() == 0: continue
                
                ts = ts.fillna(pd.Timestamp(0, tz='UTC'))
                # Fix FutureWarning by using astype
                vals = ts.astype(np.int64) // 10**9
                return torch.from_numpy(vals.values).float()
            except:
                continue
        return None

    def _extract_data(self, table, task) -> Dict:
        df = table.df
        # If the table wrapper has a name (e.g. 'train'), it's not the entity type.
        # We need the entity table name.
        # In RelBench task tables, usually they link to an entity table.
        # Heuristic: Use task.entity_table if set, or guess 'user'/'item' etc.
        
        etype = getattr(task, 'entity_table', None)
        if not etype:
            # Try to guess from dataset db tables
            # This is hard without explicit metadata. 
            # Fallback: use 'user' or 'customer' or the first table found in DB.
            db_tables = list(self._get_db().table_dict.keys())
            if 'user' in db_tables: etype = 'user'
            elif 'customer' in db_tables: etype = 'customer'
            else: etype = db_tables[0] # Best guess
        
        tcol = getattr(task, 'target_col', 'label')
        timecol = getattr(task, 'timestamp_col', None)
        
        # Entities extraction:
        # Task tables usually have a column like 'customer_id' or 'user_id'
        # We need to find that column and map it to tensor indices.
        
        entity_id_col = None
        for c in df.columns:
            if c != tcol and c != timecol and ('id' in c.lower() or c == etype):
                entity_id_col = c
                break
        
        if entity_id_col and etype in self.node_id_map:
            # Map Entity ID -> Tensor Index
            id_map = self.node_id_map[etype]
            raw_ids = df[entity_id_col]
            mapped_indices = raw_ids.map(id_map).fillna(0).astype(int).tolist()
            entities = [(etype, i) for i in mapped_indices]
        else:
            # Fallback: Sequential (Warning: this might be wrong if IDs are not aligned)
            entities = [(etype, i) for i in range(len(df))]
        
        # Labels
        if tcol in df.columns:
            labels = pd.to_numeric(df[tcol], errors='coerce').fillna(0).values
        else:
            labels = np.zeros(len(df))
            
        # Timestamps
        timestamps = [datetime.now()] * len(df)
        if timecol and timecol in df.columns:
            try:
                if pd.__version__ >= '2.0.0':
                    ts = pd.to_datetime(df[timecol], utc=True, format='mixed', errors='coerce')
                else:
                    ts = pd.to_datetime(df[timecol], utc=True, errors='coerce')
                now = pd.Timestamp.now(tz='UTC')
                ts = ts.fillna(now)
                timestamps = ts.tolist()
            except:
                pass
                
        return {'entities': entities, 'labels': labels, 'timestamps': timestamps}


class RelBenchDatabase(KumoDatabase):
    def __init__(self, relbench_db):
        super().__init__()
        self.relbench_db = relbench_db
        for name, table in relbench_db.table_dict.items():
            self.tables[name] = KumoTable(name, table.df)
        self.relationships = [] 

    def load_from_source(self, source: Any) -> None:
        pass

    def get_table(self, name): return self.tables.get(name)
    def get_schema(self): return {n: list(t.data.columns) for n,t in self.tables.items()}
    def get_relationships(self): return []


class RelBenchGraphConverter:
    """
    RelBench Graph Converter
    Restored class to satisfy imports in train scripts
    """
    def __init__(self):
        self.node_mapping = {}
        self.edge_mapping = {}

    def convert(self, database: RelBenchDatabase, dataset: Dataset) -> TemporalHeterogeneousGraph:
        """
        Convert database to graph
        Delegates to RelBenchAdapter's robust logic
        """
        adapter = RelBenchAdapter(dataset.name)
        adapter.dataset = dataset
        adapter.database = database
        return adapter.build_temporal_graph()


def get_database_schema_from_relbench(dataset) -> Dict:
    schema = {}
    db = dataset.get_db() if hasattr(dataset, 'get_db') else getattr(dataset, 'db', None)
    if not db: return {}
    
    for name, table in db.table_dict.items():
        types = {}
        for c in table.df.columns:
            dt = table.df[c].dtype
            if pd.api.types.is_numeric_dtype(dt): types[c] = 'numerical'
            elif pd.api.types.is_datetime64_any_dtype(dt): types[c] = 'time'
            else: types[c] = 'categorical'
        schema[name] = types
    return schema