"""
RelBench Dataset Adapter
(Final Fix: Explicit Path Search for Tasks)
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
    import relbench
    from relbench.base import Database, Table, Dataset, TaskType
    from relbench.datasets import get_dataset
except ImportError:
    raise ImportError("Please install RelBench: pip install relbench")

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
    
    NAME_ALIASES = {
        'amazon': ['rel-amazon', 'amazon'],
        'stack': ['rel-stack', 'stack', 'rel-stackexchange'],
        'f1': ['rel-f1', 'f1'],
        'trial': ['rel-trial', 'trial'],
        'avito': ['rel-avito', 'avito'],
        'event': ['rel-event', 'event'],
        'hm': ['rel-hm', 'hm-merchandise'],
    }

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = None
        self.database = None
        self.graph = None
        self.node_id_map = {}

    def load_dataset(self) -> Any:
        logger.info(f"Preparing RelBench dataset: {self.dataset_name}")
        candidates = self.NAME_ALIASES.get(self.dataset_name, [self.dataset_name])
        
        for name in candidates:
            try:
                self.dataset = get_dataset(name, download=False)
                logger.info(f"Successfully loaded '{name}' from local cache.")
                return self.dataset
            except Exception: continue
        
        target_name = candidates[0]
        logger.info(f"Local load failed. Downloading '{target_name}'...")
        try:
            self.dataset = get_dataset(target_name, download=True)
            return self.dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{self.dataset_name}'. Error: {e}")

    def _get_db(self):
        if hasattr(self.dataset, 'get_db'): return self.dataset.get_db()
        return getattr(self.dataset, 'db', None)

    def convert_database(self) -> KumoDatabase:
        if not self.dataset: self.load_dataset()
        db = self._get_db()
        self.database = RelBenchDatabase(db)
        return self.database

    def build_temporal_graph(self) -> TemporalHeterogeneousGraph:
        if not self.database: self.convert_database()
        logger.info("Building temporal heterogeneous graph...")
        self.graph = TemporalHeterogeneousGraph()
        db = self._get_db()
        self.node_id_map = {}
        
        for table_name, table in db.table_dict.items():
            df = table.df
            num_nodes = len(df)
            self.node_id_map[table_name] = {uid: i for i, uid in enumerate(df.index)}
            features = self._extract_node_features(table_name, df)
            timestamps = self._extract_timestamps(table_name, df)
            if timestamps is None: timestamps = torch.zeros(num_nodes)
            self.graph.add_node_type(table_name, num_nodes, features, timestamps)

        for src_table_name, src_table in db.table_dict.items():
            fkeys = getattr(src_table, 'fkey_col_to_pkey_table', {})
            for src_col, dst_table_name in fkeys.items():
                if dst_table_name not in self.node_id_map: continue
                edge_name = f"{src_table_name}_{src_col}_to_{dst_table_name}"
                src_df = src_table.df
                valid_mask = src_df[src_col].notna()
                if not valid_mask.any(): continue
                valid_df = src_df[valid_mask]
                src_indices = np.where(valid_mask)[0]
                dst_raw = valid_df[src_col].values
                dst_map = self.node_id_map[dst_table_name]
                dst_indices = pd.Series(dst_raw).map(dst_map)
                valid_edge = dst_indices.notna()
                if valid_edge.sum() == 0: continue
                final_src = src_indices[valid_edge.values]
                final_dst = dst_indices[valid_edge].astype(int).values
                edge_index = torch.stack([torch.from_numpy(final_src), torch.from_numpy(final_dst)], dim=0).long()
                edge_ts = None
                if src_table_name in self.graph.node_timestamps:
                    edge_ts = self.graph.node_timestamps[src_table_name][final_src]
                self.graph.add_edge_type(edge_name, (src_table_name, dst_table_name), edge_index, edge_ts)
                rev_name = f"{dst_table_name}_rev_{src_table_name}_{src_col}"
                rev_index = torch.stack([torch.from_numpy(final_dst), torch.from_numpy(final_src)], dim=0).long()
                self.graph.add_edge_type(rev_name, (dst_table_name, src_table_name), rev_index, edge_ts)

        logger.info(f"Graph built: {self.graph.total_nodes} nodes")
        return self.graph

    def _manual_load_split_folder(self, task_name: str) -> Dict[str, Any]:
        """
        Manually hunt for task parquet files on disk.
        """
        logger.info(f"Hunting for task file '{task_name}' on disk...")
        
        candidates = self.NAME_ALIASES.get(self.dataset_name, [self.dataset_name])
        search_paths = []
        
        cache_base = Path.home() / '.cache' / 'relbench'
        for name in candidates:
            search_paths.append(cache_base / name / 'tasks' / task_name)
            search_paths.append(Path(name) / 'tasks' / task_name)
            search_paths.append(Path('data') / name / 'tasks' / task_name)

        for task_dir in search_paths:
            if not task_dir.exists(): continue
            
            logger.info(f"Checking task folder: {task_dir}")
            splits = {}
            found = False
            
            class MockTask:
                def __init__(self):
                    self.task_type = TaskType.BINARY_CLASSIFICATION
                    self.entity_table = None
                    self.target_col = 'label'
                    self.timestamp_col = 'timestamp'
                    self.num_classes = 2
            
            mock = MockTask()
            
            for split in ['train', 'val', 'test']:
                fpath = task_dir / f"{split}.parquet"
                if fpath.exists():
                    found = True
                    df = pd.read_parquet(fpath)
                    class MockTable:
                        def __init__(self, df): self.df = df
                    splits[split] = self._extract_data(MockTable(df), mock)
            
            if found:
                logger.info(f"Successfully loaded splits from {task_dir}")
                return splits

        return None

    def get_task_config(self, task_name: str) -> TaskConfig:
        if not self.dataset: self.load_dataset()
        task = None
        if hasattr(self.dataset, 'get_task'):
            try: task = self.dataset.get_task(task_name, process=False)
            except: pass
        if task is None and hasattr(self.dataset, 'tasks'):
            task = self.dataset.tasks.get(task_name)
        if task is None:
            return TaskConfig('classification', 2, 'label', 0)

        task_type = 'regression'
        num_classes = None
        target_col = getattr(task, 'target_col', 'label')
        tt = getattr(task, 'task_type', None)
        if str(tt) == 'TaskType.BINARY_CLASSIFICATION' or tt == TaskType.BINARY_CLASSIFICATION:
            task_type = 'classification'; num_classes = 2
        elif str(tt) == 'TaskType.MULTICLASS_CLASSIFICATION' or tt == TaskType.MULTICLASS_CLASSIFICATION:
            task_type = 'classification'; num_classes = getattr(task, 'num_classes', 2)
        
        return TaskConfig(task_type, num_classes, target_col, 0)

    def get_train_test_split(self, task_name: str) -> Dict[str, Any]:
        if not self.dataset: self.load_dataset()
        logger.info(f"Retrieving task data: {task_name}")
        
        # 1. Standard API
        if hasattr(self.dataset, 'get_task'):
            try:
                task = self.dataset.get_task(task_name, process=True)
                return {
                    'train': self._extract_data(task.train_table, task),
                    'val': self._extract_data(task.val_table, task),
                    'test': self._extract_data(task.test_table, task)
                }
            except: pass
        
        # 2. Dictionary Fallback
        if hasattr(self.dataset, 'tasks'):
            task = self.dataset.tasks.get(task_name)
            if task:
                return {
                    'train': self._extract_data(task.train_table, task),
                    'val': self._extract_data(task.val_table, task),
                    'test': self._extract_data(task.test_table, task)
                }

        # 3. Manual Folder Fallback
        splits = self._manual_load_split_folder(task_name)
        if splits: return splits

        raise RuntimeError(f"Task '{task_name}' NOT found.")

    def _extract_node_features(self, table_name, df) -> torch.Tensor:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty: return torch.zeros((len(df), 16))
        return torch.from_numpy(num_df.fillna(0).values.astype(np.float32))

    def _extract_timestamps(self, table_name, df) -> Optional[torch.Tensor]:
        cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        for col in cols:
            try:
                if pd.__version__ >= '2.0.0':
                    ts = pd.to_datetime(df[col], utc=True, format='mixed', errors='coerce')
                else:
                    ts = pd.to_datetime(df[col], utc=True, errors='coerce')
                if ts.notna().sum() == 0: continue
                ts = ts.fillna(pd.Timestamp(0, tz='UTC'))
                vals = ts.astype(np.int64) // 10**9
                return torch.from_numpy(vals.values).float()
            except: continue
        return None

    def _extract_data(self, table, task) -> Dict:
        df = table.df
        etype = getattr(task, 'entity_table', None)
        if not etype:
            db_tables = list(self._get_db().table_dict.keys())
            etype = 'user' if 'user' in db_tables else db_tables[0]

        tcol = getattr(task, 'target_col', 'label')
        timecol = getattr(task, 'timestamp_col', 'timestamp')
        
        entity_id_col = None
        for c in df.columns:
            if c != tcol and c != timecol and ('id' in c.lower() or c == etype):
                entity_id_col = c
                break
        
        if entity_id_col and etype in self.node_id_map:
            id_map = self.node_id_map[etype]
            mapped = df[entity_id_col].map(id_map)
            valid = mapped.notna()
            entities = [(etype, i) for i in mapped[valid].astype(int).tolist()]
            
            if tcol in df.columns:
                labels = pd.to_numeric(df.loc[valid, tcol], errors='coerce').fillna(0).values
            else:
                labels = np.zeros(len(entities))
                
            timestamps = []
            if timecol and timecol in df.columns:
                try:
                    ts_series = pd.to_datetime(df.loc[valid, timecol], utc=True).fillna(pd.Timestamp.now(tz='UTC'))
                    timestamps = ts_series.tolist()
                except:
                    timestamps = [datetime.now()] * len(entities)
            else:
                timestamps = [datetime.now()] * len(entities)
        else:
            entities = [(etype, i) for i in range(len(df))]
            labels = np.zeros(len(df))
            timestamps = [datetime.now()] * len(df)
            
        return {'entities': entities, 'labels': labels, 'timestamps': timestamps}


class RelBenchDatabase(KumoDatabase):
    def __init__(self, relbench_db):
        super().__init__()
        self.relbench_db = relbench_db
        for name, table in relbench_db.table_dict.items():
            self.tables[name] = KumoTable(name, table.df)
        self.relationships = [] 
    def load_from_source(self, source: Any) -> None: pass
    def get_table(self, name): return self.tables.get(name)
    def get_schema(self): return {n: list(t.data.columns) for n,t in self.tables.items()}
    def get_relationships(self): return []

class RelBenchGraphConverter:
    def convert(self, database: RelBenchDatabase, dataset: Dataset) -> TemporalHeterogeneousGraph:
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
            if pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt):
                 types[c] = 'numerical'
            elif pd.api.types.is_datetime64_any_dtype(dt):
                 types[c] = 'time'
            else:
                 types[c] = 'categorical'
        schema[name] = types
    return schema