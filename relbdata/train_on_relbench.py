"""
Train KumoRFM on RelBench datasets
(Fixed: Variable Scoping, Initialization, and Robust Stats)
"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
import os
import gc
import time
import warnings
from typing import Dict, Any, List
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score

try:
    import torch_frame
    from torch_frame.data.stats import StatType
except ImportError:
    sys.exit("[ERROR] pytorch-frame missing")

import torch.multiprocessing
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path as _PathForSys
_HERE = _PathForSys(__file__).resolve()
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.model_config import KumoRFMConfig, ExperimentConfig, TaskConfig, SamplingConfig
from models.kumorfm import KumoRFM
from training.trainer import KumoRFMTrainer
from utils.training_utils import set_random_seed
from relbdata.adapter import RelBenchAdapter, get_database_schema_from_relbench
from sampling.context_label_table import InContextLabelTable, ForwardLabelSampler
from sampling.backward_sampler import BackwardSubgraphSampler
from sampling.context_sampler import ContextSampler
from data.temporal_graph import TemporalHeterogeneousGraph

logger = logging.getLogger(__name__)

TASK_CONFIG_REGISTRY: List[TaskConfig] = []
DATASET_NAME_REGISTRY: List[str] = []
NODE_TYPE_REGISTRY: List[str] = []

def get_registry_id(registry: List, item: Any) -> int:
    try: return registry.index(item)
    except ValueError:
        registry.append(item)
        return len(registry) - 1

def compute_heuristic_col_stats(schema):
    stats = {}
    current_ts = float(time.time())
    
    median_key = getattr(StatType, 'MEDIAN_TIMESTAMP', getattr(StatType, 'MEDIAN_TIME', getattr(StatType, 'MEDIAN', StatType.MEAN)))
    year_range_key = getattr(StatType, 'YEAR_RANGE', None)

    for table, cols in schema.items():
        stats[table] = {}
        for col, dtype in cols.items():
            s_dtype = str(dtype).lower()
            if 'float' in s_dtype or 'int' in s_dtype or 'numerical' in s_dtype:
                stats[table][col] = {StatType.MEAN: 0.0, StatType.STD: 1.0}
            elif 'time' in s_dtype or 'date' in s_dtype:
                ts_stats = {}
                ts_stats[median_key] = current_ts
                if year_range_key: ts_stats[year_range_key] = [2000, 2030]
                ts_stats[StatType.MEAN] = current_ts
                ts_stats[StatType.STD] = 1.0
                stats[table][col] = ts_stats
            else:
                dummy_vocab = [str(i) for i in range(10)]
                dummy_counts = [1] * 10
                stats[table][col] = {StatType.COUNT: (dummy_vocab, dummy_counts)}
    return stats

def make_graph_shared(graph):
    for nt in graph.node_features:
        if graph.node_features[nt] is not None: graph.node_features[nt].share_memory_()
    for nt in graph.node_timestamps:
        if graph.node_timestamps[nt] is not None: graph.node_timestamps[nt].share_memory_()
    for et in graph.edge_indices:
        graph.edge_indices[et].share_memory_()
    return graph

class RelBenchDataset(Dataset):
    def __init__(self, entities, labels, timestamps, task_config, dataset_name, 
                 graph, backward_sampler, context_sampler, node_type_registry, 
                 num_hops=2, max_neighbors=10, num_context=5, context_strategy='mixed'):
        self.entities = entities
        self.timestamps = timestamps
        self.task_config = task_config
        self.dataset_name = dataset_name
        self.graph = graph 
        self.backward_sampler = backward_sampler
        self.context_sampler = context_sampler
        self.num_hops = num_hops
        self.max_neighbors = max_neighbors
        self.num_context = num_context
        self.context_strategy = context_strategy
        self.node_type_registry = node_type_registry
        
        self.ds_id = get_registry_id(DATASET_NAME_REGISTRY, dataset_name)
        self.task_id = get_registry_id(TASK_CONFIG_REGISTRY, task_config)
        
        self.ent_type_ids = []
        self.ent_node_ids = []
        for et, eid in entities:
            if et not in NODE_TYPE_REGISTRY: NODE_TYPE_REGISTRY.append(et)
            self.ent_type_ids.append(NODE_TYPE_REGISTRY.index(et))
            self.ent_node_ids.append(eid)
            
        labels_array = np.asarray(labels)
        if task_config.task_type == 'classification':
            labels_str = labels_array.astype(str)
            _, inverse = np.unique(labels_str, return_inverse=True)
            self.labels = torch.tensor(inverse, dtype=torch.long)
        elif task_config.task_type == 'link_prediction':
            self.labels = torch.tensor(labels_array.astype(float), dtype=torch.float)
        else:
            raw_y = pd.to_numeric(pd.Series(labels_array.reshape(-1)), errors='coerce').fillna(0.0).values
            raw_y = torch.tensor(raw_y, dtype=torch.float)
            self.labels = torch.log1p(torch.abs(raw_y))

    def __len__(self): return len(self.entities)

    def __getitem__(self, idx):
        nt_id = self.ent_type_ids[idx]
        node_id = self.ent_node_ids[idx]
        ts_float = self.timestamps[idx].timestamp()
        node_type = self.node_type_registry[nt_id]
        target_ent = (node_type, node_id)
        ts = self.timestamps[idx]
        test_subgraph = self.backward_sampler.sample(
            self.graph, target_ent, ts, self.num_hops, self.max_neighbors
        )
        ctx_exs = self.context_sampler.sample_context(
            self.graph, target_ent, ts, self.task_config, 
            self.num_context, strategy=self.context_strategy
        )
        return {
            'ds_id': self.ds_id, 'nt_id': nt_id, 'node_id': node_id, 'ts': ts_float,
            'task_id': self.task_id, 'label': self.labels[idx],
            'test_subgraph': test_subgraph, 'target_ent': target_ent, 'ctx_exs': ctx_exs
        }

def collate_fn(batch):
    return {
        'ds_ids': torch.tensor([b['ds_id'] for b in batch], dtype=torch.long),
        'nt_ids': torch.tensor([b['nt_id'] for b in batch], dtype=torch.long),
        'node_ids': torch.tensor([b['node_id'] for b in batch], dtype=torch.long),
        'timestamps': torch.tensor([b['ts'] for b in batch], dtype=torch.float),
        'task_ids': torch.tensor([b['task_id'] for b in batch], dtype=torch.long),
        'labels': torch.stack([b['label'] for b in batch]),
        'test_subgraphs': [b['test_subgraph'] for b in batch],
        'target_ents': [b['target_ent'] for b in batch],
        'ctx_exs_list': [b['ctx_exs'] for b in batch]
    }

def apply_dataset_prefix(graph, schema, ds_name):
    prefix = f"{ds_name}__"
    new_schema = {f"{prefix}{k}": v for k, v in schema.items()}
    new_graph = TemporalHeterogeneousGraph()
    for nt in graph.node_types:
        new_nt = f"{prefix}{nt}"
        new_graph.add_node_type(new_nt, graph.node_counts[nt], graph.node_features.get(nt), graph.node_timestamps.get(nt))
    for et in graph.edge_types:
        src, rel, dst = et
        new_et = (f"{prefix}{src}", f"{prefix}{rel}", f"{prefix}{dst}")
        new_graph.add_edge_type(new_et[1], (new_et[0], new_et[2]), graph.edge_indices[et], graph.edge_timestamps.get(et))
    return new_graph, new_schema

def setup_logger(rank, log_dir):
    root = logging.getLogger()
    if root.hasHandlers(): root.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler(); sh.setFormatter(fmt); sh.setLevel(logging.INFO)
    root.addHandler(sh)
    if rank == 0: root.setLevel(logging.DEBUG)
    else: root.setLevel(logging.ERROR)

def verify_device_compatibility(device):
    if device.type == 'cpu': return True
    try:
        t = torch.ones(1, device=device)
        _ = t + t
        return True
    except RuntimeError: return False

class RelBenchTrainer:
    def __init__(self, model, config, exp_config, device, rank, is_distributed=False):
        self.model = model
        self.config = config
        self.exp_config = exp_config
        self.device = device
        self.rank = rank 
        self.is_distributed = is_distributed
        self.base_trainer = KumoRFMTrainer(model, config, exp_config, device)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        for epoch in range(self.exp_config.num_epochs):
            if self.is_distributed and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            train_loss = self._run_epoch(train_loader, epoch, True)
            val_loss = 0.0
            if val_loader:
                if self.device.type == 'cuda': torch.cuda.empty_cache()
                val_loss = self._run_epoch(val_loader, epoch, False)
            
            if self.rank == 0:
                logger.info(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")
                if val_loss < best_val_loss and val_loader:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
            
            if self.base_trainer.early_stopping(val_loss): break

    def _run_epoch(self, loader, epoch, is_train):
        if is_train: self.model.train()
        else: self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        steps = torch.tensor(0.0, device=self.device)
        
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            iterable = tqdm(loader, desc=f"Epoch {epoch+1}", disable=(self.rank != 0))
            for batch in iterable:
                ds_ids = batch['ds_ids'].to(self.device, non_blocking=True)
                nt_ids = batch['nt_ids'].to(self.device, non_blocking=True)
                node_ids = batch['node_ids'].to(self.device, non_blocking=True)
                ts = batch['timestamps'].to(self.device, non_blocking=True)
                task_ids = batch['task_ids'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                if is_train: self.base_trainer.optimizer.zero_grad()

                out = self.model(
                    dataset_ids=ds_ids, entity_type_ids=nt_ids, entity_ids=node_ids,
                    timestamps=ts, task_config_ids=task_ids,
                    test_subgraphs=batch['test_subgraphs'],
                    target_ents=batch['target_ents'],
                    ctx_exs_list=batch['ctx_exs_list']
                )
                preds = out['predictions']
                loss = torch.tensor(0.0, device=self.device)
                valid = 0
                for tid in torch.unique(task_ids):
                    mask = (task_ids == tid)
                    sub_pred = preds[mask]
                    sub_lbl = labels[mask]
                    t_conf = TASK_CONFIG_REGISTRY[tid.item()]
                    
                    if t_conf.task_type == 'classification':
                        if sub_pred.shape[-1] > t_conf.num_classes:
                            sub_pred = sub_pred[:, :t_conf.num_classes]
                        l = self.cls_criterion(sub_pred, sub_lbl.long())
                    elif t_conf.task_type == 'link_prediction':
                        l = torch.nn.functional.binary_cross_entropy_with_logits(sub_pred[:,0], sub_lbl.float())
                    else:
                        l = self.reg_criterion(sub_pred[:,0], sub_lbl.view(-1))
                    
                    loss += l * mask.sum()
                    valid += mask.sum()
                
                if valid > 0:
                    loss = loss / valid
                    if is_train:
                        loss.backward()
                        self.base_trainer.optimizer.step()
                        self.base_trainer.scheduler.step()
                    total_loss += loss.item()
                    steps += 1.0
        
        if self.is_distributed:
            dist.all_reduce(total_loss); dist.all_reduce(steps)
        return (total_loss / max(1.0, steps)).item()

    def _save_checkpoint(self, epoch, metric):
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        path = Path(self.exp_config.save_dir) / 'best_model.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'epoch': epoch, 'model': model_state}, path)

def ddp_setup(rank, world_size):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'; os.environ['MASTER_PORT'] = '29509'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        if torch.cuda.is_available(): torch.cuda.set_device(rank)

def ddp_cleanup(world_size):
    if world_size > 1: dist.destroy_process_group()

def main_worker(rank, world_size, args):
    ddp_setup(rank, world_size)
    setup_logger(rank, args.output_dir)
    
    device_candidate = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device_candidate.type == 'cuda' and not verify_device_compatibility(device_candidate):
        if rank == 0: logger.warning("CUDA incompatible (RTX 5090 vs PyTorch). Fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = device_candidate
    
    set_random_seed(42 + rank)

    str_graph_map = {}
    combined_schema = {}
    target_ds = ['amazon','stack','f1','trial','avito','event','hm'] if args.dataset == 'ALL' else args.dataset.split(',')
    
    samp_conf = SamplingConfig(num_hops=args.num_hops, max_neighbors=args.max_neighbors, max_neighbors_per_hop=[args.max_neighbors]*args.num_hops)
    label_table = InContextLabelTable()
    fwd_sampler = ForwardLabelSampler(label_table)
    
    train_datasets = []
    # [CRITICAL FIX] Initialize val_datasets
    val_datasets = []
    
    backward_sampler = BackwardSubgraphSampler(samp_conf)
    context_sampler = ContextSampler(samp_conf, backward_sampler)
    context_sampler.attach_label_table(label_table)
    
    for ds_name in target_ds:
        try:
            if rank == 0: print(f"Loading {ds_name}...")
            adapter = RelBenchAdapter(ds_name)
            ds = adapter.load_dataset()
            
            raw_g = adapter.build_temporal_graph()
            # [CRITICAL FIX] Avoid variable scoping error by directly using the function result
            pref_g, pref_s = apply_dataset_prefix(raw_g, get_database_schema_from_relbench(ds), ds_name)
            
            make_graph_shared(pref_g)
            str_graph_map[ds_name] = pref_g
            combined_schema.update(pref_s)
            
            task_list = [args.task] if args.task not in ['ALL', 'auto'] else list(ds.tasks.keys())
            if args.task == 'auto' and task_list: task_list = task_list[:1]
            
            for t_name in task_list:
                conf = adapter.get_task_config(t_name)
                splits = adapter.get_train_test_split(t_name)
                
                def fix_ent(split):
                    split['entities'] = [(f"{ds_name}__{e}", i) for e, i in split['entities']]
                    return split
                
                fwd_sampler.ingest_relbench_split(fix_ent(splits['train']))
                train_datasets.append(RelBenchDataset(
                    splits['train']['entities'], splits['train']['labels'], splits['train']['timestamps'],
                    conf, ds_name, pref_g, backward_sampler, context_sampler, NODE_TYPE_REGISTRY,
                    num_context=args.num_context, max_neighbors=args.max_neighbors
                ))
                if 'val' in splits:
                    val_datasets.append(RelBenchDataset(
                        splits['val']['entities'], splits['val']['labels'], splits['val']['timestamps'], 
                        conf, ds_name, pref_g, backward_sampler, context_sampler, NODE_TYPE_REGISTRY,
                        num_context=args.num_context, max_neighbors=args.max_neighbors
                    ))
            
            # [CRITICAL FIX] Correct cleanup
            del adapter, ds, raw_g
            gc.collect()
        except Exception as e:
            if rank == 0: logger.error(f"Failed {ds_name}: {e}")

    label_table.finalize()
    if not train_datasets:
        if rank == 0: logger.error("No datasets!")
        ddp_cleanup(world_size); return

    full_train = ConcatDataset(train_datasets)
    if world_size > 1: sampler = DistributedSampler(full_train, num_replicas=world_size, rank=rank)
    else: sampler = RandomSampler(full_train)
    
    loader = DataLoader(full_train, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=collate_fn, num_workers=0, pin_memory=(device.type=='cuda'))
    
    if val_datasets:
        full_val = ConcatDataset(val_datasets)
        if world_size > 1: val_sampler = DistributedSampler(full_val, num_replicas=world_size, rank=rank)
        else: val_sampler = SequentialSampler(full_val)
        val_loader = DataLoader(full_val, batch_size=args.batch_size, sampler=val_sampler,
                                collate_fn=collate_fn, num_workers=0, pin_memory=(device.type=='cuda'))
    else:
        val_loader = None
    
    conf = KumoRFMConfig(hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads, sampling_config=samp_conf)
    col_stats = compute_heuristic_col_stats(combined_schema)
    
    model = KumoRFM(conf, combined_schema, col_stats=col_stats)
    
    tc_map = {i: c for i, c in enumerate(TASK_CONFIG_REGISTRY)}
    ds_map = {n: i for i, n in enumerate(DATASET_NAME_REGISTRY)}
    nt_map = {n: i for i, n in enumerate(NODE_TYPE_REGISTRY)}
    model.set_global_resources(str_graph_map, ds_map, nt_map, tc_map)
    model.set_task_config(TaskConfig('classification', num_classes=2))
    
    model.backward_sampler = backward_sampler
    model.context_sampler = context_sampler
    
    model = model.to(device)
    if world_size > 1 and device.type == 'cuda':
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    trainer = RelBenchTrainer(model, conf, ExperimentConfig(save_dir=args.output_dir), device, rank, is_distributed=(world_size > 1))
    trainer.train(loader, val_loader)
    
    ddp_cleanup(world_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='trial')
    parser.add_argument('--task', default='auto')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--output-dir', default='./relbench_outputs')
    
    parser.add_argument('--max-neighbors', type=int, default=10)
    parser.add_argument('--num-context', type=int, default=5)
    parser.add_argument('--num-hops', type=int, default=2)
    parser.add_argument('--nprocs', type=int, default=None)
    
    args, _ = parser.parse_known_args()
    
    ws = args.nprocs if args.nprocs else 1
    if not torch.cuda.is_available(): ws = 1
    elif ws > torch.cuda.device_count(): ws = torch.cuda.device_count()
    
    print(f"Spawning {ws} processes...")
    if ws > 1: mp.spawn(main_worker, args=(ws, args), nprocs=ws)
    else: main_worker(0, 1, args)

if __name__ == '__main__':
    main()