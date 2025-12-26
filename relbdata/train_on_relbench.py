"""
Train KumoRFM on RelBench datasets

"""

import argparse
import logging
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
import os
import gc
import warnings
from typing import Dict, Any, List
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# [Critical] Use file_system strategy for shared memory handling
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

# --- Global Registries ---
TASK_CONFIG_REGISTRY: List[TaskConfig] = []
DATASET_NAME_REGISTRY: List[str] = []
NODE_TYPE_REGISTRY: List[str] = []

def get_registry_id(registry: List, item: Any) -> int:
    try:
        return registry.index(item)
    except ValueError:
        registry.append(item)
        return len(registry) - 1

def make_graph_shared(graph: TemporalHeterogeneousGraph):
    """
    Recursively move all tensors in the graph to shared memory.
    This allows workers to access the graph without pickling/copying data.
    """
    # 1. Node Features
    for nt in graph.node_features:
        if graph.node_features[nt] is not None:
            graph.node_features[nt].share_memory_()
            
    # 2. Node Timestamps
    for nt in graph.node_timestamps:
        if graph.node_timestamps[nt] is not None:
            graph.node_timestamps[nt].share_memory_()
            
    # 3. Edge Indices and Timestamps
    for et in graph.edge_indices:
        graph.edge_indices[et].share_memory_()
        if et in graph.edge_timestamps and graph.edge_timestamps[et] is not None:
            graph.edge_timestamps[et].share_memory_()
            
    return graph

class RelBenchDataset(Dataset):
    def __init__(self, entities, labels, timestamps, task_config, dataset_name, 
                 graph: TemporalHeterogeneousGraph,  # [Optimization] Accept Object (Shared Mem)
                 backward_sampler: BackwardSubgraphSampler,
                 context_sampler: ContextSampler,
                 node_type_registry: List[str], 
                 num_hops=2, max_neighbors=10, num_context=5, context_strategy='mixed'):
        
        self.entities = entities
        self.timestamps = timestamps
        self.task_config = task_config
        self.dataset_name = dataset_name
        
        # Graph is now a shared-memory object.
        # It is NOT copied when passed to workers, only the structure is pickled.
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
        else:
            raw_y = pd.to_numeric(pd.Series(labels_array.reshape(-1)), errors='coerce').fillna(0.0).values
            raw_y = torch.tensor(raw_y, dtype=torch.float)
            self.labels = torch.log1p(torch.abs(raw_y))

    def __len__(self):
        return len(self.entities)

    def __getitem__(self, idx):
        nt_id = self.ent_type_ids[idx]
        node_id = self.ent_node_ids[idx]
        ts_float = self.timestamps[idx].timestamp()
        node_type = self.node_type_registry[nt_id]
        
        target_ent = (node_type, node_id)
        ts = self.timestamps[idx]
        
        # Sampling directly from Shared Memory (Fast RAM Access)
        test_subgraph = self.backward_sampler.sample(
            self.graph, target_ent, ts, self.num_hops, self.max_neighbors
        )
        
        ctx_exs = self.context_sampler.sample_context(
            self.graph, target_ent, ts, self.task_config, 
            self.num_context, strategy=self.context_strategy
        )
        
        return {
            'ds_id': self.ds_id,
            'nt_id': nt_id,
            'node_id': node_id,
            'ts': ts_float,
            'task_id': self.task_id,
            'label': self.labels[idx],
            'test_subgraph': test_subgraph,
            'target_ent': target_ent,
            'ctx_exs': ctx_exs
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

def setup_logger(rank):
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    if rank == 0:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.ERROR)

class RelBenchTrainer:
    def __init__(self, model, config, exp_config, device, rank):
        self.model = model
        self.config = config
        self.exp_config = exp_config
        self.device = device
        self.rank = rank 
        self.base_trainer = KumoRFMTrainer(model, config, exp_config, device)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()

    def train(self, train_loader, val_loader):
        if self.rank == 0: logger.info("Starting DDP training loop...")

        best_val_loss = float('inf')
        for epoch in range(self.exp_config.num_epochs):
            train_loader.sampler.set_epoch(epoch)
            train_loss = self._run_epoch(train_loader, epoch, is_train=True)
            val_loss = self._run_epoch(val_loader, epoch, is_train=False)
            
            if self.rank == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
            
            if self.base_trainer.early_stopping(val_loss): break

    def _run_epoch(self, loader, epoch, is_train):
        if is_train: self.model.train()
        else: self.model.eval()
        
        total_loss = torch.tensor(0.0, device=self.device)
        steps = torch.tensor(0.0, device=self.device)
        
        if self.rank == 0:
            iterable = tqdm(loader, desc=f"Epoch {epoch+1} [{'Train' if is_train else 'Val'}]")
        else:
            iterable = loader
        
        for batch in iterable:
            ds_ids = batch['ds_ids'].to(self.device)
            nt_ids = batch['nt_ids'].to(self.device)
            node_ids = batch['node_ids'].to(self.device)
            ts = batch['timestamps'].to(self.device)
            task_ids = batch['task_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            test_subgraphs = batch['test_subgraphs']
            target_ents = batch['target_ents']
            ctx_exs_list = batch['ctx_exs_list']

            if is_train: self.base_trainer.optimizer.zero_grad()

            output = self.model(
                dataset_ids=ds_ids, 
                entity_type_ids=nt_ids, 
                entity_ids=node_ids,
                timestamps=ts, 
                task_config_ids=task_ids,
                test_subgraphs=test_subgraphs,
                target_ents=target_ents,
                ctx_exs_list=ctx_exs_list
            )
            preds = output['predictions']
            
            loss = torch.tensor(0.0, device=self.device)
            valid = 0
            unique_tasks = torch.unique(task_ids)
            
            for tid in unique_tasks:
                mask = (task_ids == tid)
                sub_pred = preds[mask]
                sub_label = labels[mask]
                t_conf = TASK_CONFIG_REGISTRY[tid.item()]
                
                if t_conf.task_type == 'classification':
                    sub_label = sub_label.long()
                    if sub_pred.shape[-1] > t_conf.num_classes:
                        sub_pred = sub_pred[:, :t_conf.num_classes]
                    l = self.cls_criterion(sub_pred, sub_label)
                else:
                    reg_pred = sub_pred[:, 0]
                    sub_label = sub_label.view_as(reg_pred)
                    l = self.reg_criterion(reg_pred, sub_label) * 0.1
                
                loss += l * mask.sum()
                valid += mask.sum()
            
            if valid > 0:
                loss = loss / valid
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.base_trainer.optimizer.step()
                    self.base_trainer.scheduler.step()
                
                total_loss += loss.item()
                steps += 1.0
                
                if self.rank == 0:
                    iterable.set_postfix({'loss': f'{loss.item():.4f}'})
        
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(steps, op=dist.ReduceOp.SUM)
        
        return (total_loss / max(1.0, steps)).item()

    def _save_checkpoint(self, epoch, metric):
        path = Path(self.exp_config.save_dir) / 'best_model.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.model.module.state_dict()
        torch.save({'epoch': epoch, 'model': state}, path)

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29506' # Changed port again for safety
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    ddp_setup(rank, world_size)
    setup_logger(rank)
    device = torch.device(f"cuda:{rank}")
    set_random_seed(42 + rank) 

    str_graph_map = {} 
    combined_schema = {}
    
    target_ds = ['amazon','stack','f1','trial','avito','event','hm'] if args.dataset == 'ALL' else args.dataset.split(',')
    sampling_config = SamplingConfig(num_hops=2, max_neighbors=10)
    label_table = InContextLabelTable()
    fwd_sampler = ForwardLabelSampler(label_table)
    max_classes = 2

    datasets = []
    backward_sampler = BackwardSubgraphSampler(sampling_config)
    context_sampler = ContextSampler(sampling_config, backward_sampler)
    context_sampler.attach_label_table(label_table)

    for i, ds_name in enumerate(target_ds):
        try:
            if rank == 0: 
                print(f"\n[Rank {rank}] Loading dataset {i+1}/{len(target_ds)}: {ds_name}...", flush=True)
            
            adapter = RelBenchAdapter(ds_name)
            ds = adapter.load_dataset()
            raw_g = adapter.build_temporal_graph()
            raw_s = get_database_schema_from_relbench(ds)
            pref_g, pref_s = apply_dataset_prefix(raw_g, raw_s, ds_name)
            
            # [CRITICAL OPTIMIZATION] Move to shared memory immediately
            make_graph_shared(pref_g)
            
            str_graph_map[ds_name] = pref_g
            combined_schema.update(pref_s)
            
            for nt in pref_g.node_types: 
                get_registry_id(NODE_TYPE_REGISTRY, nt)
            get_registry_id(DATASET_NAME_REGISTRY, ds_name)
            
            tasks = [args.task] if args.task != 'ALL' and args.task != 'auto' else list(ds.tasks.keys())
            if not tasks and args.task=='auto': tasks = list(ds.tasks.keys())[:1]
            
            for t_name in tasks:
                conf = adapter.get_task_config(t_name)
                splits = adapter.get_train_test_split(t_name)
                if conf.num_classes and conf.num_classes > max_classes: max_classes = conf.num_classes
                
                def fix_ent(split):
                    split['entities'] = [(f"{ds_name}__{e}", i) for e, i in split['entities']]
                    return split
                
                fwd_sampler.ingest_relbench_split(fix_ent(splits['train']))
                if 'val' in splits: fwd_sampler.ingest_relbench_split(fix_ent(splits['val']))
                
                datasets.append(RelBenchDataset(
                    splits['train']['entities'], 
                    splits['train']['labels'], 
                    splits['train']['timestamps'], 
                    conf, 
                    ds_name,
                    pref_g, # Pass the shared memory graph object!
                    backward_sampler, 
                    context_sampler,
                    node_type_registry=NODE_TYPE_REGISTRY
                ))
            
            # Note: We CANNOT delete pref_g here because datasets hold references to it
            # and it's needed for training.
            del adapter, ds, raw_g, raw_s
            gc.collect()
            
        except Exception as e:
            if rank == 0: logger.error(f"Error loading {ds_name}: {e}")

    label_table.finalize()
    
    if not datasets: 
        if rank == 0: logger.error("No datasets loaded. Exiting.")
        ddp_cleanup()
        return

    if rank == 0: print(f"\n[Rank {rank}] All datasets loaded. Concatenating...", flush=True)
    
    full_ds = ConcatDataset(datasets)
    sampler = DistributedSampler(full_ds, num_replicas=world_size, rank=rank)
    
    loader = DataLoader(
        full_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4, # Safe to use workers now with Shared Memory
        pin_memory=False,
        persistent_workers=True
    )
    
    conf = KumoRFMConfig(
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers, 
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    exp_config = ExperimentConfig(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=args.output_dir,
        log_dir=str(Path(args.output_dir) / "logs")
    )

    model = KumoRFM(conf, combined_schema)
    tc_map = {i: c for i, c in enumerate(TASK_CONFIG_REGISTRY)}
    ds_name_map = {n: i for i, n in enumerate(DATASET_NAME_REGISTRY)}
    nt_map = {n: i for i, n in enumerate(NODE_TYPE_REGISTRY)}
    
    model.set_global_resources(str_graph_map, ds_name_map, nt_map, tc_map)
    model.set_task_config(TaskConfig('classification', num_classes=max_classes))
    model.backward_sampler = backward_sampler
    model.context_sampler = context_sampler
    
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    trainer = RelBenchTrainer(model, conf, exp_config, device, rank)
    trainer.train(loader, loader)
    
    ddp_cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='trial')
    parser.add_argument('--task', type=str, default='auto')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='./relbench_outputs')
    parser.add_argument('--num-workers', type=int, default=4)
    args, _ = parser.parse_known_args()

    world_size = torch.cuda.device_count()
    print(f"Spawning {world_size} processes for DDP training...", flush=True)
    
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size)

if __name__ == '__main__':
    main()