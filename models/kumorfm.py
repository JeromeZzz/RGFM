"""
KumoRFM Main Model - Refactored for RelGT Integration

"""
import torch
import torch.nn as nn
from typing import Dict, Any, List
from datetime import datetime, timezone
import math

from config.model_config import KumoRFMConfig, TaskConfig
from data.temporal_graph import TemporalHeterogeneousGraph

# Use the new Adapter
from .relgt.relgt_model import RelGTAdapter
from .icl.icl_module import ICLModule, ClassificationHead, RegressionHead, LinkPredictionHead
from .icl.dual_context import DualContextMechanism

class KumoRFM(nn.Module):
    def __init__(self, config: KumoRFMConfig, database_schema: Dict, col_stats: Dict = None):
        super().__init__()
        self.config = config
        self.database_schema = database_schema
        self.graph_map = {}
        self.task_config_map = {}
        self.node_type_to_id = {}
        self.id_to_node_type = {}
        
        # Samplers are attached externally by the training script
        self.backward_sampler = None
        self.context_sampler = None

        # Initialize RelGT Adapter
        # This handles all subgraph encoding: Tokenization, Feature Encoding, GNN layers
        self.relgt_adapter = RelGTAdapter(config, database_schema, col_stats)
        
        # In-Context Learning Modules
        self.icl_module = ICLModule(config)
        self.dual_context = DualContextMechanism(config)
        
        # Task Heads
        self._register_heads()
        self.global_max_dim = 2 

    def set_global_resources(self, graph_map, ds_map, nt_map, tc_map):
        self.graph_map = {ds_map[k]: v for k, v in graph_map.items()}
        self.id_to_dataset_name = {v: k for k, v in ds_map.items()}
        self.id_to_node_type = {v: k for k, v in nt_map.items()}
        self.node_type_to_id = nt_map
        self.task_config_map = tc_map

    def _register_heads(self):
        self.icl_module.register_task_head('classification', ClassificationHead(self.config, 2))
        self.icl_module.register_task_head('regression', RegressionHead(self.config))
        self.icl_module.register_task_head('link_prediction', LinkPredictionHead(self.config))

    def set_task_config(self, config):
        if config.task_type == 'classification' and config.num_classes:
            self.icl_module.register_task_head('classification', ClassificationHead(self.config, config.num_classes))
        self.global_max_dim = self._get_global_max_dim()

    def _get_global_max_dim(self):
        max_d = 1
        for name, head in self.icl_module.task_heads.items():
            if isinstance(head, ClassificationHead):
                try: max_d = max(max_d, head.projection[-1].out_features)
                except: max_d = max(max_d, 2)
            elif isinstance(head, RegressionHead):
                max_d = max(max_d, 1)
        return max(max_d, 2)

    def forward(self, dataset_ids, entity_type_ids, entity_ids, timestamps, task_config_ids, 
                test_subgraphs=None, target_ents=None, ctx_exs_list=None):
        device = dataset_ids.device
        batch_size = len(dataset_ids)

        # Buffers for Context processing
        all_ctx_subgraphs = []
        all_ctx_targets_ent = []
        all_ctx_ts_vals = []
        
        batch_target_types, batch_target_ids = [], []
        batch_ctx_types, batch_ctx_ids = [], []
        batch_ctx_timestamps, batch_ctx_labels = [], []
        task_types = []

        # 1. Unpack Batch Data
        for i in range(batch_size):
            nt_id = entity_type_ids[i].item()
            node_id = entity_ids[i].item()
            task_conf = self.task_config_map[task_config_ids[i].item()]
            
            task_types.append(task_conf.task_type)
            batch_target_types.append(nt_id)
            batch_target_ids.append(node_id)
            
            ctx_exs = ctx_exs_list[i]
            c_types, c_ids, c_ts, c_lbls = [], [], [], []
            for ctx in ctx_exs:
                # Accumulate for Batch Encoding
                all_ctx_subgraphs.append(ctx.subgraph)
                all_ctx_targets_ent.append(ctx.entity)
                all_ctx_ts_vals.append(ctx.timestamp.timestamp())
                
                # Metadata for ICL
                c_nt_name, c_nid = ctx.entity
                c_types.append(self.node_type_to_id.get(c_nt_name, 0))
                c_ids.append(c_nid)
                c_ts.append(ctx.timestamp.timestamp())
                c_lbls.append(ctx.label)
            
            batch_ctx_types.append(c_types)
            batch_ctx_ids.append(c_ids)
            batch_ctx_timestamps.append(c_ts)
            batch_ctx_labels.append(c_lbls)

        # 2. Encode Test Subgraphs (Target) via RelGT
        # Returns [B, HiddenDim]
        test_embs = self.relgt_adapter(test_subgraphs, target_ents, timestamps)

        # 3. Encode Context Subgraphs via RelGT
        if all_ctx_subgraphs:
            ctx_ts_tensor = torch.tensor(all_ctx_ts_vals, dtype=torch.float, device=device)
            ctx_flat_embs = self.relgt_adapter(all_ctx_subgraphs, all_ctx_targets_ent, ctx_ts_tensor)
            
            if batch_size > 0:
                # Reshape [B * num_ctx, D] -> [B, num_ctx, D]
                # Assuming constant num_context per batch (standard practice)
                real_num_ctx = len(all_ctx_subgraphs) // batch_size
                ctx_embs = ctx_flat_embs.view(batch_size, real_num_ctx, -1)
            else:
                ctx_embs = torch.zeros(0, 0, self.config.hidden_dim, device=device)
        else:
            ctx_embs = torch.zeros(batch_size, 0, self.config.hidden_dim, device=device)

        # 4. In-Context Learning (Vectorized)
        # Prepare Tensors
        # Pad lists to tensors if lengths differ (here assuming equal length for simplicity or lists)
        c_ts_tensor = torch.tensor(batch_ctx_timestamps, device=device)
        t_ts_tensor = timestamps.view(batch_size, 1)
        
        # Intra-Entity Mask: 1 if context node is same as target node
        # (Requires mapping types/ids to same space, handled by set_global_resources)
        t_types = torch.tensor(batch_target_types, device=device).view(batch_size, 1)
        t_ids = torch.tensor(batch_target_ids, device=device).view(batch_size, 1)
        c_types_t = torch.tensor(batch_ctx_types, device=device)
        c_ids_t = torch.tensor(batch_ctx_ids, device=device)
        
        intra_mask = ((t_types == c_types_t) & (t_ids == c_ids_t)).float()

        # Handle Mixed Tasks
        if len(set(task_types)) == 1:
            # Optimized path for single task type in batch
            task_type = task_types[0]
            label_tensor = self._batch_process_labels(batch_ctx_labels, task_type).to(device)
            
            # Dual Context Mechanism (Retrieval / Augmentation)
            enhanced_test, _ = self.dual_context(
                ctx_embs, label_tensor.unsqueeze(-1), test_embs.unsqueeze(1),
                c_ts_tensor, t_ts_tensor, intra_mask
            )
            
            # ICL Transformer
            final_preds = self._run_icl_vectorized(ctx_embs, label_tensor, enhanced_test.squeeze(1), task_type)
        else:
            # Loop for mixed tasks
            final_preds_list = []
            for i in range(batch_size):
                task_conf = self.task_config_map[task_config_ids[i].item()]
                task_type = task_conf.task_type
                
                t_e = test_embs[i].unsqueeze(0).unsqueeze(1)
                c_e = ctx_embs[i].unsqueeze(0)
                l_raw = batch_ctx_labels[i]
                
                # Process labels per task
                l_ts = self._process_labels(l_raw, task_conf).to(device)
                curr_intra = intra_mask[i].unsqueeze(0)
                
                enh, _ = self.dual_context(
                    c_e, l_ts, t_e, 
                    c_ts_tensor[i].unsqueeze(0), 
                    t_ts_tensor[i].unsqueeze(0), 
                    curr_intra
                )
                
                # ICL Forward (using lists for module compatibility)
                c_list = [c_e[0, k] for k in range(len(l_raw))]
                pred = self.icl_module(c_list, l_raw, enh.squeeze(0).squeeze(0), task_type)
                
                if pred.dim() == 0: pred = pred.view(1) 
                if pred.dim() == 1: pred = pred.view(1, -1)
                final_preds_list.append(pred)
            
            final_preds = torch.cat(self._smart_pad(final_preds_list, task_types, device), dim=0)

        # Align dimensions
        final_preds = self._smart_pad_tensor(final_preds, self.global_max_dim)

        return {'predictions': final_preds}

    def _smart_pad_tensor(self, tensor, target_dim):
        current_dim = tensor.shape[-1]
        if current_dim < target_dim:
            pad_size = target_dim - current_dim
            pad = torch.zeros((tensor.shape[0], pad_size), device=tensor.device)
            return torch.cat([tensor, pad], dim=-1)
        return tensor

    def _run_icl_vectorized(self, ctx_embs, ctx_labels, test_emb, task_type):
        """Helper to run ICL Module in batch mode"""
        label_emb = self.icl_module.label_encoder(ctx_labels, task_type, device=ctx_embs.device)
        full_ctx = ctx_embs + label_emb
        full_test = test_emb.unsqueeze(1)
        
        B = ctx_embs.size(0)
        query = self.icl_module.query_token.expand(B, -1, -1)
        
        # Seq: [Ctx1, Ctx2, ..., Test, Query]
        seq = torch.cat([full_ctx, full_test, query], dim=1)
        
        # Positional Encoding & Truncation
        seq_len = seq.size(1)
        max_len = self.icl_module.position_encoding.size(1)
        if seq_len > max_len:
             seq = seq[:, -max_len:, :]
             seq_len = max_len
        seq = seq + self.icl_module.position_encoding[:, :seq_len, :]
        
        # Transformer Layers
        for layer in self.icl_module.icl_layers:
            seq = layer(seq)
        
        # Query Token Output
        q_out = seq[:, -1, :]
        
        # Task Head
        pred = q_out
        if task_type in self.icl_module.task_heads:
            pred = self.icl_module.task_heads[task_type](q_out)
        
        if pred.dim() == 1:
            pred = pred.view(-1, 1)
            
        return pred

    def _smart_pad(self, preds_list, task_types, device):
        max_dim = max(p.shape[-1] for p in preds_list)
        out = []
        for i, p in enumerate(preds_list):
            if p.shape[-1] < max_dim:
                pad_val = -1e9 if task_types[i] == 'classification' else 0.0
                pad = torch.full((p.shape[0], max_dim - p.shape[-1]), pad_val, device=device)
                p = torch.cat([p, pad], dim=-1)
            out.append(p)
        return out

    def _batch_process_labels(self, labels_list, task_type):
        flat = [item for sublist in labels_list for item in sublist]
        dtype = torch.long if task_type == 'classification' else torch.float
        return torch.tensor(flat, dtype=dtype).view(len(labels_list), len(labels_list[0]))

    def _process_labels(self, labels, config):
        if config.task_type == 'classification':
            return torch.tensor(self._normalize_labels(labels), dtype=torch.long).unsqueeze(0)
        return torch.tensor(labels, dtype=torch.float).unsqueeze(0)

    def _normalize_labels(self, labels):
        return [int(l) if isinstance(l, (int, float)) else 0 for l in labels]