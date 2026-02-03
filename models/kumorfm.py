"""
KumoRFM Main Model - Fixed Method Indentation
Ensures helper methods are part of the class to fix AttributeError.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import math
import logging
import torch.nn.functional as F

from config.model_config import KumoRFMConfig, TaskConfig
from data.temporal_graph import TemporalHeterogeneousGraph
from .encoders.multimodal_encoder import MultiModalEncoder
from .encoders.table_encoder import MultiTableEncoder
from .relgt.relgt_model import RelGTWrapper
from .icl.icl_module import ICLModule, ClassificationHead, RegressionHead, LinkPredictionHead
from utils.debug_probe import DebugProbe

logger = logging.getLogger(__name__)

class NeighborNodeTypeEncoder(nn.Module):
    def __init__(self, node_type_map, embedding_dim):
        super().__init__()
        num_types = (max(node_type_map.values()) + 2) if node_type_map else 5002
        self.embedding = nn.Embedding(num_types, embedding_dim)
        nn.init.orthogonal_(self.embedding.weight)
    def forward(self, x): return self.embedding(x.clamp(max=self.embedding.num_embeddings-1))

class NeighborHopEncoder(nn.Module):
    def __init__(self, max_hop, dim):
        super().__init__()
        self.embedding = nn.Embedding(max_hop + 10, dim)
        nn.init.xavier_uniform_(self.embedding.weight)
    def forward(self, x): return self.embedding((x+1).clamp(0, self.embedding.num_embeddings-1))

class NeighborTimeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.project = nn.Linear(dim, dim)
        nn.init.normal_(self.project.weight, std=0.01)
    def forward(self, rel_time):
        device = rel_time.device
        if rel_time.dim() == 1: rel_time = rel_time.unsqueeze(0).unsqueeze(-1)
        elif rel_time.dim() == 2: rel_time = rel_time.unsqueeze(-1)
        pos = rel_time
        d = torch.arange(self.dim, device=device).float()
        div = torch.exp(-(d//2)*math.log(10000.0)/max(self.dim-1, 1))
        sinus = torch.zeros(pos.shape[0], pos.shape[1], self.dim, device=device)
        sinus[..., 0::2] = torch.sin(pos * div[0::2])
        sinus[..., 1::2] = torch.cos(pos * div[:self.dim//2])
        return self.project(sinus)

class StructureEncoder(nn.Module):
    def __init__(self, dim, max_degree=128):
        super().__init__()
        self.degree_embedding = nn.Embedding(max_degree + 1, dim)
        nn.init.xavier_uniform_(self.degree_embedding.weight)
    def forward(self, edge_index, num_nodes, device):
        if edge_index.numel() == 0: deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
        else:
            row, col = edge_index
            deg = torch.bincount(row, minlength=num_nodes) + torch.bincount(col, minlength=num_nodes)
        deg = deg.clamp(max=self.degree_embedding.num_embeddings - 1)
        return self.degree_embedding(deg)

class KumoRFM(nn.Module):
    def __init__(self, config: KumoRFMConfig, database_schema: Dict):
        super().__init__()
        self.config = config
        self.database_schema = database_schema
        self.graph_map = {}
        self.task_config_map = {}
        self.node_type_to_id = {}
        
        self._init_encoders()
        
        self.relgt = RelGTWrapper(config, len(database_schema)+5000)
        self.icl_module = ICLModule(config)
        self._register_heads()
        
        self.feature_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.feature_norm = nn.LayerNorm(config.hidden_dim)
        self.final_emb_norm = nn.LayerNorm(config.hidden_dim)
        self.global_max_dim = 2 

    def set_global_resources(self, graph_map, ds_map, nt_map, tc_map):
        self.graph_map = {ds_map[k]: v for k, v in graph_map.items()}
        self.node_type_to_id = nt_map
        self.relgt_type_encoder = NeighborNodeTypeEncoder(nt_map, self.config.hidden_dim)
        self.task_config_map = tc_map

    def _init_encoders(self):
        from config.model_config import ColumnEncoderConfig
        self.multimodal_encoder = MultiModalEncoder(ColumnEncoderConfig(), self.config.hidden_dim)
        for t, cols in self.database_schema.items():
            for c, ctype in cols.items(): self.multimodal_encoder.register_column(f"{t}.{c}", ctype)
        self.table_encoder = MultiTableEncoder(self.config, list(self.database_schema.keys()))
        dim = self.config.hidden_dim
        self.relgt_type_encoder = NeighborNodeTypeEncoder({}, dim)
        self.relgt_hop_encoder = NeighborHopEncoder(5, dim)
        self.relgt_time_encoder = NeighborTimeEncoder(dim)
        self.relgt_struct_encoder = StructureEncoder(dim)

    def _register_heads(self):
        self.icl_module.register_task_head('classification', ClassificationHead(self.config, 2))
        self.icl_module.register_task_head('regression', RegressionHead(self.config))
        self.icl_module.register_task_head('link_prediction', LinkPredictionHead(self.config))

    def set_task_config(self, config):
        if config.task_type == 'classification' and config.num_classes:
            self.icl_module.register_task_head('classification', ClassificationHead(self.config, config.num_classes))
        self.global_max_dim = self._get_global_max_dim()
    
    def _get_global_max_dim(self): return 2

    def forward(self, dataset_ids, entity_type_ids, entity_ids, timestamps, task_config_ids, 
                test_subgraphs=None, target_ents=None, ctx_exs_list=None,
                test_dst_subgraphs=None, target_dst_ents=None):
        
        device = dataset_ids.device
        batch_size = len(dataset_ids)

        all_ctx_subgraphs, all_ctx_targets_ent, all_ctx_labels, all_ctx_task_types = [], [], [], []
        batch_split_indices = [0]
        curr_count = 0

        for i in range(batch_size):
            ctx_exs = ctx_exs_list[i]
            task_idx = task_config_ids[i].item()
            task_conf = self.task_config_map.get(task_idx, list(self.task_config_map.values())[0])
            for ctx in ctx_exs:
                all_ctx_subgraphs.append(ctx.subgraph)
                all_ctx_targets_ent.append(ctx.entity)
                all_ctx_labels.append(ctx.label)
                all_ctx_task_types.append(task_conf.task_type)
            curr_count += len(ctx_exs)
            batch_split_indices.append(curr_count)

        ctx_flat_embs = torch.zeros(0, self.config.hidden_dim, device=device)
        if all_ctx_subgraphs:
            ctx_flat_embs = self._batch_encode_subgraphs(
                all_ctx_subgraphs, all_ctx_targets_ent, None, device, 
                inject_labels=all_ctx_labels, task_types=all_ctx_task_types, tag="CTX"
            )

        test_embs = self._batch_encode_subgraphs(
            test_subgraphs, target_ents, timestamps, device, tag="TEST"
        )
        if test_dst_subgraphs:
             dst_embs = self._batch_encode_subgraphs(test_dst_subgraphs, target_dst_ents, timestamps, device, tag="DST")
             test_embs = (test_embs + dst_embs) / 2.0

        final_preds_list = []
        for i in range(batch_size):
            start, end = batch_split_indices[i], batch_split_indices[i+1]
            if start < end:
                c_seq = ctx_flat_embs[start:end] 
                c_lbls = all_ctx_labels[start:end]
            else:
                c_seq = torch.zeros(0, self.config.hidden_dim, device=device)
                c_lbls = []
            
            t_seq = test_embs[i].unsqueeze(0)
            task_idx = task_config_ids[i].item()
            task_conf = self.task_config_map.get(task_idx)
            
            c_list = torch.unbind(c_seq, dim=0) 
            pred = self.icl_module(c_list, c_lbls, t_seq.squeeze(0), task_conf.task_type)
            if pred.dim() == 0: pred = pred.view(1) 
            if pred.dim() == 1: pred = pred.view(1, -1)
            final_preds_list.append(pred)

        batch_task_types = [self.task_config_map.get(tid.item()).task_type for tid in task_config_ids]
        final_preds = torch.cat(self._smart_pad(final_preds_list, batch_task_types, device), dim=0)
        DebugProbe.check_tensor("Final_Preds", final_preds)
        return {'predictions': final_preds}

    def _batch_encode_subgraphs(self, subgraphs, target_entities, timestamps, device, 
                              inject_labels: Optional[List[Any]] = None,
                              task_types: Optional[List[str]] = None, tag=""):
        if not subgraphs: return torch.zeros(0, self.config.hidden_dim, device=device)

        batch_node_features = {}
        for sub in subgraphs:
            for nt in sub.node_types:
                if nt not in batch_node_features: batch_node_features[nt] = []
                f = sub.node_features.get(nt)
                if f is None: f = torch.zeros(sub.node_counts[nt], self.config.hidden_dim)
                if f.dim() == 1: f = f.unsqueeze(-1)
                if f.shape[-1] < self.config.hidden_dim:
                    pad = torch.zeros(f.shape[0], self.config.hidden_dim - f.shape[-1])
                    f = torch.cat([f, pad], dim=-1)
                elif f.shape[-1] > self.config.hidden_dim: f = f[:, :self.config.hidden_dim]
                batch_node_features[nt].append(f)

        big_input = {}
        for nt, feats in batch_node_features.items():
            if feats:
                cat_f = torch.cat(feats, dim=0).to(device)
                big_input[nt] = cat_f.unsqueeze(0).unsqueeze(2) if cat_f.dim()==2 else cat_f
        
        big_enc = self.table_encoder(big_input)
        nt_offsets = {nt: 0 for nt in big_enc.keys()}
        
        giant_tokens, giant_nt, giant_ei, giant_et = [], [], [], []
        giant_batch_idx = [] 
        tgt_indices, curr_off = [], 0
        VIRTUAL_LABEL_TYPE_ID = len(self.node_type_to_id) + 1 
        VIRTUAL_EDGE_TYPE_ID = 999 

        for i, sub in enumerate(subgraphs):
            g_f, g_t, g_map, g_c = [], [], {}, 0
            for nt in sub.node_types:
                cnt = sub.node_counts[nt]
                if nt in big_enc:
                    start = nt_offsets[nt]
                    end = start + cnt
                    g_f.append(big_enc[nt].squeeze(0)[start:end])
                    nt_offsets[nt] = end
                else: g_f.append(torch.zeros(cnt, self.config.hidden_dim, device=device))
                g_t.extend([self.node_type_to_id.get(nt,0)]*cnt)
                g_map[nt] = (g_c, g_c+cnt)
                g_c += cnt
            
            tokens = torch.cat(g_f, dim=0) if g_f else torch.zeros(0, self.config.hidden_dim, device=device)
            nt_tsr = torch.tensor(g_t, device=device)
            
            edges, ets = [], []
            for idx, et in enumerate(sub.edge_types):
                if et in sub.edge_indices:
                    e = sub.edge_indices[et].to(device).clone()
                    e[0] += g_map[et[0]][0]
                    e[1] += g_map[et[2]][0]
                    edges.append(e)
                    ets.extend([idx]*e.shape[1])
                    e_rev = torch.stack([e[1], e[0]], dim=0)
                    edges.append(e_rev)
                    ets.extend([idx]*e.shape[1])
            
            if inject_labels is not None:
                label_val = inject_labels[i]
                val_scalar = label_val.item() if hasattr(label_val, 'item') else label_val
                l_tsr = torch.tensor([[val_scalar]], device=device)
                l_emb_3d = self.icl_module.label_encoder(l_tsr, task_types[i], device=device) 
                l_emb = l_emb_3d.reshape(1, self.config.hidden_dim)
                
                if i == 0: DebugProbe.check_tensor(f"{tag}_Label_Emb", l_emb)

                tokens = torch.cat([tokens, l_emb], dim=0)
                nt_tsr = torch.cat([nt_tsr, torch.tensor([VIRTUAL_LABEL_TYPE_ID], device=device)], dim=0)
                t_type, _ = target_entities[i]
                t_local_id = getattr(sub, 'target_local_index', 0)
                if t_type in g_map:
                    t_idx_in_graph = g_map[t_type][0] + t_local_id
                    label_node_idx = tokens.shape[0] - 1
                    new_edge = torch.tensor([[t_idx_in_graph, label_node_idx], [label_node_idx, t_idx_in_graph]], device=device)
                    edges.append(new_edge)
                    ets.extend([VIRTUAL_EDGE_TYPE_ID, VIRTUAL_EDGE_TYPE_ID])
            
            ei = torch.cat(edges, dim=1).to(device) if edges else torch.zeros((2,0), dtype=torch.long, device=device)
            et_tsr = torch.tensor(ets, dtype=torch.long, device=device) if edges else torch.zeros(0, dtype=torch.long, device=device)

            ts_val = timestamps[i] if timestamps is not None else datetime.now(timezone.utc)
            if isinstance(ts_val, torch.Tensor): ts_val = datetime.fromtimestamp(ts_val.item(), tz=timezone.utc)
            time = self._get_time_differences(sub, ts_val, g_map, scale_to_days=True).to(device)
            if tokens.shape[0] > time.shape[0]:
                pad_t = torch.zeros(tokens.shape[0] - time.shape[0], device=device)
                time = torch.cat([time, pad_t], dim=0)

            hop = self._get_hop_distances(sub, target_entities[i], g_map).to(device)
            if tokens.shape[0] > hop.shape[0]:
                pad_h = torch.zeros(tokens.shape[0] - hop.shape[0], dtype=torch.long, device=device)
                hop = torch.cat([hop, pad_h], dim=0)

            tokens = self.feature_norm(tokens)
            tokens_proj = self.feature_proj(tokens)
            t_emb = self.relgt_type_encoder(nt_tsr)
            h_emb = self.relgt_hop_encoder(hop)
            tm_emb = self.relgt_time_encoder(time.unsqueeze(0).float()).squeeze(0)
            s_emb = self.relgt_struct_encoder(ei, tokens.shape[0], device)

            if i == 0: 
                DebugProbe.check_tensor(f"{tag}_Comp_Content", tokens_proj)
                DebugProbe.check_tensor(f"{tag}_Comp_Type", t_emb)
                DebugProbe.check_tensor(f"{tag}_Comp_Hop", h_emb)
                DebugProbe.check_tensor(f"{tag}_Comp_Time", tm_emb)
                DebugProbe.check_tensor(f"{tag}_Comp_Struct", s_emb)

            tokens = tokens_proj + t_emb + h_emb + tm_emb + s_emb
            tokens = self.final_emb_norm(tokens)
            
            t_type, _ = target_entities[i]
            t_local_id = getattr(sub, 'target_local_index', 0)
            readout_idx = 0
            if t_type in g_map: readout_idx = g_map[t_type][0] + t_local_id
            
            giant_tokens.append(tokens)
            giant_nt.append(nt_tsr)
            giant_ei.append(ei + curr_off)
            giant_et.append(et_tsr)
            giant_batch_idx.append(torch.full((tokens.shape[0],), i, dtype=torch.long, device=device))
            tgt_indices.append(curr_off + readout_idx)
            curr_off += len(tokens)
        
        input_giant = torch.cat(giant_tokens, 0)
        batch_giant = torch.cat(giant_batch_idx, 0)
        
        DebugProbe.check_tensor(f"{tag}_Pre_RelGT", input_giant)

        out_giant = self.relgt(
            input_giant, 
            torch.cat(giant_ei, 1), 
            torch.cat(giant_nt, 0), 
            torch.cat(giant_et, 0),
            batch_index=batch_giant
        )
        
        final_embs = out_giant[torch.tensor(tgt_indices, device=device)]
        DebugProbe.check_tensor(f"{tag}_Post_RelGT", final_embs)
        
        return final_embs

    # --- Helpers ---
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

    def _smart_pad_tensor(self, tensor, target_dim):
        current_dim = tensor.shape[-1]
        if current_dim < target_dim:
            pad_size = target_dim - current_dim
            pad = torch.zeros((tensor.shape[0], pad_size), device=tensor.device)
            return torch.cat([tensor, pad], dim=-1)
        return tensor

    def _get_hop_distances(self, subgraph, target, mapping):
        hops = torch.full((sum(subgraph.node_counts.values()),), 3, dtype=torch.long)
        for nt in subgraph.node_types:
            if hasattr(subgraph, f'{nt}_hop_distances'):
                h = getattr(subgraph, f'{nt}_hop_distances')
                s, e = mapping[nt]
                hops[s:e] = h
        return hops

    def _get_time_differences(self, subgraph, ts, mapping, scale_to_days=False):
        diffs = torch.zeros(sum(subgraph.node_counts.values()))
        base = ts.timestamp()
        for nt in subgraph.node_types:
            if nt in subgraph.node_timestamps:
                t = subgraph.node_timestamps[nt]
                s, e = mapping[nt]
                delta = base - t
                if scale_to_days: delta = (delta / 86400.0)
                diffs[s:e] = delta
        return diffs