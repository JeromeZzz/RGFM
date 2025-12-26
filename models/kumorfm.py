"""
KumoRFM Main Model - High Performance Batching

"""
import torch
import torch.nn as nn
from typing import Dict, Any, List
from datetime import datetime, timezone
import math

from config.model_config import KumoRFMConfig, TaskConfig
from data.temporal_graph import TemporalHeterogeneousGraph
from .encoders.multimodal_encoder import MultiModalEncoder
from .encoders.table_encoder import MultiTableEncoder
from .encoders.positional_encoding import MultiElementTokenizer
from .relgt.relgt_model import RelGTWrapper
from .icl.icl_module import ICLModule, ClassificationHead, RegressionHead, LinkPredictionHead
from .icl.dual_context import DualContextMechanism

class NeighborNodeTypeEncoder(nn.Module):
    def __init__(self, node_type_map, embedding_dim):
        super().__init__()
        num_types = (max(node_type_map.values()) + 1) if node_type_map else 5000
        safe_num = max(num_types, 5000)
        self.embedding = nn.Embedding(safe_num, embedding_dim)
        self.num_embeddings = safe_num
    def forward(self, x):
        if x.max() >= self.num_embeddings: x = x.clamp(max=self.num_embeddings-1)
        return self.embedding(x)

class NeighborHopEncoder(nn.Module):
    def __init__(self, max_hop, dim):
        super().__init__()
        self.embedding = nn.Embedding(max_hop + 10, dim)
    def forward(self, x): 
        return self.embedding((x+1).clamp(0, self.embedding.num_embeddings-1))

class NeighborTimeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
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
        return sinus

class KumoRFM(nn.Module):
    def __init__(self, config: KumoRFMConfig, database_schema: Dict):
        super().__init__()
        self.config = config
        self.database_schema = database_schema
        self.graph_map = {}
        self.task_config_map = {}
        self.id_to_node_type = {}
        self.node_type_to_id = {}
        
        self.backward_sampler = None
        self.context_sampler = None

        self._init_encoders()
        self.relgt = RelGTWrapper(config, len(database_schema)+5000)
        self.icl_module = ICLModule(config)
        self.dual_context = DualContextMechanism(config)
        self._register_heads()
        
        self.global_max_dim = 2 

    def set_global_resources(self, graph_map, ds_map, nt_map, tc_map):
        self.graph_map = {ds_map[k]: v for k, v in graph_map.items()}
        self.id_to_dataset_name = {v: k for k, v in ds_map.items()}
        self.id_to_node_type = {v: k for k, v in nt_map.items()}
        self.node_type_to_id = nt_map
        self.task_config_map = tc_map
        self.relgt_type_encoder = NeighborNodeTypeEncoder(nt_map, self.config.hidden_dim)

    def _init_encoders(self):
        from config.model_config import ColumnEncoderConfig
        self.multimodal_encoder = MultiModalEncoder(ColumnEncoderConfig(), self.config.hidden_dim)
        for t, cols in self.database_schema.items():
            for c, ctype in cols.items(): self.multimodal_encoder.register_column(f"{t}.{c}", ctype)
        self.table_encoder = MultiTableEncoder(self.config, list(self.database_schema.keys()))
        self.tokenizer = MultiElementTokenizer(self.config, len(self.database_schema))
        dim = self.config.hidden_dim
        self.relgt_type_encoder = NeighborNodeTypeEncoder({}, dim)
        self.relgt_hop_encoder = NeighborHopEncoder(2, dim)
        self.relgt_time_encoder = NeighborTimeEncoder(dim)

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

        all_ctx_subgraphs = []
        all_ctx_targets_ent = []
        
        batch_target_types, batch_target_ids = [], []
        batch_ctx_types, batch_ctx_ids = [], []
        batch_ctx_timestamps, batch_ctx_labels = [], []
        task_types = []

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
                all_ctx_subgraphs.append(ctx.subgraph)
                all_ctx_targets_ent.append(ctx.entity)
                
                c_nt_name, c_nid = ctx.entity
                c_types.append(self.node_type_to_id.get(c_nt_name, 0))
                c_ids.append(c_nid)
                c_ts.append(ctx.timestamp.timestamp())
                c_lbls.append(ctx.label)
            
            batch_ctx_types.append(c_types)
            batch_ctx_ids.append(c_ids)
            batch_ctx_timestamps.append(c_ts)
            batch_ctx_labels.append(c_lbls)

        # 2. Batched Encoding
        test_embs = self._batch_encode_subgraphs(test_subgraphs, target_ents, timestamps, device)
        
        if all_ctx_subgraphs:
            ctx_flat_embs = self._batch_encode_subgraphs(all_ctx_subgraphs, all_ctx_targets_ent, None, device)
            if batch_size > 0:
                real_num_ctx = len(all_ctx_subgraphs) // batch_size
                ctx_embs = ctx_flat_embs.view(batch_size, real_num_ctx, -1)
            else:
                ctx_embs = torch.zeros(0, 0, self.config.hidden_dim, device=device)
        else:
            ctx_embs = torch.zeros(batch_size, 0, self.config.hidden_dim, device=device)

        # 3. Vectorized ICL & Dual Context
        t_types = torch.tensor(batch_target_types, device=device).view(batch_size, 1)
        t_ids = torch.tensor(batch_target_ids, device=device).view(batch_size, 1)
        c_types = torch.tensor(batch_ctx_types, device=device)
        c_ids = torch.tensor(batch_ctx_ids, device=device)
        c_ts = torch.tensor(batch_ctx_timestamps, device=device)
        t_ts = timestamps.view(batch_size, 1)

        intra_mask = ((t_types == c_types) & (t_ids == c_ids)).float()

        if len(set(task_types)) == 1:
            task_type = task_types[0]
            label_tensor = self._batch_process_labels(batch_ctx_labels, task_type).to(device)
            enhanced_test, _ = self.dual_context(
                ctx_embs, label_tensor.unsqueeze(-1), test_embs.unsqueeze(1),
                c_ts, t_ts, intra_mask
            )
            final_preds = self._run_icl_vectorized(ctx_embs, label_tensor, enhanced_test.squeeze(1), task_type)
        else:
            final_preds_list = []
            for i in range(batch_size):
                t_e = test_embs[i].unsqueeze(0).unsqueeze(1)
                c_e = ctx_embs[i].unsqueeze(0)
                l_raw = batch_ctx_labels[i]
                l_ts = self._process_labels(l_raw, self.task_config_map[task_config_ids[i].item()]).to(device)
                curr_intra = intra_mask[i].unsqueeze(0)
                enh, _ = self.dual_context(c_e, l_ts, t_e, c_ts[i].unsqueeze(0), t_ts[i].unsqueeze(0), curr_intra)
                
                task_conf = self.task_config_map[task_config_ids[i].item()]
                c_list = [c_e[0, k] for k in range(len(l_raw))]
                pred = self.icl_module(c_list, l_raw, enh.squeeze(0).squeeze(0), task_conf.task_type)
                if pred.dim() == 0: pred = pred.view(1) 
                if pred.dim() == 1: pred = pred.view(1, -1)
                final_preds_list.append(pred)
            
            final_preds = torch.cat(self._smart_pad(final_preds_list, task_types, device), dim=0)

        final_preds = self._smart_pad_tensor(final_preds, self.global_max_dim)

        return {'predictions': final_preds}

    def _smart_pad_tensor(self, tensor, target_dim):
        current_dim = tensor.shape[-1]
        if current_dim < target_dim:
            pad_size = target_dim - current_dim
            pad = torch.zeros((tensor.shape[0], pad_size), device=tensor.device)
            return torch.cat([tensor, pad], dim=-1)
        return tensor

    def _batch_encode_subgraphs(self, subgraphs, target_entities, timestamps, device):
        if not subgraphs: return torch.zeros(0, self.config.hidden_dim, device=device)

        # [PERFORMANCE FIX] Collect everything on CPU first
        batch_node_features = {}
        batch_node_counts = {}
        
        for sub in subgraphs:
            for nt in sub.node_types:
                if nt not in batch_node_features:
                    batch_node_features[nt] = []
                    batch_node_counts[nt] = []
                
                # Fetch features on CPU (do not move to device yet)
                f = sub.node_features.get(nt)
                if f is None:
                    # Create placeholder on CPU
                    f = torch.zeros(sub.node_counts[nt], self.config.hidden_dim)
                
                # Ensure dimension on CPU
                if f.dim() == 1: f = f.unsqueeze(-1)
                if f.shape[-1] < self.config.hidden_dim:
                    pad = torch.zeros(f.shape[0], self.config.hidden_dim - f.shape[-1])
                    f = torch.cat([f, pad], dim=-1)
                elif f.shape[-1] > self.config.hidden_dim:
                    f = f[:, :self.config.hidden_dim]
                    
                batch_node_features[nt].append(f)
                batch_node_counts[nt].append(sub.node_counts[nt])

        # Batch encode via TableEncoder
        big_input = {}
        for nt, feats in batch_node_features.items():
            if feats:
                # [PERFORMANCE FIX] Concatenate on CPU, THEN move to GPU once
                cat_f = torch.cat(feats, dim=0).to(device)
                big_input[nt] = cat_f.unsqueeze(0).unsqueeze(2) if cat_f.dim()==2 else cat_f
        
        big_enc = self.table_encoder(big_input)

        # Reconstruct graph structure
        nt_offsets = {nt: 0 for nt in big_enc.keys()}
        giant_tokens, giant_nt, giant_ei, giant_et, tgt_indices, curr_off = [], [], [], [], [], 0
        
        for i, sub in enumerate(subgraphs):
            g_f, g_t, g_map, g_c = [], [], {}, 0
            
            # 1. Collect Node Features (Already on GPU from big_enc)
            for nt in sub.node_types:
                cnt = sub.node_counts[nt]
                if nt in big_enc:
                    start = nt_offsets[nt]
                    end = start + cnt
                    g_f.append(big_enc[nt].squeeze(0)[start:end])
                    nt_offsets[nt] = end
                else:
                     # Fallback for missing type in batch (rare)
                     g_f.append(torch.zeros(cnt, self.config.hidden_dim, device=device))

                g_t.extend([self.node_type_to_id.get(nt,0)]*cnt)
                g_map[nt] = (g_c, g_c+cnt)
                g_c += cnt
            
            if not g_f: 
                tgt_indices.append(curr_off)
                giant_tokens.append(torch.zeros(1,self.config.hidden_dim,device=device))
                giant_nt.append(torch.tensor([0],device=device))
                curr_off+=1; continue
            
            tokens = torch.cat(g_f, dim=0)
            nt_tsr = torch.tensor(g_t, device=device)
            
            # 2. Collect Edges
            # [Optimization] Collect edges on CPU then move? 
            # Subgraphs are small enough that creating tensors here is usually ok, 
            # but let's optimize if we can. 
            # For now, standard list comprehension is fine as edges are indices (long).
            edges, ets = [], []
            for idx, et in enumerate(sub.edge_types):
                if et in sub.edge_indices:
                    e = sub.edge_indices[et].clone() # CPU
                    e[0] += g_map[et[0]][0]
                    e[1] += g_map[et[2]][0]
                    edges.append(e)
                    ets.extend([idx]*e.shape[1])
            
            if edges:
                # Concat CPU then move
                ei = torch.cat(edges, dim=1).to(device)
                et_tsr = torch.tensor(ets, dtype=torch.long, device=device)
            else:
                ei = torch.zeros((2,0), dtype=torch.long, device=device)
                et_tsr = torch.zeros(0, dtype=torch.long, device=device)
            
            # 3. Time & Hop Encodings
            ts_val = timestamps[i] if timestamps is not None else datetime.now(timezone.utc)
            if isinstance(ts_val, torch.Tensor): ts_val = datetime.fromtimestamp(ts_val.item(), tz=timezone.utc)

            # Move auxiliary data to GPU once
            hop = self._get_hop_distances(sub, target_entities[i], g_map).to(device)
            time = self._get_time_differences(sub, ts_val, g_map).to(device)
            
            tokens = tokens + self.relgt_type_encoder(nt_tsr) + self.relgt_hop_encoder(hop) + self.relgt_time_encoder(time.unsqueeze(0).float()).squeeze(0)
            
            t_type, t_id = target_entities[i]
            if t_type in g_map:
                loc = g_map[t_type][0] + t_id
                if 0 < loc < len(tokens):
                    # Smart permutation to keep target first
                    perm = torch.cat([torch.tensor([loc],device=device), torch.arange(0,loc,device=device), torch.arange(loc+1,len(tokens),device=device)])
                    tokens = tokens[perm]
                    nt_tsr = nt_tsr[perm]
                    # Inverse permutation for edges
                    inv = torch.zeros_like(perm)
                    inv[perm] = torch.arange(len(perm), device=device)
                    ei = inv[ei]
            
            giant_tokens.append(tokens)
            giant_nt.append(nt_tsr)
            giant_ei.append(ei + curr_off)
            giant_et.append(et_tsr)
            
            tgt_indices.append(curr_off)
            curr_off += len(tokens)
            
        # 4. Final RelGT Call
        out_all = self.relgt(
            torch.cat(giant_tokens, 0), 
            torch.cat(giant_ei, 1), 
            torch.cat(giant_nt, 0), 
            torch.cat(giant_et, 0)
        )
        return out_all[torch.tensor(tgt_indices, device=device)]

    def _run_icl_vectorized(self, ctx_embs, ctx_labels, test_emb, task_type):
        label_emb = self.icl_module.label_encoder(ctx_labels, task_type, device=ctx_embs.device)
        full_ctx = ctx_embs + label_emb
        full_test = test_emb.unsqueeze(1)
        B = ctx_embs.size(0)
        query = self.icl_module.query_token.expand(B, -1, -1)
        seq = torch.cat([full_ctx, full_test, query], dim=1)
        seq_len = seq.size(1)
        if seq_len > self.icl_module.position_encoding.size(1):
             seq = seq[:, -self.icl_module.position_encoding.size(1):, :]
             seq_len = seq.size(1)
        seq = seq + self.icl_module.position_encoding[:, :seq_len, :]
        for layer in self.icl_module.icl_layers:
            seq = layer(seq)
        q_out = seq[:, -1, :]
        
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

    def _get_hop_distances(self, subgraph, target, mapping):
        # Return on CPU, move later
        hops = torch.full((sum(subgraph.node_counts.values()),), 3, dtype=torch.long)
        for nt in subgraph.node_types:
            if hasattr(subgraph, f'{nt}_hop_distances'):
                h = getattr(subgraph, f'{nt}_hop_distances')
                s, e = mapping[nt]
                hops[s:e] = h
        return hops

    def _get_time_differences(self, subgraph, ts, mapping):
        # Return on CPU, move later
        diffs = torch.zeros(sum(subgraph.node_counts.values()))
        base = ts.timestamp()
        for nt in subgraph.node_types:
            if nt in subgraph.node_timestamps:
                t = subgraph.node_timestamps[nt]
                s, e = mapping[nt]
                diffs[s:e] = base - t
        return diffs
    
    def _process_labels(self, labels, config):
        if config.task_type == 'classification':
            return torch.tensor(self._normalize_labels(labels), dtype=torch.long).unsqueeze(0)
        return torch.tensor(labels, dtype=torch.float).unsqueeze(0)

    def _normalize_labels(self, labels):
        return [int(l) if isinstance(l, (int, float)) else 0 for l in labels]