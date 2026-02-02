"""
RelGT Adapter
(Final Fix: Align Dummy Feature Dimensions + CPU/GPU Device Safety)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from .relgt.model import RelGT

class RelGTAdapter(nn.Module):
    def __init__(self, config, database_schema, col_stats_dict=None):
        super().__init__()
        self.config = config
        
        self.sample_node_len = config.sampling_config.max_neighbors
        self.max_hops = config.sampling_config.num_hops
        self.hidden_dim = config.hidden_dim
        
        self.node_type_map = {nt: i for i, nt in enumerate(database_schema.keys())}
        self.inv_node_type_map = {i: nt for nt, i in self.node_type_map.items()}
        
        self.original_num_cols = {}
        self.time_cols = {}
        self.col_names_dict = {}
        
        for table, cols in database_schema.items():
            self.original_num_cols[table] = []
            self.time_cols[table] = []
            self.col_names_dict[table] = {}
            temp_cat = []
            
            for col, dtype in cols.items():
                s_type = self._map_dtype_to_stype(dtype)
                if s_type == stype.numerical:
                    s_dtype_str = str(dtype).lower()
                    if 'time' in s_dtype_str or 'date' in s_dtype_str:
                        self.time_cols[table].append(col)
                    else:
                        self.original_num_cols[table].append(col)
                elif s_type == stype.categorical:
                    temp_cat.append(col)
            
            full_num_cols = self.original_num_cols[table] + self.time_cols[table]
            if full_num_cols:
                self.col_names_dict[table][stype.numerical] = full_num_cols
            if temp_cat:
                self.col_names_dict[table][stype.categorical] = temp_cat

        if col_stats_dict is None:
            col_stats_dict = self._get_dummy_stats(self.col_names_dict)

        self.model = RelGT(
            num_nodes=100000, 
            max_neighbor_hop=self.max_hops,
            node_type_map=self.node_type_map,
            col_names_dict=self.col_names_dict,
            col_stats_dict=col_stats_dict,
            local_num_layers=config.num_layers,
            channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            global_dim=config.hidden_dim,
            heads=config.num_heads,
            ff_dropout=config.dropout_rate,
            attn_dropout=config.dropout_rate,
            conv_type="full", 
            num_centroids=1024, 
            sample_node_len=self.sample_node_len
        )

    def _map_dtype_to_stype(self, dtype):
        s_dtype = str(dtype).lower()
        if 'float' in s_dtype or 'int' in s_dtype or 'numerical' in s_dtype: return stype.numerical
        if 'cat' in s_dtype or 'str' in s_dtype: return stype.categorical
        if 'time' in s_dtype or 'date' in s_dtype: return stype.numerical
        return stype.categorical

    def _get_dummy_stats(self, col_names):
        stats = {}
        for table, stypes in col_names.items():
            stats[table] = {}
            for s_type, cols in stypes.items():
                for col in cols:
                    if s_type == stype.numerical:
                        stats[table][col] = {StatType.MEAN: 0.0, StatType.STD: 1.0}
                    elif s_type == stype.categorical:
                        stats[table][col] = {StatType.COUNT: ([str(i) for i in range(10)], [1]*10)}
        return stats

    def forward(self, subgraphs: List[Any], target_ents: List[Tuple[str, int]], prediction_times: torch.Tensor):
        device = prediction_times.device
        B = len(subgraphs)
        K = self.sample_node_len
        
        # Structure tensors directly to GPU
        neighbor_types = torch.zeros((B, K), dtype=torch.long, device=device)
        neighbor_hops = torch.full((B, K), self.max_hops, dtype=torch.long, device=device)        
        neighbor_times = torch.zeros((B, K), dtype=torch.float, device=device)
        
        flat_batch_idx = []
        flat_nbr_idx = []
        grouped_features = {nt: [] for nt in self.node_type_map}
        grouped_timestamps = {nt: [] for nt in self.node_type_map}
        
        # [CRITICAL 1] Determine Padding Types and Dimensions
        # We need the first available type to use as "padding type"
        first_type = next(iter(self.node_type_map))
        first_type_idx = self.node_type_map[first_type]
        
        # We need to know the feature dimension for this type to create matching zeros.
        # We check the first subgraph.
        ref_graph = subgraphs[0]
        padding_feat_dim = self.hidden_dim # Default fallback
        
        # Try to find actual dimension from data
        if ref_graph.node_features.get(first_type) is not None:
             # Ensure we get the feature width (dim 1)
             if ref_graph.node_features[first_type].dim() > 1:
                 padding_feat_dim = ref_graph.node_features[first_type].shape[1]
             else:
                 padding_feat_dim = 1 # Scalar features
        elif len(self.original_num_cols.get(first_type, [])) > 0:
             # If no data but schema says cols exist, trust schema count
             padding_feat_dim = len(self.original_num_cols[first_type])
        
        batch_vec_list = []
        
        for i, (subgraph, target_ent, p_time) in enumerate(zip(subgraphs, target_ents, prediction_times)):
            target_type, target_id = target_ent
            p_time_val = p_time.item()
            
            all_nodes = [] 
            for nt in subgraph.node_features.keys():
                count = subgraph.node_counts[nt]
                feats = subgraph.node_features[nt]
                ts = subgraph.node_timestamps[nt]
                hops = getattr(subgraph, f'{nt}_hop_distances', torch.full((count,), self.max_hops, dtype=torch.long))
                
                is_target_type = (nt == target_type)
                
                for local_idx in range(count):
                    curr_hop = hops[local_idx].item()
                    if curr_hop > self.max_hops: curr_hop = self.max_hops
                    is_seed = (is_target_type and curr_hop == 0)
                    
                    # Handle features
                    if feats is not None:
                        feat = feats[local_idx]
                    else:
                        # Fallback if specific type has no features in this subgraph
                        # Try to match expected dimension or 0
                        target_dim = len(self.original_num_cols.get(nt, []))
                        feat = torch.zeros(target_dim if target_dim > 0 else self.hidden_dim)

                    t_val = ts[local_idx].item() if ts is not None else 0.0
                    all_nodes.append({'type': nt, 'feat': feat, 'time': t_val, 'hop': curr_hop, 'is_seed': is_seed})

            all_nodes.sort(key=lambda x: (not x['is_seed'], x['hop'], -x['time']))
            valid_nodes = all_nodes[:K]
            num_valid = len(valid_nodes)
            
            for j in range(K):
                flat_batch_idx.append(i)
                flat_nbr_idx.append(j)
                
                if j < num_valid:
                    node = valid_nodes[j]
                    neighbor_types[i, j] = self.node_type_map[node['type']]
                    neighbor_hops[i, j] = node['hop']
                    neighbor_times[i, j] = float(node['time']) - p_time_val
                    
                    grouped_features[node['type']].append(node['feat'])
                    grouped_timestamps[node['type']].append(node['time'])
                else:
                    # [CRITICAL 2] Padding Logic
                    neighbor_types[i, j] = first_type_idx
                    neighbor_hops[i, j] = self.max_hops
                    neighbor_times[i, j] = 0.0
                    
                    # Create dummy feat on CPU with CORRECT DIMENSION
                    dummy_feat = torch.zeros(padding_feat_dim) 
                    grouped_features[first_type].append(dummy_feat)
                    grouped_timestamps[first_type].append(0.0)
            
            batch_vec_list.append(torch.full((K,), i, dtype=torch.long, device=device))

        grouped_tfs = {}
        grouped_indices = {}
        type_to_flat_indices = {nt: [] for nt in self.node_type_map}
        
        for global_idx, (b_i, n_j) in enumerate(zip(flat_batch_idx, flat_nbr_idx)):
            type_idx = neighbor_types[b_i, n_j].item()
            if type_idx < len(self.inv_node_type_map):
                type_str = self.inv_node_type_map[type_idx]
                type_to_flat_indices[type_str].append(global_idx)

        for nt, feats_list in grouped_features.items():
            if len(feats_list) > 0:
                # [CRITICAL 3] Stack on CPU, then move to Device
                # This prevents "Expected all tensors to be on same device"
                stacked_feats = torch.stack(feats_list).to(device)
                num_nodes = stacked_feats.size(0)
                
                final_num_tensor = stacked_feats
                
                if len(self.time_cols.get(nt, [])) > 0:
                    ts_list = grouped_timestamps[nt]
                    # Convert float list to tensor on Device
                    stacked_ts = torch.tensor(ts_list, device=device).unsqueeze(1)
                    num_time_cols = len(self.time_cols[nt])
                    if num_time_cols > 1: stacked_ts = stacked_ts.repeat(1, num_time_cols)
                    final_num_tensor = torch.cat([stacked_feats, stacked_ts], dim=1)

                tf_dict = {stype.numerical: final_num_tensor}

                if stype.categorical in self.col_names_dict[nt]:
                    num_cat_cols = len(self.col_names_dict[nt][stype.categorical])
                    dummy_cats = torch.zeros((num_nodes, num_cat_cols), dtype=torch.long, device=device)
                    tf_dict[stype.categorical] = dummy_cats

                grouped_tfs[self.node_type_map[nt]] = TensorFrame(
                    feat_dict=tf_dict,
                    col_names_dict=self.col_names_dict[nt],
                    y=None
                )
                
                grouped_indices[self.node_type_map[nt]] = type_to_flat_indices[nt]

        grouped_tf_dict = {"grouped_tfs": grouped_tfs, "grouped_indices": grouped_indices, "flat_batch_idx": flat_batch_idx, "flat_nbr_idx": flat_nbr_idx}

        if len(batch_vec_list) > 0:
            pe_batch = torch.cat(batch_vec_list)
            # Edge Index construction
            pe_edge_index = torch.arange(pe_batch.size(0), device=device).unsqueeze(0).repeat(2, 1)
        else:
            # Should be impossible due to padding
            pe_batch = torch.zeros(1, dtype=torch.long, device=device)
            pe_edge_index = torch.zeros((2, 1), dtype=torch.long, device=device)

        node_indices = torch.arange(B, device=device)
        
        out = self.model(
            neighbor_types=neighbor_types,
            node_indices=node_indices,
            neighbor_hops=neighbor_hops,
            neighbor_times=neighbor_times,
            grouped_tf_dict=grouped_tf_dict,
            edge_index=pe_edge_index,
            batch=pe_batch
        )
        
        return out