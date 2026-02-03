import torch
import torch.nn.functional as F
import logging
import os
import numpy as np

# 配置独立 Logger
debug_logger = logging.getLogger("DeepDebug")
debug_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

# 文件 Handler (写到当前运行目录下的 debug_deep.log)
file_handler = logging.FileHandler('debug_deep.log', mode='w', encoding='utf-8')
file_handler.setFormatter(formatter)
debug_logger.addHandler(file_handler)

# 控制台 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
debug_logger.addHandler(console_handler)

class DebugProbe:
    STEP = 0
    
    @staticmethod
    def set_step(step):
        DebugProbe.STEP = step

    @staticmethod
    def check_tensor(tag, tensor, force=False):
        if not force and DebugProbe.STEP % 50 != 0: return
        if tensor is None or tensor.numel() == 0: return

        with torch.no_grad():
            t = tensor.detach().float()
            
            # [FIX] Handle 3D+ Tensors (e.g. [Batch, Nodes, Dim])
            # Flatten to [N, Dim] for statistics
            if t.dim() > 2:
                t = t.reshape(-1, t.shape[-1])
            
            # NaN / Inf Check
            if torch.isnan(t).any():
                debug_logger.critical(f"!!! [STEP {DebugProbe.STEP}] {tag} CONTAINS NaN !!!")
                return
            if torch.isinf(t).any():
                debug_logger.critical(f"!!! [STEP {DebugProbe.STEP}] {tag} CONTAINS Inf !!!")
                return
            
            mean = t.mean().item()
            norm_std = 0.0
            
            if t.dim() > 1: 
                norms = t.norm(p=2, dim=-1)
                norm_avg = norms.mean().item()
                if norms.numel() > 1:
                    norm_std = norms.std().item()
            else: 
                norm_avg = t.norm().item()

            sim_avg = 0.0
            if t.dim() >= 2 and t.shape[0] > 1:
                # Sample to avoid OOM on large batches
                sample_n = min(t.shape[0], 256)
                # Filter out zero vectors (padding) to avoid NaN in similarity
                t_sample = t[:sample_n]
                mask_pad = (t_sample.abs().sum(dim=-1) > 1e-6)
                t_sample = t_sample[mask_pad]

                if t_sample.shape[0] > 1:
                    t_n = F.normalize(t_sample, p=2, dim=-1)
                    sim_mat = torch.mm(t_n, t_n.t())
                    # Mask diagonal
                    mask_diag = ~torch.eye(t_sample.shape[0], device=t.device).bool()
                    if mask_diag.sum() > 0:
                        sim_avg = sim_mat[mask_diag].mean().item()

            status = ""
            if sim_avg > 0.90: status = " <<< [COLLAPSE]"
            elif sim_avg < 0.01 and t.dim() > 1: status = " <<< [ORTHOGONAL]"
            elif norm_avg > 50: status = " <<< [EXPLOSION]"

            debug_logger.info(f"[STEP {DebugProbe.STEP}] {tag:<25} | Norm: {norm_avg:6.2f}+/-{norm_std:<4.2f} | Sim: {sim_avg:4.3f} | Mean: {mean:6.3f}{status}")

    @staticmethod
    def check_grad(model, force=False):
        if not force and DebugProbe.STEP % 50 != 0: return
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        debug_logger.info(f"[STEP {DebugProbe.STEP}] GRAD CHECK | Total Norm: {total_norm:.4f}")

    @staticmethod
    def check_batch_labels(labels, task_ids):
        if DebugProbe.STEP % 50 != 0: return
        unique_tasks = torch.unique(task_ids)
        for tid in unique_tasks:
            mask = (task_ids == tid)
            l = labels[mask].float()
            if l.numel() > 0:
                mean = l.mean().item()
                if mean == 0 or mean == 1:
                    debug_logger.warning(f"!!! [STEP {DebugProbe.STEP}] Task {tid} Batch Labels TRIVIAL (All {mean})")