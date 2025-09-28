"""
训练工具函数
提供训练过程中的辅助功能
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
import random
import logging
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42):
    """
    设置随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"随机种子设置为: {seed}")


class EarlyStopping:
    """
    早停机制
    """

    def __init__(self,
                 patience: int = 10,
                 mode: str = 'min',
                 delta: float = 0.0001,
                 verbose: bool = True):
        """
        Args:
            patience: 容忍的epoch数
            mode: 'min'或'max'
            delta: 最小改进量
            verbose: 是否打印信息
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.is_better = lambda a, b: a < b - delta
        else:
            self.is_better = lambda a, b: a > b + delta

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前分数

        Returns:
            是否早停
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        if self.verbose:
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                logger.info("Early stopping triggered")
            return True

        return False

    def reset(self):
        """重置状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """
    模型检查点保存器
    """

    def __init__(self,
                 save_dir: str,
                 save_top_k: int = 3,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Args:
            save_dir: 保存目录
            save_top_k: 保存最好的k个模型
            mode: 'min'或'max'
            verbose: 是否打印信息
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_top_k = save_top_k
        self.mode = mode
        self.verbose = verbose

        self.checkpoints = []  # [(score, path), ...]

        if mode == 'min':
            self.is_better = lambda a, b: a < b
        else:
            self.is_better = lambda a, b: a > b

    def save(self,
             model: nn.Module,
             score: float,
             epoch: int,
             optimizer: Optional[Optimizer] = None,
             metrics: Optional[Dict[str, float]] = None):
        """
        保存模型检查点

        Args:
            model: 模型
            score: 分数
            epoch: 轮次
            optimizer: 优化器
            metrics: 其他指标
        """
        # 创建检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
            'timestamp': datetime.now().isoformat()
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metrics:
            checkpoint['metrics'] = metrics

        # 生成文件名
        filename = f"checkpoint_epoch{epoch}_score{score:.4f}.pt"
        filepath = self.save_dir / filename

        # 保存
        torch.save(checkpoint, filepath)

        if self.verbose:
            logger.info(f"模型已保存到 {filepath}")

        # 更新检查点列表
        self.checkpoints.append((score, filepath))

        # 保持top k
        self._maintain_top_k()

    def _maintain_top_k(self):
        """维护top k个检查点"""
        if len(self.checkpoints) <= self.save_top_k:
            return

        # 排序
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))

        # 删除多余的
        for score, path in self.checkpoints[self.save_top_k:]:
            if path.exists():
                path.unlink()
                if self.verbose:
                    logger.info(f"删除检查点 {path}")

        # 保留top k
        self.checkpoints = self.checkpoints[:self.save_top_k]

    def load_best(self, model: nn.Module,
                  optimizer: Optional[Optimizer] = None) -> Dict[str, Any]:
        """
        加载最佳模型

        Args:
            model: 模型
            optimizer: 优化器

        Returns:
            检查点字典
        """
        if not self.checkpoints:
            raise ValueError("没有可用的检查点")

        # 找到最佳检查点
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
        best_score, best_path = self.checkpoints[0]

        # 加载
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.verbose:
            logger.info(f"从 {best_path} 加载最佳模型，分数: {best_score}")

        return checkpoint


class GradientAccumulator:
    """
    梯度累积器
    用于模拟更大的批次
    """

    def __init__(self, accumulation_steps: int = 1):
        """
        Args:
            accumulation_steps: 累积步数
        """
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_step(self) -> bool:
        """是否应该更新参数"""
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0

    def reset(self):
        """重置计数器"""
        self.step_count = 0


class MetricTracker:
    """
    指标跟踪器
    """

    def __init__(self, metrics: List[str]):
        """
        Args:
            metrics: 要跟踪的指标名称列表
        """
        self.metrics = metrics
        self.history = defaultdict(list)
        self.current_batch = defaultdict(list)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        更新指标

        Args:
            predictions: 预测值
            targets: 目标值
        """
        # 移到CPU
        predictions = predictions.detach().cpu()
        targets = targets.detach().cpu()

        # 计算各种指标
        for metric in self.metrics:
            if metric == 'accuracy':
                if predictions.dim() > 1:
                    pred_classes = predictions.argmax(dim=1)
                else:
                    pred_classes = (predictions > 0.5).long()
                correct = (pred_classes == targets).float().mean().item()
                self.current_batch['accuracy'].append(correct)

            elif metric == 'f1':
                # 简化的F1计算
                if predictions.dim() > 1:
                    pred_classes = predictions.argmax(dim=1)
                else:
                    pred_classes = (predictions > 0.5).long()

                # 二分类F1
                tp = ((pred_classes == 1) & (targets == 1)).sum().float()
                fp = ((pred_classes == 1) & (targets == 0)).sum().float()
                fn = ((pred_classes == 0) & (targets == 1)).sum().float()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                self.current_batch['f1'].append(f1.item())

            elif metric == 'auc':
                # AUC需要概率值
                if predictions.dim() > 1 and predictions.size(1) == 2:
                    probs = torch.softmax(predictions, dim=1)[:, 1]
                else:
                    probs = torch.sigmoid(predictions)

                # 简化的AUC计算（实际应使用sklearn）
                self.current_batch['auc'].append(0.5)  # 占位符

            elif metric == 'mae':
                # 平均绝对误差
                mae = torch.abs(predictions - targets).mean().item()
                self.current_batch['mae'].append(mae)

            elif metric == 'rmse':
                # 均方根误差
                mse = ((predictions - targets) ** 2).mean().item()
                rmse = np.sqrt(mse)
                self.current_batch['rmse'].append(rmse)

    def compute(self) -> Dict[str, float]:
        """
        计算当前批次的平均指标

        Returns:
            指标字典
        """
        results = {}

        for metric in self.metrics:
            if metric in self.current_batch and self.current_batch[metric]:
                results[metric] = np.mean(self.current_batch[metric])

        return results

    def reset(self):
        """重置当前批次"""
        self.current_batch.clear()

    def epoch_end(self):
        """epoch结束，保存历史"""
        epoch_metrics = self.compute()
        for metric, value in epoch_metrics.items():
            self.history[metric].append(value)
        self.reset()

    def plot_history(self, save_path: Optional[str] = None):
        """绘制指标历史"""
        if not self.history:
            logger.warning("没有历史数据可绘制")
            return

        fig, axes = plt.subplots(len(self.history), 1, figsize=(10, 4 * len(self.history)))

        if len(self.history) == 1:
            axes = [axes]

        for ax, (metric, values) in zip(axes, self.history.items()):
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, 'b-', label=metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} History')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"指标图表已保存到 {save_path}")

        plt.close()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    计算模型参数数量

    Args:
        model: 模型
        trainable_only: 是否只计算可训练参数

    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def save_config(config: Any, path: str):
    """
    保存配置

    Args:
        config: 配置对象
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为字典
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config

    # 保存
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"配置已保存到 {path}")


def create_optimizer(model: nn.Module,
                     lr: float = 1e-4,
                     optimizer_type: str = 'adamw',
                     weight_decay: float = 0.01,
                     **kwargs) -> Optimizer:
    """
    创建优化器

    Args:
        model: 模型
        lr: 学习率
        optimizer_type: 优化器类型
        weight_decay: 权重衰减
        **kwargs: 其他参数

    Returns:
        优化器
    """
    # 获取参数
    params = model.parameters()

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay,
                                    momentum=kwargs.get('momentum', 0.9), **kwargs)
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    logger.info(f"创建{optimizer_type}优化器，学习率: {lr}")

    return optimizer


def create_scheduler(optimizer: Optimizer,
                     num_training_steps: int,
                     scheduler_type: str = 'cosine',
                     num_warmup_steps: int = 0,
                     **kwargs) -> _LRScheduler:
    """
    创建学习率调度器

    Args:
        optimizer: 优化器
        num_training_steps: 总训练步数
        scheduler_type: 调度器类型
        num_warmup_steps: 预热步数
        **kwargs: 其他参数

    Returns:
        调度器
    """
    if scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, **kwargs)

    elif scheduler_type == 'linear':
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - step) /
                       float(max(1, num_training_steps - num_warmup_steps)))

        scheduler = LambdaLR(optimizer, lr_lambda, **kwargs)

    elif scheduler_type == 'constant':
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lambda _: 1.0, **kwargs)

    elif scheduler_type == 'exponential':
        from torch.optim.lr_scheduler import ExponentialLR
        scheduler = ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.95))

    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")

    logger.info(f"创建{scheduler_type}学习率调度器")

    return scheduler