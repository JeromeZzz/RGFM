"""
KumoRFM训练器
处理模型的训练循环
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import os

from config.model_config import KumoRFMConfig, ExperimentConfig, TaskConfig
from models.kumorfm import KumoRFM
from utils.training_utils import (
    EarlyStopping,
    ModelCheckpoint,
    MetricTracker,
    create_optimizer,
    create_scheduler
)

logger = logging.getLogger(__name__)


class KumoRFMTrainer:
    """
    KumoRFM训练器
    管理训练循环和验证
    """

    def __init__(self,
                 model: KumoRFM,
                 config: KumoRFMConfig,
                 experiment_config: ExperimentConfig,
                 device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.experiment_config = experiment_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将模型移到设备
        self.model = self.model.to(self.device)

        # 初始化优化器和调度器
        self.optimizer = create_optimizer(
            self.model,
            self.config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            num_training_steps=1000  # 将在fit时更新
        )

        # 初始化工具
        self.early_stopping = EarlyStopping(
            patience=experiment_config.early_stopping_patience,
            mode='min'
        )
        self.checkpoint = ModelCheckpoint(
            save_dir=experiment_config.save_dir,
            save_top_k=3,
            mode='min'
        )
        self.metric_tracker = MetricTracker(
            metrics=experiment_config.metrics
        )

        # 损失函数
        self.criterion = self._get_criterion()

        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {}
        }

        # 当前epoch
        self.current_epoch = 0

    def _get_criterion(self) -> nn.Module:
        """获取损失函数"""
        # 这里可以根据任务类型选择不同的损失函数
        # 简化处理，使用交叉熵作为默认
        return nn.CrossEntropyLoss()

    def fit(self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数

        Returns:
            训练历史
        """
        num_epochs = num_epochs or self.experiment_config.num_epochs

        # 更新调度器的总步数
        total_steps = len(train_loader) * num_epochs
        self.scheduler = create_scheduler(
            self.optimizer,
            num_training_steps=total_steps
        )

        logger.info(f"开始训练，共 {num_epochs} 轮")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练批次数: {len(train_loader)}")
        if val_loader:
            logger.info(f"验证批次数: {len(val_loader)}")

        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            self.current_epoch = epoch

            # 训练阶段
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history['train_loss'].append(train_loss)

            for metric_name, value in train_metrics.items():
                if metric_name not in self.train_history['train_metrics']:
                    self.train_history['train_metrics'][metric_name] = []
                self.train_history['train_metrics'][metric_name].append(value)

            # 验证阶段
            if val_loader:
                val_loss, val_metrics = self.validate(val_loader)
                self.train_history['val_loss'].append(val_loss)

                for metric_name, value in val_metrics.items():
                    if metric_name not in self.train_history['val_metrics']:
                        self.train_history['val_metrics'][metric_name] = []
                    self.train_history['val_metrics'][metric_name].append(value)

                # 早停检查
                if self.early_stopping(val_loss):
                    logger.info(f"早停触发，停止训练")
                    break

                # 模型检查点
                self.checkpoint.save(
                    self.model,
                    val_loss,
                    epoch,
                    self.optimizer,
                    metrics=val_metrics
                )

            # 打印进度
            self._log_epoch_summary(epoch, train_loss, train_metrics, val_loss, val_metrics)

        logger.info("训练完成")
        return self.train_history

    def train_epoch(self,
                    train_loader: DataLoader,
                    epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        训练一个epoch

        Returns:
            (平均损失, 指标字典)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 重置指标跟踪器
        self.metric_tracker.reset()

        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # 解包批次数据
            graph, target_entities, timestamps, labels, task_configs = self._unpack_batch(batch)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            batch_loss = 0.0
            predictions = []

            for i in range(len(target_entities)):
                # 单个样本的前向传播
                output = self.model(
                    graph[i],
                    target_entities[i],
                    timestamps[i],
                    task_configs[i]
                )

                # 提取预测
                if 'predicted_class' in output:
                    pred = output['predicted_class']
                elif 'predicted_value' in output:
                    pred = output['predicted_value']
                else:
                    pred = output.get('predictions', 0)

                predictions.append(pred)

                # 计算损失
                # 这里简化处理，实际应该根据任务类型使用不同的损失
                if isinstance(pred, torch.Tensor):
                    loss = self.criterion(pred.unsqueeze(0), labels[i:i + 1])
                else:
                    loss = torch.tensor(0.0, device=self.device)

                batch_loss += loss

            # 平均损失
            batch_loss = batch_loss / len(target_entities)

            # 反向传播
            batch_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

            # 更新参数
            self.optimizer.step()
            self.scheduler.step()

            # 记录损失
            total_loss += batch_loss.item()
            num_batches += 1

            # 更新指标
            if predictions:
                predictions_tensor = torch.stack([
                    p if isinstance(p, torch.Tensor) else torch.tensor(p)
                    for p in predictions
                ])
                self.metric_tracker.update(predictions_tensor, labels)

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        # 计算平均损失和指标
        avg_loss = total_loss / num_batches
        metrics = self.metric_tracker.compute()

        return avg_loss, metrics

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        验证模型

        Returns:
            (平均损失, 指标字典)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # 重置指标跟踪器
        self.metric_tracker.reset()

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")

            for batch in pbar:
                # 解包批次数据
                graph, target_entities, timestamps, labels, task_configs = self._unpack_batch(batch)

                batch_loss = 0.0
                predictions = []

                for i in range(len(target_entities)):
                    # 前向传播
                    output = self.model(
                        graph[i],
                        target_entities[i],
                        timestamps[i],
                        task_configs[i]
                    )

                    # 提取预测
                    if 'predicted_class' in output:
                        pred = output['predicted_class']
                    elif 'predicted_value' in output:
                        pred = output['predicted_value']
                    else:
                        pred = output.get('predictions', 0)

                    predictions.append(pred)

                    # 计算损失
                    if isinstance(pred, torch.Tensor):
                        loss = self.criterion(pred.unsqueeze(0), labels[i:i + 1])
                    else:
                        loss = torch.tensor(0.0, device=self.device)

                    batch_loss += loss

                # 平均损失
                batch_loss = batch_loss / len(target_entities)
                total_loss += batch_loss.item()
                num_batches += 1

                # 更新指标
                if predictions:
                    predictions_tensor = torch.stack([
                        p if isinstance(p, torch.Tensor) else torch.tensor(p)
                        for p in predictions
                    ])
                    self.metric_tracker.update(predictions_tensor, labels)

                # 更新进度条
                pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})

        # 计算平均损失和指标
        avg_loss = total_loss / num_batches
        metrics = self.metric_tracker.compute()

        return avg_loss, metrics

    def _unpack_batch(self, batch: Dict[str, Any]) -> Tuple:
        """解包批次数据"""
        # 这里的实现取决于数据加载器的格式
        # 简化示例
        graph = batch.get('graphs', [])
        target_entities = batch.get('target_entities', [])
        timestamps = batch.get('timestamps', [])
        labels = batch.get('labels', torch.zeros(len(graph)))
        task_configs = batch.get('task_configs', [TaskConfig(task_type='classification')] * len(graph))

        # 移到设备
        if isinstance(labels, torch.Tensor):
            labels = labels.to(self.device)

        return graph, target_entities, timestamps, labels, task_configs

    def _log_epoch_summary(self,
                           epoch: int,
                           train_loss: float,
                           train_metrics: Dict[str, float],
                           val_loss: Optional[float] = None,
                           val_metrics: Optional[Dict[str, float]] = None):
        """记录epoch总结"""
        summary = f"\nEpoch {epoch} Summary:\n"
        summary += f"  Train Loss: {train_loss:.4f}\n"

        for metric, value in train_metrics.items():
            summary += f"  Train {metric}: {value:.4f}\n"

        if val_loss is not None:
            summary += f"  Val Loss: {val_loss:.4f}\n"

            if val_metrics:
                for metric, value in val_metrics.items():
                    summary += f"  Val {metric}: {value:.4f}\n"

        logger.info(summary)

    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'config': self.config,
            'experiment_config': self.experiment_config
        }
        torch.save(checkpoint, path)
        logger.info(f"检查点已保存到 {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_history = checkpoint['train_history']

        logger.info(f"检查点已从 {path} 加载，epoch: {self.current_epoch}")