"""
KumoRFM微调模块
支持任务特定的微调
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
from tqdm import tqdm

from config.model_config import KumoRFMConfig, TaskConfig, ExperimentConfig
from models.kumorfm import KumoRFM
from data.temporal_graph import TemporalHeterogeneousGraph
from data.database import Database
from pql.parser import PQLQuery
from .trainer import KumoRFMTrainer


class FinetuneDataset(Dataset):
    """
    微调数据集
    从PQL查询生成训练样本
    """

    def __init__(self,
                 database: Database,
                 pql_query: PQLQuery,
                 entities: List[Tuple[Any, datetime]],
                 config: KumoRFMConfig):
        self.database = database
        self.pql_query = pql_query
        self.entities = entities
        self.config = config

        # 预计算所有样本
        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """准备所有训练样本"""
        samples = []

        for entity_value, timestamp in self.entities:
            # 创建样本
            sample = {
                'entity': (self.pql_query.entity_table, entity_value),
                'timestamp': timestamp,
                'target_column': self.pql_query.target_column,
                'aggregation': self.pql_query.aggregation
            }

            # 生成标签（这里需要实际的前向采样器）
            # 简化处理：生成模拟标签
            sample['label'] = self._generate_label(entity_value, timestamp)

            samples.append(sample)

        return samples

    def _generate_label(self, entity_value: Any, timestamp: datetime) -> Any:
        """生成标签（简化实现）"""
        # 实际应该使用前向采样器从数据库生成真实标签
        # 这里返回模拟值
        if self.pql_query.aggregation in ['sum', 'mean', 'max', 'min']:
            return np.random.randn()  # 回归任务
        elif self.pql_query.aggregation == 'count':
            return np.random.randint(0, 2)  # 二分类任务
        else:
            return 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class KumoRFMFinetuner:
    """
    KumoRFM微调器
    处理任务特定的微调
    """

    def __init__(self,
                 model: KumoRFM,
                 config: KumoRFMConfig,
                 task_config: TaskConfig):
        self.model = model
        self.config = config
        self.task_config = task_config

        # 设置模型的任务配置
        self.model.set_task_config(task_config)

        # 冻结部分层（可选）
        self.freeze_layers = []

        # 优化器（只优化未冻结的参数）
        self.optimizer = None
        self.criterion = self._get_criterion()

    def _get_criterion(self) -> nn.Module:
        """根据任务类型获取损失函数"""
        if self.task_config.task_type == 'classification':
            return nn.CrossEntropyLoss()
        elif self.task_config.task_type == 'regression':
            return nn.MSELoss()
        elif self.task_config.task_type == 'multilabel':
            return nn.BCEWithLogitsLoss()
        elif self.task_config.task_type == 'link_prediction':
            return nn.CosineEmbeddingLoss()
        else:
            return nn.MSELoss()

    def freeze_backbone(self, freeze_relgt: bool = True, freeze_encoders: bool = True):
        """
        冻结骨干网络

        Args:
            freeze_relgt: 是否冻结RelGT层
            freeze_encoders: 是否冻结编码器层
        """
        if freeze_relgt:
            for param in self.model.relgt.parameters():
                param.requires_grad = False
            self.freeze_layers.append('relgt')

        if freeze_encoders:
            for param in self.model.multimodal_encoder.parameters():
                param.requires_grad = False
            for param in self.model.table_encoder.parameters():
                param.requires_grad = False
            self.freeze_layers.extend(['multimodal_encoder', 'table_encoder'])

        # 创建优化器（只包含可训练参数）
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate * 0.1  # 使用较小的学习率
        )

        # 打印可训练参数信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params_count:,}")
        print(f"冻结层: {self.freeze_layers}")

    def finetune_from_pql(self,
                          database: Database,
                          graph: TemporalHeterogeneousGraph,
                          pql_query: PQLQuery,
                          experiment_config: ExperimentConfig) -> Dict[str, Any]:
        """
        从PQL查询进行微调

        Args:
            database: 数据库
            graph: 时序异构图
            pql_query: PQL查询
            experiment_config: 实验配置

        Returns:
            微调结果
        """
        # 准备数据
        train_loader, val_loader, test_loader = self._prepare_data(
            database, graph, pql_query, experiment_config
        )

        # 创建训练器
        trainer = KumoRFMTrainer(
            self.model,
            self.config,
            experiment_config
        )

        # 微调
        print(f"开始微调任务: {self.task_config.task_type}")
        train_history = trainer.fit(train_loader, val_loader)

        # 评估
        test_results = self.evaluate(test_loader)

        return {
            'train_history': train_history,
            'test_results': test_results
        }

    def _prepare_data(self,
                      database: Database,
                      graph: TemporalHeterogeneousGraph,
                      pql_query: PQLQuery,
                      experiment_config: ExperimentConfig,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        准备训练、验证和测试数据
        """
        # 获取所有实体和时间戳
        all_entities = self._get_all_entities(database, pql_query)

        # 随机划分
        np.random.shuffle(all_entities)
        n = len(all_entities)

        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_entities = all_entities[:train_size]
        val_entities = all_entities[train_size:train_size + val_size]
        test_entities = all_entities[train_size + val_size:]

        # 创建数据集
        train_dataset = FinetuneDataset(database, pql_query, train_entities, self.config)
        val_dataset = FinetuneDataset(database, pql_query, val_entities, self.config)
        test_dataset = FinetuneDataset(database, pql_query, test_entities, self.config)

        # 自定义collate函数
        def collate_fn(batch):
            return {
                'graphs': [self._create_mock_graph() for _ in batch],  # 简化：创建模拟图
                'target_entities': [item['entity'] for item in batch],
                'timestamps': [item['timestamp'] for item in batch],
                'labels': torch.tensor([item['label'] for item in batch]),
                'task_configs': [self.task_config] * len(batch)
            }

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        return train_loader, val_loader, test_loader

    def _get_all_entities(self,
                          database: Database,
                          pql_query: PQLQuery) -> List[Tuple[Any, datetime]]:
        """获取所有符合条件的实体"""
        entities = []

        # 从查询中获取实体值
        if pql_query.entity_values:
            entity_values = pql_query.entity_values
        else:
            # 从数据库获取所有实体
            # 这里简化处理
            entity_values = list(range(100))  # 模拟100个实体

        # 生成时间戳
        base_time = datetime.now()
        for i in range(50):  # 50个时间点
            timestamp = base_time - timedelta(days=i * 7)  # 每周一个点
            for entity_value in entity_values[:10]:  # 限制数量
                entities.append((entity_value, timestamp))

        return entities

    def _create_mock_graph(self) -> TemporalHeterogeneousGraph:
        """创建模拟图（简化实现）"""
        graph = TemporalHeterogeneousGraph()

        # 添加一些模拟节点和边
        graph.add_node_type("users", 10)
        graph.add_node_type("items", 20)
        graph.add_edge_type("user_item", ("users", "items"))

        return graph

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        评估微调后的模型

        Returns:
            评估指标
        """
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                graphs = batch['graphs']
                target_entities = batch['target_entities']
                timestamps = batch['timestamps']
                labels = batch['labels']
                task_configs = batch['task_configs']

                batch_loss = 0.0

                for i in range(len(graphs)):
                    # 前向传播
                    output = self.model(
                        graphs[i],
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
                        pred = 0

                    all_predictions.append(pred)
                    all_labels.append(labels[i].item())

                    # 计算损失
                    if self.task_config.task_type == 'regression':
                        loss = self.criterion(
                            torch.tensor([pred], dtype=torch.float),
                            torch.tensor([labels[i]], dtype=torch.float)
                        )
                    else:
                        loss = torch.tensor(0.0)

                    batch_loss += loss.item()

                total_loss += batch_loss / len(graphs)

        # 计算指标
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        metrics = {}
        metrics['loss'] = total_loss / len(test_loader)

        if self.task_config.task_type == 'classification':
            # 分类指标
            from sklearn.metrics import accuracy_score, f1_score
            metrics['accuracy'] = accuracy_score(labels, predictions)
            metrics['f1'] = f1_score(labels, predictions, average='weighted')

        elif self.task_config.task_type == 'regression':
            # 回归指标
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            metrics['mae'] = mean_absolute_error(labels, predictions)
            metrics['rmse'] = np.sqrt(mean_squared_error(labels, predictions))

        return metrics


def finetune_kumorfm(model: KumoRFM,
                     database: Database,
                     graph: TemporalHeterogeneousGraph,
                     pql_query: PQLQuery,
                     config: KumoRFMConfig,
                     task_config: TaskConfig,
                     experiment_config: ExperimentConfig,
                     freeze_backbone: bool = True) -> Dict[str, Any]:
    """
    微调KumoRFM的便捷函数

    Args:
        model: 预训练的KumoRFM模型
        database: 数据库
        graph: 时序异构图
        pql_query: PQL查询定义任务
        config: 模型配置
        task_config: 任务配置
        experiment_config: 实验配置
        freeze_backbone: 是否冻结骨干网络

    Returns:
        微调结果
    """
    # 创建微调器
    finetuner = KumoRFMFinetuner(model, config, task_config)

    # 冻结骨干网络（如果需要）
    if freeze_backbone:
        finetuner.freeze_backbone()

    # 执行微调
    results = finetuner.finetune_from_pql(
        database, graph, pql_query, experiment_config
    )

    return results


from datetime import timedelta