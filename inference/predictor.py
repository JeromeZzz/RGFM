"""
KumoRFM推理预测器
提供简单的推理接口
"""

import torch
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import json
from pathlib import Path

from config.model_config import KumoRFMConfig, TaskConfig
from models.kumorfm import KumoRFM
from data.temporal_graph import TemporalHeterogeneousGraph
from data.database import Database
from pql.parser import PQLQuery, PQLParser

logger = logging.getLogger(__name__)


class KumoRFMPredictor:
    """
    KumoRFM预测器
    提供高级推理接口
    """

    def __init__(self,
                 model: KumoRFM,
                 config: KumoRFMConfig,
                 device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将模型移到设备并设置为评估模式
        self.model = self.model.to(self.device)
        self.model.eval()

        # PQL解析器
        self.pql_parser = PQLParser()

        # 缓存
        self.prediction_cache = {}
        self.cache_enabled = True

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_path: str,
                        database_schema: Dict[str, Dict[str, str]],
                        device: Optional[torch.device] = None) -> 'KumoRFMPredictor':
        """
        从检查点加载预测器

        Args:
            checkpoint_path: 检查点路径
            database_schema: 数据库模式
            device: 设备

        Returns:
            预测器实例
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 提取配置
        config = checkpoint.get('config', KumoRFMConfig())

        # 创建模型
        model = KumoRFM(config, database_schema)

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])

        # 创建预测器
        predictor = cls(model, config, device)

        logger.info(f"从 {checkpoint_path} 加载模型成功")

        return predictor

    def predict(self,
                graph: TemporalHeterogeneousGraph,
                target_entity: Union[Tuple[str, int], List[Tuple[str, int]]],
                prediction_time: Union[datetime, List[datetime]],
                task_config: TaskConfig,
                context_strategy: str = 'mixed',
                num_context: int = 10,
                return_attention: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        进行预测

        Args:
            graph: 时序异构图
            target_entity: 目标实体或实体列表
            prediction_time: 预测时间或时间列表
            task_config: 任务配置
            context_strategy: 上下文策略
            num_context: 上下文数量
            return_attention: 是否返回注意力权重

        Returns:
            预测结果或结果列表
        """
        # 处理批量预测
        if isinstance(target_entity, list):
            if not isinstance(prediction_time, list):
                prediction_time = [prediction_time] * len(target_entity)

            results = []
            for entity, time in zip(target_entity, prediction_time):
                result = self._predict_single(
                    graph, entity, time, task_config,
                    context_strategy, num_context, return_attention
                )
                results.append(result)

            return results
        else:
            # 单个预测
            return self._predict_single(
                graph, target_entity, prediction_time, task_config,
                context_strategy, num_context, return_attention
            )

    def _predict_single(self,
                        graph: TemporalHeterogeneousGraph,
                        target_entity: Tuple[str, int],
                        prediction_time: datetime,
                        task_config: TaskConfig,
                        context_strategy: str,
                        num_context: int,
                        return_attention: bool) -> Dict[str, Any]:
        """单个实体的预测"""
        # 生成缓存键
        cache_key = self._generate_cache_key(
            target_entity, prediction_time, task_config
        )

        # 检查缓存
        if self.cache_enabled and cache_key in self.prediction_cache:
            logger.debug(f"使用缓存预测: {cache_key}")
            return self.prediction_cache[cache_key]

        # 执行预测
        with torch.no_grad():
            output = self.model(
                graph,
                target_entity,
                prediction_time,
                task_config,
                context_strategy=context_strategy,
                num_context=num_context
            )

        # 处理输出
        result = self._process_output(output, task_config)

        # 添加元信息
        result['entity'] = target_entity
        result['prediction_time'] = prediction_time.isoformat()
        result['task_type'] = task_config.task_type

        # 可选：添加注意力权重
        if not return_attention and 'attention_weights' in result:
            del result['attention_weights']

        # 缓存结果
        if self.cache_enabled:
            self.prediction_cache[cache_key] = result

        return result

    def predict_from_pql(self,
                         database: Database,
                         graph: TemporalHeterogeneousGraph,
                         pql_query: str,
                         prediction_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        从PQL查询进行预测

        Args:
            database: 数据库
            graph: 时序异构图
            pql_query: PQL查询字符串
            prediction_time: 预测时间（默认为当前时间）

        Returns:
            预测结果
        """
        # 解析PQL查询
        parsed_query = self.pql_parser.parse(pql_query)

        # 验证查询
        schema = database.get_schema()
        errors = self.pql_parser.validate(parsed_query, schema)
        if errors:
            raise ValueError(f"PQL查询无效: {'; '.join(errors)}")

        # 转换为任务配置
        task_config = self.pql_parser.to_task_config(parsed_query)

        # 设置预测时间
        if prediction_time is None:
            prediction_time = datetime.now()

        # 预测所有指定的实体
        results = {}
        for entity_value in parsed_query.entity_values:
            # 构建实体元组
            entity = (parsed_query.entity_table, entity_value)

            # 执行预测
            result = self.predict(
                graph,
                entity,
                prediction_time,
                task_config
            )

            results[str(entity_value)] = result

        # 聚合结果（如果需要）
        if parsed_query.aggregation and len(results) > 1:
            aggregated = self._aggregate_results(results, parsed_query.aggregation)
            results['_aggregated'] = aggregated

        return results

    def batch_predict(self,
                      graph: TemporalHeterogeneousGraph,
                      entities: List[Tuple[str, int]],
                      prediction_times: Union[datetime, List[datetime]],
                      task_config: TaskConfig,
                      batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        批量预测

        Args:
            graph: 时序异构图
            entities: 实体列表
            prediction_times: 预测时间（单个或列表）
            task_config: 任务配置
            batch_size: 批次大小

        Returns:
            预测结果列表
        """
        if not isinstance(prediction_times, list):
            prediction_times = [prediction_times] * len(entities)

        results = []

        # 分批处理
        for i in range(0, len(entities), batch_size):
            batch_entities = entities[i:i + batch_size]
            batch_times = prediction_times[i:i + batch_size]

            # 批量预测
            batch_results = self.predict(
                graph,
                batch_entities,
                batch_times,
                task_config
            )

            results.extend(batch_results)

        return results

    def _process_output(self, output: Dict[str, Any], task_config: TaskConfig) -> Dict[str, Any]:
        """处理模型输出"""
        # 基本结果已经在模型中处理
        result = output.copy()

        # 添加置信度信息
        if 'probabilities' in result:
            probs = result['probabilities']
            if isinstance(probs, np.ndarray):
                result['confidence'] = float(np.max(probs))

        # 格式化预测值
        if task_config.task_type == 'regression' and 'predicted_value' in result:
            # 确保是标量
            value = result['predicted_value']
            if hasattr(value, 'item'):
                result['predicted_value'] = float(value.item())
            else:
                result['predicted_value'] = float(value)

        return result

    def _generate_cache_key(self,
                            entity: Tuple[str, int],
                            prediction_time: datetime,
                            task_config: TaskConfig) -> str:
        """生成缓存键"""
        entity_str = f"{entity[0]}:{entity[1]}"
        time_str = prediction_time.strftime("%Y%m%d%H%M%S")
        task_str = f"{task_config.task_type}:{task_config.target_column}"

        return f"{entity_str}_{time_str}_{task_str}"

    def _aggregate_results(self,
                           results: Dict[str, Dict[str, Any]],
                           aggregation: str) -> Dict[str, Any]:
        """聚合多个预测结果"""
        # 提取预测值
        values = []
        for entity_result in results.values():
            if 'predicted_value' in entity_result:
                values.append(entity_result['predicted_value'])
            elif 'predicted_class' in entity_result:
                values.append(entity_result['predicted_class'])

        if not values:
            return {'error': 'No values to aggregate'}

        # 执行聚合
        import numpy as np
        values = np.array(values)

        if aggregation == 'mean':
            agg_value = float(np.mean(values))
        elif aggregation == 'sum':
            agg_value = float(np.sum(values))
        elif aggregation == 'max':
            agg_value = float(np.max(values))
        elif aggregation == 'min':
            agg_value = float(np.min(values))
        elif aggregation == 'count':
            agg_value = len(values)
        else:
            agg_value = values.tolist()

        return {
            'aggregation': aggregation,
            'value': agg_value,
            'num_entities': len(values)
        }

    def enable_cache(self, enabled: bool = True):
        """启用/禁用缓存"""
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()

    def clear_cache(self):
        """清空预测缓存"""
        self.prediction_cache.clear()
        logger.info("预测缓存已清空")

    def save_predictions(self, predictions: Union[Dict, List], path: str):
        """保存预测结果到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)

        logger.info(f"预测结果已保存到 {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'config': self.config.__dict__,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self.prediction_cache)
        }

        return info


import numpy as np