"""
PQL (Predictive Query Language) 解析器接口
保留接口，不实现具体的解析功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from config.model_config import TaskConfig


@dataclass
class PQLQuery:
    """
    PQL查询的结构化表示
    """
    # PREDICT子句
    target_column: str
    aggregation: str  # 'mean', 'sum', 'max', 'min', 'count'
    time_window: Optional[Tuple[int, int]] = None  # (start_offset, end_offset)

    # FOR子句
    entity_table: str
    entity_column: str
    entity_values: List[Any]  # 要预测的实体值列表

    # WHERE子句（可选）
    conditions: Optional[List[Dict[str, Any]]] = None

    # 其他
    task_type: Optional[str] = None  # 从查询推断的任务类型


class PQLParser(ABC):
    """
    PQL解析器抽象接口
    负责解析PQL查询字符串并转换为结构化表示
    """

    @abstractmethod
    def parse(self, query: str) -> PQLQuery:
        """
        解析PQL查询字符串

        Args:
            query: PQL查询字符串

        Returns:
            PQLQuery: 解析后的查询对象

        示例查询:
        - PREDICT SUM(sales, -7, 0) > 1000 FOR EACH store_id WHERE region = 'North'
        - PREDICT AVG(rating) FOR user_id IN (123, 456, 789)
        - PREDICT COUNT(clicks) FOR ALL products WHERE category = 'Electronics'
        """
        raise NotImplementedError("Subclasses must implement parse method")

    @abstractmethod
    def validate(self, query: PQLQuery, database_schema: Dict[str, List[str]]) -> List[str]:
        """
        验证查询的有效性

        Args:
            query: 解析后的查询对象
            database_schema: 数据库模式 {table_name: [column_names]}

        Returns:
            错误列表，如果查询有效则返回空列表
        """
        raise NotImplementedError("Subclasses must implement validate method")

    @abstractmethod
    def to_task_config(self, query: PQLQuery) -> TaskConfig:
        """
        将PQL查询转换为任务配置

        Args:
            query: 解析后的查询对象

        Returns:
            TaskConfig: 对应的任务配置
        """
        raise NotImplementedError("Subclasses must implement to_task_config method")


class MockPQLParser(PQLParser):
    """
    模拟PQL解析器，用于测试
    """

    def parse(self, query: str) -> PQLQuery:
        """
        模拟解析过程
        实际实现应该：
        1. 词法分析
        2. 语法分析
        3. 语义分析
        4. 构建查询对象
        """
        # 返回一个模拟的查询对象
        return PQLQuery(
            target_column="sales",
            aggregation="sum",
            time_window=(-7, 0),
            entity_table="stores",
            entity_column="store_id",
            entity_values=[1, 2, 3],
            conditions=[{"column": "region", "operator": "=", "value": "North"}],
            task_type="regression"
        )

    def validate(self, query: PQLQuery, database_schema: Dict[str, List[str]]) -> List[str]:
        """模拟验证"""
        errors = []

        # 检查表是否存在
        if query.entity_table not in database_schema:
            errors.append(f"Table '{query.entity_table}' not found")

        # 检查列是否存在
        if query.entity_table in database_schema:
            columns = database_schema[query.entity_table]
            if query.entity_column not in columns:
                errors.append(f"Column '{query.entity_column}' not found in table '{query.entity_table}'")

        return errors

    def to_task_config(self, query: PQLQuery) -> TaskConfig:
        """转换为任务配置"""
        # 推断任务类型
        if query.task_type:
            task_type = query.task_type
        else:
            # 基于聚合函数推断
            if query.aggregation in ['sum', 'mean', 'max', 'min']:
                task_type = 'regression'
            elif query.aggregation == 'count':
                task_type = 'classification'
            else:
                task_type = 'regression'

        config = TaskConfig(
            task_type=task_type,
            target_column=query.target_column,
            aggregation=query.aggregation
        )

        # 设置时间窗口
        if query.time_window:
            config.time_window_start = query.time_window[0]
            config.time_window_end = query.time_window[1]

        return config


class PQLBuilder:
    """
    PQL查询构建器
    提供流式API来构建PQL查询
    """

    def __init__(self):
        self._predict_clause = None
        self._for_clause = None
        self._where_clause = None

    def predict(self, target: str, aggregation: str = 'mean',
                time_window: Optional[Tuple[int, int]] = None) -> 'PQLBuilder':
        """设置PREDICT子句"""
        self._predict_clause = {
            'target': target,
            'aggregation': aggregation,
            'time_window': time_window
        }
        return self

    def for_each(self, entity_table: str, entity_column: str,
                 entity_values: Optional[List[Any]] = None) -> 'PQLBuilder':
        """设置FOR子句"""
        self._for_clause = {
            'entity_table': entity_table,
            'entity_column': entity_column,
            'entity_values': entity_values or []
        }
        return self

    def where(self, conditions: List[Dict[str, Any]]) -> 'PQLBuilder':
        """设置WHERE子句"""
        self._where_clause = conditions
        return self

    def build(self) -> PQLQuery:
        """构建查询对象"""
        if self._predict_clause is None:
            raise ValueError("Missing PREDICT clause")

        if self._for_clause is None:
            raise ValueError("Missing FOR clause")

        return PQLQuery(
            target_column=self._predict_clause['target'],
            aggregation=self._predict_clause['aggregation'],
            time_window=self._predict_clause['time_window'],
            entity_table=self._for_clause['entity_table'],
            entity_column=self._for_clause['entity_column'],
            entity_values=self._for_clause['entity_values'],
            conditions=self._where_clause
        )