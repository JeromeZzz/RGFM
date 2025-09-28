"""
数据库接口（保留接口，不实现具体功能）
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd


class Table:
    """表的抽象表示"""

    def __init__(self, name: str, data: pd.DataFrame):
        self.name = name
        self.data = data
        self.columns = list(data.columns)

    def get_column_type(self, column: str) -> str:
        """获取列的数据类型"""
        # 这里只是接口，实际实现应该推断列的语义类型
        return "unknown"


class Database(ABC):
    """
    数据库抽象接口
    用于访问关系型数据库中的表和数据
    """

    def __init__(self):
        self.tables: Dict[str, Table] = {}
        self.relationships: List[Dict[str, Any]] = []  # 主外键关系

    @abstractmethod
    def load_from_source(self, source: Any) -> None:
        """
        从数据源加载数据库
        source: 数据源（可以是连接字符串、文件路径等）
        """
        raise NotImplementedError("Subclasses must implement load_from_source")

    @abstractmethod
    def get_table(self, table_name: str) -> Table:
        """获取指定表"""
        raise NotImplementedError("Subclasses must implement get_table")

    @abstractmethod
    def get_schema(self) -> Dict[str, List[str]]:
        """
        获取数据库模式
        返回: {table_name: [column_names]}
        """
        raise NotImplementedError("Subclasses must implement get_schema")

    @abstractmethod
    def get_relationships(self) -> List[Dict[str, Any]]:
        """
        获取表之间的关系
        返回: [{
            'source_table': str,
            'source_column': str,
            'target_table': str,
            'target_column': str,
            'relationship_type': str  # 'one-to-one', 'one-to-many', 'many-to-many'
        }]
        """
        raise NotImplementedError("Subclasses must implement get_relationships")

    def get_column_metadata(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """
        获取列的元数据
        返回列的统计信息、数据类型等
        """
        # 接口预留，具体实现应返回详细的列元数据
        return {
            'table': table_name,
            'column': column_name,
            'dtype': 'unknown',
            'semantic_type': 'unknown'
        }


class MockDatabase(Database):
    """
    模拟数据库实现，用于测试
    """

    def load_from_source(self, source: Any) -> None:
        """模拟加载"""
        # 创建一些模拟数据用于测试
        pass

    def get_table(self, table_name: str) -> Table:
        """返回模拟表"""
        if table_name not in self.tables:
            # 创建一个空的DataFrame作为模拟
            df = pd.DataFrame()
            self.tables[table_name] = Table(table_name, df)
        return self.tables[table_name]

    def get_schema(self) -> Dict[str, List[str]]:
        """返回模拟模式"""
        return {name: table.columns for name, table in self.tables.items()}

    def get_relationships(self) -> List[Dict[str, Any]]:
        """返回模拟关系"""
        return self.relationships