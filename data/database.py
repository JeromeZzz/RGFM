"""
数据库接口（保留接口，不实现具体功能）
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd


class Table:
    """Abstract representation of a table"""

    def __init__(self, name: str, data: pd.DataFrame):
        self.name = name
        self.data = data
        self.columns = list(data.columns)

    def get_column_type(self, column: str) -> str:
        """Get column data type"""
        # This is just an interface, actual implementation should infer semantic type of column
        return "unknown"


class Database(ABC):
    """
    Database Abstract Interface
    Used for accessing tables and data in relational databases
    """

    def __init__(self):
        self.tables: Dict[str, Table] = {}
        self.relationships: List[Dict[str, Any]] = []  # Primary-foreign key relationships

    @abstractmethod
    def load_from_source(self, source: Any) -> None:
        """
        Load database from data source
        source: Data source (can be connection string, file path, etc.)
        """
        raise NotImplementedError("Subclasses must implement load_from_source")

    @abstractmethod
    def get_table(self, table_name: str) -> Table:
        """Get specified table"""
        raise NotImplementedError("Subclasses must implement get_table")

    @abstractmethod
    def get_schema(self) -> Dict[str, List[str]]:
        """
        Get database schema
        Returns: {table_name: [column_names]}
        """
        raise NotImplementedError("Subclasses must implement get_schema")

    @abstractmethod
    def get_relationships(self) -> List[Dict[str, Any]]:
        """
        Get relationships between tables
        Returns: [{
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
        Get column metadata
        Returns statistical information, data types, etc. of columns
        """
        # Interface reserved, concrete implementation should return detailed column metadata
        return {
            'table': table_name,
            'column': column_name,
            'dtype': 'unknown',
            'semantic_type': 'unknown'
        }


class MockDatabase(Database):
    """
    Mock database implementation for testing
    """

    def load_from_source(self, source: Any) -> None:
        """Mock loading"""
        # Create some mock data for testing
        pass

    def get_table(self, table_name: str) -> Table:
        """Return mock table"""
        if table_name not in self.tables:
            # Create an empty DataFrame as mock
            df = pd.DataFrame()
            self.tables[table_name] = Table(table_name, df)
        return self.tables[table_name]

    def get_schema(self) -> Dict[str, List[str]]:
        """Return mock schema"""
        return {name: table.columns for name, table in self.tables.items()}

    def get_relationships(self) -> List[Dict[str, Any]]:
        """Return mock relationships"""
        return self.relationships