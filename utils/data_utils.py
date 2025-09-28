"""
数据工具函数
提供数据处理和转换的辅助功能
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_csv_to_database(csv_path: str,
                         table_name: str,
                         database: 'Database',
                         date_columns: Optional[List[str]] = None,
                         categorical_columns: Optional[List[str]] = None) -> None:
    """
    从CSV加载数据到数据库

    Args:
        csv_path: CSV文件路径
        table_name: 表名
        database: 数据库实例
        date_columns: 日期列名列表
        categorical_columns: 类别列名列表
    """
    # 读取CSV
    df = pd.read_csv(csv_path)

    # 处理日期列
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

    # 处理类别列
    if categorical_columns:
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')

    # 添加到数据库
    from data.database import Table
    table = Table(table_name, df)
    database.tables[table_name] = table

    logger.info(f"从 {csv_path} 加载表 {table_name}，形状: {df.shape}")


def infer_column_type(series: pd.Series) -> str:
    """
    推断列的语义类型

    Args:
        series: pandas Series

    Returns:
        列类型: 'numerical', 'categorical', 'text', 'time', 'embedding'
    """
    dtype = series.dtype

    # 时间类型
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'time'

    # 数值类型
    if pd.api.types.is_numeric_dtype(series):
        # 检查是否是类别型（唯一值较少）
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.05 and series.nunique() < 100:
            return 'categorical'
        else:
            return 'numerical'

    # 字符串类型
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        # 检查平均长度
        avg_length = series.astype(str).str.len().mean()
        if avg_length > 50:
            return 'text'
        else:
            return 'categorical'

    # 列表/数组类型（可能是嵌入）
    if series.dtype == object and len(series) > 0:
        first_non_null = series.dropna().iloc[0] if len(series.dropna()) > 0 else None
        if isinstance(first_non_null, (list, np.ndarray)):
            return 'embedding'

    # 默认为类别型
    return 'categorical'


def split_temporal_data(df: pd.DataFrame,
                        time_column: str,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序划分数据集

    Args:
        df: 数据框
        time_column: 时间列名
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        (训练集, 验证集, 测试集)
    """
    # 按时间排序
    df = df.sort_values(by=time_column)

    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    logger.info(f"数据集划分 - 训练: {len(train_df)}, 验证: {len(val_df)}, 测试: {len(test_df)}")

    return train_df, val_df, test_df


def create_temporal_features(df: pd.DataFrame,
                             time_column: str,
                             feature_types: List[str] = ['hour', 'dayofweek', 'month', 'quarter']) -> pd.DataFrame:
    """
    创建时间特征

    Args:
        df: 数据框
        time_column: 时间列名
        feature_types: 要创建的特征类型

    Returns:
        添加了时间特征的数据框
    """
    df = df.copy()

    # 确保时间列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])

    # 创建特征
    if 'hour' in feature_types:
        df[f'{time_column}_hour'] = df[time_column].dt.hour

    if 'dayofweek' in feature_types:
        df[f'{time_column}_dayofweek'] = df[time_column].dt.dayofweek

    if 'day' in feature_types:
        df[f'{time_column}_day'] = df[time_column].dt.day

    if 'month' in feature_types:
        df[f'{time_column}_month'] = df[time_column].dt.month

    if 'quarter' in feature_types:
        df[f'{time_column}_quarter'] = df[time_column].dt.quarter

    if 'year' in feature_types:
        df[f'{time_column}_year'] = df[time_column].dt.year

    if 'is_weekend' in feature_types:
        df[f'{time_column}_is_weekend'] = df[time_column].dt.dayofweek.isin([5, 6]).astype(int)

    return df


def handle_missing_values(df: pd.DataFrame,
                          strategy: Dict[str, str] = None,
                          default_numerical: str = 'mean',
                          default_categorical: str = 'mode') -> pd.DataFrame:
    """
    处理缺失值

    Args:
        df: 数据框
        strategy: {列名: 策略} 字典
        default_numerical: 数值列的默认策略
        default_categorical: 类别列的默认策略

    Returns:
        处理后的数据框
    """
    df = df.copy()
    strategy = strategy or {}

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue

        # 获取列的策略
        col_strategy = strategy.get(col)
        if col_strategy is None:
            # 根据数据类型选择默认策略
            if pd.api.types.is_numeric_dtype(df[col]):
                col_strategy = default_numerical
            else:
                col_strategy = default_categorical

        # 应用策略
        if col_strategy == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif col_strategy == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif col_strategy == 'mode':
            mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
            df[col].fillna(mode_value, inplace=True)
        elif col_strategy == 'ffill':
            df[col].fillna(method='ffill', inplace=True)
        elif col_strategy == 'bfill':
            df[col].fillna(method='bfill', inplace=True)
        elif col_strategy == 'drop':
            df = df.dropna(subset=[col])
        else:
            # 使用提供的值
            df[col].fillna(col_strategy, inplace=True)

    return df


def encode_categorical_features(df: pd.DataFrame,
                                categorical_columns: List[str],
                                encoding_type: str = 'label',
                                encoder_dict: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    编码类别特征

    Args:
        df: 数据框
        categorical_columns: 类别列名列表
        encoding_type: 编码类型 ('label', 'onehot')
        encoder_dict: 已有的编码器字典

    Returns:
        (编码后的数据框, 编码器字典)
    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    df = df.copy()
    encoder_dict = encoder_dict or {}

    for col in categorical_columns:
        if col not in df.columns:
            continue

        if encoding_type == 'label':
            # 标签编码
            if col not in encoder_dict:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                encoder_dict[col] = encoder
            else:
                encoder = encoder_dict[col]
                # 处理未见过的类别
                df[col] = df[col].apply(
                    lambda x: encoder.transform([str(x)])[0]
                    if str(x) in encoder.classes_
                    else len(encoder.classes_)
                )

        elif encoding_type == 'onehot':
            # 独热编码
            if col not in encoder_dict:
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[[col]])
                encoder_dict[col] = encoder
            else:
                encoder = encoder_dict[col]
                encoded = encoder.transform(df[[col]])

            # 创建新列
            feature_names = [f"{col}_{i}" for i in range(encoded.shape[1])]
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

            # 替换原列
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

    return df, encoder_dict


def normalize_numerical_features(df: pd.DataFrame,
                                 numerical_columns: List[str],
                                 method: str = 'standard',
                                 scaler_dict: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    归一化数值特征

    Args:
        df: 数据框
        numerical_columns: 数值列名列表
        method: 归一化方法 ('standard', 'minmax', 'robust')
        scaler_dict: 已有的缩放器字典

    Returns:
        (归一化后的数据框, 缩放器字典)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    df = df.copy()
    scaler_dict = scaler_dict or {}

    # 选择缩放器
    scaler_class = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }.get(method, StandardScaler)

    for col in numerical_columns:
        if col not in df.columns:
            continue

        if col not in scaler_dict:
            scaler = scaler_class()
            df[col] = scaler.fit_transform(df[[col]])
            scaler_dict[col] = scaler
        else:
            scaler = scaler_dict[col]
            df[col] = scaler.transform(df[[col]])

    return df, scaler_dict


def create_graph_features(edge_list: List[Tuple[int, int]],
                          num_nodes: int) -> Dict[str, np.ndarray]:
    """
    创建图特征

    Args:
        edge_list: 边列表 [(source, target), ...]
        num_nodes: 节点总数

    Returns:
        图特征字典
    """
    # 度数特征
    in_degree = np.zeros(num_nodes)
    out_degree = np.zeros(num_nodes)

    for src, tgt in edge_list:
        out_degree[src] += 1
        in_degree[tgt] += 1

    # 计算其他特征
    total_degree = in_degree + out_degree
    degree_centrality = total_degree / (num_nodes - 1) if num_nodes > 1 else total_degree

    features = {
        'in_degree': in_degree,
        'out_degree': out_degree,
        'total_degree': total_degree,
        'degree_centrality': degree_centrality,
        'in_out_ratio': np.divide(in_degree, out_degree + 1e-8)
    }

    return features


def save_processed_data(data: Any,
                        path: str,
                        format: str = 'pickle') -> None:
    """
    保存处理后的数据

    Args:
        data: 要保存的数据
        path: 保存路径
        format: 保存格式 ('pickle', 'json', 'csv')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    elif format == 'json':
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    elif format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        else:
            raise ValueError("CSV格式只支持DataFrame")

    else:
        raise ValueError(f"不支持的格式: {format}")

    logger.info(f"数据已保存到 {path}")


def load_processed_data(path: str,
                        format: str = 'pickle') -> Any:
    """
    加载处理后的数据

    Args:
        path: 数据路径
        format: 数据格式 ('pickle', 'json', 'csv')

    Returns:
        加载的数据
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if format == 'pickle':
        with open(path, 'rb') as f:
            data = pickle.load(f)

    elif format == 'json':
        with open(path, 'r') as f:
            data = json.load(f)

    elif format == 'csv':
        data = pd.read_csv(path)

    else:
        raise ValueError(f"不支持的格式: {format}")

    logger.info(f"从 {path} 加载数据")

    return data


def create_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    创建数据集统计信息

    Args:
        df: 数据框

    Returns:
        统计信息字典
    """
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }

    # 数值列统计
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        stats['numerical_stats'] = df[numerical_cols].describe().to_dict()

    # 类别列统计
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        stats['categorical_stats'] = {}
        for col in categorical_cols:
            stats['categorical_stats'][col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(10).to_dict()
            }

    return stats