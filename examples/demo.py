"""
KumoRFM演示示例
展示如何使用KumoRFM进行预测和微调
"""

import torch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from config.model_config import KumoRFMConfig, TaskConfig, ExperimentConfig, SamplingConfig
from data.database import MockDatabase
from data.graph_converter import MockGraphConverter
from data.temporal_graph import TemporalHeterogeneousGraph
from pql.parser import PQLBuilder, MockPQLParser
from models.kumorfm import KumoRFM
from inference.predictor import KumoRFMPredictor
from training.finetune import finetune_kumorfm
from utils.data_utils import create_dataset_statistics
from utils.training_utils import set_random_seed


def create_mock_data():
    """创建模拟数据用于演示"""
    print("创建模拟数据...")

    # 创建用户表
    users_data = {
        'user_id': range(100),
        'age': np.random.randint(18, 65, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'city': np.random.choice(['北京', '上海', '广州', '深圳'], 100),
        'registration_date': pd.date_range('2020-01-01', periods=100, freq='D')
    }
    users_df = pd.DataFrame(users_data)

    # 创建商品表
    items_data = {
        'item_id': range(200),
        'category': np.random.choice(['电子', '服装', '食品', '图书'], 200),
        'price': np.random.uniform(10, 1000, 200),
        'brand': np.random.choice(['品牌A', '品牌B', '品牌C'], 200)
    }
    items_df = pd.DataFrame(items_data)

    # 创建交易表
    transactions = []
    for _ in range(1000):
        user_id = np.random.randint(0, 100)
        item_id = np.random.randint(0, 200)
        timestamp = datetime.now() - timedelta(days=np.random.randint(0, 365))
        amount = items_df.loc[item_id, 'price'] * np.random.uniform(0.8, 1.2)

        transactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'timestamp': timestamp,
            'amount': amount,
            'quantity': np.random.randint(1, 5)
        })

    transactions_df = pd.DataFrame(transactions)

    return users_df, items_df, transactions_df


def demo_basic_prediction():
    """演示基本预测功能"""
    print("\n=== 基本预测演示 ===")

    # 设置随机种子
    set_random_seed(42)

    # 创建模拟数据
    users_df, items_df, transactions_df = create_mock_data()

    # 创建数据库
    database = MockDatabase()
    database.tables['users'] = users_df
    database.tables['items'] = items_df
    database.tables['transactions'] = transactions_df

    # 定义数据库模式
    database_schema = {
        'users': {
            'user_id': 'categorical',
            'age': 'numerical',
            'gender': 'categorical',
            'city': 'categorical',
            'registration_date': 'time'
        },
        'items': {
            'item_id': 'categorical',
            'category': 'categorical',
            'price': 'numerical',
            'brand': 'categorical'
        },
        'transactions': {
            'user_id': 'categorical',
            'item_id': 'categorical',
            'timestamp': 'time',
            'amount': 'numerical',
            'quantity': 'numerical'
        }
    }

    # 创建图转换器
    converter = MockGraphConverter()
    graph = converter.convert(database)

    print(f"创建的图: {graph}")

    # 创建模型配置
    config = KumoRFMConfig(
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        max_neighbors=50,
        num_hops=2
    )

    # 创建模型
    model = KumoRFM(config, database_schema)

    # 创建预测器
    predictor = KumoRFMPredictor(model, config)

    # 定义任务：预测用户下个月的购买金额
    task_config = TaskConfig(
        task_type='regression',
        target_column='amount',
        aggregation='sum',
        time_window_start=-30,
        time_window_end=0
    )

    # 预测
    target_entity = ('users', 0)  # 用户0
    prediction_time = datetime.now()

    print(f"\n预测用户 {target_entity[1]} 在 {prediction_time.date()} 的购买金额...")

    result = predictor.predict(
        graph,
        target_entity,
        prediction_time,
        task_config,
        num_context=5
    )

    print(f"预测结果: {result['predicted_value']:.2f}")
    print(f"使用的上下文数量: {result['num_context_used']}")

    return model, database, graph


def demo_pql_prediction():
    """演示使用PQL进行预测"""
    print("\n=== PQL预测演示 ===")

    # 创建模型和数据
    model, database, graph = demo_basic_prediction()

    # 创建PQL查询
    pql_query = """
    PREDICT SUM(amount, -30, 0) > 1000
    FOR EACH user_id IN (0, 1, 2, 3, 4)
    WHERE city = '北京'
    """

    print(f"\nPQL查询: {pql_query}")

    # 创建预测器
    predictor = KumoRFMPredictor(model, model.config)

    # 执行PQL预测
    results = predictor.predict_from_pql(
        database,
        graph,
        pql_query
    )

    print("\n预测结果:")
    for user_id, result in results.items():
        if user_id != '_aggregated':
            print(f"  用户 {user_id}: {result.get('predicted_value', 'N/A'):.2f}")

    if '_aggregated' in results:
        print(f"\n聚合结果: {results['_aggregated']}")


def demo_batch_prediction():
    """演示批量预测"""
    print("\n=== 批量预测演示 ===")

    # 创建模型和数据
    model, database, graph = demo_basic_prediction()

    # 准备批量预测
    entities = [('users', i) for i in range(10)]
    prediction_time = datetime.now()

    task_config = TaskConfig(
        task_type='classification',
        target_column='will_purchase',
        num_classes=2
    )

    # 创建预测器
    predictor = KumoRFMPredictor(model, model.config)

    print(f"\n批量预测 {len(entities)} 个用户...")

    results = predictor.batch_predict(
        graph,
        entities,
        prediction_time,
        task_config,
        batch_size=5
    )

    # 统计结果
    positive_count = sum(1 for r in results if r.get('predicted_class', 0) == 1)
    print(f"\n预测结果统计:")
    print(f"  预测会购买的用户数: {positive_count}")
    print(f"  预测不会购买的用户数: {len(results) - positive_count}")

    # 显示前5个结果
    print("\n前5个预测结果:")
    for i, result in enumerate(results[:5]):
        print(f"  用户 {i}: 类别={result['predicted_class']}, "
              f"置信度={result.get('confidence', 0):.3f}")


def demo_finetuning():
    """演示模型微调"""
    print("\n=== 模型微调演示 ===")

    # 创建模型和数据
    model, database, graph = demo_basic_prediction()

    # 创建PQL查询定义任务
    pql_builder = PQLBuilder()
    pql_query = pql_builder \
        .predict('amount', 'sum', (-7, 0)) \
        .for_each('users', 'user_id') \
        .build()

    # 任务配置
    task_config = TaskConfig(
        task_type='regression',
        target_column='amount',
        aggregation='sum',
        time_window_start=-7,
        time_window_end=0
    )

    # 实验配置
    experiment_config = ExperimentConfig(
        num_epochs=3,
        save_dir='./checkpoints_demo'
    )

    print("\n开始微调模型...")

    # 执行微调
    results = finetune_kumorfm(
        model,
        database,
        graph,
        pql_query,
        model.config,
        task_config,
        experiment_config,
        freeze_backbone=True
    )

    print("\n微调完成!")
    print(f"训练损失历史: {results['train_history']['train_loss']}")
    print(f"测试结果: {results['test_results']}")


def demo_attention_visualization():
    """演示注意力可视化"""
    print("\n=== 注意力可视化演示 ===")

    # 创建模型和数据
    model, database, graph = demo_basic_prediction()

    # 创建预测器
    predictor = KumoRFMPredictor(model, model.config)

    # 定义任务
    task_config = TaskConfig(
        task_type='regression',
        target_column='amount'
    )

    # 预测并获取注意力权重
    target_entity = ('users', 0)
    prediction_time = datetime.now()

    result = predictor.predict(
        graph,
        target_entity,
        prediction_time,
        task_config,
        return_attention=True
    )

    if 'attention_weights' in result:
        print("\n注意力权重:")

        # 实体内注意力
        if 'intra_entity' in result['attention_weights']:
            weights = result['attention_weights']['intra_entity']
            print(f"  实体内注意力形状: {weights.shape if hasattr(weights, 'shape') else 'N/A'}")

        # 子图间注意力
        if 'inter_subgraph' in result['attention_weights']:
            weights = result['attention_weights']['inter_subgraph']
            print(f"  子图间注意力形状: {weights.shape if hasattr(weights, 'shape') else 'N/A'}")

        # 相关性分数
        if 'relevance_scores' in result['attention_weights']:
            scores = result['attention_weights']['relevance_scores']
            print(f"  相关性分数: {scores if hasattr(scores, '__len__') and len(scores) < 10 else 'N/A'}")
    else:
        print("未返回注意力权重")


def demo_save_and_load():
    """演示模型保存和加载"""
    print("\n=== 模型保存和加载演示 ===")

    # 创建模型和数据
    model, database, graph = demo_basic_prediction()

    # 保存检查点
    checkpoint_path = './demo_checkpoint.pt'

    print(f"\n保存模型到 {checkpoint_path}...")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'database_schema': database_schema
    }
    torch.save(checkpoint, checkpoint_path)

    print("模型已保存")

    # 加载模型
    print(f"\n从 {checkpoint_path} 加载模型...")

    loaded_predictor = KumoRFMPredictor.from_checkpoint(
        checkpoint_path,
        database_schema
    )

    print("模型已加载")

    # 验证加载的模型
    task_config = TaskConfig(task_type='regression')
    target_entity = ('users', 0)
    prediction_time = datetime.now()

    result = loaded_predictor.predict(
        graph,
        target_entity,
        prediction_time,
        task_config
    )

    print(f"\n加载模型的预测结果: {result.get('predicted_value', 'N/A')}")

    # 清理
    Path(checkpoint_path).unlink()


def main():
    """运行所有演示"""
    print("=" * 50)
    print("KumoRFM 演示")
    print("=" * 50)

    # 运行各个演示
    demo_basic_prediction()
    demo_pql_prediction()
    demo_batch_prediction()
    demo_finetuning()
    demo_attention_visualization()
    demo_save_and_load()

    print("\n所有演示完成！")


if __name__ == "__main__":
    # 设置数据库模式（全局变量）
    database_schema = {
        'users': {
            'user_id': 'categorical',
            'age': 'numerical',
            'gender': 'categorical',
            'city': 'categorical',
            'registration_date': 'time'
        },
        'items': {
            'item_id': 'categorical',
            'category': 'categorical',
            'price': 'numerical',
            'brand': 'categorical'
        },
        'transactions': {
            'user_id': 'categorical',
            'item_id': 'categorical',
            'timestamp': 'time',
            'amount': 'numerical',
            'quantity': 'numerical'
        }
    }

    main()