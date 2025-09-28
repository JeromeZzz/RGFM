"""
KumoRFM基础测试
"""

import pytest
import torch
import numpy as np
from datetime import datetime, timedelta

from config.model_config import KumoRFMConfig, TaskConfig
from data.temporal_graph import TemporalHeterogeneousGraph
from models.kumorfm import KumoRFM
from sampling.backward_sampler import BackwardSubgraphSampler
from sampling.context_sampler import ContextSampler


class TestTemporalGraph:
    """测试时序异构图"""

    def test_create_graph(self):
        """测试创建图"""
        graph = TemporalHeterogeneousGraph()

        # 添加节点类型
        graph.add_node_type("users", 100)
        graph.add_node_type("items", 200)

        assert len(graph.node_types) == 2
        assert graph.node_counts["users"] == 100
        assert graph.node_counts["items"] == 200
        assert graph.total_nodes == 300

    def test_add_edges(self):
        """测试添加边"""
        graph = TemporalHeterogeneousGraph()
        graph.add_node_type("users", 10)
        graph.add_node_type("items", 20)

        # 添加边类型
        edge_index = torch.tensor([[0, 1, 2], [5, 10, 15]])
        graph.add_edge_type("user_item", ("users", "items"), edge_index)

        assert len(graph.edge_types) == 1
        assert graph.edge_indices[("users", "user_item", "items")].shape == (2, 3)


class TestSampling:
    """测试采样模块"""

    def test_backward_sampling(self):
        """测试后向采样"""
        from config.model_config import SamplingConfig

        # 创建图
        graph = create_test_graph()

        # 创建采样器
        config = SamplingConfig()
        sampler = BackwardSubgraphSampler(config)

        # 采样
        target_entity = ("users", 0)
        prediction_time = datetime.now()

        subgraph = sampler.sample(
            graph, target_entity, prediction_time,
            num_hops=2, max_nodes=50
        )

        assert isinstance(subgraph, TemporalHeterogeneousGraph)
        assert subgraph.total_nodes <= 50

    def test_context_sampling(self):
        """测试上下文采样"""
        from config.model_config import SamplingConfig

        # 创建图和采样器
        graph = create_test_graph()
        config = SamplingConfig()
        backward_sampler = BackwardSubgraphSampler(config)
        context_sampler = ContextSampler(config, backward_sampler)

        # 采样上下文
        target_entity = ("users", 0)
        prediction_time = datetime.now()
        task_config = TaskConfig(task_type="regression")

        contexts = context_sampler.sample_context(
            graph, target_entity, prediction_time,
            task_config, num_examples=5
        )

        assert len(contexts) <= 5
        for ctx in contexts:
            assert hasattr(ctx, 'subgraph')
            assert hasattr(ctx, 'entity')
            assert hasattr(ctx, 'timestamp')
            assert hasattr(ctx, 'label')


class TestModel:
    """测试模型"""

    def test_model_creation(self):
        """测试模型创建"""
        config = KumoRFMConfig(hidden_dim=64, num_layers=2)
        database_schema = {
            'users': {'user_id': 'categorical', 'age': 'numerical'},
            'items': {'item_id': 'categorical', 'price': 'numerical'}
        }

        model = KumoRFM(config, database_schema)

        assert model.config.hidden_dim == 64
        assert model.config.num_layers == 2

        # 检查子模块
        assert hasattr(model, 'multimodal_encoder')
        assert hasattr(model, 'table_encoder')
        assert hasattr(model, 'relgt')
        assert hasattr(model, 'icl_module')

    def test_forward_pass(self):
        """测试前向传播"""
        config = KumoRFMConfig(hidden_dim=64, num_layers=1)
        database_schema = {
            'users': {'user_id': 'categorical'},
            'items': {'item_id': 'categorical'}
        }

        model = KumoRFM(config, database_schema)
        graph = create_test_graph()

        # 前向传播
        target_entity = ("users", 0)
        prediction_time = datetime.now()
        task_config = TaskConfig(task_type="classification", num_classes=2)

        output = model(
            graph, target_entity, prediction_time,
            task_config, num_context=2
        )

        assert isinstance(output, dict)
        assert 'predicted_class' in output or 'predictions' in output


class TestEncoders:
    """测试编码器"""

    def test_numerical_encoder(self):
        """测试数值编码器"""
        from models.encoders.column_encoder import NumericalEncoder
        from config.model_config import ColumnEncoderConfig

        config = ColumnEncoderConfig()
        encoder = NumericalEncoder(config)

        # 测试输入
        x = torch.randn(10, 5)  # batch_size=10, seq_len=5
        output = encoder(x)

        assert output.shape == (10, 5, config.numerical_embedding_dim)

    def test_categorical_encoder(self):
        """测试类别编码器"""
        from models.encoders.column_encoder import CategoricalEncoder
        from config.model_config import ColumnEncoderConfig

        config = ColumnEncoderConfig()
        vocab_size = 100
        encoder = CategoricalEncoder(config, vocab_size)

        # 测试输入
        x = torch.randint(0, vocab_size, (10, 5))
        output = encoder(x)

        assert output.shape == (10, 5, config.categorical_embedding_dim)


def create_test_graph():
    """创建测试用的图"""
    graph = TemporalHeterogeneousGraph()

    # 添加节点
    graph.add_node_type("users", 50)
    graph.add_node_type("items", 100)

    # 添加边
    num_edges = 200
    src = torch.randint(0, 50, (num_edges,))
    dst = torch.randint(0, 100, (num_edges,))
    edge_index = torch.stack([src, dst])

    # 添加时间戳
    timestamps = torch.rand(num_edges) * 365 * 24 * 3600  # 一年内的随机时间

    graph.add_edge_type(
        "user_item",
        ("users", "items"),
        edge_index,
        timestamps
    )

    return graph


if __name__ == "__main__":
    pytest.main([__file__, "-v"])