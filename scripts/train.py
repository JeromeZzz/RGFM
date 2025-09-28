"""
KumoRFM训练脚本
支持从命令行训练模型
"""

import argparse
import logging
from pathlib import Path
import torch
from datetime import datetime
import json

from utils.config_loader import load_config_from_yaml
from utils.training_utils import set_random_seed
from models.kumorfm import KumoRFM
from training.trainer import KumoRFMTrainer
from training.finetune import finetune_kumorfm
from data.database import MockDatabase
from data.graph_converter import MockGraphConverter
from pql.parser import MockPQLParser

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train KumoRFM model')

    # 基本参数
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'finetune'],
                        default='train', help='Training mode')

    # 数据参数
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing data files')
    parser.add_argument('--database', type=str,
                        help='Path to database file')
    parser.add_argument('--graph', type=str,
                        help='Path to graph file')

    # 训练参数
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float,
                        help='Learning rate (overrides config)')

    # 微调参数
    parser.add_argument('--checkpoint', type=str,
                        help='Checkpoint path for fine-tuning')
    parser.add_argument('--pql', type=str,
                        help='PQL query for fine-tuning task')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone during fine-tuning')

    # 输出参数
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--experiment-name', type=str,
                        default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help='Experiment name')

    # 其他参数
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                        default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')

    args = parser.parse_args()

    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    # 设置随机种子
    set_random_seed(args.seed)

    # 加载配置
    logger.info(f"从 {args.config} 加载配置")
    configs = load_config_from_yaml(args.config)

    model_config = configs['model_config']
    training_config = configs['training_config']
    experiment_config = configs['experiment_config']
    task_configs = configs['task_configs']
    database_schema = configs['database_schema']

    # 覆盖配置（如果提供了命令行参数）
    if args.epochs:
        experiment_config.num_epochs = args.epochs
    if args.batch_size:
        model_config.batch_size = args.batch_size
        training_config['batch_size'] = args.batch_size
    if args.lr:
        model_config.learning_rate = args.lr
        training_config['learning_rate'] = args.lr

    # 创建输出目录
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_config.save_dir = str(output_dir / 'checkpoints')
    experiment_config.log_dir = str(output_dir / 'logs')

    # 保存配置
    config_save_path = output_dir / 'config.json'
    with open(config_save_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'model_config': model_config.__dict__,
            'experiment_config': experiment_config.__dict__
        }, f, indent=2)
    logger.info(f"配置已保存到 {config_save_path}")

    # 加载数据
    logger.info("加载数据...")
    database = load_database(args.database or args.data_dir)
    graph = load_graph(args.graph or args.data_dir, database)

    if args.mode == 'train':
        # 训练模式
        logger.info("开始训练...")

        # 创建模型
        model = KumoRFM(model_config, database_schema)

        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(
            database, graph, task_configs, training_config
        )

        # 创建训练器
        trainer = KumoRFMTrainer(model, model_config, experiment_config, device)

        # 训练
        train_history = trainer.fit(train_loader, val_loader)

        # 保存最终模型
        final_checkpoint_path = output_dir / 'final_model.pt'
        trainer.save_checkpoint(str(final_checkpoint_path))
        logger.info(f"最终模型已保存到 {final_checkpoint_path}")

        # 保存训练历史
        history_path = output_dir / 'train_history.json'
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=2)

    elif args.mode == 'finetune':
        # 微调模式
        logger.info("开始微调...")

        if not args.checkpoint:
            raise ValueError("微调模式需要提供 --checkpoint 参数")

        if not args.pql:
            raise ValueError("微调模式需要提供 --pql 参数")

        # 加载预训练模型
        logger.info(f"从 {args.checkpoint} 加载模型")
        model = load_pretrained_model(args.checkpoint, database_schema)

        # 解析PQL查询
        parser = MockPQLParser()
        pql_query = parser.parse(args.pql)

        # 获取任务配置
        task_config = parser.to_task_config(pql_query)

        # 执行微调
        results = finetune_kumorfm(
            model,
            database,
            graph,
            pql_query,
            model_config,
            task_config,
            experiment_config,
            freeze_backbone=args.freeze_backbone
        )

        # 保存结果
        results_path = output_dir / 'finetune_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"微调结果已保存到 {results_path}")

    logger.info("训练完成！")


def load_database(path):
    """加载数据库（简化实现）"""
    # 实际应该从文件或数据库加载
    database = MockDatabase()
    # 这里可以添加从CSV或其他格式加载数据的逻辑
    return database


def load_graph(path, database):
    """加载图（简化实现）"""
    # 实际应该从文件加载或使用转换器生成
    converter = MockGraphConverter()
    graph = converter.convert(database)
    return graph


def create_data_loaders(database, graph, task_configs, training_config):
    """创建数据加载器（简化实现）"""
    # 实际应该创建真实的数据加载器
    # 这里返回模拟的加载器
    from torch.utils.data import DataLoader, TensorDataset

    # 创建模拟数据
    num_samples = 1000
    mock_data = torch.randn(num_samples, 10)
    mock_labels = torch.randint(0, 2, (num_samples,))

    dataset = TensorDataset(mock_data, mock_labels)

    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False
    )

    return train_loader, val_loader


def load_pretrained_model(checkpoint_path, database_schema):
    """加载预训练模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 提取配置
    config = checkpoint.get('config', KumoRFMConfig())

    # 创建模型
    model = KumoRFM(config, database_schema)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


if __name__ == '__main__':
    main()