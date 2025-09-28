"""
配置文件加载器
从YAML文件加载配置
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

from config.model_config import (
    KumoRFMConfig,
    SamplingConfig,
    ColumnEncoderConfig,
    TaskConfig,
    ExperimentConfig
)


class ConfigLoader:
    """配置加载器"""

    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 处理环境变量替换
        config = ConfigLoader._replace_env_vars(config)

        return config

    @staticmethod
    def _replace_env_vars(config: Any) -> Any:
        """递归替换环境变量"""
        if isinstance(config, dict):
            return {k: ConfigLoader._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [ConfigLoader._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            # 环境变量格式: ${VAR_NAME:default_value}
            var_spec = config[2:-1]
            if ':' in var_spec:
                var_name, default_value = var_spec.split(':', 1)
            else:
                var_name, default_value = var_spec, None

            return os.environ.get(var_name, default_value)
        else:
            return config

    @staticmethod
    def create_model_config(yaml_path: str) -> KumoRFMConfig:
        """从YAML创建模型配置"""
        config_dict = ConfigLoader.load_yaml(yaml_path)
        model_dict = config_dict.get('model', {})

        # 创建采样配置
        sampling_dict = model_dict.pop('sampling', {})
        sampling_config = SamplingConfig(**sampling_dict)

        # 创建主配置
        config = KumoRFMConfig(**model_dict)
        config.sampling_config = sampling_config

        return config

    @staticmethod
    def create_training_config(yaml_path: str) -> Dict[str, Any]:
        """从YAML创建训练配置"""
        config_dict = ConfigLoader.load_yaml(yaml_path)
        training_dict = config_dict.get('training', {})

        return training_dict

    @staticmethod
    def create_experiment_config(yaml_path: str) -> ExperimentConfig:
        """从YAML创建实验配置"""
        config_dict = ConfigLoader.load_yaml(yaml_path)
        experiment_dict = config_dict.get('experiment', {})

        # 移除wandb配置（如果存在）
        experiment_dict.pop('wandb', None)

        return ExperimentConfig(**experiment_dict)

    @staticmethod
    def create_task_configs(yaml_path: str) -> Dict[str, TaskConfig]:
        """从YAML创建任务配置"""
        config_dict = ConfigLoader.load_yaml(yaml_path)
        tasks_dict = config_dict.get('tasks', {})

        task_configs = {}
        for task_name, task_dict in tasks_dict.items():
            # 重命名字段以匹配TaskConfig
            if 'type' in task_dict:
                task_dict['task_type'] = task_dict.pop('type')

            task_configs[task_name] = TaskConfig(**task_dict)

        return task_configs

    @staticmethod
    def create_encoder_config(yaml_path: str) -> ColumnEncoderConfig:
        """从YAML创建编码器配置"""
        config_dict = ConfigLoader.load_yaml(yaml_path)
        encoders_dict = config_dict.get('encoders', {})

        # 扁平化配置
        flat_config = {}
        for encoder_type, encoder_config in encoders_dict.items():
            for key, value in encoder_config.items():
                flat_key = f"{encoder_type}_{key}"
                flat_config[flat_key] = value

        return ColumnEncoderConfig(**flat_config)

    @staticmethod
    def load_database_schema(yaml_path: str) -> Dict[str, Dict[str, str]]:
        """从YAML加载数据库模式"""
        config_dict = ConfigLoader.load_yaml(yaml_path)
        return config_dict.get('database_schema', {})

    @staticmethod
    def create_all_configs(yaml_path: str) -> Dict[str, Any]:
        """创建所有配置"""
        return {
            'model_config': ConfigLoader.create_model_config(yaml_path),
            'training_config': ConfigLoader.create_training_config(yaml_path),
            'experiment_config': ConfigLoader.create_experiment_config(yaml_path),
            'task_configs': ConfigLoader.create_task_configs(yaml_path),
            'encoder_config': ConfigLoader.create_encoder_config(yaml_path),
            'database_schema': ConfigLoader.load_database_schema(yaml_path)
        }


def load_config_from_yaml(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    便捷函数：从YAML文件加载所有配置

    Args:
        config_path: YAML配置文件路径

    Returns:
        包含所有配置的字典
    """
    return ConfigLoader.create_all_configs(config_path)


# 使用示例
if __name__ == "__main__":
    # 加载配置
    configs = load_config_from_yaml("config.yaml")

    # 获取各种配置
    model_config = configs['model_config']
    training_config = configs['training_config']
    experiment_config = configs['experiment_config']
    task_configs = configs['task_configs']
    database_schema = configs['database_schema']

    print(f"模型配置: hidden_dim={model_config.hidden_dim}, "
          f"num_layers={model_config.num_layers}")
    print(f"训练配置: batch_size={training_config['batch_size']}, "
          f"learning_rate={training_config['learning_rate']}")
    print(f"任务数量: {len(task_configs)}")
    print(f"数据库表数量: {len(database_schema)}")