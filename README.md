# RGFM: 关系数据基础模型

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

RGFM 是一个基于深度学习的关系数据基础模型，专门设计用于处理时序异构图数据。它结合了最先进的图神经网络技术和上下文学习机制，能够在各种关系数据预测任务上实现优异的性能。

## 主要特性

- **零样本泛化**: 无需任务特定训练即可处理新的预测任务
- **多模态编码**: 支持数值、类别、文本、时间等多种数据类型
- **上下文学习**: 通过历史示例进行模式识别和预测
- **灵活的任务支持**: 分类、回归、链接预测等多种任务
- **PQL查询接口**: 使用预测查询语言(PQL)定义任务
- **集成RelGT**: 利用关系图变换器进行深度图表示学习

##  架构概览

RGFM采用五阶段处理流程：

1. **动态子图采样**: 从时序异构图中采样相关的历史子图
2. **多模态特征编码**: 统一编码不同类型的数据
3. **关系图变换器(RelGT)**: 深度图神经网络处理
4. **上下文学习(ICL)**: 利用历史模式进行预测
5. **任务特定输出**: 根据任务类型生成预测结果

##  安装

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/JeromeZzz/RGFM.git
cd RGFM

# 安装依赖
pip install -r requirements.txt

# 安装KumoRFM
pip install -e .
```

### 安装RelGT（可选但推荐）

```bash
# 克隆RelGT
git clone https://github.com/snap-stanford/relgt.git models/relgt/relgt

# 安装RelGT依赖
pip install -r models/relgt/relgt/requirements.txt
```

### 开发安装

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 安装可视化工具
pip install -e ".[viz]"

# 安装高级功能
pip install -e ".[advanced]"
```

##  快速开始

### 基本预测示例

```python
from kumorfm import KumoRFM, KumoRFMPredictor
from kumorfm.config import KumoRFMConfig, TaskConfig
from datetime import datetime

# 创建配置
config = KumoRFMConfig(
    hidden_dim=256,
    num_layers=4,
    num_heads=8
)

# 定义数据库模式
database_schema = {
    'users': {'user_id': 'categorical', 'age': 'numerical'},
    'items': {'item_id': 'categorical', 'price': 'numerical'},
    'transactions': {'user_id': 'categorical', 'item_id': 'categorical', 
                     'amount': 'numerical', 'timestamp': 'time'}
}

# 创建模型
model = KumoRFM(config, database_schema)

# 创建预测器
predictor = KumoRFMPredictor(model, config)

# 定义任务：预测用户下周的购买金额
task_config = TaskConfig(
    task_type='regression',
    target_column='amount',
    aggregation='sum',
    time_window_start=-7,
    time_window_end=0
)

# 执行预测
result = predictor.predict(
    graph,  # 时序异构图
    target_entity=('users', 123),  # 用户123
    prediction_time=datetime.now(),
    task_config=task_config
)

print(f"预测金额: {result['predicted_value']:.2f}")
```

### 使用PQL查询

```python
# 使用PQL定义预测任务
pql_query = """
PREDICT SUM(sales, -30, 0) > 1000
FOR EACH store_id IN (1, 2, 3, 4, 5)
WHERE region = 'North'
"""

# 执行PQL预测
results = predictor.predict_from_pql(
    database,
    graph,
    pql_query
)

# 查看结果
for store_id, result in results.items():
    print(f"店铺 {store_id}: {result['predicted_value']}")
```

### 模型微调

```python
from kumorfm.training import finetune_kumorfm

# 为特定任务微调模型
results = finetune_kumorfm(
    model,
    database,
    graph,
    pql_query,
    config,
    task_config,
    experiment_config,
    freeze_backbone=True  # 冻结骨干网络
)

print(f"测试MAE: {results['test_results']['mae']:.4f}")
```

##  运行演示

```bash
# 运行完整演示
python examples/demo.py

# 或使用命令行
kumorfm-demo
```

演示包括：
- 基本预测
- PQL查询
- 批量预测
- 模型微调
- 注意力可视化
- 模型保存和加载

## 🔧 配置说明

### 模型配置 (KumoRFMConfig)

```python
config = KumoRFMConfig(
    # 模型架构
    hidden_dim=256,          # 隐藏层维度
    num_layers=4,            # RelGT层数
    num_heads=8,             # 注意力头数
    
    # 采样参数
    max_neighbors=300,       # 最大邻居数
    num_hops=2,              # 采样跳数
    context_window_size=10,  # 上下文示例数
    
    # 训练参数
    learning_rate=1e-4,      # 学习率
    dropout_rate=0.3,        # Dropout率
    batch_size=256           # 批次大小
)
```

### 任务配置 (TaskConfig)

```python
task_config = TaskConfig(
    task_type='classification',    # 任务类型
    num_classes=5,                 # 类别数
    target_column='label',         # 目标列
    aggregation='mean',            # 聚合方式
    time_window_start=-7,          # 时间窗口开始（天）
    time_window_end=0              # 时间窗口结束（天）
)
```

##  项目结构

```
kumorfm/
├── config/              # 配置模块
├── data/                # 数据处理
├── pql/                 # PQL解析器
├── sampling/            # 子图采样
├── models/              # 模型实现
│   ├── encoders/        # 编码器
│   ├── relgt/           # RelGT集成
│   ├── icl/             # 上下文学习
│   └── kumorfm.py       # 主模型
├── training/            # 训练模块
├── inference/           # 推理模块
├── utils/               # 工具函数
└── examples/            # 示例代码
```

##  技术细节

### 多元素Token化

每个节点被分解为5个元素：
1. **节点特征**: 多模态编码后的特征
2. **节点类型**: 表类型的one-hot编码
3. **跳跃距离**: 与目标节点的结构距离
4. **相对时间**: 与预测时间的时间差
5. **子图结构**: 局部图结构编码

### 双重上下文机制

- **实体内上下文**: 关注目标实体自身的历史
- **子图间上下文**: 捕获不同历史快照间的模式
