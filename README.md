# RGFM：关系图基础模型

RGFM（Relational Graph Foundation Model）是一个端到端的关系图基础模型框架，专门用于处理多表多列的关系数据并在时序异构图上进行训练和推理。项目集成了 RelBench 数据集、真实的 RelGT（Relational Graph Transformer）编码器以及上下文学习（ICL）预测头，能够在无需编写任务专用代码的情况下完成多种预测任务。

> **说明**：原始代码基于 “Kumo” 命名，本版本已统一更名为 “RGFM”，功能保持不变。

## 核心特性

- **RelBench 数据支持**：自动加载并转换官方 RelBench 数据集，构建时序异构图并完成任务配置。
- **RelGT 五元素编码**：直接接入 RelGT 官方 LocalModule，通过特征、类型、跳数、时间、结构等五种元素形成子图序列后再编码。
- **ICL 预测头**：结合上下文采样、双重上下文注意力和标签编码，实现分类、回归或链接预测等任务。
- **Dry-Run 流水线**：无需 RelBench 数据即可运行完整训练流程，方便调试与验证。
- **健壮的训练工具**：AdamW 优化器、余弦学习率调度、指标记录、模型保存以及可调超参数。


1. **数据适配器**（`relbdata/adapter.py`）  
   加载 RelBench 数据、推断数据库模式、构建时序异构图、生成训练/验证/测试拆分。
2. **在线上下文标签生成**（`sampling/context_label_table.py`）  
   `ForwardLabelSampler` 将 train/val 标签导入 `InContextLabelTable`，支持 `uniform`、`most_recent`、`fixed_interval` 策略。
3. **子图采样**（`sampling/backward_sampler.py`, `sampling/context_sampler.py`）  
   Backward 采样器抽取目标子图；ContextSampler 优先使用标签表生成上下文，缺失时回退旧策略。
4. **RelGT 编码**（`models/kumorfm.py`）  
   表级特征合并成 token，叠加节点类型/跳数/时间编码后输入真实的 RelGT LocalModule。
5. **ICL 模块**（`models/icl/`）  
   构建上下文序列、编码标签、通过 ICL Transformer 与任务头得到 logits / 输出。
6. **训练脚本**（`relbdata/train_on_relbench.py`）  
   统一的 CLI，支持 dry-run、自定义所有核心超参，并将模型/日志/结果写入 `relbench_outputs/`。

## 安装步骤

```bash
# 克隆仓库
git clone https://github.com/JeromeZzz/RGFM.git
cd RGFM

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装本项目
pip install -e .
```

### 可选：安装官方 RelGT 仓库

```bash
git clone https://github.com/snap-stanford/relgt.git models/relgt/relgt
pip install -r models/relgt/relgt/requirements.txt
```

> 如果环境中缺少 `torch_geometric` / `torch_frame`，RGFM 会自动退回到内置的轻量编码器以保证流程可运行。

## 准备 RelBench 数据

1. 按官方指南下载 RelBench 数据集，放置在 `relbdata/adapter.py` 可访问的位置（默认 `~/.relbench` 或通过环境变量指定目录）。
2. 支持的数据集名称：`amazon`、`stack`、`f1`、`trial`、`avito`、`event`、`hm`（公共版建议使用 `trial`）。
3. 任务名称依据数据集而定，参见 `relbdata/__init__.py:DATASET_TASKS`；若不确定，可使用 `--task auto` 自动推断或回退。

## 运行训练

主要入口为 `relbdata/train_on_relbench.py`。

### 1）Dry-Run（无需 RelBench 数据）

```bash
python -u relbdata/train_on_relbench.py \
  --dataset trial --task auto \
  --epochs 1 --batch-size 4 \
  --device cpu --dry-run
```

该命令会构建一个极小图，跑完训练/验证/测试流程，并在 `relbench_outputs/trial_auto_<时间戳>/` 下生成输出。

### 2）RelBench 真实训练（GPU）

```bash
python -u relbdata/train_on_relbench.py \
  --dataset trial \
  --task auto \
  --hidden-dim 256 \
  --num-layers 4 \
  --num-heads 8 \
  --dropout 0.3 \
  --epochs 20 \
  --batch-size 16 \
  --lr 2e-4 \
  --early-stopping-patience 6 \
  --device cuda \
  --seed 42 \
  --output-dir ./relbench_outputs \
  --dry-run             # 如需快速链路验证可开启，真实训练请移除
```

可调节参数总览：

| 参数 | 说明 |
| --- | --- |
| `--dataset {amazon,stack,f1,trial,avito,event,hm}` | RelBench 数据集名称 |
| `--task TASK` | 任务名称（`auto` 自动推断） |
| `--hidden-dim` | 模型隐藏维度 |
| `--num-layers` | RelGT 层数 |
| `--num-heads` | 注意力头数 |
| `--dropout` | Dropout 概率 |
| `--epochs` | 训练轮数 |
| `--batch-size` | 批大小 |
| `--lr` | 学习率（AdamW） |
| `--early-stopping-patience` | 早停耐心 |
| `--device {cuda,cpu}` | 训练设备 |
| `--seed` | 随机种子 |
| `--output-dir` | 结果/模型输出目录 |
| `--dry-run` | 启用小型合成数据验证链接 |

常用超参数范围：

| 参数 | 推荐区间 | 说明 |
| --- | --- | --- |
| `--hidden-dim` | 128–384 | RelGT 与 ICL 层的宽度 |
| `--num-layers` | 2–6 | RelGT 层数 |
| `--num-heads` | 4–8 | 注意力头数 |
| `--dropout` | 0.1–0.5 | 正则力度，过拟合时调高 |
| `--lr` | 1e-4 – 3e-4 | AdamW 学习率 |
| `--batch-size` | 8–64 | 视显存而定 |

### 3）定位 CUDA 设备断言

在 PowerShell 中启用同步执行：

```powershell
$env:CUDA_LAUNCH_BLOCKING = '1'
python -u relbdata/train_on_relbench.py --dataset trial --task auto --epochs 1 --batch-size 8 --device cuda
Remove-Item Env:CUDA_LAUNCH_BLOCKING -ErrorAction SilentlyContinue
```

## 仓库结构

```
RGFM/
├── config/                 # 模型与任务配置
├── data/                   # 时序图数据结构
├── relbdata/               # RelBench 适配器、训练脚本、分析工具
├── sampling/               # 子图与上下文采样
├── models/
│   ├── kumorfm.py          # RGFM 主模型（RelGT + ICL）
│   ├── encoders/           # 表/多模态编码器
│   ├── relgt/              # RelGT 集成与封装
│   └── icl/                # 上下文学习模块
├── training/               # 通用训练工具
├── inference/              # 推理接口（可选）
├── utils/                  # 日志与训练辅助
└── examples/               # 示例脚本
```

## 全流程示意

```
多源关系数据库
       │
       ▼
数据预处理 → 图表示 G(V,E)
       │
       ▼
表无关编码器 → 多模态嵌入
       │
       ▼
Relational Graph Transformer
       │
       ▼
上下文采样生成 {(G≤t̂[ê], ŷ)}
       │
       ▼
上下文学习训练 (预训练)
       │
       ├──► 推理阶段 (ICL)
       │       └─ 动态预测 + 可解释输出
       │
       └──► 微调阶段 (Fine-tuning)
               └─ 任务特化训练 + 缓存加速
```



## 常见问题与排查

- **无法找到数据集**：确认 RelBench 数据已下载且路径正确，适配器会依次尝试 `trial`、`rel-trial`、`stack` 等名称。
- **标签越界**：RGFM 会将分类标签映射为连续索引并动态扩展标签嵌入，如仍报错，可检查 `relbdata/train_on_relbench.py:569-572` 的推断逻辑。
- **子图过小导致 RelGT 不稳定**：封装器会在序列长度小于 2 时自动复制 token 并在小批量下切换到 eval 模式。如图过于稀疏，请增大采样跳数或邻居数。
- **仅需推理**：可使用 `inference/predictor.py` 加载已训练模型执行预测（同样以 RGFM 命名）。

## 许可证

本项目以 MIT License 方式开源，详细内容见 [`LICENSE`](LICENSE)。
